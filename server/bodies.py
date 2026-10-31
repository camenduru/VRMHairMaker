"""Avatar bodies (VRM files) the hair can be attached to, and the head-frame transform of each one.

A body is a folder data/bodies/<id>/ with model.vrm, meta.json (hairmaker/blender/user_body_boxes.py output) and
body.json in the hairdeck.vrm.body format: h, key, vrm, dir, fit, k, b, file_tag. Unlike hairdeck.vrm.body.build,
which matches the body's Blender hair bbox to its CHARM-dataset hair.glb, the fit here is the identity similarity
(dataset = axis_map(Blender world); the add-on imports the VRM exactly as it is used), so any VRM 0.x avatar works
without a dataset extraction; the face bbox comes from a face-named mesh (VRoid 'Face') or the approximate
head-region box (face_method 'head_region').
"""
import json
import os
import re
import shutil
import struct
import subprocess
import threading
import time
import uuid

import numpy as np

from . import config
from .config import BODIES_DIR, BLENDER_SCRIPTS

SX, SZ = 1, -1  # axis signs of the glTF export (dataset = s * (Bx, Bz, -By) + t), as in hairdeck.data.head_frame
_LOCK = threading.Lock()
_TABLES = {}  # id -> body dict cache


def axis_map(p):
    """Blender world (x, y, z), Z up -> dataset axes (x, y, z), Y up, face toward +z."""
    p = np.asarray(p, np.float64)
    return np.stack([SX * p[..., 0], p[..., 2], SZ * p[..., 1]], -1)


def anchor(f):
    """Head-frame origin of a fit: face-bbox centre in x and z, face-bbox top in y (hairdeck.data.head_frame.anchor)."""
    o = np.array(f['o'], np.float64)
    return np.array([o[0], o[1] + f['h'] / 2, o[2]])


def canonical_head():
    """W, O of the canonical head (data/head_frame_train.npz)."""
    from hairdeck import paths
    hz = np.load(paths.data('head_frame_train.npz'))
    return float(hz['W']), hz['O'].astype(np.float64)


def vrm_version(path):
    """'1.0' or '0.x' from the glTF JSON chunk's extensionsUsed (VRMC_vrm vs VRM), 'unknown' when unreadable."""
    try:
        with open(path, 'rb') as f:
            magic, ver, length = struct.unpack('<III', f.read(12))
            if magic != 0x46546C67:
                return 'unknown'
            jl, jt = struct.unpack('<II', f.read(8))
            if jt != 0x4E4F534A:
                return 'unknown'
            g = json.loads(f.read(jl))
        used = g.get('extensionsUsed', [])
        if 'VRMC_vrm' in used:
            return '1.0'
        if 'VRM' in used:
            return '0.x'
        return 'unknown'
    except Exception:  # noqa: BLE001
        return 'unknown'


def _tag(name):
    """File tag of an upload: its stem, ascii-safe."""
    stem = re.sub(r'\.[vV][rR][mM]$', '', os.path.basename(name or 'model'))
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('._') or 'model'
    return stem[:40]


def _colliders_world(path, version):
    """Head-region colliders of a VRM in the browser world frame (what hairdeck axes call the dataset frame:
    y up, face +z). The file's node transforms already place the model as three.js loads it; VRM 0.x files are
    shown rotated 180 deg about Y (VRMUtils.rotateVRM0), so their colliders get the same rotation. Returns
    (colliders, head_file_pos, head_world_pos) — the head node position in both frames, for verification."""
    colliders, head_file = parse_head_colliders(path)
    if version == '0.x':
        for c in colliders:
            c['center'][0] = round(-c['center'][0], 5)
            c['center'][2] = round(-c['center'][2], 5)
        head_file = [-head_file[0], head_file[1], -head_file[2]]
    return colliders, head_file


def make_body_json(mid, meta, vrm_path, h_index):
    """body.json fields of one meta.json (user_body_boxes.py output): identity fit + k, b."""
    fb = np.array(meta['face_rest_bbox'], np.float64)
    fc = axis_map((fb[0] + fb[1]) / 2)
    fe = np.abs(axis_map(fb[1] - fb[0]))
    hj = axis_map(np.array(meta['head_rest_matrix'])[:3, 3])
    fit = dict(status='ok', o=fc.tolist(), w=float(fe[0]), h=float(fe[1]), d=float(fe[2]), head_joint=hj.tolist(),
               sx=SX, sz=SZ, s=1.0, t=[0.0, 0.0, 0.0], s_axes=[1.0, 1.0, 1.0], resid=0.0,
               method=meta.get('face_method', 'face_mesh'))
    if 'hair_rest_bbox' in meta:
        fit['hair_lo'] = axis_map(meta['hair_rest_bbox'][0]).tolist()
        fit['hair_hi'] = axis_map(meta['hair_rest_bbox'][1]).tolist()
    W, O = canonical_head()
    k = W / fit['w']
    b = O - k * anchor(fit)
    return dict(h=int(h_index), key=mid, vrm=os.path.abspath(vrm_path), dir=os.path.dirname(os.path.abspath(vrm_path)),
                fit=fit, k=float(k), b=b.tolist(), file_tag=_tag(mid) if mid.startswith(('default', 'avatar')) else _tag(mid) + '_' + mid[-6:])


def blender_body_meta(vrm_path, out_json):
    """Run user_body_boxes.py on a VRM (Blender) and return its meta dict."""
    cmd = [os.environ['HAIRDECK_BLENDER'], '-b', '--python', os.path.join(BLENDER_SCRIPTS, 'user_body_boxes.py'),
           '--', vrm_path, out_json]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not os.path.exists(out_json):
        raise RuntimeError('user_body_boxes failed: ' + (r.stdout + r.stderr)[-2000:])
    return json.load(open(out_json))


def _write(mid, body):
    d = os.path.join(BODIES_DIR, mid)
    json.dump(body, open(os.path.join(d, 'body.json'), 'w'), indent=1)


def _read(mid):
    d = os.path.join(BODIES_DIR, mid)
    b = json.load(open(os.path.join(d, 'body.json')))
    b.update(vrm=os.path.abspath(b['vrm']), dir=os.path.abspath(b['dir']), k=float(b['k']),
             b=[float(x) for x in b['b']], h=int(b['h']))
    meta = {}
    mp = os.path.join(d, 'meta.json')
    if os.path.exists(mp):
        meta = json.load(open(mp))
    b['meta'] = meta
    return b


def add_default():
    """Seed the two built-in bodies: 'default' (the rtmlib sample avatar, body-built here) and 'avatar' (HairDeck's
    own calibrated sample avatar). Existing folders are kept."""
    os.makedirs(BODIES_DIR, exist_ok=True)
    d = os.path.join(BODIES_DIR, 'default')
    if not os.path.exists(os.path.join(d, 'body.json')):
        os.makedirs(d, exist_ok=True)
        vrm = os.path.join(d, 'model.vrm')
        if not os.path.exists(vrm):
            with open(config.DEFAULT_MODEL_SRC, 'rb') as s, open(vrm, 'wb') as o:
                o.write(s.read())
        if not os.path.exists(os.path.join(d, 'meta.json')):
            meta = blender_body_meta(vrm, os.path.join(d, 'meta.json'))
        else:
            meta = json.load(open(os.path.join(d, 'meta.json')))
        body = make_body_json('default', meta, vrm, 1)
        body['name'] = 'Sample Avatar B'
        body['builtin'] = True
        _write('default', body)
    d2 = os.path.join(BODIES_DIR, 'avatar')
    if not os.path.exists(os.path.join(d2, 'body.json')):
        os.makedirs(d2, exist_ok=True)
        from hairdeck import paths
        src = json.load(open(paths.data('avatar', 'body.json')))
        vrm2 = os.path.abspath(paths.resolve(src['vrm']))
        local2 = os.path.join(d2, 'model.vrm')
        if not os.path.exists(local2):
            shutil.copy2(vrm2, local2)   # served at /bodies/avatar/model.vrm
        body = dict(h=900, key='avatar', vrm=local2,
                    dir=os.path.abspath(paths.resolve(src['dir'])), fit=src['fit'], k=float(src['k']),
                    b=[float(x) for x in src['b']], file_tag='avatar')
        body['name'] = 'HairDeck avatar'
        body['builtin'] = True
        _write('avatar', body)


def _summary(mid, b):
    meta = b.get('meta') or {}
    fit = b.get('fit') or {}
    return dict(id=mid, name=b.get('name') or b.get('key', mid), file_tag=b['file_tag'], k=b['k'], b=b['b'],
                fit=dict(s=fit.get('s', 1.0), t=fit.get('t', [0.0, 0.0, 0.0]),
                         sx=fit.get('sx', SX), sz=fit.get('sz', SZ)),
                head_joint=fit.get('head_joint'),
                vrm_version=meta.get('vrm_version') or vrm_version(b['vrm']),
                face_method=(b['fit'] or {}).get('method', 'face_mesh'),
                head_bone=meta.get('head_bone'), n_hair_objects=meta.get('n_hair_objects'),
                url=f'/bodies/{mid}/model.vrm')


def build_uploaded(mid, filename, data):
    """Save an uploaded VRM and build its body files (Blender run included, ~30-60 s). Returns the summary dict."""
    d = os.path.join(BODIES_DIR, mid)
    os.makedirs(d, exist_ok=True)
    vrm = os.path.join(d, 'model.vrm')
    with open(vrm, 'wb') as f:
        f.write(data)
    ver = vrm_version(vrm)
    try:
        meta = blender_body_meta(vrm, os.path.join(d, 'meta.json'))
    except Exception as e:  # noqa: BLE001
        body = dict(h=None, key=mid, vrm=vrm, dir=d, fit=None, k=None, b=None, file_tag=_tag(mid),
                    name=_tag(filename), error=f'Body fit failed: {e}', vrm_version=ver)
        _write(mid, body)
        raise
    body = make_body_json(mid, meta, vrm, _next_h())
    body['name'] = _tag(filename)
    _write(mid, body)
    return summary(mid)


def _next_h():
    used = []
    for mid in os.listdir(BODIES_DIR):
        p = os.path.join(BODIES_DIR, mid, 'body.json')
        if os.path.exists(p):
            try:
                used.append(int(json.load(open(p)).get('h') or 0))
            except Exception:  # noqa: BLE001
                pass
    return max([1000] + used) + 1


def summary(mid):
    with _LOCK:
        if mid not in _TABLES:
            _TABLES[mid] = _read(mid)
    return _summary(mid, _TABLES[mid])


def body(mid):
    """Full body dict (hairdeck.vrm.body compatible, absolute paths)."""
    with _LOCK:
        if mid not in _TABLES:
            _TABLES[mid] = _read(mid)
        return _TABLES[mid]


def body_path(mid):
    return os.path.join(BODIES_DIR, mid, 'body.json')


def all_bodies():
    out = []
    if os.path.isdir(BODIES_DIR):
        for mid in sorted(os.listdir(BODIES_DIR)):
            if os.path.exists(os.path.join(BODIES_DIR, mid, 'body.json')):
                try:
                    b = _read(mid)
                except Exception:  # noqa: BLE001
                    continue
                out.append(_summary(mid, b))
    return out


def new_upload_id():
    return 'u' + uuid.uuid4().hex[:8]


def wait_for_build(mid, timeout=600):
    """Block until body.json of mid exists (a build that runs in another thread)."""
    t = time.time()
    p = os.path.join(BODIES_DIR, mid, 'body.json')
    while not os.path.exists(p):
        if time.time() - t > timeout:
            raise TimeoutError(f'body build of {mid} timed out')
        time.sleep(0.5)
    return summary(mid)
