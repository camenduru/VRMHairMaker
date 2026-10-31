"""The HairDeck image pipeline (hairdeck.infer.image.Pipeline) behind a lock, plus the spring-rig data of a generated
result on a body (hairdeck.vrm.prep.prepare) for the in-browser spring-bone preview."""
import json
import os
import threading

import numpy as np

from . import config  # noqa: F401  (sets HAIRDECK_* before hairdeck imports)

_LOCK = threading.Lock()
_EXPORT_LOCK = threading.Lock()
_P = None


def pipeline():
    """The loaded Pipeline (loads on first use, ~1 min: SAM 3, BiRefNet, DINOv3, the image and chain DiTs)."""
    global _P
    with _LOCK:
        if _P is None:
            from hairdeck.infer import image as PL
            # results land directly under HAIRDECK_RESULTS (served at /results/<out_id>)
            _P = PL.Pipeline(out_root=os.environ['HAIRDECK_RESULTS'])
        return _P


def chain_error():
    return getattr(pipeline(), 'chain_error', None)


def generate(image_path, seed=0, steps=50, cfg=2.0, n_samples=1, chains=True, out_dir=None):
    """One generate call; the prep panel is written next to the result as prep.png. Returns the run() dict."""
    p = pipeline()
    with _LOCK:
        r = p.run(image_path, seed=int(seed), steps=int(steps), cfg=float(cfg), n_samples=int(n_samples),
                  chains=bool(chains), out_dir=out_dir)
    vis = r.get('prep_vis')
    if vis is not None:
        try:
            vis.save(os.path.join(r['out_dir'], 'prep.png'))
        except Exception:  # noqa: BLE001
            pass
    return r


def add_chains(out_dir, seed):
    p = pipeline()
    with _LOCK:
        return p.add_chains(out_dir, int(seed))


def _round(a, n=6):
    return np.round(np.asarray(a, np.float64), n).tolist()


def hair_rig(out_dir, seed, body):
    """Skinned hair mesh + chains of one sample on one body, in the body's dataset frame (y up, face +z, metres) —
    the input of the browser spring-bone preview. Uses hairdeck.vrm.prep (the same geometric skinning the Blender
    export binds), cached per (out_dir, seed, body). Returns a JSON-able dict."""
    from hairdeck.vrm import prep as VP
    from hairdeck.infer.glb import srgb_to_linear
    d = os.path.join(out_dir, f'rig_seed{seed}', body['file_tag'])
    cache = os.path.join(d, 'rig.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    os.makedirs(d, exist_ok=True)
    info = json.load(open(os.path.join(out_dir, 'info.json')))
    colour = info['hair_color']
    VP.prepare(out_dir, int(seed), body, srgb_to_linear(colour), os.path.join(d, 'run'), d)
    z = np.load(os.path.join(d, f"{body['file_tag']}_prep.npz"))
    V = np.asarray(z['V'], np.float64)
    V_ds = np.stack([V[:, 0], V[:, 2], -V[:, 1]], -1)  # Blender world -> dataset (identity fit)
    hj = np.asarray(z['head_joint'], np.float64)
    hj_ds = np.array([hj[0], hj[2], -hj[1]])
    vch = z['vchain'].astype(np.int32)
    vseg = z['vseg'].astype(np.int32)
    vseg2 = z['vseg2'].astype(np.int32)
    vw = np.asarray(z['vw'], np.float64)
    val = np.asarray(z['valpha'], np.float64)
    chain_ptr = z['chain_ptr'].astype(np.int32)
    # chain joints in the same frame as the vertices: the npz keeps them in Blender world (chain_J) and in the
    # body's dataset frame (chain_J_ds, which only coincides for an identity fit) — the browser needs Blender world
    Jb = np.asarray(z['chain_J'], np.float64)
    J = np.stack([Jb[:, 0], Jb[:, 2], -Jb[:, 1]], -1)  # same Blender -> dataset/world map as V
    n_chains = len(chain_ptr) - 1
    # per chain: the bone index of its segment 0 inside the global bone list (bone 0 = head bone)
    cum = np.cumsum(np.diff(chain_ptr))  # cum[c] = joints of chains 0..c
    offs = np.concatenate([[1], cum[:-1] + 1]).tolist()
    # approximate skull collider for the browser spring sim (the Blender export reuses the body's own VRM
    # colliders instead): a sphere over the skull, radius from the VRoid head-collider ratio (~0.55 face widths,
    # 0.102 m on the sample avatar); chains rooted inside it get a rigid prefix there, like the export's rigid roots
    W_m = float(z['W']) if 'W' in z else 0.15
    radius = 0.55 * W_m
    center = [float(hj_ds[0]), float(hj_ds[1]) + 0.6 * W_m, float(hj_ds[2])]
    rig = dict(
        V=_round(V_ds, 5), F=z['F'].astype(np.int32).tolist(), UV=_round(z['UV'], 5),
        vchain=vch.tolist(), vseg=vseg.tolist(), vseg2=vseg2.tolist(), vw=_round(vw, 4), valpha=_round(val, 4),
        chainOffsets=[0] + offs,  # bone index of segment 0 of chain c = 1 + sum of previous joint counts
        J=_round(J, 5), chainPtr=chain_ptr.tolist(), stiffness=_round(z['stiffness'], 4),
        nChains=int(n_chains), headJoint=_round(hj_ds, 5),
        headCollider=dict(center=[round(c, 5) for c in center], radius=round(radius, 5),
                          prefixRadius=round(radius + 0.02, 5)),  # +HIT like the export's rigid-root test
        colourSrgb=[round(float(c), 4) for c in colour],
        colourLinear=[round(float(c), 6) for c in np.asarray(z['colour'], np.float64)],
        nCards=int(len(z['vcard'])) if 'vcard' in z else 0)
    with open(cache, 'w') as f:
        json.dump(rig, f)
    return rig
