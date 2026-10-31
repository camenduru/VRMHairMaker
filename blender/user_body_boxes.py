"""Rest-pose boxes of a user-uploaded avatar body for the VRM hair maker (Blender 4.2 LTS with the VRM Add-on, MIT).

Like blender/body_boxes.py, but tolerant of avatars that do not follow the VRoid Studio mesh naming:

  hair_rest_bbox    bbox of hair-named meshes when present ('hair' in the object or mesh name), else omitted
  face_rest_bbox    bbox of face-named meshes when present ('face' in the object or mesh name); when no mesh is
                    named that way, the bbox of the mesh vertices in the head region around the humanoid head bone
                    (head_face_method = 'head_region', approximate)
  head_rest_matrix  rest matrix of the humanoid head bone in the Blender world
  head_bone         its bone name
  vrm_version       '0.x' when the file (or the imported armature) carries VRM 0.x data, else '1.0'

The server combines this with an identity Blender -> dataset similarity (the body is used exactly as the add-on
imports it), so the dataset hair.glb alignment of hairdeck.data.head_frame.fit_one is not needed here.

blender -b --python user_body_boxes.py -- VRM OUT_JSON
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'code', 'blender'))  # HairDeck's vrm_scene
import vrm_scene as S  # noqa: E402
import bpy  # noqa: E402


def _vrm_version(arm):
    """The file's VRM version as the add-on sees it (vrm1 data present -> '1.0', else the vrm0 struct -> '0.x')."""
    try:
        ext = arm.data.vrm_addon_extension
        v1 = getattr(ext, 'vrm1', None)
        if v1 is not None and getattr(v1.humanoid, 'human_bones', None):
            return '1.0'
        v0 = getattr(ext, 'vrm0', None)
        if v0 is not None and getattr(v0.humanoid, 'human_bones', None):
            return '0.x'
    except Exception:  # noqa: BLE001
        pass
    return 'unknown'


def _human_bones(arm):
    """{role: bone_name} of the humanoid map, vrm0 first (the export path's format), vrm1 as the fallback."""
    try:
        hb = S.human_bones(arm)
        if hb:
            return hb
    except Exception:  # noqa: BLE001
        pass
    out = {}
    try:
        for hb1 in arm.data.vrm_addon_extension.vrm1.humanoid.human_bones:
            if getattr(hb1, 'bone_name', None):
                out[str(hb1.human_bone_name).rsplit('.', 1)[-1].lower()] = hb1.bone_name
    except Exception:  # noqa: BLE001
        pass
    return out


def _head_bone(arm):
    hb = _human_bones(arm)
    if 'head' in hb:
        return hb['head']
    for guess in ('J_Bip_C_Head', 'head', 'Head', 'head_top'):
        if guess in arm.data.bones:
            return guess
    raise RuntimeError('no humanoid head bone in ' + arm.name)


def _mesh_pts(objs):
    objs = [o for o in objs if o.type == 'MESH']
    if not objs:
        return None
    return S.eval_points(objs)


def _face_meshes():
    """Mesh objects whose object or mesh-data name contains 'face' (VRoid Studio: 'Face')."""
    out = []
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        if 'face' in o.name.lower() or 'face' in (o.data.name or '').lower():
            out.append(o)
    return out


def _head_region_box(head_world, H):
    """Approximate face bbox from the mesh vertices around the head bone. Calibration (w/h/d ratios and the z band)
    comes from VRoid 8-head proportions: the face mesh spans roughly from the brow line to the chin plus the hairline,
    width ~0.73 of the head-region width. Returns (lo, hi) in the Blender world."""
    pts = _mesh_pts([o for o in bpy.data.objects if o.type == 'MESH'])
    if pts is None or len(pts) == 0:
        return None
    hx, hy, hz = head_world
    # vertical band around the head: head joint to +0.115 H (skull top ~ 0.13 H above the joint on 8-head rigs)
    band = pts[(pts[:, 2] > hz - 0.02 * H) & (pts[:, 2] < hz + 0.115 * H)]
    # keep the blob near the head axis (drops shoulders/ears of long hair passing the band)
    near = band[np.linalg.norm(band[:, :2] - np.array([hx, hy]), axis=1) < 0.11 * H]
    if len(near) < 50:
        return None
    lo, hi = near.min(0), near.max(0)
    c, e = (lo + hi) / 2, hi - lo
    # face bbox is narrower and lower than the whole head blob (empirical on VRoid avatars)
    return np.array([c[0] - e[0] * 0.365, c[1] - e[1] * 0.365, hz - 0.005 * H]), \
        np.array([c[0] + e[0] * 0.365, c[1] + e[1] * 0.365, hz + 0.085 * H])


def main():
    vrm, out = sys.argv[sys.argv.index('--') + 1:][:2]
    arm = S.import_vrm(vrm)
    version = _vrm_version(arm)
    hb = _head_bone(arm)
    M = arm.matrix_world @ arm.data.bones[hb].matrix_local
    head_world = np.array(M.translation, np.float64)

    hair = S.hair_objects()
    meta = dict(vrm=vrm, vrm_version=version, head_bone=hb, face_method='face_mesh',
                head_rest_matrix=[list(map(float, r)) for r in M])
    if hair:
        pts = _mesh_pts(hair)
        meta['n_hair_objects'] = len(hair)
        meta['hair_objects'] = [o.name for o in hair]
        meta['hair_rest_bbox'] = [pts.min(0).tolist(), pts.max(0).tolist()]

    face = _face_meshes()
    fp = _mesh_pts(face) if face else None
    if fp is not None and len(fp) > 0:
        meta['face_objects'] = [o.name for o in face]
        meta['face_rest_bbox'] = [fp.min(0).tolist(), fp.max(0).tolist()]
    else:  # not VRoid-named: the approximate head-region box
        H = None
        pts_all = _mesh_pts([o for o in bpy.data.objects if o.type == 'MESH'])
        if pts_all is not None and len(pts_all):
            H = float(pts_all[:, 2].max() - pts_all[:, 2].min())
        box = _head_region_box(head_world, H or 1.6)
        if box is None:
            raise RuntimeError('no face meshes and no head-region vertices found')
        meta['face_method'] = 'head_region'
        meta['face_rest_bbox'] = [box[0].tolist(), box[1].tolist()]

    json.dump(meta, open(out, 'w'), indent=1)
    print('BODY_META', json.dumps(meta))


main()
