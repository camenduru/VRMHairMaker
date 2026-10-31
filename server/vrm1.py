"""VRM 1.0 spring-bone injection for exported hair.

The Blender VRM add-on keeps a VRM 1.0 body in VRM 1.0 on export: the generated hair mesh and its HairGen_* bones
round-trip, but the VRM 0.x spring groups (hairdeck/blender/vrm_hair writes vrm0.secondaryAnimation) do not become
VRMC_springBone springs. This module adds them back into the exported glTF JSON directly: one spring per generated
chain, joints = the chain's segment bones (named HairGen_c{c:02d}_{k:02d} by vrm_hair), with the chain's predicted
stiffness and the export's spring constants (drag 0.4, hit radius 0.02, no gravity). The body's own collider groups
that hang on the head node are attached to the springs, so the hair still avoids the head and chest.
"""
import json
import struct


def _json_chunk(data):
    jl = struct.unpack('<I', data[12:16])[0]
    return json.loads(data[20:20 + jl]), data[20 + jl:]


def _write_chunk(j, rest):
    js = json.dumps(j, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    js += b' ' * (-len(js) % 4)
    body = struct.pack('<II', len(js), 0x4E4F534A) + js + rest
    return struct.pack('<III', 0x46546C67, 2, 12 + len(body)) + body


def inject_springs(path, rig):
    """Add the generated chains as VRMC_springBone springs of an exported VRM 1.0 file (in place).

    rig: the /api/hairrig dict (chainPtr over the kept chains, stiffness per chain) — the same chains the Blender
    export turned into HairGen_* bones.
    """
    data = open(path, 'rb').read()
    j, rest = _json_chunk(data)
    ext = j.setdefault('extensions', {})
    sb = ext.setdefault('VRMC_springBone', {'specVersion': '1.0', 'colliders': [], 'colliderGroups': [], 'springs': []})
    if 'VRMC_springBone' not in j.setdefault('extensionsUsed', []):
        j['extensionsUsed'].append('VRMC_springBone')

    nodes = j.get('nodes', [])
    by_name = {}
    for i, n in enumerate(nodes):
        by_name.setdefault(n.get('name'), i)
    humanoid = j.get('extensions', {}).get('VRMC_vrm', {}).get('humanoid', {})
    head_node = (humanoid.get('humanBones', {}).get('head') or {}).get('node')
    colliders = sb.get('colliders', [])
    # a collider group counts as a head collider when any of its colliders sits on the head node
    # (the add-on writes groups as {name, colliders: [indices into sb['colliders']]}, each collider has a node)
    head_groups = []
    if head_node is not None:
        for gi, cg in enumerate(sb.get('colliderGroups', [])):
            if any(isinstance(ci, int) and ci < len(colliders) and colliders[ci].get('node') == head_node
                   for ci in cg.get('colliders', [])):
                head_groups.append(gi)

    ptr, stiff = rig['chainPtr'], rig['stiffness']
    added = 0
    for c in range(rig['nChains']):
        n = ptr[c + 1] - ptr[c]          # joints of the chain (cleaned); segments = n - 1, plus the _end leaf
        if n < 2:
            continue
        names = [f'HairGen_c{c:02d}_{k:02d}' for k in range(n - 1)]
        if any(nm not in by_name for nm in names):
            continue
        s = stiff[c] if c < len(stiff) else 0.5
        spring = dict(name=f'HairGen_{c:02d}', joints=[
            dict(node=by_name[nm], hitRadius=0.02, stiffness=float(s), gravityPower=0.0,
                 gravityDir=dict(x=0, y=0, z=-1), dragForce=0.4)
            for nm in names], colliderGroups=head_groups)
        if head_groups:
            spring['center'] = None
        sb.setdefault('springs', []).append(spring)
        added += 1
    open(path, 'wb').write(_write_chunk(j, rest))
    return added
