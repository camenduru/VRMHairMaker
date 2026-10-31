// VRM hair maker — browser app.
// VRM preview follows the rtmlib web app (three.js + @pixiv/three-vrm v3: VRMLoaderPlugin, rotateVRM0, VRMA idle,
// ACES tone mapping, orbit controls). Hair comes from the HairDeck pipeline: GLB cards in the head frame are placed
// on the avatar with the body's head-frame transform (p_dataset = (p_head - b) / k), and the spring-bone preview
// runs the verlet sim of spring.js on the rig data of /api/hairrig (the same skinning the Blender export binds).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { createVRMAnimationClip, VRMAnimationLoaderPlugin } from '@pixiv/three-vrm-animation';
import { HairSim, motionFrame, NF, ident3 } from './spring.js';

const $ = (id) => document.getElementById(id);
window.__logs = [];
for (const level of ['error', 'warn']) {
  const orig = console[level].bind(console);
  console[level] = (...args) => {
    try { window.__logs.push(level + ': ' + args.map((a) => (a instanceof Error ? a.stack : String(a))).join(' ').slice(0, 500)); } catch {}
    orig(...args);
  };
}
const els = {
  vrmDrop: $('vrm-drop'), vrmFile: $('vrm-file'), vrmName: $('vrm-current-name'), vrmSub: $('vrm-current-sub'),
  vrmStatus: $('vrm-status'), vrmModels: $('vrm-models'),
  imageDrop: $('image-drop'), imageFile: $('image-file'), imagePreview: $('image-preview'), imageHint: $('image-hint'),
  examplesGrid: $('examples-grid'),
  gen: $('gen-btn'), pipelineStatus: $('pipeline-status'),
  seed: $('seed'), cfg: $('cfg'), cfgOut: $('cfg-out'),
  showHead: $('show-head'), showChains: $('show-chains'), motionButtons: $('motion-buttons'),
  variation: $('variation-pick'), viewerStatus: $('viewer-status'), badge: $('viewer-badge'),
  exportBtn: $('export-btn'), turnBtn: $('turn-btn'), wearBtn: $('wear-btn'), exportInfo: $('export-info'),
  fileList: $('file-list'), zipBtn: $('zip-btn'), prepVis: $('prep-vis'), prepNone: $('prep-none'),
};

const state = {
  models: [], bodyId: null, body: null,
  vrm: null, vrmUrl: null, mixer: null, idleClip: null,
  headNode: null, headRestInv: null,
  imageFile: null,
  result: null, outId: null, samples: [], seedIdx: 0,
  hair: null,           // { kind: 'glb'|'spring'|'exported', root, sim, posAttr, restGeo }
  motion: null, motionF: 0, motionHold: false,
  busy: false,
};

// ---------------------------------------------------------------- scene (rtmlib-style)
const canvas = $('viewer');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080c);
scene.fog = new THREE.Fog(0x07080c, 5.5, 14);

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 60);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.95, 0);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.minDistance = 0.8; controls.maxDistance = 12;
controls.maxPolarAngle = Math.PI / 2 - 0.02;
let userMovedCamera = false;
controls.addEventListener('start', () => { userMovedCamera = true; });

function fitCamera(force = false) {
  const box = new THREE.Box3();
  if (state.vrm) box.setFromObject(state.vrm.scene);
  else box.set(new THREE.Vector3(-0.4, 0, -0.4), new THREE.Vector3(0.4, 1.6, 0.4));
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const h = Math.max(size.y, 0.5), fov = (camera.fov * Math.PI) / 180;
  const dist = Math.min(14, Math.max(0.8, h / 2 / Math.tan(fov / 2) * 1.12));
  const dir = camera.position.clone().sub(controls.target);
  if (dir.lengthSq() < 1e-6 || force) dir.set(0, 0.35, 1).normalize();
  dir.normalize().multiplyScalar(dist);
  controls.target.copy(center);
  camera.position.copy(center).add(dir);
}
controls.addEventListener('change', () => {});

scene.add(new THREE.HemisphereLight(0x9dc8ff, 0x0a0c12, 1.15));
const key = new THREE.DirectionalLight(0xffffff, 2.6);
key.position.set(1.6, 3.2, 2.4);
key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.bias = -0.0006;
Object.assign(key.shadow.camera, { left: -2, right: 2, top: 3, bottom: -0.5 });
scene.add(key);
const rim = new THREE.DirectionalLight(0x62f2d2, 1.2);
rim.position.set(-2.2, 2.0, -2.6);
scene.add(rim);
const fill = new THREE.PointLight(0xff7aa8, 4.0, 10);
fill.position.set(-1.4, 1.2, 2.2);
scene.add(fill);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(2.6, 48).rotateX(-Math.PI / 2),
  new THREE.MeshStandardMaterial({ color: 0x12141c, roughness: 0.95, metalness: 0 }),
);
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(6, 24, 0x2c3140, 0x1a1e29);
grid.position.y = 0.002;
scene.add(grid);

const avatarRoot = new THREE.Group();
scene.add(avatarRoot);
function row3(m4) {
  // THREE.Matrix4.elements are column-major with te[4] = row1col2 (see Matrix4.set), so the row-major
  // 3x3 (row r, column c = e[c*4 + r]) is:
  const e = m4.elements;
  return [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]];
}

window.mocapDebug = { state, camera, controls, scene, avatarRoot, renderer, fitCamera: () => { userMovedCamera = false; fitCamera(true); } };

function resize() {
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = canvas.clientHeight || canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($('viewer-wrap'));

// ---------------------------------------------------------------- loaders
const gltfLoader = new GLTFLoader();
const vrmLoader = new GLTFLoader();
vrmLoader.register((parser) => new VRMLoaderPlugin(parser));
const vrmaLoader = new GLTFLoader();
vrmaLoader.register((parser) => new VRMAnimationLoaderPlugin(parser));

function loadGLTF(loader, url, onProgress) {
  return new Promise((resolve, reject) => {
    loader.load(url, resolve, onProgress, reject);
  });
}

async function loadIdle(url) {
  try {
    const gltf = await loadGLTF(vrmaLoader, url);
    const anim = gltf.userData.vrmAnimations?.[0];
    if (!anim || !state.vrm) return;
    const clip = createVRMAnimationClip(anim, state.vrm);
    if (state.mixer) { state.mixer.stopAllAction(); }
    state.mixer = new THREE.AnimationMixer(state.vrm.scene);
    state.mixer.clipAction(clip).play();
  } catch (e) {
    console.warn('[idle] animation unavailable', e);
  }
}

async function loadAvatar(url, { idle = true } = {}) {
  // remove the previous avatar and hair
  clearHair();
  if (state.vrm) {
    avatarRoot.remove(state.vrm.scene);
    VRMUtils.deepDispose(state.vrm.scene);
    state.vrm = null;
  }
  const gltf = await loadGLTF(vrmLoader, url, (e) => {
    if (e.lengthComputable) setBadge(`loading model ${Math.round((e.loaded / e.total) * 100)}%`);
  });
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error('not a VRM file');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  avatarRoot.add(vrm.scene);
  state.vrm = vrm;
  state.vrmUrl = url;
  state.mixer = null;
  // the raw head node drives the hair's rigid motion; captured in the rest pose
  vrm.scene.updateMatrixWorld(true);
  state.headNode = vrm.humanoid?.getRawBoneNode?.('head') || null;
  // captured at the authored rest pose, BEFORE the idle animation moves the rig: every later head motion
  // (idle sway included) is then expressed by the head-delta matrix the hair rig follows
  // the head's rest transform relative to the avatar root — the hair meshes are children of that root, so the
  // sim must track only the head's motion WITHIN it (a jump/sway moves the whole root and must not displace
  // the hair a second time through D)
  state.headRestInv = state.headNode
    ? new THREE.Matrix4().copy(avatarRoot.matrixWorld).invert().multiply(state.headNode.matrixWorld).invert()
    : null;
  if (vrm.lookAt && state.headNode) { /* keep default lookAt */ }
  setViewerEmpty(false);
  setOwnHairVisible(true);
  if (idle) await loadIdle('/models/idle_loop.vrma');
  fitCamera(true);
  userMovedCamera = false;
  setBadge(null);
  return vrm;
}

// ---------------------------------------------------------------- hair on the avatar
// head frame -> the model's world frame. p_dataset = (p_head - b) / k (hairdeck.vrm.prep.head_to_body), then the
// body's Blender->dataset similarity inverted (fit): p_world = ((p - b) / k - t) / s — the axis permutations of
// to_blender + axis_map cancel, leaving one uniform scale and shift.
function headFrameObject() {
  const k = state.body?.k ?? 1;
  const b = state.body?.b ?? [0, 0, 0];
  const fit = state.body?.fit || { s: 1, t: [0, 0, 0] };
  const sc = 1 / (k * fit.s);
  const root = new THREE.Group();
  root.scale.setScalar(sc);
  root.position.set(
    -b[0] * sc - fit.t[0] / fit.s,
    -b[1] * sc - fit.t[1] / fit.s,
    -b[2] * sc - fit.t[2] / fit.s,
  );
  return root;
}

function clearHair() {
  if (state.hair?.root) {
    avatarRoot.remove(state.hair.root);
    state.hair.root.traverse((o) => {
      o.geometry?.dispose?.();
      const m = o.material;
      if (m) (Array.isArray(m) ? m : [m]).forEach((x) => { x.map?.dispose?.(); x.dispose?.(); });
    });
  }
  state.hair = null;
  state.motion = null; state.motionF = 0;
  els.motionButtons.hidden = true;
  if (state.vrm) setOwnHairVisible(true);
}

async function showHairGLB(sample, { proxy, chains }) {
  clearHair();
  const url = chains ? (proxy ? sample.glb_chains : sample.glb_nohead_chains)
                     : (proxy ? sample.glb : sample.glb_nohead);
  if (!url) { setViewerStatus(); return; }
  const gltf = await loadGLTF(gltfLoader, url);
  const root = headFrameObject();
  root.add(gltf.scene);
  gltf.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  avatarRoot.add(root);
  root.updateMatrix();
  const rest = root.matrix.clone();
  root.matrixAutoUpdate = false;
  state.hair = { kind: 'glb', root, restMatrix: rest };
  setOwnHairVisible(false);
  setViewerEmpty(false);
  setViewerStatus();
}

async function showHairSpring(sample) {
  // rig data (the geometric skinning of hairdeck.vrm.prep) + the verlet sim, drawn over the avatar
  clearHair();
  setBadge('building spring rig…');
  let rig;
  try {
    const r = await fetch(`/api/hairrig/${state.outId}/${sample.seed}?vrm=${state.bodyId}`);
    if (!r.ok) throw new Error(await r.text());
    rig = await r.json();
  } catch (e) {
    console.warn(e);
    setBadge(null);
    toast('The spring rig could not be built for this model; showing the static hair.');
    await showHairGLB(sample, { proxy: els.showHead.checked, chains: els.showChains.checked });
    return;
  }
  if (!rig.V?.length) { setBadge(null); return showHairGLB(sample, { proxy: els.showHead.checked, chains: els.showChains.checked }); }
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(rig.V.length * 3);
  rig.V.forEach((v, i) => { pos[i * 3] = v[0]; pos[i * 3 + 1] = v[1]; pos[i * 3 + 2] = v[2]; });
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(rig.UV.flat()), 2));
  geo.setIndex(rig.F.flat());   // F arrives as a list of triangles
  geo.computeVertexNormals();
  const c = rig.colourSrgb;
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace),
    roughness: 0.75, metalness: 0, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true; mesh.frustumCulled = false;
  avatarRoot.add(mesh);          // dataset frame == world frame of the rotated VRM
  const sim = new HairSim(rig);
  state.hair = { kind: 'spring', root: mesh, sim, posAttr: geo.getAttribute('position') };
  setOwnHairVisible(false);
  setViewerEmpty(false);
  els.motionButtons.hidden = false;
  setBadge(null);
  setViewerStatus();
}

async function wearExported(vrmUrl) {
  clearHair();
  setBadge('loading exported VRM…');
  const gltf = await loadGLTF(vrmLoader, vrmUrl);
  if (state.vrm) {
    avatarRoot.remove(state.vrm.scene);
    VRMUtils.deepDispose(state.vrm.scene);
    state.vrm = null;
  }
  const vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  avatarRoot.add(vrm.scene);
  state.vrm = vrm;
  vrm.scene.updateMatrixWorld(true);
  state.headNode = vrm.humanoid?.getRawBoneNode?.('head') || null;
  // the head's rest transform relative to the avatar root — the hair meshes are children of that root, so the
  // sim must track only the head's motion WITHIN it (a jump/sway moves the whole root and must not displace
  // the hair a second time through D)
  state.headRestInv = state.headNode
    ? new THREE.Matrix4().copy(avatarRoot.matrixWorld).invert().multiply(state.headNode.matrixWorld).invert()
    : null;
  state.hair = { kind: 'exported', root: null };
  state.mixer = null;
  setViewerEmpty(false);
  await loadIdle('/models/idle_loop.vrma');
  els.motionButtons.hidden = false;   // the exported VRM simulates its own spring bones
  fitCamera(true);
  setBadge(null);
  setViewerStatus();
}

// ---------------------------------------------------------------- animation loop
const clock = new THREE.Clock();
const _m = new THREE.Matrix4();
const _piv = new THREE.Vector3();
const _r = new THREE.Matrix3();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, Math.max(0.001, clock.getDelta()));

  // motion override on the normalized head bone (the exported VRM's spring bones react via three-vrm).
  // turn: swing 0.35 s, hold ~1.2 s, then ease back to the rest pose over 0.6 s — an instant reset would whip
  // the spring hair while the head snaps, which reads as the hair not following the head.
  const RETURN_FRAMES = 36;
  if (state.motion && state.vrm) {
    const head = state.vrm.humanoid?.getNormalizedBoneNode('head');
    const g = motionGroup();
    if (state.motionF > NF) {   // the motion curve is over: hold, then ease back to rest
      state.returnF = (state.returnF ?? 0) + 1;
      const holdFrames = state.motion === 'turn' ? 72 : 12;
      if (state.returnF <= holdFrames) {
        // hold the final pose
      } else if (state.returnF <= holdFrames + RETURN_FRAMES && head) {
        const f = motionFrame(state.motion, NF);   // the pose the curve ended on
        const x = (state.returnF - holdFrames) / RETURN_FRAMES;
        const ease = x * x * (3 - 2 * x);
        if (state.motion === 'turn') head.rotation.y = f.angleY * (1 - ease);
        else if (state.motion === 'nod') head.rotation.x = f.angleX * (1 - ease);
        else if (state.motion === 'jump') g.position.y = f.T[1] * (1 - ease);
        else if (state.motion === 'sway') g.position.x = f.T[0] * (1 - ease);
      } else {
        stopMotion();   // full reset (head rotation 0, group position 0, springs keep settling)
      }
    } else {
      state.returnF = 0;
      const f = motionFrame(state.motion, state.motionF);
      state.motionF += 1;
      if (head) {
        if (state.motion === 'turn') head.rotation.y = f.angleY;
        else if (state.motion === 'nod') head.rotation.x = f.angleX;
      }
      if (state.motion === 'jump') g.position.y = f.T[1];
      else if (state.motion === 'sway') g.position.x = f.T[0];
    }
  }

  if (state.mixer && !state.motion) state.mixer.update(dt);  // the idle clip would fight the motion override
  if (state.vrm) state.vrm.update(dt);
  avatarRoot.updateMatrixWorld(true);

  // rigid head motion -> the hair follows (static GLB) or feeds the verlet sim (spring preview)
  const hair = state.hair;
  if (hair && state.headNode) {
    _m.copy(avatarRoot.matrixWorld).invert().multiply(state.headNode.matrixWorld).multiply(state.headRestInv);
    if (hair.kind === 'glb') {
      hair.root.matrix.multiplyMatrices(_m, hair.restMatrix);
    } else if (hair.kind === 'spring') {
      const R = row3(_m);
      // the rigid carry of a rest point p is D @ p = R @ (p - piv) + D @ piv; spring.js applies
      // R @ (p - piv) + piv + T, so T must be D @ piv - piv (NOT the translation column of D — the
      // head rotation acts about the model origin, 1.3 m from the head joint, and the difference
      // arcs the whole hair backward whenever the head pitches or yaws)
      _piv.set(...hair.sim.piv).applyMatrix4(_m);
      hair.sim.R = R;
      hair.sim.T = [_piv.x - hair.sim.piv[0], _piv.y - hair.sim.piv[1], _piv.z - hair.sim.piv[2]];
      hair.sim.step(R, hair.sim.T, dt);
      hair.sim.deform(hair.posAttr.array);
      hair.posAttr.needsUpdate = true;
    }
    // kind === 'exported': three-vrm's springBoneManager (inside vrm.update) does the work
  }
  controls.update();
  renderer.render(scene, camera);
}

function motionGroup() {
  return state.vrm ? state.vrm.scene.parent || avatarRoot : avatarRoot;
}

function stopMotion(silent = false) {
  state.motion = null; state.motionF = 0; state.returnF = 0;
  if (state.vrm) {
    const head = state.vrm.humanoid?.getNormalizedBoneNode('head');
    if (head) { head.rotation.set(0, 0, 0); }
  }
  motionGroup().position.set(0, 0, 0);
  if (state.hair?.sim) state.hair.sim.reset();
  if (!silent) document.querySelectorAll('.motion').forEach((b) => b.classList.remove('on'));
}
document.querySelectorAll('.motion').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.motion').forEach((x) => x.classList.remove('on'));
  b.classList.add('on');
  state.motion = b.dataset.motion;
  state.motionF = 0;
  if (state.hair?.sim) state.hair.sim.reset();
}));
$('view-reset').addEventListener('click', () => { userMovedCamera = false; fitCamera(true); });
$('motion-stop').addEventListener('click', () => {
  document.querySelectorAll('.motion').forEach((x) => x.classList.remove('on'));
  stopMotion();
});

// ---------------------------------------------------------------- UI helpers
function setBadge(text) {
  els.badge.hidden = !text;
  els.badge.textContent = text || '';
}
function setViewerEmpty(empty) {
  $('viewer-empty').style.display = empty ? '' : 'none';
}

// the model's own hair meshes (name contains 'hair', the export removes the same set) are hidden while the
// generated hair is shown
function setOwnHairVisible(visible) {
  if (!state.vrm) return;
  state.vrm.scene.traverse((o) => {
    if (!o.isMesh) return;
    const nm = (o.name || '') + (o.geometry?.name ? ' ' + o.geometry.name : '');
    if (!nm.toLowerCase().includes('hair')) return;
    o.visible = visible;
    o.castShadow = visible;
  });
}
function toast(msg) {
  els.viewerStatus.innerHTML = `<span class="status-warn">${msg}</span>`;
}
const NOTES = {
  matte_uncertain: 'The segmentation of this image may be inaccurate.',
  box_from_low_scores: 'The hair or face was detected with low confidence, so the crop may be off.',
  no_hair: 'No hair was detected; the crop is based on the face.',
  no_person_mask: 'No person was detected; the whole image was used.',
};
function setStatus() {
  const r = state.result;
  if (!r) { els.viewerStatus.textContent = 'Add an image to get started.'; return; }
  const done = r.samples.filter((s) => s.n_meshed > 0);
  let html = '';
  if (r.message) html += `<span class="status-warn">${r.message}</span>`;
  if (done.length) {
    html += `<b>${done.length} variation${done.length > 1 ? 's' : ''} ready</b>` +
      ` <span class="muted">· ${(r.timings?.total ?? 0).toFixed(1)} s</span>`;
  }
  const notes = [...new Set([...(r.notes || [])].map((n) => NOTES[n]).filter(Boolean))];
  if (notes.length) html += ` <span class="muted">${notes.join(' ')}</span>`;
  els.viewerStatus.innerHTML = html || 'No hair cards were generated for this image.';
}
function setViewerStatus() {
  const s = state.samples[state.seedIdx];
  if (!s) { els.badge.hidden = true; return; }
  const mode = state.hair?.kind === 'exported' ? 'exported VRM' :
    state.hair?.kind === 'spring' ? 'live spring bones' : 'hair cards';
  setBadge(`${s.label} · ${mode}`);
}

// ---------------------------------------------------------------- data plumbing
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || body.message || r.statusText);
  return body;
}

async function refreshModels(selectId) {
  state.models = await fetchJSON('/api/models');
  els.vrmModels.innerHTML = '';
  for (const m of state.models) {
    const chip = document.createElement('button');
    chip.className = 'mini' + (m.id === state.bodyId ? ' on' : '');
    chip.textContent = m.name || m.id;
    chip.title = `${m.file_tag} · VRM ${m.vrm_version} · face: ${m.face_method}` +
      (m.n_hair_objects != null ? ` · ${m.n_hair_objects} hair mesh(es) found` : '');
    chip.addEventListener('click', () => selectModel(m.id));
    els.vrmModels.appendChild(chip);
  }
  const cur = state.models.find((m) => m.id === (selectId || state.bodyId)) || state.models[0];
  if (cur) await selectModel(cur.id);
}

async function selectModel(id) {
  if (state.bodyId === id && state.vrm) return;
  const m = state.models.find((x) => x.id === id);
  if (!m) return;
  state.bodyId = id;
  state.body = await fetchJSON(`/api/models/${id}`);
  els.vrmName.textContent = m.name || m.id;
  const bits = [`VRM ${m.vrm_version}`];
  if (m.face_method === 'head_region') bits.push('approximate head fit');
  if (m.n_hair_objects != null) bits.push(`${m.n_hair_objects} hair mesh(es) to replace`);
  els.vrmSub.textContent = bits.join(' · ');
  els.vrmStatus.textContent = m.vrm_version === '1.0'
    ? 'VRM 1.0 model: the hair export writes VRM 1.0 with injected spring bones.' : '';
  els.vrmModels.querySelectorAll('.mini').forEach((c) => c.classList.toggle('on', c.textContent === (m.name || m.id)));
  try {
    await loadAvatar(m.url);
  } catch (e) {
    console.error(e);
    toast('The model could not be loaded: ' + (e.message || e));
    return;
  }
  // hair placement depends on the body's head transform: reload the current hair
  if (state.result && state.samples[state.seedIdx]) await refreshHair();
  document.querySelectorAll('#vrm-models .mini').forEach((chip, i) => chip.classList.toggle('on', state.models[i]?.id === id));
}

async function refreshExamples() {
  const list = await fetchJSON('/api/examples');
  els.examplesGrid.innerHTML = '';
  for (const url of list.slice(0, 12)) {
    const img = new Image();
    img.src = url;
    img.title = 'use this example';
    img.addEventListener('click', () => setImageFromUrl(url));
    els.examplesGrid.appendChild(img);
  }
}

async function setImageFromUrl(url) {
  const blob = await (await fetch(url)).blob();
  const name = url.split('/').pop() || 'example.png';
  state.imageFile = new File([blob], name, { type: blob.type });
  showImagePreview(URL.createObjectURL(blob));
}

function showImagePreview(url) {
  els.imagePreview.src = url;
  els.imagePreview.hidden = false;
  els.imageHint.style.display = 'none';
}

function fmtSize(bytes) {
  return bytes > 1e6 ? (bytes / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(bytes / 1e3)) + ' kB';
}

async function head(url) {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    return { ok: r.ok, size: Number(r.headers.get('content-length') || 0) };
  } catch { return { ok: false, size: 0 }; }
}

async function refreshFiles() {
  const out = [];
  const s = state.samples[state.seedIdx];
  const base = state.outId ? `/results/${state.outId}` : null;
  if (s && base) {
    for (const [key, label] of [
      ['glb', 'hair cards + reference head (GLB)'], ['glb_nohead', 'hair cards (GLB)'],
      ['glb_chains', 'hair cards + head + chains (GLB)'], ['glb_nohead_chains', 'hair cards + chains (GLB)'],
      ['json', 'hair-card geometry (JSON)'], ['chains_json', 'spring-bone chains (JSON)'],
    ]) {
      if (s[key]) out.push({ url: s[key], label });
    }
  }
  if (state.exportedUrl) out.push({ url: state.exportedUrl, label: 'VRM avatar with the generated hair' });
  if (state.turnUrl) out.push({ url: state.turnUrl, label: 'head-turn preview (MP4)' });
  if (state.outId) out.push({ url: `/api/download_all/${state.outId}`, label: 'everything of this result (zip)' });
  els.fileList.innerHTML = '';
  if (!out.length) { els.fileList.innerHTML = '<li class="muted">No files yet.</li>'; els.zipBtn.disabled = true; return; }
  for (const f of out) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = f.url; a.download = '';
    a.textContent = f.label;
    li.appendChild(a);
    const h = await head(f.url);
    if (h.size) {
      const sz = document.createElement('span');
      sz.className = 'fsize';
      sz.textContent = fmtSize(h.size);
      li.appendChild(sz);
    }
    els.fileList.appendChild(li);
  }
  els.zipBtn.disabled = !state.outId;
}

function refreshVariations() {
  els.variation.innerHTML = '';
  const done = state.samples.filter((s) => s.n_meshed > 0);
  if (!done.length) {
    els.variation.disabled = true;
    els.variation.innerHTML = '<option>— generate hair first —</option>';
    return;
  }
  for (let i = 0; i < state.samples.length; i++) {
    const s = state.samples[i];
    if (s.n_meshed <= 0) continue;
    const o = document.createElement('option');
    o.value = i;
    o.textContent = s.label;
    els.variation.appendChild(o);
  }
  els.variation.disabled = false;
  els.variation.value = String(state.seedIdx);
  els.exportBtn.disabled = false;
  els.turnBtn.disabled = false;
  els.wearBtn.disabled = !state.exportedUrl;
}

async function applyResult(result, { keepSeed = false } = {}) {
  state.result = result;
  state.outId = result.out_id;
  state.samples = result.samples || [];
  state.seedIdx = keepSeed ? Math.min(state.seedIdx, Math.max(0, state.samples.length - 1)) : 0;
  state.exportedUrl = null; state.turnUrl = null;
  els.wearBtn.disabled = true;
  els.exportInfo.textContent = 'Export puts this hair (with its spring bones) onto the selected model as a downloadable VRM 0.x avatar.';
  refreshVariations();
  setStatus();
  await refreshFiles();
  if (result.prepVis) {
    els.prepVis.src = result.prepVis; els.prepVis.hidden = false; els.prepNone.hidden = true;
  } else {
    els.prepVis.hidden = true; els.prepNone.hidden = false;
  }
  await refreshHair();
  if (!userMovedCamera) fitCamera(true);
}

async function refreshHair() {
  const s = state.samples[state.seedIdx];
  if (!s || s.n_meshed <= 0) { clearHair(); return; }
  const chains = els.showChains.checked;
  if (chains && s.chains_pending) {
    setBadge('generating spring-bone chains…');
    try {
      const upd = await fetchJSON('/api/chains', { method: 'POST', body: new URLSearchParams({ out_id: state.outId, seed: s.seed }) });
      Object.assign(s, upd);
    } catch (e) { toast('Spring-bone chains could not be generated: ' + e.message); }
    setBadge(null);
  }
  if (chains) await showHairSpring(s);
  else await showHairGLB(s, { proxy: els.showHead.checked, chains: false });
}

// ---------------------------------------------------------------- events
function bindDrop(zone, input, handler) {
  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault(); zone.classList.remove('drag');
    if (e.dataTransfer.files?.[0]) handler(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files?.[0]) handler(input.files[0]); });
}

bindDrop(els.vrmDrop, els.vrmFile, uploadVrm);
bindDrop(els.imageDrop, els.imageFile, (f) => {
  if (!f.type.startsWith('image/')) { toast('Please choose an image file.'); return; }
  state.imageFile = f;
  showImagePreview(URL.createObjectURL(f));
});

async function uploadVrm(file) {
  if (!file.name.toLowerCase().endsWith('.vrm')) { toast('Please drop a .vrm file.'); return; }
  setBadge(`registering ${file.name}…`);
  els.vrmStatus.textContent = 'Computing the head fit (Blender, up to a minute)…';
  try {
    const fd = new FormData();
    fd.append('file', file);
    const m = await fetchJSON('/api/models', { method: 'POST', body: fd });
    els.vrmStatus.textContent = '';
    await refreshModels(m.id);
  } catch (e) {
    els.vrmStatus.textContent = e.message;
    toast(e.message);
  }
  setBadge(null);
}

$('cfg').addEventListener('input', () => { els.cfgOut.textContent = Number($('cfg').value).toFixed(2); });

els.gen.addEventListener('click', async () => {
  if (state.busy) return;
  if (!state.imageFile) { toast('Please add a source image first.'); return; }
  state.busy = true;
  els.gen.disabled = true;
  els.gen.classList.add('busy');
  els.gen.textContent = 'Generating…';
  setBadge('generating hair…');
  try {
    const fd = new FormData();
    fd.append('image', state.imageFile);
    fd.append('vrm', state.bodyId || 'default');
    fd.append('seed', els.seed.value || '0');
    fd.append('steps', document.querySelector('input[name="steps"]:checked').value);
    fd.append('cfg', $('cfg').value);
    fd.append('n_samples', document.querySelector('input[name="n_samples"]:checked').value);
    fd.append('chains', 'true');
    const r = await fetchJSON('/api/generate', { method: 'POST', body: fd });
    stopMotion(true);
    await applyResult(r);
  } catch (e) {
    toast('Generation failed: ' + e.message);
  }
  state.busy = false;
  els.gen.disabled = false;
  els.gen.classList.remove('busy');
  els.gen.textContent = 'Generate hair';
  setBadge(null);
});

els.variation.addEventListener('change', async () => {
  stopMotion(true);
  state.seedIdx = Number(els.variation.value) || 0;
  state.exportedUrl = null;
  els.wearBtn.disabled = true;
  await refreshFiles();
  await refreshHair();
});

els.showHead.addEventListener('change', refreshHair);
els.showChains.addEventListener('change', refreshHair);

els.exportBtn.addEventListener('click', async () => {
  if (state.busy) return;
  const s = state.samples[state.seedIdx];
  if (!s) return;
  state.busy = true;
  els.exportBtn.disabled = true;
  els.exportInfo.textContent = 'Exporting the VRM avatar. This can take up to a minute (Blender).';
  setBadge('exporting VRM…');
  try {
    const r = await fetchJSON('/api/export_vrm', {
      method: 'POST', body: new URLSearchParams({ out_id: state.outId, seed: s.seed, vrm: state.bodyId || 'default' }),
    });
    state.exportedUrl = r.vrm;
    const oktxt = r.ok ? 'The file passed validation.' : 'Note: ' + (r.problems?.[0] || 'the file did not pass every validation check.');
    const ver = state.body?.vrm_version === '1.0' ? 'VRM 1.0' : 'VRM 0.x';
    els.exportInfo.innerHTML = `Seed ${s.seed} · ${ver} · ${r.size_mb} MB · ${r.n_chains ?? 0} spring chain(s). ${oktxt}`;
    els.wearBtn.disabled = false;
    await wearExported(r.vrm);
    await refreshFiles();
  } catch (e) {
    els.exportInfo.textContent = 'Export failed: ' + e.message;
  }
  state.busy = false;
  els.exportBtn.disabled = false;
  setBadge(null);
});

els.wearBtn.addEventListener('click', async () => {
  if (state.exportedUrl) await wearExported(state.exportedUrl);
});

els.turnBtn.addEventListener('click', async () => {
  const s = state.samples[state.seedIdx];
  if (!s) return;
  els.turnBtn.disabled = true;
  els.exportInfo.textContent = 'Rendering the head-turn video. This can take up to a minute.';
  try {
    const r = await fetchJSON('/api/turn', {
      method: 'POST', body: new URLSearchParams({ out_id: state.outId, seed: s.seed, vrm: state.bodyId || 'default' }),
    });
    state.turnUrl = r.video;
    els.exportInfo.innerHTML = `Head-turn preview rendered (${r.seconds ?? 3} s` +
      (r.max_motion_cm != null ? `, max motion ${r.max_motion_cm.toFixed(1)} cm` : '') + ').';
    let vid = $('turn-video');
    if (!vid) {
      vid = document.createElement('video');
      vid.id = 'turn-video';
      vid.controls = true; vid.loop = true; vid.autoplay = true; vid.muted = true;
      vid.style.maxWidth = '100%'; vid.style.borderRadius = '10px';
      els.exportInfo.after(vid);
    }
    vid.src = r.video;
    await refreshFiles();
  } catch (e) {
    els.exportInfo.textContent = 'The head-turn render is unavailable here (' + e.message + ') — use the live motion preview in the 3D viewer.';
  }
  els.turnBtn.disabled = false;
});

els.zipBtn.addEventListener('click', () => { if (state.outId) location.href = `/api/download_all/${state.outId}`; });

document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === t));
  $('tab-downloads').hidden = t.dataset.tab !== 'downloads';
  $('tab-input').hidden = t.dataset.tab !== 'input';
}));

// ---------------------------------------------------------------- boot
async function boot() {
  resize();
  animate();
  try {
    const st = await fetchJSON('/api/status');
    state.models = st.bodies || [];
    for (const m of state.models) if (!m.url) m.url = `/models/${m.id}.vrm`;
  } catch { /* server not ready yet */ }
  await refreshExamples();
  await refreshModels('default');
  setInterval(async () => {
    try {
      const st = await fetchJSON('/api/status');
      els.pipelineStatus.textContent = st.ready ? 'Hair model ready.' :
        st.pipeline === 'error' ? 'Hair model failed to load — see the server log.' : 'Loading the hair model…';
      els.pipelineStatus.classList.toggle('err', st.pipeline === 'error');
      els.gen.disabled = state.busy || !st.ready;
    } catch { els.pipelineStatus.textContent = 'Server unreachable.'; }
  }, 2500);
}
boot();
