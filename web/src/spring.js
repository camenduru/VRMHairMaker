// Spring-bone simulation of the generated hair driven by chains, in the browser.
// A faithful port of hairdeck/spring/sim.py (UniVRM VRMSpringBoneLogic: verlet with drag, stiffness pull toward the
// parent-rotated rest direction, length constraint; no colliders) plus its deform(). The rig data comes from
// /api/hairrig (hairdeck.vrm.prep: the same geometric skinning the Blender export binds).
//
// All positions are in the dataset frame: y up, face toward +z, metres — the frame a VRM stands in (feet at the
// origin) after VRMUtils.rotateVRM0.

export const DT = 1 / 60;
export const DRAG = 0.4;
export const NF = 150; // frames of the hairdeck motion curves
// gentle pull toward each joint's rest position (carried by the head motion): bounds the wander of long idle
// sessions so the hair always settles back into the generated style, while fast head motion still swings it
export const SHAPE_KP = 2.5;

export function ident3() { return [1, 0, 0, 0, 1, 0, 0, 0, 1]; }
export function mat3mul(a, b) {
  const o = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  }
  return o;
}
export function mat3vec(a, v) {
  return [a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
          a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
          a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
}
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
export function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
export function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
export function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }

function deg(a) { return a * Math.PI / 180; }
export function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return [c, 0, s, 0, 1, 0, -s, 0, c]; }
export function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, -s, 0, s, c]; }

// rotdiff(a, b): rotation matrix (row-major 3x3) taking unit vector a to unit vector b (shortest arc)
export function rotdiff(a, b) {
  const v = cross(a, b);
  const s = norm(v), c = dot(a, b);
  if (s < 1e-12) return ident3();
  const k = scale(v, 1 / s);
  const ang = Math.atan2(s, c);
  const K = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0];
  const KK = mat3mul(K, K);
  const sa = Math.sin(ang), ca = 1 - Math.cos(ang);
  const out = new Array(9);
  for (let i = 0; i < 9; i++) out[i] = (i % 4 === 0 ? 1 : 0) + sa * K[i] + ca * KK[i];
  return out;
}

// motion curves of sim.py: turn (yaw 50 deg in 0.35 s, hold), nod (pitch 25 deg), jump (0.15 m up), sway (6 cm side)
export function motionFrame(name, f) {
  // f: 1..NF -> {R: 3x3, T: [x,y,z], angleY, angleX} in the dataset frame
  const t = f / 60;
  let R = ident3(), T = [0, 0, 0], angleY = 0, angleX = 0;
  if (name === 'turn') {
    const x = Math.min(t / 0.35, 1);
    angleY = deg(50) * x * x * (3 - 2 * x);
    R = rotY(angleY);
  } else if (name === 'nod') {
    angleX = t < 0.5 ? deg(25) * Math.sin(Math.PI * t / 0.5) : 0;
    R = rotX(angleX);
  } else if (name === 'jump') {
    T[1] = t < 0.6 ? 0.15 * 4 * (t / 0.6) * (1 - t / 0.6) : 0;
  } else if (name === 'sway') {
    T[0] = t < 4 / 3 ? 0.06 * Math.sin(2 * Math.PI * 1.5 * t) : 0;
  }
  return { R, T, angleY, angleX };
}

// ---- the rig: hair vertices + skin weights + chains, from /api/hairrig ----
export class HairSim {
  constructor(rig) {
    this.rig = rig;
    this.V = rig.V;                       // rest vertices (dataset frame)
    this.piv = rig.headJoint;             // head joint (dataset frame)
    this.chains = [];
    const J = rig.J, ptr = rig.chainPtr;
    const col = rig.headCollider || null;
    this.collider = col;   // {center, radius} in the dataset/world frame
    for (let c = 0; c < rig.nChains; c++) {
      const ch = buildChain(J.slice(ptr[c], ptr[c + 1]), rig.stiffness[c] ?? 0.5);
      if (ch && col) {
        // leading joints inside the head sphere at rest follow the head rigidly (the export's rigid prefix,
        // with the same +HIT tolerance: distance < radius + 0.02)
        let n = 0;
        const pr = col.prefixRadius ?? col.radius;
        while (n + 1 < ch.J.length && dist(ch.J[n], col.center) < pr) n++;
        ch.rigidN = n;       // number of rigid segments (their bases are joints 0..n-1)
      } else {
        ch.rigidN = 0;
      }
      this.chains.push(ch);
    }
    // per-vertex skin binding (sim.bind output): chain weight w on the assigned chain, split between
    // segment vseg (alpha) and its neighbour vseg2 (1 - alpha); w also blends against the rigid head
    const n = this.V.length;
    this.w = new Float32Array(n);
    this.alpha = new Float32Array(n).fill(1);
    this.segA = new Int32Array(n).fill(-1);
    this.segB = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      const c = rig.vchain[i];
      this.w[i] = rig.vw[i];
      if (c >= 0 && c < this.chains.length && this.chains[c]) {
        this.segA[i] = rig.vseg[i];
        this.segB[i] = rig.vseg2[i];
        this.alpha[i] = rig.valpha[i];
      } else {
        this.w[i] = 0;
      }
    }
    this.R = ident3();
    this.T = [0, 0, 0];
  }

  reset() {
    for (const c of this.chains) {
      if (!c) continue;
      for (let k = 0; k < c.L.length; k++) {
        if (k < c.rigidN) {
          c.heads[k] = c.J[k].slice();
          c.rots[k] = ident3();
          c.cur[k] = add(c.J[k], scale(c.D[k], c.L[k]));
        } else {
          c.cur[k] = add(c.J[k], scale(c.D[k], c.L[k]));
        }
        c.prev[k] = c.cur[k].slice();
        if (k >= c.rigidN) { c.heads[k] = c.J[k].slice(); c.rots[k] = ident3(); }
      }
    }
  }

  // one verlet step (sim.simulate) with the export's rigid-scalp prefix: segments rooted inside the head
  // collider follow the head rigidly; the rest verlet-follows with stiffness/drag, pushed out of the head sphere
  step(R, T = [0, 0, 0], dt = DT) {
    const col = this.collider;
    const colC = col ? rigidPoint(this.piv, col.center, R, T) : null;
    for (const c of this.chains) {
      if (!c) continue;
      const piv = this.piv;
      let Wp = R;
      let head = null;
      for (let k = 0; k < c.L.length; k++) {
        if (k < c.rigidN) {   // scalp prefix: rigid with the head
          head = rigidPoint(piv, c.J[k + 1], R, T);
          c.heads[k] = rigidPoint(piv, c.J[k], R, T);
          c.rots[k] = R;
          c.cur[k] = head.slice(); c.prev[k] = head.slice();
          continue;
        }
        if (head === null) {  // the first simulated segment hangs from its rigidly carried base joint
          head = rigidPoint(piv, c.J[k], R, T);
        }
        const rest = mat3vec(Wp, c.D[k]);
        const cur = c.cur[k], prev = c.prev[k];
        const nx0 = [cur[0] + (cur[0] - prev[0]) * (1 - c.drag) + rest[0] * (c.stiff * dt),
                     cur[1] + (cur[1] - prev[1]) * (1 - c.drag) + rest[1] * (c.stiff * dt),
                     cur[2] + (cur[2] - prev[2]) * (1 - c.drag) + rest[2] * (c.stiff * dt)];
        let v0 = sub(nx0, head);
        const n = norm(v0);
        const v = n > 1e-12 ? scale(v0, 1 / n) : rest;
        let nx = add(head, scale(v, c.L[k]));
        if (col) {   // keep the joint outside the (head-carried) skull sphere
          const dc = sub(nx, colC);
          const dn = norm(dc);
          if (dn < col.radius) nx = add(colC, dn > 1e-9 ? scale(dc, col.radius / dn) : [0, col.radius, 0]);
        }
        if (SHAPE_KP > 0) {   // shape memory: the rest joint position carried by the head motion
          const rp = rigidPoint(piv, c.J[k + 1], R, T);
          nx = [nx[0] + (rp[0] - nx[0]) * SHAPE_KP * dt, nx[1] + (rp[1] - nx[1]) * SHAPE_KP * dt,
                nx[2] + (rp[2] - nx[2]) * SHAPE_KP * dt];
        }
        c.prev[k] = cur; c.cur[k] = nx;
        const Wk = rotdiff(rest, v);
        c.heads[k] = head; c.rots[k] = Wk;
        Wp = Wk; head = nx;
      }
    }
  }

  // deform() of sim.py: write simulated positions into a Float32Array (xyz per vertex)
  deform(out) {
    const V = this.V, n = V.length;
    const R = this.R, T = this.T, piv = this.piv;
    const px = piv[0], py = piv[1], pz = piv[2];
    const tx = T[0], ty = T[1], tz = T[2];
    for (let i = 0; i < n; i++) {
      const v = V[i];
      const rx = v[0] - px, ry = v[1] - py, rz = v[2] - pz;
      let X0 = R[0] * rx + R[1] * ry + R[2] * rz + px + tx;
      let X1 = R[3] * rx + R[4] * ry + R[5] * rz + py + ty;
      let X2 = R[6] * rx + R[7] * ry + R[8] * rz + pz + tz;
      const c = this.rig.vchain[i];
      const w = this.w[i];
      if (c >= 0 && w > 0) {
        const ch = this.chains[c];
        if (ch && ch.heads) {
          const a = this.alpha[i];
          const kA = this.segA[i];
          const fA = segPoint(ch, kA, v);
          let F0 = fA[0], F1 = fA[1], F2 = fA[2];
          if (kA !== this.segB[i] && a < 1) {
            const fB = segPoint(ch, this.segB[i], v);
            F0 = a * F0 + (1 - a) * fB[0];
            F1 = a * F1 + (1 - a) * fB[1];
            F2 = a * F2 + (1 - a) * fB[2];
          }
          X0 = (1 - w) * X0 + w * F0;
          X1 = (1 - w) * X1 + w * F1;
          X2 = (1 - w) * X2 + w * F2;
        }
      }
      out[i * 3] = X0; out[i * 3 + 1] = X1; out[i * 3 + 2] = X2;
    }
  }
}

function rigidPoint(piv, p, R, T) {
  const dx = p[0] - piv[0], dy = p[1] - piv[1], dz = p[2] - piv[2];
  return [R[0] * dx + R[1] * dy + R[2] * dz + piv[0] + T[0],
          R[3] * dx + R[4] * dy + R[5] * dz + piv[1] + T[1],
          R[6] * dx + R[7] * dy + R[8] * dz + piv[2] + T[2]];
}
function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }

function buildChain(Jc, stiff) {
  if (!Jc || Jc.length < 2) return null;
  // sim.clean: drop zero-length segments first; every index below refers to the cleaned polyline
  const J = [Jc[0].slice()];
  for (let k = 1; k < Jc.length; k++) {
    if (norm(sub(Jc[k], J[J.length - 1])) > 1e-6) J.push(Jc[k].slice());
  }
  if (J.length < 2) return null;
  const D = [], L = [], cur = [], prev = [], heads = [], rots = [];
  for (let k = 0; k + 1 < J.length; k++) {
    const dd = sub(J[k + 1], J[k]);
    const l = norm(dd);
    if (l < 1e-6) continue;
    D.push(scale(dd, 1 / l)); L.push(l);
    cur.push(J[k + 1].slice()); prev.push(J[k + 1].slice());
    heads.push(J[k].slice()); rots.push(ident3());
  }
  if (!L.length) return null;
  return { J, D, L, stiff, drag: DRAG, cur, prev, heads, rots };
}

// head_k + R_k @ (V - J[k]) for segment k of chain ch (sim.deform's follow term)
function segPoint(ch, k, v) {
  const head = ch.heads[k], Rk = ch.rots[k], J = ch.J[k];
  const dx = v[0] - J[0], dy = v[1] - J[1], dz = v[2] - J[2];
  return [head[0] + Rk[0] * dx + Rk[1] * dy + Rk[2] * dz,
          head[1] + Rk[3] * dx + Rk[4] * dy + Rk[5] * dz,
          head[2] + Rk[6] * dx + Rk[7] * dy + Rk[8] * dz];
}
