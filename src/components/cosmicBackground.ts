import * as THREE from 'three';

export function init(container: HTMLElement): void {

// ══════════════════════════════════════════════════════════════
// 场景 & 渲染器
// ══════════════════════════════════════════════════════════════
const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x00010c);

const camera = new THREE.PerspectiveCamera(
  75, window.innerWidth / window.innerHeight, 0.5, 6000
);
camera.position.set(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

// ══════════════════════════════════════════════════════════════
// 工具函数
// ══════════════════════════════════════════════════════════════
const rnd  = (a: number, b: number) => Math.random() * (b - a) + a;
const rndI = (a: number, b: number) => Math.floor(rnd(a, b));

// 球面均匀随机方向
function sphereDir(): THREE.Vector3 {
  const phi   = Math.acos(2 * Math.random() - 1);
  const theta = Math.random() * Math.PI * 2;
  return new THREE.Vector3(
    Math.sin(phi) * Math.cos(theta),
    Math.sin(phi) * Math.sin(theta),
    Math.cos(phi)
  );
}

// ══════════════════════════════════════════════════════════════
// 星点纹理
// ══════════════════════════════════════════════════════════════
function makeSprite(): THREE.Texture {
  const cv  = document.createElement('canvas');
  cv.width  = cv.height = 64;
  const ctx = cv.getContext('2d')!;
  const g   = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0.00, 'rgba(255,255,255,1)');
  g.addColorStop(0.12, 'rgba(220,235,255,0.95)');
  g.addColorStop(0.35, 'rgba(140,180,255,0.3)');
  g.addColorStop(1.00, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
}
const spriteTex = makeSprite();

// ══════════════════════════════════════════════════════════════
// 星星数据池
// ══════════════════════════════════════════════════════════════
const N    = 2500;                // 总星数（对象池）
const NEAR = 100;                 // 最近生成距离
const FAR  = 2800;                // 最远生成距离

// Flat arrays — 世界坐标 & 颜色 & 大小
const starPos = new Float32Array(N * 3);   // 世界坐标
const starCol = new Float32Array(N * 3);   // RGB
const starSz  = new Float32Array(N);       // base point size

// 星星类型分布，决定颜色 & 大小
function randomStarColor(out: Float32Array, i: number) {
  const t  = Math.random();
  let r: number, g: number, b: number, sz: number;
  if (t < 0.60) {
    // 白/淡黄恒星
    const bright = rnd(0.55, 1.0);
    r = bright; g = bright * rnd(0.93, 1.0); b = bright * rnd(0.88, 1.0);
    sz = rnd(1.5, 4.5);
  } else if (t < 0.80) {
    // 蓝白热星
    r = rnd(0.35, 0.75); g = rnd(0.60, 0.90); b = 1.0;
    sz = rnd(2.0, 7.0);
  } else if (t < 0.95) {
    // 橙红冷星
    r = 1.0; g = rnd(0.28, 0.62); b = rnd(0.04, 0.18);
    sz = rnd(2.5, 8.0);
  } else {
    // 特亮稀有星（白矮星/超巨星）
    r = 1.0; g = 1.0; b = rnd(0.85, 1.0);
    sz = rnd(5.0, 12.0);
  }
  out[i*3]   = r;
  out[i*3+1] = g;
  out[i*3+2] = b;
  return sz;
}

// 在相机前方半球 cone 内随机生成一个位置
function spawnAhead(
  outPos: Float32Array,
  idx: number,
  camPos: THREE.Vector3,
  camQuat: THREE.Quaternion
) {
  // 在前方 ±100° 锥角内随机
  const halfAngle = Math.PI * 0.58;
  const phi   = Math.acos(1 - Math.random() * (1 - Math.cos(halfAngle)));
  const theta = Math.random() * Math.PI * 2;
  const local = new THREE.Vector3(
    Math.sin(phi) * Math.cos(theta),
    Math.sin(phi) * Math.sin(theta),
    -Math.cos(phi)    // 局部 -Z 是前方
  );
  local.applyQuaternion(camQuat);
  const dist = rnd(NEAR, FAR);
  outPos[idx*3]   = camPos.x + local.x * dist;
  outPos[idx*3+1] = camPos.y + local.y * dist;
  outPos[idx*3+2] = camPos.z + local.z * dist;
}

// 初始化：球面均匀分布（出生在原点周围所有方向）
for (let i = 0; i < N; i++) {
  const dir  = sphereDir();
  const dist = rnd(NEAR, FAR);
  starPos[i*3]   = dir.x * dist;
  starPos[i*3+1] = dir.y * dist;
  starPos[i*3+2] = dir.z * dist;
  starSz[i] = randomStarColor(starCol, i);
}

// ══════════════════════════════════════════════════════════════
// Points 几何体（正常飞行视图，speed 0→0.85 可见）
// ══════════════════════════════════════════════════════════════
const pGeo = new THREE.BufferGeometry();
// 使用同一 buffer，直接引用
const pPosBuf = new THREE.BufferAttribute(starPos, 3);
const pColBuf = new THREE.BufferAttribute(starCol, 3);
const pSzBuf  = new THREE.BufferAttribute(starSz,  1);
pPosBuf.setUsage(THREE.DynamicDrawUsage);
pGeo.setAttribute('position', pPosBuf);
pGeo.setAttribute('color',    pColBuf);
pGeo.setAttribute('aSize',    pSzBuf);

const pMat = new THREE.ShaderMaterial({
  uniforms: {
    uTex:   { value: spriteTex },
    uSpeed: { value: 0.0 },
  },
  vertexShader: /* glsl */`
    attribute float aSize;
    uniform float uSpeed;
    varying vec3 vCol;
    void main() {
      vCol = color;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      // 高速时点稍微放大（因为要淡出，先给点存在感）
      float scale = 1.0 + uSpeed * 0.4;
      gl_PointSize = aSize * scale * (280.0 / -mv.z);
      gl_Position  = projectionMatrix * mv;
    }
  `,
  fragmentShader: /* glsl */`
    uniform sampler2D uTex;
    uniform float uSpeed;
    varying vec3 vCol;
    void main() {
      vec4 t = texture2D(uTex, gl_PointCoord);
      if (t.a < 0.004) discard;
      // speed 0.7→1.0 渐隐，为光条让位
      float fade = 1.0 - smoothstep(0.65, 0.95, uSpeed);
      if (fade < 0.005) discard;
      // 轻微蓝移
      vec3 col = mix(vCol, vec3(0.7, 0.88, 1.0), uSpeed * uSpeed * 0.4);
      gl_FragColor = vec4(col, t.a * fade);
    }
  `,
  transparent:  true,
  depthWrite:   false,
  blending:     THREE.AdditiveBlending,
  vertexColors: true,
});
scene.add(new THREE.Points(pGeo, pMat));

// ══════════════════════════════════════════════════════════════
// LineSegments 几何体（光速条纹，speed > 0.6 渐显）
// 每颗星 2 个顶点：[head=星的世界坐标，tail=向相机方向延伸]
// ══════════════════════════════════════════════════════════════
const sPosBuf = new Float32Array(N * 2 * 3);
const sColBuf = new Float32Array(N * 2 * 3);
const sRole   = new Float32Array(N * 2);    // 0=tail, 1=head

for (let i = 0; i < N; i++) {
  // head & tail 初始位置相同（speed=0 时线段长度为0，不可见）
  for (let v = 0; v < 2; v++) {
    sPosBuf[(i*2+v)*3]   = starPos[i*3];
    sPosBuf[(i*2+v)*3+1] = starPos[i*3+1];
    sPosBuf[(i*2+v)*3+2] = starPos[i*3+2];
    sColBuf[(i*2+v)*3]   = starCol[i*3];
    sColBuf[(i*2+v)*3+1] = starCol[i*3+1];
    sColBuf[(i*2+v)*3+2] = starCol[i*3+2];
    sRole[i*2+v] = v; // 0=tail, 1=head
  }
}

const sGeo = new THREE.BufferGeometry();
const sPosBufAttr = new THREE.BufferAttribute(sPosBuf, 3);
const sColBufAttr = new THREE.BufferAttribute(sColBuf, 3);
const sRoleBuf    = new THREE.BufferAttribute(sRole, 1);
sPosBufAttr.setUsage(THREE.DynamicDrawUsage);
sGeo.setAttribute('position', sPosBufAttr);
sGeo.setAttribute('color',    sColBufAttr);
sGeo.setAttribute('aRole',    sRoleBuf);

const sMat = new THREE.ShaderMaterial({
  uniforms: {
    uSpeed:  { value: 0.0 },
    uCamPos: { value: new THREE.Vector3() },
  },
  vertexShader: /* glsl */`
    attribute float aRole;
    uniform float uSpeed;
    uniform vec3  uCamPos;
    varying vec3  vCol;
    varying float vRole;
    void main() {
      vCol  = color;
      vRole = aRole;
      vec3 p = position;

      if (aRole < 0.5) {
        // tail 顶点：沿「星→相机」方向延伸，创造向视角中心汇聚的光条
        vec3 toCamera = uCamPos - position;
        float dist    = length(toCamera);
        vec3 dir      = toCamera / max(dist, 0.001);
        // 延伸长度 = speed² × (距离/FAR) × 最大延伸量
        // 近处短、远处长，近似真实光行差效果
        float stretch = uSpeed * uSpeed * min(dist * 0.18, 160.0);
        p = position + dir * stretch;
      }

      gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    uniform float uSpeed;
    varying vec3  vCol;
    varying float vRole;
    void main() {
      // 整体淡入：speed 0.6→0.85 渐显
      float show = smoothstep(0.55, 0.82, uSpeed);
      if (show < 0.005) discard;
      // tail 端淡出（向相机方向越来越透明，避免硬截断）
      float alpha = mix(0.06, 1.0, vRole) * show;
      // 蓝白偏移：速度越快颜色越白蓝
      float s2  = uSpeed * uSpeed;
      vec3  col = mix(vCol, vec3(0.65, 0.88, 1.0), s2 * 0.88);
      // 高速时增亮
      col *= 1.0 + s2 * 0.9;
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent:  true,
  depthWrite:   false,
  blending:     THREE.AdditiveBlending,
  vertexColors: true,
});
scene.add(new THREE.LineSegments(sGeo, sMat));

// ══════════════════════════════════════════════════════════════
// 相机旋转控制（鼠标）
// ══════════════════════════════════════════════════════════════
let yaw = 0, pitch = 0, targetYaw = 0, targetPitch = 0;

window.addEventListener('mousemove', (e: MouseEvent) => {
  targetYaw   = -(e.clientX / window.innerWidth  - 0.5) * Math.PI * 0.5;
  targetPitch = -(e.clientY / window.innerHeight - 0.5) * Math.PI * 0.3;
  if (velocity > 0.04) isDecelerating = true;
  lastMoveTime = performance.now();
});

// ══════════════════════════════════════════════════════════════
// 速度控制
// ══════════════════════════════════════════════════════════════
let velocity       = 0;    // 0~1 归一化
let idleTimer      = 0;    // 静止累计时间（秒）
let lastMoveTime   = performance.now();
let isDecelerating = false;

const IDLE_DELAY  = 1.5;   // 静止多久后开始加速
const DECEL_RATE  = 0.88;  // 每帧速度衰减系数（减速时）
const MAX_SPEED   = 8.0;   // 最大世界速度（单位/帧 @60fps）

// ══════════════════════════════════════════════════════════════
// 动画循环
// ══════════════════════════════════════════════════════════════
const camPos  = camera.position;
const camQuat = camera.quaternion;
const camDir  = new THREE.Vector3();   // 当前前向
const _v3     = new THREE.Vector3();   // 复用临时向量

let lastT = performance.now();

function animate() {
  requestAnimationFrame(animate);

  const now   = performance.now();
  const delta = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;

  // ── 相机旋转插值 ──────────────────────────────────────────
  yaw   += (targetYaw   - yaw)   * 0.055;
  pitch += (targetPitch - pitch) * 0.055;
  camQuat.setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
  camDir.set(0, 0, -1).applyQuaternion(camQuat);

  // ── 速度更新（指数加速 + 指数减速） ──────────────────────
  const idle = (now - lastMoveTime) / 1000;
  if (isDecelerating) {
    velocity *= Math.pow(DECEL_RATE, delta * 60);
    if (velocity < 0.008) { velocity = 0; isDecelerating = false; idleTimer = 0; }
  } else if (idle > IDLE_DELAY) {
    idleTimer += delta;
    // 指数曲线：t=2→0.12, t=4→0.38, t=6→0.72, t=8→≈1.0
    velocity = Math.min(1.0, (Math.exp(0.5 * idleTimer) - 1) * 0.07);
  } else {
    idleTimer = 0;
    velocity  = Math.max(0, velocity - delta * 0.5);
  }

  // ── 相机移动 ──────────────────────────────────────────────
  const worldSpeed = velocity * MAX_SPEED;
  if (worldSpeed > 0.001) {
    camPos.addScaledVector(camDir, worldSpeed);
  }

  // ── 粒子回收 & 重生 ───────────────────────────────────────
  // 回收条件：
  //   1. 在相机背后（dot < -80）
  //   2. 太远（dist > FAR × 1.4）
  let pNeedsUpdate = false;
  let sNeedsUpdate = false;
  const CULL_BEHIND = 80;
  const FAR_SQ      = FAR * FAR * 1.96;  // (FAR×1.4)²

  for (let i = 0; i < N; i++) {
    const ix = i * 3;
    const dx = starPos[ix]   - camPos.x;
    const dy = starPos[ix+1] - camPos.y;
    const dz = starPos[ix+2] - camPos.z;

    const dotFwd = dx * camDir.x + dy * camDir.y + dz * camDir.z;
    const dist2  = dx*dx + dy*dy + dz*dz;

    if (dotFwd < -CULL_BEHIND || dist2 > FAR_SQ) {
      // 回收并重新在前方生成
      spawnAhead(starPos, i, camPos, camQuat);
      starSz[i] = randomStarColor(starCol, i);

      // 同步 Points buffers
      pPosBuf.array[ix]   = starPos[ix];
      pPosBuf.array[ix+1] = starPos[ix+1];
      pPosBuf.array[ix+2] = starPos[ix+2];
      pColBuf.array[ix]   = starCol[ix];
      pColBuf.array[ix+1] = starCol[ix+1];
      pColBuf.array[ix+2] = starCol[ix+2];
      pSzBuf.array[i]     = starSz[i];

      // 同步 LineSegments buffers（head & tail 均更新为新位置）
      const si = i * 2;
      for (let v = 0; v < 2; v++) {
        const six = (si + v) * 3;
        sPosBuf[six]   = starPos[ix];
        sPosBuf[six+1] = starPos[ix+1];
        sPosBuf[six+2] = starPos[ix+2];
        sColBuf[six]   = starCol[ix];
        sColBuf[six+1] = starCol[ix+1];
        sColBuf[six+2] = starCol[ix+2];
      }
      pNeedsUpdate = true;
      sNeedsUpdate = true;
    }
  }

  if (pNeedsUpdate) {
    pPosBuf.needsUpdate = true;
    pColBuf.needsUpdate = true;
    pSzBuf.needsUpdate  = true;
  }
  if (sNeedsUpdate) {
    sPosBufAttr.needsUpdate = true;
    sColBufAttr.needsUpdate = true;
  }

  // ── 更新 uniforms ─────────────────────────────────────────
  pMat.uniforms.uSpeed.value         = velocity;
  sMat.uniforms.uSpeed.value         = velocity;
  sMat.uniforms.uCamPos.value.copy(camPos);

  renderer.render(scene, camera);
}
animate();

// ── 窗口适配 ─────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

} // end init
