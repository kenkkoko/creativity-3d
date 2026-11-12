// 3D 星圖：刺激詞在原點，半徑 = 真實語義距離（SPEC 4.1）
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const SCALE = 6;                 // API 座標半徑約 0–1.2 → 場景單位約 0–7.2
const RINGS = [0.25, 0.5, 0.75, 1.0];
const COLOR_ME = new THREE.Color("#d6e6ff");
const COLOR_BORDER = new THREE.Color("#ffe2c4");
const COLOR_BAD = new THREE.Color("#6b6b73");
const COLOR_HUMAN = new THREE.Color("#ffffff");
const COLOR_SEED = new THREE.Color("#f0a46a");
const COLOR_ACCENT = new THREE.Color("#f0a46a");
const VERDICT_TEXT = { reasonable: "合理", borderline: "勉強", unreasonable: "不合理" };

const easeOutExpo = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function glowTexture() {
  const size = 128;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.15, "rgba(255,255,255,.85)");
  grd.addColorStop(0.4, "rgba(255,255,255,.22)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// 以 index 為種子的穩定亂數，讓聯想橋的偏移每次都一樣
function seeded(n) {
  let s = (n + 1) * 9301 + 49297;
  return () => ((s = (s * 9301 + 49297) % 233280) / 233280);
}

// CSS2DRenderer 會覆寫外層元素的 transform，所以樣式與位移放在內層
function makeLabel(cls, text) {
  const wrap = document.createElement("div");
  const inner = document.createElement("div");
  inner.className = cls;
  inner.textContent = text;
  wrap.appendChild(inner);
  return { obj: new CSS2DObject(wrap), el: inner };
}

function toVec(p) {
  return new THREE.Vector3(p.x * SCALE, p.y * SCALE, p.z * SCALE);
}

export function webglAvailable() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch {
    return false;
  }
}

/**
 * @param {HTMLElement} container
 * @param {object} data AnalysisResponse
 * @param {{onSelect?: (index: number|null) => void}} opts
 */
export function createStarMap(container, data, opts = {}) {
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  let width = container.clientWidth || 800;
  let height = container.clientHeight || 600;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(width, height);
  Object.assign(labelRenderer.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none", zIndex: "2" });
  container.appendChild(labelRenderer.domElement);

  const card = document.createElement("div");
  card.className = "star-card";
  container.appendChild(card);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x08080a, 0.018);
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 400);
  camera.position.set(3, 5.5, 15);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 4;
  controls.maxDistance = 40;
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 0.35;
  controls.enablePan = false;

  const tex = glowTexture();
  const disposables = [tex];
  const track = (o) => (disposables.push(o), o);

  // ---- 遠方背景星 ----
  {
    const n = 900;
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(60 + Math.random() * 80);
      arr.set([v.x, v.y, v.z], i * 3);
    }
    const g = track(new THREE.BufferGeometry());
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    const m = track(new THREE.PointsMaterial({ color: 0x9a9aa6, size: 0.35, sizeAttenuation: true, transparent: true, opacity: 0.45, map: tex, depthWrite: false, fog: false }));
    scene.add(new THREE.Points(g, m));
  }

  // ---- 中心：刺激詞 ----
  const center = new THREE.Group();
  scene.add(center);
  const core = new THREE.Mesh(track(new THREE.SphereGeometry(0.32, 32, 32)), track(new THREE.MeshBasicMaterial({ color: 0xfff3e6 })));
  center.add(core);
  const halo = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: tex, color: COLOR_ACCENT, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })));
  halo.scale.setScalar(3.4);
  center.add(halo);
  const halo2 = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: tex, color: COLOR_ACCENT, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false })));
  halo2.scale.setScalar(9);
  center.add(halo2);
  {
    center.add(makeLabel("center-label", data.stimulus?.word ?? "").obj);
  }

  // ---- 距離環 ----
  RINGS.forEach((r, i) => {
    const pts = [];
    for (let k = 0; k <= 128; k++) {
      const a = (k / 128) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * r * SCALE, 0, Math.sin(a) * r * SCALE));
    }
    const g = track(new THREE.BufferGeometry().setFromPoints(pts));
    const m = track(new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.05 + i * 0.012, depthWrite: false }));
    scene.add(new THREE.Line(g, m));
    const lbl = makeLabel("ring-label", r.toFixed(2)).obj;
    lbl.position.set(r * SCALE * Math.SQRT1_2, 0, r * SCALE * Math.SQRT1_2);
    scene.add(lbl);
  });

  // ---- 背景星雲：其他人（白）+ AI 參考（暖橘） ----
  const cloud = Array.isArray(data.cloud) ? data.cloud : [];
  const cloudUniforms = { uTime: { value: 0 }, uGrow: { value: reduceMotion ? 1 : 0 }, uMap: { value: tex }, uPixel: { value: renderer.getPixelRatio() } };
  if (cloud.length) {
    const pos = new Float32Array(cloud.length * 3);
    const col = new Float32Array(cloud.length * 3);
    const size = new Float32Array(cloud.length);
    const phase = new Float32Array(cloud.length);
    cloud.forEach((p, i) => {
      const v = toVec(p.position);
      pos.set([v.x, v.y, v.z], i * 3);
      const c = p.source === "seed" ? COLOR_SEED : COLOR_HUMAN;
      col.set([c.r, c.g, c.b], i * 3);
      size[i] = (p.source === "seed" ? 0.55 : 0.5) * (1 + Math.log2(Math.max(1, p.count || 1)) * 0.45);
      phase[i] = Math.random() * Math.PI * 2;
    });
    const g = track(new THREE.BufferGeometry());
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
    g.setAttribute("aPhase", new THREE.BufferAttribute(phase, 1));
    const m = track(new THREE.ShaderMaterial({
      uniforms: cloudUniforms,
      vertexShader: `
        attribute float aSize; attribute float aPhase; attribute vec3 color;
        uniform float uTime; uniform float uGrow; uniform float uPixel;
        varying vec3 vColor; varying float vAlpha;
        void main() {
          vec3 p = position * uGrow;
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          float tw = 0.75 + 0.25 * sin(uTime * 1.6 + aPhase);
          gl_PointSize = aSize * uPixel * (220.0 / -mv.z) * tw;
          vColor = color; vAlpha = 0.55 * tw * uGrow;
        }`,
      fragmentShader: `
        uniform sampler2D uMap; varying vec3 vColor; varying float vAlpha;
        void main() {
          vec4 t = texture2D(uMap, gl_PointCoord);
          gl_FragColor = vec4(vColor, t.a * vAlpha);
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    scene.add(new THREE.Points(g, m));
  }

  // ---- 自己的聯想 ----
  const assocs = Array.isArray(data.associations) ? data.associations : [];
  const stars = []; // {a, group, sprite, hit, label, target}
  const hitTargets = [];
  assocs.forEach((a, i) => {
    const bad = a.verdict === "unreasonable" || a.weight === 0;
    const eff = Math.max(0, Math.min(1.2, a.effective_distance || 0));
    const target = toVec(a.position || { x: 0, y: 0, z: 0 });
    const group = new THREE.Group();
    group.position.set(0, 0, 0);
    scene.add(group);

    const color = bad ? COLOR_BAD : a.verdict === "borderline" ? COLOR_BORDER : COLOR_ME;
    const mat = track(new THREE.SpriteMaterial({
      map: tex, color, transparent: true,
      opacity: bad ? 0.45 : 0.75 + eff * 0.25,
      blending: bad ? THREE.NormalBlending : THREE.AdditiveBlending,
      depthWrite: false,
    }));
    const sprite = new THREE.Sprite(mat);
    const baseScale = bad ? 0.35 : 0.75 + eff * 1.1;
    sprite.scale.setScalar(baseScale);
    group.add(sprite);

    if (!bad) {
      const dot = new THREE.Mesh(track(new THREE.SphereGeometry(0.05 + eff * 0.05, 12, 12)), track(new THREE.MeshBasicMaterial({ color: 0xffffff })));
      group.add(dot);
    }

    const hit = new THREE.Mesh(track(new THREE.SphereGeometry(0.42, 8, 8)), track(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false })));
    hit.userData.index = i;
    group.add(hit);
    hitTargets.push(hit);

    const { obj: label, el } = makeLabel("star-label" + (bad ? " bad" : ""), a.text);
    group.add(label);

    stars.push({ a, i, group, sprite, baseScale, label, el, target, delay: reduceMotion ? 0 : 0.25 + i * 0.06 });
  });

  // ---- 聯想橋 ----
  const bridgeGroup = new THREE.Group();
  scene.add(bridgeGroup);
  let bridgeAnim = null;

  function clearBridge() {
    bridgeGroup.children.slice().forEach((o) => {
      bridgeGroup.remove(o);
      o.traverse((c) => {
        c.geometry?.dispose();
        c.material?.dispose();
        if (c.isCSS2DObject) c.element.remove();
      });
    });
    bridgeAnim = null;
  }

  function drawBridge(star) {
    clearBridge();
    const a = star.a;
    const end = star.target.clone();
    const steps = Array.isArray(a.bridge) ? a.bridge : [];
    const rnd = seeded(star.i);
    const dir = end.clone().normalize();
    const len = end.length();
    // 和方向垂直的兩個軸，用來加偏移（API 沒有給中間概念的座標）
    const helper = Math.abs(dir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(dir, helper).normalize();
    const v = new THREE.Vector3().crossVectors(dir, u).normalize();
    const pts = [new THREE.Vector3(0, 0, 0)];
    const mids = [];
    const n = steps.length;
    if (n === 0) {
      const mid = end.clone().multiplyScalar(0.5).addScaledVector(u, len * 0.12);
      pts.push(mid);
    } else {
      const real = Array.isArray(a.bridge_positions) && a.bridge_positions.length === n;
      steps.forEach((_, k) => {
        if (real) {
          // 後端提供的真實語義位置（SPEC 4.1）
          const p = toVec(a.bridge_positions[k]);
          pts.push(p);
          mids.push(p);
          return;
        }
        const t = (k + 1) / (n + 1);
        const off = len * (0.12 + rnd() * 0.12) * Math.sin(Math.PI * t);
        const ang = rnd() * Math.PI * 2;
        const p = end.clone().multiplyScalar(t)
          .addScaledVector(u, Math.cos(ang) * off)
          .addScaledVector(v, Math.sin(ang) * off);
        pts.push(p);
        mids.push(p);
      });
    }
    pts.push(end);
    const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
    const segs = 120;
    const isBad = a.verdict === "unreasonable";
    const color = isBad ? COLOR_BAD : COLOR_ACCENT;

    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segs, 0.028, 8, false),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: isBad ? 0.35 : 0.95, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    const glow = new THREE.Mesh(
      new THREE.TubeGeometry(curve, segs, 0.11, 8, false),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: isBad ? 0.06 : 0.16, blending: THREE.AdditiveBlending, depthWrite: false }),
    );
    bridgeGroup.add(glow, tube);
    const total = tube.geometry.index.count;
    tube.geometry.setDrawRange(0, 0);
    glow.geometry.setDrawRange(0, 0);

    const nodes = mids.map((p, k) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: COLOR_ACCENT, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
      s.scale.setScalar(0.6);
      s.position.copy(p);
      const { obj: l, el } = makeLabel("bridge-label", steps[k]);
      el.style.opacity = "0";
      s.add(l);
      bridgeGroup.add(s);
      return { s, el, t: (k + 1) / (n + 1) };
    });
    bridgeAnim = { start: performance.now(), dur: reduceMotion ? 1 : 900, tube, glow, total, nodes };
  }

  // ---- 互動 ----
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hovered = null;
  let locked = null;
  let active = null;

  function fmt(n, d = 2) {
    return n === null || n === undefined || Number.isNaN(n) ? "—" : Number(n).toFixed(d);
  }

  function renderCard(star) {
    const a = star.a;
    const bridge = Array.isArray(a.bridge) && a.bridge.length
      ? `<div class="sc-bridge">${esc(data.stimulus?.word)} → ${a.bridge.map(esc).join(" → ")} → ${esc(a.text)}</div>`
      : "";
    card.innerHTML = `
      <div class="sc-text">${esc(a.text)}</div>
      <dl class="sc-grid">
        <dt>語義距離</dt><dd>${fmt(a.distance)}</dd>
        <dt>有效距離</dt><dd>${fmt(a.effective_distance)}</dd>
        <dt>稀有度</dt><dd>${a.rarity === null || a.rarity === undefined ? "樣本不足" : fmt(a.rarity)}</dd>
        <dt>合理性</dt><dd><span class="verdict ${esc(a.verdict)}">${VERDICT_TEXT[a.verdict] || esc(a.verdict)}</span></dd>
      </dl>
      ${bridge}
      ${a.reason ? `<div class="sc-reason">${esc(a.reason)}</div>` : ""}
      ${a.note ? `<div class="sc-note">「${esc(a.note)}」</div>` : ""}`;
  }

  function positionCard(star) {
    const p = new THREE.Vector3();
    star.group.getWorldPosition(p);
    p.project(camera);
    const x = (p.x * 0.5 + 0.5) * width;
    const y = (-p.y * 0.5 + 0.5) * height;
    const cw = card.offsetWidth || 250;
    const ch = card.offsetHeight || 160;
    let left = x + 18;
    let top = y - ch / 2;
    if (left + cw > width - 12) left = x - cw - 18;
    left = Math.max(12, left);
    top = Math.max(12, Math.min(height - ch - 12, top));
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
  }

  function setActive(star) {
    if (active === star) return;
    active = star;
    stars.forEach((s) => {
      s.el.classList.toggle("on", s === star);
      s.el.classList.toggle("dim", !!star && s !== star);
    });
    if (star) {
      renderCard(star);
      card.classList.add("show");
      drawBridge(star);
    } else {
      card.classList.remove("show");
      clearBridge();
    }
    opts.onSelect?.(star ? star.i : null);
  }

  function pick(ev) {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(hitTargets, false)[0];
    return hit ? stars[hit.object.userData.index] : null;
  }

  function onMove(ev) {
    if (ev.pointerType === "touch") return;
    const s = pick(ev);
    renderer.domElement.style.cursor = s ? "pointer" : "grab";
    if (s !== hovered) {
      hovered = s;
      if (!locked) setActive(s);
    }
  }
  let downAt = null;
  function onDown(ev) {
    downAt = { x: ev.clientX, y: ev.clientY };
    pauseAutoRotate();
  }
  function onUp(ev) {
    if (!downAt) return;
    const moved = Math.hypot(ev.clientX - downAt.x, ev.clientY - downAt.y);
    downAt = null;
    if (moved > 5) return; // 拖曳不算點擊
    const s = pick(ev);
    locked = s;
    setActive(s);
  }
  function onLeave() {
    hovered = null;
    if (!locked) setActive(null);
  }
  renderer.domElement.addEventListener("pointermove", onMove);
  renderer.domElement.addEventListener("pointerdown", onDown);
  renderer.domElement.addEventListener("pointerup", onUp);
  renderer.domElement.addEventListener("pointerleave", onLeave);
  renderer.domElement.style.cursor = "grab";
  renderer.domElement.style.touchAction = "none";

  // 閒置時緩慢自轉
  let idleTimer = null;
  function pauseAutoRotate() {
    controls.autoRotate = false;
    clearTimeout(idleTimer);
    if (!reduceMotion) idleTimer = setTimeout(() => { if (!locked) controls.autoRotate = true; }, 4000);
  }
  controls.addEventListener("start", pauseAutoRotate);
  renderer.domElement.addEventListener("wheel", pauseAutoRotate, { passive: true });

  // ---- 尺寸 ----
  function resize() {
    width = container.clientWidth || width;
    height = container.clientHeight || height;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labelRenderer.setSize(width, height);
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);

  // ---- 動畫迴圈 ----
  const t0 = performance.now();
  let raf = 0;
  let disposed = false;
  function frame(now) {
    if (disposed) return;
    raf = requestAnimationFrame(frame);
    const t = (now - t0) / 1000;

    cloudUniforms.uTime.value = t;
    cloudUniforms.uGrow.value = reduceMotion ? 1 : easeOutExpo(Math.min(1, t / 2.2));

    const pulse = 1 + Math.sin(t * 2) * 0.06;
    halo.scale.setScalar(3.4 * pulse);

    for (const s of stars) {
      const k = reduceMotion ? 1 : easeOutExpo(Math.max(0, Math.min(1, (t - s.delay) / 1.6)));
      s.group.position.copy(s.target).multiplyScalar(k);
      s.label.visible = k > 0.6;
      const tw = 1 + Math.sin(t * 2.4 + s.i) * 0.05;
      const boost = s === active ? 1.45 : 1;
      s.sprite.scale.setScalar(s.baseScale * tw * boost * Math.max(0.2, k));
    }

    if (bridgeAnim) {
      const p = Math.min(1, (now - bridgeAnim.start) / bridgeAnim.dur);
      const e = easeOutExpo(p);
      const count = Math.floor((bridgeAnim.total * e) / 3) * 3;
      bridgeAnim.tube.geometry.setDrawRange(0, count);
      bridgeAnim.glow.geometry.setDrawRange(0, count);
      for (const nd of bridgeAnim.nodes) {
        const on = e >= nd.t ? 1 : 0;
        nd.s.material.opacity = on * 0.9;
        nd.el.style.opacity = String(on);
      }
    }

    controls.update();
    if (active) positionCard(active);
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
  }
  raf = requestAnimationFrame(frame);

  return {
    /** 從外部（例如面板上的聯想清單）選取一顆星 */
    select(index) {
      const s = index === null || index === undefined ? null : stars[index];
      locked = s;
      if (s) pauseAutoRotate();
      setActive(s);
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      clearTimeout(idleTimer);
      ro.disconnect();
      clearBridge();
      controls.dispose();
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      renderer.domElement.removeEventListener("pointerleave", onLeave);
      disposables.forEach((d) => d.dispose?.());
      renderer.dispose();
      renderer.domElement.remove();
      labelRenderer.domElement.remove();
      card.remove();
    },
  };
}
