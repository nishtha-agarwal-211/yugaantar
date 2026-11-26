'use strict';

/* ------------------------------------------------------------------ *
 * Multiverse of Monsoon — interactive dashboard
 *
 * Rendering is a plain canvas equirectangular map of the contract grid
 * (32 x 36 cells, lat 6.5..37.5, lon 65.5..100.5). Every rollout shown
 * comes from a live POST to the model server's /simulate endpoint; the
 * timeline only indexes into the 10 returned steps.
 * ------------------------------------------------------------------ */

const NC = 6;
const STEPS = 10;
const GRID = { nlat: 32, nlon: 36, lat0: 6.5, lon0: 65.5 };

const VIEWS = {
  t2m: {
    label: '2 m temperature', short: 't2m', unit: '°C', channels: [0],
    min: 6, max: 44, warp: 'lin',
    stops: [[0, [26, 40, 106]], [0.2, [47, 111, 208]], [0.37, [67, 176, 184]],
            [0.53, [143, 211, 106]], [0.68, [242, 199, 68]], [0.84, [238, 122, 43]],
            [1, [168, 31, 31]]],
  },
  tp: {
    label: 'Daily rainfall', short: 'tp', unit: 'mm/day', channels: [1],
    min: 0, max: 45, warp: 'sqrt',
    stops: [[0, [8, 16, 26]], [0.08, [18, 52, 84]], [0.2, [31, 111, 139]],
            [0.36, [56, 163, 165]], [0.55, [127, 208, 122]], [0.74, [242, 217, 78]],
            [0.9, [232, 100, 44]], [1, [181, 35, 107]]],
  },
  wind: {
    label: '10 m wind', short: 'u10 / v10', unit: 'm/s', channels: [2, 3],
    min: 0, max: 12, warp: 'lin', flow: true,
    stops: [[0, [16, 35, 61]], [0.22, [29, 95, 138]], [0.45, [47, 158, 154]],
            [0.68, [168, 217, 90]], [0.85, [242, 178, 60]], [1, [224, 82, 58]]],
  },
  msl: {
    label: 'Mean sea-level pressure', short: 'msl', unit: 'hPa', channels: [4],
    min: 996, max: 1016, warp: 'lin',
    stops: [[0, [122, 47, 138]], [0.25, [156, 86, 140]], [0.45, [217, 154, 138]],
            [0.55, [233, 227, 216]], [0.72, [168, 200, 232]], [1, [63, 119, 181]]],
  },
  tcwv: {
    label: 'Column water vapour', short: 'tcwv', unit: 'kg/m²', channels: [5],
    min: 5, max: 70, warp: 'lin',
    stops: [[0, [59, 42, 27]], [0.2, [122, 90, 55]], [0.38, [168, 154, 90]],
            [0.56, [111, 174, 114]], [0.74, [63, 143, 158]], [0.88, [43, 95, 158]],
            [1, [41, 48, 122]]],
  },
};

const CHANNEL_BUTTONS = [
  ['t2m', 'Temperature'],
  ['tp', 'Rainfall'],
  ['wind', 'Wind'],
  ['msl', 'Pressure'],
  ['tcwv', 'Water vapour'],
];

const DIFF_STOPS = [
  [0, [47, 111, 208]], [0.3, [143, 184, 230]], [0.5, [232, 238, 243]],
  [0.7, [240, 176, 106]], [1, [210, 69, 47]],
];

const S = {
  apiBase: 'http://localhost:8000',
  grid: GRID,
  presets: [],
  shockTypes: {},
  windows: [],
  windowId: null,
  init: null,
  initJSON: '',
  doyJSON: '',
  prime: null,
  multi: null,
  multiSeq: 0,
  steps: STEPS,
  view: 't2m',
  day: 0,
  playing: false,
  playTimer: null,
  shock: {
    active: true,
    center: { lat: 13.5, lon: 88 },
    radius: 6,
    components: [{ type: 'thermal', value: 3 }, { type: 'moisture', value: 15 }],
  },
  activePreset: 'bob_warm',
  pending: false,
  latency: null,
  inferenceMs: null,
  error: null,
  health: null,
  geo: [],
  diffCache: null,
  seq: 0,
  // impact co-pilot
  primeNested: null,
  multiNested: null,
  initNested: null,
  impact: null,
  impactError: null,
  impactPending: false,
  impactSeq: 0,
  impactMetric: 'risk',
  impactScenario: 'prime',
};

const views = [];
const byId = (id) => document.getElementById(id);

window.addEventListener('error', (ev) => {
  console.error('[dashboard] window error:', ev.message, ev.error && ev.error.stack);
});
window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  console.error('[dashboard] unhandled rejection:', (r && r.stack) || r);
});

/* ------------------------------------------------------------------ maths */

function idx(d, c, i, j) {
  return ((d * NC + c) * GRID.nlat + i) * GRID.nlon + j;
}

function flatten(nested) {
  const d = nested.length, c = nested[0].length;
  const i = nested[0][0].length, j = nested[0][0][0].length;
  const out = new Float32Array(d * c * i * j);
  let k = 0;
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < c; b++) {
      const plane = nested[a][b];
      for (let r = 0; r < i; r++) {
        const row = plane[r];
        for (let q = 0; q < j; q++) out[k++] = row[q];
      }
    }
  }
  return out;
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function rampColor(stops, t) {
  t = clamp(t, 0, 1);
  for (let k = 1; k < stops.length; k++) {
    if (t <= stops[k][0] || k === stops.length - 1) {
      const t0 = stops[k - 1][0], t1 = stops[k][0] || 1;
      const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
      const a = stops[k - 1][1], b = stops[k][1];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    }
  }
  return [0, 0, 0];
}

function rampCss(stops) {
  const parts = stops.map(([t, c]) => `rgb(${c[0]},${c[1]},${c[2]}) ${(t * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

function normalize(v, lo, hi, warp) {
  let t = (v - lo) / ((hi - lo) || 1);
  if (!isFinite(t)) t = 0;
  t = clamp(t, 0, 1);
  if (warp === 'sqrt') t = Math.sqrt(t);
  return t;
}

/** Raw channel value -> display unit (K -> °C, Pa -> hPa). */
function displayValue(raw, view) {
  if (view === 't2m') return raw - 273.15;
  if (view === 'msl') return raw / 100;
  return raw;
}

/** Scalar a panel shows for the active view: speed for wind, else the channel. */
function panelValue(arr, view, day, i, j) {
  if (!arr) return NaN;
  if (view === 'wind') {
    return Math.hypot(arr[idx(day, 2, i, j)], arr[idx(day, 3, i, j)]);
  }
  return displayValue(arr[idx(day, VIEWS[view].channels[0], i, j)], view);
}

function diffAt(day, i, j) {
  if (!S.prime || !S.multi) return 0;
  const a = panelValue(S.multi, S.view, day, i, j);
  const b = panelValue(S.prime, S.view, day, i, j);
  return a - b;
}

function fmt(v, digits) {
  if (!isFinite(v)) return '–';
  const d = digits === undefined ? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2) : digits;
  return v.toFixed(d);
}

function signed(v, digits) {
  return (v > 0 ? '+' : '') + fmt(v, digits);
}

/* ------------------------------------------------------------------ views */

function makeView(el) {
  const base = el.querySelector('canvas.base');
  const flowCanvas = el.querySelector('canvas.flow');
  const wrap = el.querySelector('.canvas-wrap');
  const v = {
    el,
    mode: el.dataset.mode,
    wrap,
    tooltip: el.querySelector('.tooltip'),
    base,
    ctx: base.getContext('2d'),
    flowCanvas,
    flowCtx: flowCanvas ? flowCanvas.getContext('2d') : null,
    off: document.createElement('canvas'),
    cssW: 0, cssH: 0, dpr: 1, proj: null,
    flow: null,
  };
  v.off.width = GRID.nlon;
  v.off.height = GRID.nlat;
  v.offCtx = v.off.getContext('2d');
  v.img = v.offCtx.createImageData(GRID.nlon, GRID.nlat);
  if (flowCanvas) v.flow = makeFlow(v);

  const resize = () => {
    const w = Math.max(40, wrap.clientWidth);
    const h = Math.max(40, wrap.clientHeight);
    const dpr = window.devicePixelRatio || 1;
    v.cssW = w; v.cssH = h; v.dpr = dpr;
    for (const cvs of [base, flowCanvas]) {
      if (!cvs) continue;
      cvs.width = Math.round(w * dpr);
      cvs.height = Math.round(h * dpr);
      const c = cvs.getContext('2d');
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    if (v.flow) v.flow.reset();
    paint(v);
  };

  if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
  else window.addEventListener('resize', resize);

  wrap.addEventListener('mousemove', (ev) => onHover(v, ev));
  wrap.addEventListener('mouseleave', () => { v.tooltip.hidden = true; });
  wrap.addEventListener('click', (ev) => onClickMap(v, ev));

  v.resize = resize;
  return v;
}

function projection(w, h) {
  const latSpan = GRID.nlat;              // 32 degrees
  const lonSpan = GRID.nlon;              // 36 degrees
  const cosLat = Math.cos((22 * Math.PI) / 180);
  const scale = Math.min(w / (lonSpan * cosLat), h / latSpan);
  const gw = lonSpan * cosLat * scale;
  const gh = latSpan * scale;
  return {
    gx: (w - gw) / 2, gy: (h - gh) / 2, gw, gh, scale, cosLat,
    cellW: gw / GRID.nlon, cellH: gh / GRID.nlat,
    px: (lon) => (w - gw) / 2 + (lon - (GRID.lon0 - 0.5)) * cosLat * scale,
    py: (lat) => (h - gh) / 2 + (GRID.lat0 + GRID.nlat + 0.5 - lat) * scale,
  };
}

/* ------------------------------------------------------------------ paint */

function paint(v) {
  const { ctx, cssW, cssH } = v;
  if (!cssW || !cssH) return;
  const p = projection(cssW, cssH);
  v.proj = p;

  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = '#070b12';
  ctx.fillRect(0, 0, cssW, cssH);

  if (!S.prime) {
    drawNote(ctx, cssW, cssH, S.pending ? 'running rollout…' : 'waiting for a live rollout');
    return;
  }

  const view = VIEWS[S.view];
  const isDiff = v.mode === 'diff';
  const stops = isDiff ? DIFF_STOPS : view.stops;
  const lo = isDiff ? -diffScale() : view.min;
  const hi = isDiff ? diffScale() : view.max;
  const data = v.img.data;

  for (let i = 0; i < GRID.nlat; i++) {
    for (let j = 0; j < GRID.nlon; j++) {
      const val = isDiff
        ? diffAt(S.day, i, j)
        : panelValue(v.mode === 'prime' ? S.prime : (S.multi || S.prime), S.view, S.day, i, j);
      const rgb = rampColor(stops, normalize(val, lo, hi, isDiff ? 'lin' : view.warp));
      const o = ((GRID.nlat - 1 - i) * GRID.nlon + j) * 4;
      data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2]; data[o + 3] = 255;
    }
  }
  v.offCtx.putImageData(v.img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(v.off, p.gx, p.gy, p.gw, p.gh);

  drawGraticule(ctx, p, cssW, cssH);
  drawCoast(ctx, p);
  drawShock(ctx, p, cssW, cssH);

  if (v.flow) {
    const showFlow = view.flow && v.mode !== 'diff' && !!(v.mode === 'prime' ? S.prime : (S.multi || S.prime));
    v.flow.setActive(showFlow, v.mode === 'prime' ? S.prime : (S.multi || S.prime));
  }
}

function drawNote(ctx, w, h, text) {
  ctx.fillStyle = 'rgba(200, 220, 235, 0.45)';
  ctx.font = '12px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(text, w / 2, h / 2);
  ctx.textAlign = 'left';
}

function drawGraticule(ctx, p, w, h) {
  ctx.save();
  ctx.strokeStyle = 'rgba(180, 205, 230, 0.10)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (let lat = 10; lat <= 35; lat += 5) {
    ctx.moveTo(p.gx, p.py(lat)); ctx.lineTo(p.gx + p.gw, p.py(lat));
  }
  for (let lon = 70; lon <= 100; lon += 5) {
    ctx.moveTo(p.px(lon), p.gy); ctx.lineTo(p.px(lon), p.gy + p.gh);
  }
  ctx.stroke();

  ctx.strokeStyle = 'rgba(150, 180, 210, 0.22)';
  ctx.strokeRect(p.gx, p.gy, p.gw, p.gh);

  ctx.fillStyle = 'rgba(190, 212, 232, 0.42)';
  ctx.font = '9px ui-monospace, monospace';
  for (let lat = 10; lat <= 35; lat += 10) ctx.fillText(`${lat}°N`, p.gx + 3, p.py(lat) - 3);
  for (let lon = 70; lon <= 100; lon += 10) ctx.fillText(`${lon}°E`, p.px(lon) - 9, p.gy + p.gh - 3);
  ctx.restore();
}

function drawCoast(ctx, p) {
  if (!S.geo.length) return;
  ctx.save();
  ctx.lineJoin = 'round';
  for (const ring of S.geo) {
    ctx.beginPath();
    ring.forEach(([lon, lat], k) => {
      const x = p.px(lon), y = p.py(lat);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = 'rgba(9, 14, 21, 0.65)';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.strokeStyle = 'rgba(216, 234, 246, 0.72)';
    ctx.lineWidth = 1.1;
    ctx.stroke();
  }
  ctx.restore();
}

function drawShock(ctx, p, w, h) {
  if (!S.shock.active) return;
  const { lat, lon } = S.shock.center;
  const x = p.px(lon), y = p.py(lat);
  const r = S.shock.radius * p.scale;

  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(240, 163, 94, 0.10)';
  ctx.fill();
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = 'rgba(245, 190, 120, 0.9)';
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.moveTo(x - 6, y); ctx.lineTo(x + 6, y);
  ctx.moveTo(x, y - 6); ctx.lineTo(x, y + 6);
  ctx.strokeStyle = '#ffd9a3';
  ctx.lineWidth = 1.4;
  ctx.stroke();

  ctx.fillStyle = 'rgba(255, 226, 186, 0.92)';
  ctx.font = '9.5px ui-monospace, monospace';
  const label = `${S.shock.center.lat.toFixed(1)}N ${S.shock.center.lon.toFixed(1)}E`;
  const tw = ctx.measureText(label).width;
  const lx = clamp(x + 9, 2, w - tw - 4);
  const ly = clamp(y - 8, 11, h - 4);
  ctx.fillText(label, lx, ly);
  ctx.restore();
}

/* -------------------------------------------------------------- wind flow */

function makeFlow(v) {
  const N = 900;
  const px = new Float32Array(N);
  const py = new Float32Array(N);
  const age = new Uint16Array(N);
  const st = { active: false, field: null, last: 0, ready: false };

  function seed(k) {
    px[k] = Math.random() * GRID.nlon;
    py[k] = Math.random() * GRID.nlat;
    age[k] = (Math.random() * 120) | 0;
  }
  for (let k = 0; k < N; k++) seed(k);

  st.reset = () => {
    for (let k = 0; k < N; k++) seed(k);
    st.ready = false;
    if (v.flowCtx) v.flowCtx.clearRect(0, 0, v.cssW, v.cssH);
  };

  st.setActive = (on, field) => {
    st.active = !!on;
    st.field = field || null;
    if (v.flowCanvas) v.flowCanvas.style.opacity = on ? '1' : '0';
    if (!on && v.flowCtx) v.flowCtx.clearRect(0, 0, v.cssW, v.cssH);
  };

  function sample(field, fx, fy) {
    // fx in [0, 36) lon-index space, fy in [0, 32) row space from the north.
    const ifloat = 31.5 - fy;
    const jfloat = fx - 0.5;
    const i0 = clamp(Math.floor(ifloat), 0, GRID.nlat - 1);
    const j0 = clamp(Math.floor(jfloat), 0, GRID.nlon - 1);
    const i1 = clamp(i0 + 1, 0, GRID.nlat - 1);
    const j1 = clamp(j0 + 1, 0, GRID.nlon - 1);
    const fi = clamp(ifloat - i0, 0, 1);
    const fj = clamp(jfloat - j0, 0, 1);
    const d = S.day;
    const w00 = (1 - fi) * (1 - fj), w10 = fi * (1 - fj), w01 = (1 - fi) * fj, w11 = fi * fj;
    const u = field[idx(d, 2, i0, j0)] * w00 + field[idx(d, 2, i1, j0)] * w10
            + field[idx(d, 2, i0, j1)] * w01 + field[idx(d, 2, i1, j1)] * w11;
    const w = field[idx(d, 3, i0, j0)] * w00 + field[idx(d, 3, i1, j0)] * w10
            + field[idx(d, 3, i0, j1)] * w01 + field[idx(d, 3, i1, j1)] * w11;
    return [u, w];
  }

  st.step = (dt) => {
    const ctx = v.flowCtx;
    if (!ctx || !st.active || !st.field || !v.proj) return;
    const p = v.proj;

    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = `rgba(0,0,0,${clamp(0.11 + dt * 0.6, 0.06, 0.3)})`;
    ctx.fillRect(0, 0, v.cssW, v.cssH);
    ctx.restore();

    if (!st.ready) { st.resetParticles = true; st.ready = true; }

    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(238, 250, 255, 0.62)';
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    const speedFactor = 0.55;     // m/s -> cells/s, exaggerated for readability
    for (let k = 0; k < N; k++) {
      const [u, w] = sample(st.field, px[k], py[k]);
      const vx = u * speedFactor * dt;
      const vy = -w * speedFactor * dt;
      const nx = px[k] + vx, ny = py[k] + vy;
      const x0 = p.gx + px[k] * p.cellW, y0 = p.gy + py[k] * p.cellH;
      const x1 = p.gx + nx * p.cellW, y1 = p.gy + ny * p.cellH;
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      px[k] = nx; py[k] = ny; age[k] += 1;
      if (nx < 0 || nx > GRID.nlon || ny < 0 || ny > GRID.nlat || age[k] > 170) seed(k);
    }
    ctx.stroke();
    ctx.restore();
  };

  return st;
}

let lastFrame = 0;
function animationLoop(ts) {
  requestAnimationFrame(animationLoop);
  const dt = lastFrame ? Math.min(0.25, (ts - lastFrame) / 1000) : 0.016;
  if (ts - lastFrame < 45) return;
  lastFrame = ts;
  for (const v of views) {
    if (v.flow && v.flow.active) v.flow.step(dt);
  }
}

/* ------------------------------------------------------------- diff scale */

function defaultDiffScale(view) {
  return { t2m: 1.5, tp: 2.0, wind: 0.6, msl: 2.0, tcwv: 2.0 }[view] || 1;
}

function diffScale() {
  const key = `${S.view}:${S.multiSeq}:${S.windowId}`;
  if (S.diffCache && S.diffCache.key === key) return S.diffCache.scale;
  let max = 0;
  if (S.prime && S.multi) {
    for (let d = 0; d < S.steps; d++) {
      for (let i = 0; i < GRID.nlat; i++) {
        for (let j = 0; j < GRID.nlon; j++) {
          const a = Math.abs(diffAt(d, i, j));
          if (a > max) max = a;
        }
      }
    }
  }
  const scale = max > 1e-6 ? max : defaultDiffScale(S.view);
  S.diffCache = { key, scale, max };
  return scale;
}

/* ------------------------------------------------------------ shock model */

function buildPerturbation() {
  if (!S.shock.active || !S.init) return null;
  const out = [];
  for (let c = 0; c < NC; c++) {
    const plane = [];
    for (let i = 0; i < GRID.nlat; i++) plane.push(new Array(GRID.nlon).fill(0));
    out.push(plane);
  }
  const comps = S.shock.components;
  const R = S.shock.radius;
  const clat = S.shock.center.lat;
  const clon = S.shock.center.lon;

  for (let i = 0; i < GRID.nlat; i++) {
    const lat = GRID.lat0 + i;
    const dLat = lat - clat;
    for (let j = 0; j < GRID.nlon; j++) {
      const lon = GRID.lon0 + j;
      const dLon = (lon - clon) * Math.cos((lat * Math.PI) / 180);
      const dist = Math.hypot(dLat, dLon);
      if (dist > R) continue;
      const wgt = 0.5 * (1 + Math.cos((Math.PI * dist) / R));
      for (const comp of comps) {
        const spec = S.shockTypes[comp.type];
        if (!spec) continue;
        const base = comp.value * wgt;
        for (const c of spec.channels) {
          if (spec.mode === 'percent') {
            out[c][i][j] += (base / 100) * S.init[idx(2, c, i, j)];
          } else {
            out[c][i][j] += base * (spec.unit_scale || 1);
          }
        }
      }
    }
  }
  return out;
}

/* --------------------------------------------------------------- requests */

async function postSimulate(perturbation, signal) {
  const body = '{"init_states":' + S.initJSON
    + ',"doy_sincos":' + S.doyJSON
    + ',"steps":' + S.steps
    + ',"perturbation":' + JSON.stringify(perturbation) + '}';
  const t0 = performance.now();
  const res = await fetch(S.apiBase + '/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${txt.slice(0, 160)}`);
  }
  const json = await res.json();
  const ms = performance.now() - t0;
  return {
    rollout: flatten(json.rollout),
    nested: json.rollout,
    ms,
    inferenceMs: json.inference_ms,
  };
}

let inFlight = null;

async function runBaseline() {
  S.prime = null;
  S.multi = null;
  S.diffCache = null;
  S.pending = true;
  setBusy(true);
  render();
  try {
    const r = await postSimulate(null, null);
    S.prime = r.rollout;
    S.multi = r.rollout;
    S.primeNested = r.nested;
    S.multiNested = r.nested;
    S.multiSeq += 1;
    S.latency = r.ms;
    S.inferenceMs = r.inferenceMs;
    S.error = null;
  } catch (err) {
    S.error = String(err.message || err);
    S.prime = null;
    S.multi = null;
    S.primeNested = null;
    S.multiNested = null;
  } finally {
    S.pending = false;
    setBusy(false);
    render();
    if (S.prime) runShock();
  }
}

let shockTimer = null;
function scheduleShock(delay) {
  if (!S.prime) return;
  clearTimeout(shockTimer);
  shockTimer = setTimeout(runShock, delay === undefined ? 130 : delay);
}

async function runShock() {
  if (!S.prime) return;
  if (!S.shock.active) {
    S.multi = S.prime;
    S.multiNested = S.primeNested;
    S.multiSeq += 1;
    S.diffCache = null;
    S.latency = 0;
    render();
    refreshImpact();
    return;
  }
  const seq = ++S.seq;
  if (inFlight) inFlight.abort();
  inFlight = new AbortController();
  S.pending = true;
  setBusy(true);
  try {
    const r = await postSimulate(buildPerturbation(), inFlight.signal);
    if (seq !== S.seq) return;
    S.multi = r.rollout;
    S.multiNested = r.nested;
    S.multiSeq += 1;
    S.diffCache = null;
    S.latency = r.ms;
    S.inferenceMs = r.inferenceMs;
    S.error = null;
  } catch (err) {
    if (err && err.name === 'AbortError') return;
    if (seq !== S.seq) return;
    S.error = String(err.message || err);
    S.multi = null;
    S.multiNested = null;
  } finally {
    if (seq === S.seq) {
      S.pending = false;
      setBusy(false);
      updateStatus();
      render();
      refreshImpact();
    }
  }
}

/* ------------------------------------------------------------------- boot */

async function boot() {
  views.push(...Array.from(document.querySelectorAll('.view')).map(makeView));
  requestAnimationFrame(animationLoop);

  try {
    const boot = await (await fetch('/api/bootstrap')).json();
    S.apiBase = boot.api_base || S.apiBase;
    S.presets = boot.presets || [];
    S.shockTypes = boot.shock_types || {};
    S.windows = boot.windows || [];
    Object.assign(GRID, boot.grid || {});

    const qs = new URLSearchParams(location.search);
    if (qs.get('api')) S.apiBase = qs.get('api');

    await loadGeo();
    initImpact();
    renderChannelButtons();
    renderWindowButtons();
    renderShockTypeSelect();
    const first = S.presets.find((p) => p.id === 'bob_warm') || S.presets[0];
    if (first) applyPreset(first);
    syncShockUi();
    updateLegend();

    await loadWindow(boot.default_window || S.windows[0].id);
    checkHealth();
    setInterval(checkHealth, 20000);
  } catch (err) {
    console.error('[dashboard] boot failed:', (err && err.stack) || err);
    S.error = String(err.message || err);
    updateStatus();
    for (const v of views) paint(v);
  }
}

async function loadGeo() {
  try {
    const doc = await (await fetch('/static/geo/india.json')).json();
    const rings = [];
    for (const f of doc.features || []) {
      const g = f.geometry || {};
      const polys = g.type === 'Polygon' ? [g.coordinates] : (g.coordinates || []);
      for (const poly of polys) if (poly && poly[0]) rings.push(poly[0]);
    }
    S.geo = rings;
  } catch (err) {
    S.geo = [];
  }
}

async function loadWindow(id) {
  S.windowId = id;
  S.pending = true;
  setBusy(true);
  render();
  const res = await fetch('/api/window/' + encodeURIComponent(id));
  if (!res.ok) throw new Error(`window ${id}: HTTP ${res.status}`);
  const win = await res.json();
  S.init = flatten(win.init);
  S.initNested = win.init;
  S.initJSON = JSON.stringify(win.init);
  S.doyJSON = JSON.stringify(win.doy_sincos);
  S.multi = null;
  S.primeNested = null;
  S.multiNested = null;
  S.impact = null;
  S.diffCache = null;
  renderWindowButtons();
  byId('windowHint').textContent = `${win.label} · 3-day analysis at day-of-year ${win.doy} (synthetic).`;
  await runBaseline();
}

async function checkHealth() {
  try {
    const h = await (await fetch('/api/health')).json();
    S.health = h;
    updateStatus();
    updateBanner();
  } catch (err) {
    S.health = { reachable: false, error: String(err.message || err) };
    updateStatus();
  }
}

/* ------------------------------------------------------------------- UI */

function setBusy(on) {
  document.body.classList.toggle('busy', !!on);
  document.body.classList.toggle('busy-done', !on && !!S.prime);
  updateStatus();
}

function updateStatus() {
  const dot = byId('dot');
  const text = byId('statusText');
  const lat = byId('latency');
  dot.className = 'dot';
  lat.textContent = '';

  if (S.error) {
    dot.classList.add('bad');
    text.textContent = 'error · ' + S.error.slice(0, 90);
    return;
  }
  if (!S.health) {
    text.textContent = 'connecting…';
    return;
  }
  if (!S.health.reachable) {
    dot.classList.add('bad');
    text.textContent = 'model server offline — start python serve.py --port 8000';
    return;
  }
  if (S.pending) {
    dot.classList.add('busy');
    text.textContent = 'running /simulate…';
    return;
  }
  if ((S.health.status || '') !== 'ready') {
    dot.classList.add('busy');
    text.textContent = `model ${S.health.status || 'loading'}`;
    return;
  }
  dot.classList.add('ok');
  text.textContent = S.health.model_name || 'model ready';
  if (S.latency !== null) {
    lat.textContent = `${Math.round(S.latency)} ms`;
    lat.title = S.inferenceMs ? `server inference ${S.inferenceMs.toFixed(1)} ms` : '';
  }
}

function updateBanner() {
  const el = byId('banner');
  const name = (S.health && S.health.model_name) || '';
  if (name.includes('fallback')) {
    el.hidden = false;
    el.textContent = 'model_weights.pt is not loaded, so /simulate is serving its persistence fallback: '
      + 'the difference panel shows the injected shock but no wave propagation.';
  } else {
    el.hidden = true;
  }
}

function renderChannelButtons() {
  const host = byId('channels');
  host.innerHTML = '';
  for (const [key, label] of CHANNEL_BUTTONS) {
    const b = document.createElement('button');
    b.dataset.channel = key;
    b.innerHTML = `${label}<span class="k">${VIEWS[key].short}</span>`;
    b.classList.toggle('on', key === S.view);
    b.addEventListener('click', () => {
      S.view = key;
      renderChannelButtons();
      updateLegend();
      for (const v of views) { if (v.flow) v.flow.reset(); }
      render();
    });
    host.appendChild(b);
  }
}

function renderWindowButtons() {
  const host = byId('windows');
  host.innerHTML = '';
  for (const w of S.windows) {
    const b = document.createElement('button');
    b.textContent = w.label;
    b.classList.toggle('on', w.id === S.windowId);
    b.addEventListener('click', () => {
      if (w.id === S.windowId) return;
      loadWindow(w.id).catch((err) => { S.error = String(err.message || err); updateStatus(); });
    });
    host.appendChild(b);
  }
}

function renderPresets() {
  const host = byId('presets');
  host.innerHTML = '';
  for (const p of S.presets) {
    const b = document.createElement('button');
    b.className = 'preset' + (p.id === S.activePreset ? ' on' : '');
    b.innerHTML = `${p.label}<small>${p.detail}</small>`;
    b.addEventListener('click', () => applyPreset(p));
    host.appendChild(b);
  }
}

function renderShockTypeSelect() {
  const sel = byId('shockType');
  sel.innerHTML = '';
  for (const [key, spec] of Object.entries(S.shockTypes)) {
    const o = document.createElement('option');
    o.value = key;
    o.textContent = `${spec.label} (${spec.unit})`;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => {
    const spec = S.shockTypes[sel.value];
    S.shock.components = [{ type: sel.value, value: spec.default }];
    S.shock.active = true;
    S.activePreset = null;
    renderPresets();
    renderShockSliders();
    scheduleShock(0);
  });
}

function renderShockSliders() {
  const host = byId('shockSliders');
  host.innerHTML = '';
  S.shock.components.forEach((comp, k) => {
    const spec = S.shockTypes[comp.type];
    if (!spec) return;
    const row = document.createElement('div');
    row.className = 'slider-row';
    const top = document.createElement('div');
    top.className = 'row-top';
    const name = document.createElement('span');
    name.textContent = `${spec.label} (${spec.unit})`;
    const val = document.createElement('b');
    const fmtVal = () => {
      val.textContent = (spec.unit === '%' ? '' : (comp.value > 0 ? '+' : '')) + comp.value.toFixed(spec.step < 1 ? 2 : 0) + ' ' + spec.unit;
    };
    fmtVal();
    top.appendChild(name);
    top.appendChild(val);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.step;
    input.value = comp.value;
    input.addEventListener('input', () => {
      comp.value = parseFloat(input.value);
      fmtVal();
      S.shock.active = true;
      S.activePreset = null;
      renderPresets();
      scheduleShock();
    });
    row.appendChild(top);
    row.appendChild(input);
    host.appendChild(row);
  });
}

function applyPreset(p) {
  S.shock.active = true;
  S.shock.center = { lat: p.center.lat, lon: p.center.lon };
  S.shock.radius = p.radius;
  S.shock.components = p.components.map((c) => ({ type: c.type, value: c.value }));
  S.activePreset = p.id;
  renderPresets();
  const sel = byId('shockType');
  if (sel && p.components.length) sel.value = p.components[0].type;
  renderShockSliders();
  syncShockUi();
  scheduleShock(0);
}

function syncShockUi() {
  byId('centerLabel').textContent =
    `${S.shock.center.lat.toFixed(1)}°N ${S.shock.center.lon.toFixed(1)}°E`;
  byId('radiusLabel').textContent = `${S.shock.radius.toFixed(1)}°`;
  const rr = byId('radiusRange');
  if (rr) rr.value = S.shock.radius;
  byId('clearShock').textContent = S.shock.active ? 'Remove shock' : 'Restore shock';
  byId('dayRange').value = S.day + 1;
  byId('dayLabel').textContent = `Day ${S.day + 1}`;
}

function updateLegend() {
  const view = VIEWS[S.view];
  byId('legendBar').style.background = rampCss(view.stops);
  byId('legendMin').textContent = `${view.min}`;
  byId('legendMid').textContent = `${Math.round((view.min + view.max) / 2)}`;
  byId('legendMax').textContent = `${view.max}`;
  byId('legendUnit').textContent = `${view.label} · ${view.unit}${view.flow ? ' · streamlines show u10/v10' : ''}`;

  const scale = diffScale();
  byId('diffBar').style.background = rampCss(DIFF_STOPS);
  byId('diffLo').textContent = signed(-scale, 2);
  byId('diffHi').textContent = signed(scale, 2);
}

/* ----------------------------------------------------------- interaction */

function cellFromEvent(v, ev) {
  const p = v.proj;
  if (!p) return null;
  const rect = v.base.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  const lon = GRID.lon0 - 0.5 + (x - p.gx) / p.cellW;
  const lat = 38 - (y - p.gy) / p.cellH;
  if (lon < GRID.lon0 - 0.6 || lon > GRID.lon0 + GRID.nlon - 0.4) return null;
  if (lat < GRID.lat0 - 0.6 || lat > GRID.lat0 + GRID.nlat - 0.4) return null;
  const i = clamp(Math.floor(38 - lat), 0, GRID.nlat - 1);
  const j = clamp(Math.floor(lon - (GRID.lon0 - 0.5)), 0, GRID.nlon - 1);
  return { i, j, lat: GRID.lat0 + i, lon: GRID.lon0 + j };
}

function onHover(v, ev) {
  const cell = cellFromEvent(v, ev);
  if (!cell) { v.tooltip.hidden = true; return; }
  const view = VIEWS[S.view];
  const unit = view.unit;
  const prime = panelValue(S.prime, S.view, S.day, cell.i, cell.j);
  const multi = panelValue(S.multi, S.view, S.day, cell.i, cell.j);
  const lines = [
    `${cell.lat.toFixed(1)}°N ${cell.lon.toFixed(1)}°E`,
    `prime&nbsp;&nbsp;${fmt(prime)} ${unit}`,
    `multi&nbsp;&nbsp;${fmt(multi)} ${unit}`,
    `delta&nbsp;&nbsp;${signed(multi - prime, 2)} ${unit}`,
  ];
  v.tooltip.innerHTML = lines.join('<br>');
  v.tooltip.hidden = false;
  const rect = v.wrap.getBoundingClientRect();
  const tw = v.tooltip.offsetWidth || 130;
  const left = Math.min(Math.max(6, ev.clientX - rect.left + 12), rect.width - tw - 6);
  const top = Math.min(Math.max(6, ev.clientY - rect.top + 12), rect.height - 70);
  v.tooltip.style.left = left + 'px';
  v.tooltip.style.top = top + 'px';
}

function onClickMap(v, ev) {
  const cell = cellFromEvent(v, ev);
  if (!cell) return;
  S.shock.center = { lat: cell.lat, lon: cell.lon };
  S.shock.active = true;
  S.activePreset = null;
  renderPresets();
  syncShockUi();
  scheduleShock(0);
}

/* ---------------------------------------------------------------- render */

function render() {
  for (const v of views) paint(v);
  renderCaptions();
  updateLegend();
  syncShockUi();
  updateStatus();
  renderImpact();
}

function renderCaptions() {
  const view = VIEWS[S.view];
  const label = `${view.short} · ${view.unit}`;
  byId('cap-prime').textContent = `baseline · ${label}`;
  byId('cap-multi').textContent = `perturbed · ${label}`;

  const meanOf = (arr) => {
    if (!arr) return NaN;
    let sum = 0;
    for (let i = 0; i < GRID.nlat; i++) {
      for (let j = 0; j < GRID.nlon; j++) sum += panelValue(arr, S.view, S.day, i, j);
    }
    return sum / (GRID.nlat * GRID.nlon);
  };

  const p = S.prime ? meanOf(S.prime) : NaN;
  const m = S.multi ? meanOf(S.multi) : NaN;
  byId('val-prime').textContent = isFinite(p) ? `mean ${fmt(p)} ${view.unit}` : '';
  byId('val-multi').textContent = isFinite(m) ? `mean ${fmt(m)} ${view.unit}` : '';

  const scale = diffScale();
  const maxAbs = diffScaleMax();
  byId('val-diff').textContent = maxAbs < 1e-6
    ? 'zero anomaly'
    : `max |Δ| ${fmt(maxAbs)} ${view.unit}`;
  byId('diffLo').textContent = signed(-scale, 2);
  byId('diffHi').textContent = signed(scale, 2);

  byId('footMeta').textContent = S.multiSeq
    ? `${S.steps} leads · ${GRID.nlat}×${GRID.nlon}×${NC} grid cells · api ${S.apiBase}`
    : `api ${S.apiBase}`;
}

function diffScaleMax() {
  if (!S.prime || !S.multi) return 0;
  let max = 0;
  for (let d = 0; d < S.steps; d++) {
    for (let i = 0; i < GRID.nlat; i++) {
      for (let j = 0; j < GRID.nlon; j++) {
        const a = Math.abs(diffAt(d, i, j));
        if (a > max) max = a;
      }
    }
  }
  return max;
}

/* ------------------------------------------------------------ interactions */

function wireControls() {
  const range = byId('dayRange');
  range.addEventListener('input', () => {
    S.day = clamp(parseInt(range.value, 10) - 1, 0, S.steps - 1);
    byId('dayLabel').textContent = `Day ${S.day + 1}`;
    render();
  });

  const play = byId('play');
  play.addEventListener('click', () => {
    S.playing = !S.playing;
    play.textContent = S.playing ? '❚❚' : '▶';
    clearInterval(S.playTimer);
    if (S.playing) {
      S.playTimer = setInterval(() => {
        S.day = (S.day + 1) % S.steps;
        render();
      }, 900);
    }
  });

  const rr = byId('radiusRange');
  rr.addEventListener('input', () => {
    S.shock.radius = parseFloat(rr.value);
    byId('radiusLabel').textContent = `${S.shock.radius.toFixed(1)}°`;
    S.activePreset = null;
    renderPresets();
    scheduleShock();
  });

  byId('clearShock').addEventListener('click', () => {
    S.shock.active = !S.shock.active;
    S.activePreset = S.shock.active ? S.activePreset : null;
    renderPresets();
    syncShockUi();
    if (!S.shock.active) {
      clearTimeout(shockTimer);
      if (inFlight) inFlight.abort();
      S.seq += 1;
      S.multi = S.prime;
      S.multiNested = S.primeNested;
      S.multiSeq += 1;
      S.diffCache = null;
      S.pending = false;
      setBusy(false);
      render();
      refreshImpact();
    } else {
      scheduleShock(0);
    }
  });

  byId('rerun').addEventListener('click', () => { runBaseline(); });

  document.addEventListener('keydown', (ev) => {
    const tag = (ev.target && ev.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (ev.key === 'ArrowRight') {
      S.day = clamp(S.day + 1, 0, S.steps - 1);
      render();
    } else if (ev.key === 'ArrowLeft') {
      S.day = clamp(S.day - 1, 0, S.steps - 1);
      render();
    } else if (ev.key === ' ') {
      ev.preventDefault();
      byId('play').click();
    }
  });
}

wireControls();
boot();
