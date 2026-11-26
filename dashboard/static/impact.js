'use strict';

/* ------------------------------------------------------------------ *
 * Agricultural impact co-pilot
 *
 * Draws the district-level risk heatmap, summary stat cards, the
 * multiverse comparison callout, the affected-area lead-day chart and
 * the district ranking table from the /api/impact response.  All indices
 * are computed server-side (impact/indices.py) from the same live
 * rollout the three map panels use.
 * ------------------------------------------------------------------ */

const IMPACT_METRICS = {
  risk: {
    label: 'Composite risk', short: 'risk',
    note: '0.5 · crop stress + 0.3 · rainfall deficit + 0.2 · heat',
    stops: [[0, [24, 42, 58]], [0.2, [37, 110, 116]], [0.4, [142, 188, 96]],
            [0.62, [238, 193, 74]], [0.82, [230, 115, 52]], [1, [154, 32, 46]]],
  },
  crop: {
    label: 'Kharif crop water stress', short: 'crop',
    note: 'Hargreaves demand vs cumulative rain, dry-day streak, heat and monsoon flow',
    stops: [[0, [21, 38, 52]], [0.25, [45, 112, 128]], [0.45, [147, 196, 110]],
            [0.65, [240, 203, 84]], [0.85, [226, 120, 52]], [1, [162, 38, 52]]],
  },
  deficit: {
    label: 'Rainfall deficit (CRDI)', short: 'deficit',
    note: '−CRDI vs 2005–2019 climatology · 0 normal → 100 total deficit',
    stops: [[0, [26, 48, 72]], [0.25, [60, 128, 150]], [0.5, [196, 192, 150]],
            [0.7, [232, 160, 80]], [0.88, [212, 92, 54]], [1, [143, 27, 42]]],
  },
  heat: {
    label: 'Wet-bulb heat stress', short: 'heat',
    note: 'Stull wet-bulb: 0 at 24 °C → 100 at 34 °C (danger 31 °C)',
    stops: [[0, [24, 40, 58]], [0.25, [46, 110, 138]], [0.5, [233, 206, 104]],
            [0.72, [236, 138, 56]], [0.88, [204, 64, 72]], [1, [104, 30, 84]]],
  },
};

const IMPACT = {
  canvas: null, ctx: null, wrap: null,
  off: null, offCtx: null, img: null,
  series: null, seriesCtx: null,
  cssW: 0, cssH: 0, dpr: 1, proj: null,
  ready: false,
};

/* ------------------------------------------------------------- helpers */

function impactEscape(text) {
  return String(text).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function fmtArea(km2) {
  if (!isFinite(km2)) return '–';
  if (km2 >= 1e6) return (km2 / 1e6).toFixed(2) + ' M km²';
  if (km2 >= 1000) return Math.round(km2 / 1000) + 'k km²';
  return Math.round(km2) + ' km²';
}

function fmtPeople(n) {
  if (!isFinite(n)) return '–';
  if (n >= 1e7) return (n / 1e6).toFixed(1) + ' M';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' M';
  if (n >= 1000) return Math.round(n / 1000) + 'k';
  return Math.round(n) + '';
}

function impactCategory(risk) {
  if (risk >= 65) return 'severe';
  if (risk >= 45) return 'high';
  if (risk >= 25) return 'moderate';
  return 'low';
}

function impactValue(cell, metric) {
  if (!cell) return NaN;
  if (metric === 'deficit') return Math.max(0, -cell.crdi);
  return cell[metric];
}

function impactSource() {
  if (!S.impact) return null;
  if (S.impactScenario === 'multi' && S.impact.cells_multi) return S.impact.cells_multi;
  return S.impact.cells;
}

function impactDistricts() {
  if (!S.impact) return null;
  if (S.impactScenario === 'multi' && S.impact.districts_multi) return S.impact.districts_multi;
  return S.impact.districts;
}

function impactSummary() {
  if (!S.impact) return null;
  if (S.impactScenario === 'multi' && S.impact.summary_multi) return S.impact.summary_multi;
  return S.impact.summary;
}

function impactDeltaMap(day) {
  const map = new Map();
  if (!S.impact || !S.impact.districts_multi) return map;
  const prime = new Map(S.impact.districts.map((d) => [d.id, d]));
  for (const d of S.impact.districts_multi) {
    const p = prime.get(d.id);
    if (!p) continue;
    map.set(d.id, d.series.crop[day] - p.series.crop[day]);
  }
  return map;
}

/* ------------------------------------------------------------- controls */

function renderImpactMetricButtons() {
  const host = byId('impactMetrics');
  if (!host) return;
  host.innerHTML = '';
  for (const [key, spec] of Object.entries(IMPACT_METRICS)) {
    const b = document.createElement('button');
    b.dataset.metric = key;
    b.innerHTML = `${spec.label}<span class="k">${spec.short}</span>`;
    b.classList.toggle('on', key === S.impactMetric);
    b.addEventListener('click', () => {
      S.impactMetric = key;
      renderImpactMetricButtons();
      renderImpact();
    });
    host.appendChild(b);
  }
}

function renderImpactScenarioButtons() {
  const host = byId('impactScenario');
  if (!host) return;
  host.innerHTML = '';
  const hasMulti = !!(S.impact && S.impact.cells_multi);
  if (!hasMulti && S.impactScenario === 'multi') S.impactScenario = 'prime';
  const options = [['prime', 'Timeline Prime'], ['multi', 'Multiverse']];
  for (const [key, label] of options) {
    const b = document.createElement('button');
    b.textContent = label;
    b.classList.toggle('on', key === S.impactScenario);
    b.disabled = key === 'multi' && !hasMulti;
    if (b.disabled) b.style.opacity = '0.4';
    b.addEventListener('click', () => {
      if (key === 'multi' && !hasMulti) return;
      S.impactScenario = key;
      renderImpactScenarioButtons();
      renderImpact();
    });
    host.appendChild(b);
  }
}

/* --------------------------------------------------------------- drawing */

function resizeImpact() {
  const cv = IMPACT.canvas;
  if (!cv || !IMPACT.wrap) return;
  const w = Math.max(40, IMPACT.wrap.clientWidth);
  const h = Math.max(40, IMPACT.wrap.clientHeight);
  const dpr = window.devicePixelRatio || 1;
  IMPACT.cssW = w;
  IMPACT.cssH = h;
  IMPACT.dpr = dpr;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  paintImpact();
  drawImpactSeries();
}

function paintImpact() {
  const ctx = IMPACT.ctx;
  if (!ctx || !IMPACT.cssW) return;
  const w = IMPACT.cssW, h = IMPACT.cssH;
  const p = projection(w, h);
  IMPACT.proj = p;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#070b12';
  ctx.fillRect(0, 0, w, h);

  const cells = impactSource();
  if (!cells || !S.impact) {
    drawNote(ctx, w, h, S.impactPending ? 'computing impact indices…' : 'waiting for a live rollout');
    return;
  }

  const day = clamp(S.day, 0, cells.length - 1);
  const cellDay = cells[day];
  const spec = IMPACT_METRICS[S.impactMetric];
  const stops = spec.stops;
  const lo = 0;
  const hi = 100;
  const data = IMPACT.img.data;
  const districtIndex = S.impact.district_index;

  for (let i = 0; i < GRID.nlat; i++) {
    for (let j = 0; j < GRID.nlon; j++) {
      const o = ((GRID.nlat - 1 - i) * GRID.nlon + j) * 4;
      if (districtIndex[i][j] < 0) {
        data[o] = 10; data[o + 1] = 16; data[o + 2] = 24; data[o + 3] = 255;
        continue;
      }
      const val = impactValue({
        crdi: cellDay.crdi[i][j],
        crop: cellDay.crop[i][j],
        heat: cellDay.heat[i][j],
        risk: cellDay.risk[i][j],
      }, S.impactMetric);
      const rgb = rampColor(stops, normalize(val, lo, hi, 'lin'));
      data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2]; data[o + 3] = 255;
    }
  }
  IMPACT.offCtx.putImageData(IMPACT.img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(IMPACT.off, p.gx, p.gy, p.gw, p.gh);

  drawGraticule(ctx, p, w, h);
  drawImpactDistricts(ctx, p, w);
  drawCoast(ctx, p);
}

function drawImpactDistricts(ctx, p, w) {
  if (!S.impact) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(220, 238, 250, 0.20)';
  ctx.lineWidth = 0.7;
  for (const d of S.impact.districts) {
    const [lat0, lat1, lon0, lon1] = d.bbox;
    const x = p.px(lon0 - 0.5);
    const y = p.py(lat1 + 0.5);
    ctx.strokeRect(x, y, p.px(lon1 + 0.5) - x, p.py(lat0 - 0.5) - y);
  }
  if (p.gw > 380) {
    ctx.fillStyle = 'rgba(222, 238, 250, 0.58)';
    ctx.font = '7.5px ui-monospace, monospace';
    ctx.textAlign = 'center';
    for (const d of S.impact.districts) {
      const x = p.px(d.center[1]);
      const y = p.py(d.center[0]);
      if (x < 2 || x > w - 2) continue;
      ctx.fillText(d.name, x, y);
    }
    ctx.textAlign = 'left';
  }
  ctx.restore();
}

function drawImpactSeries() {
  const cv = IMPACT.series;
  if (!cv) return;
  const w = Math.max(160, cv.clientWidth || 260);
  const h = 118;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const summary = impactSummary();
  const multi = S.impact && S.impact.summary_multi ? S.impact.summary_multi : null;
  if (!summary) {
    ctx.fillStyle = 'rgba(160, 185, 205, 0.4)';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('no rollout yet', 8, h / 2);
    return;
  }

  const left = 34, right = 7, top = 9, bottom = 15;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  const primeSeries = summary.series.affected_km2;
  const multiSeries = multi ? multi.series.affected_km2 : null;
  const all = multiSeries ? primeSeries.concat(multiSeries) : primeSeries;
  const yMax = Math.max(1, ...all);
  const n = primeSeries.length;
  const xAt = (k) => left + (n <= 1 ? plotW / 2 : (k / (n - 1)) * plotW);
  const yAt = (v) => top + plotH - (v / yMax) * plotH;

  ctx.strokeStyle = 'rgba(180, 205, 230, 0.12)';
  ctx.lineWidth = 0.6;
  for (let g = 0; g <= 2; g++) {
    const y = top + (plotH * g) / 2;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotW, y);
    ctx.stroke();
  }
  ctx.fillStyle = 'rgba(160, 185, 205, 0.6)';
  ctx.font = '8.5px ui-monospace, monospace';
  ctx.fillText(fmtArea(yMax), 0, top + 3);
  ctx.fillText('0', 0, top + plotH);

  const line = (series, color, dash) => {
    ctx.beginPath();
    series.forEach((v, k) => {
      const x = xAt(k), y = yAt(v);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.setLineDash([]);
  };
  if (multiSeries) line(multiSeries, '#f0a35e', [4, 3]);
  line(primeSeries, '#4fc3c8', []);

  const k = clamp(S.day, 0, n - 1);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  ctx.moveTo(xAt(k), top);
  ctx.lineTo(xAt(k), top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(200, 222, 238, 0.75)';
  ctx.fillText(`d${k + 1}`, xAt(k) - 6, h - 3);
  ctx.textAlign = 'right';
  ctx.fillText(`d${n}`, left + plotW, h - 3);
  ctx.textAlign = 'left';
}

/* ------------------------------------------------------------- DOM render */

function statCard(label, value, note, tone) {
  return `<div class="statcard${tone ? ' ' + tone : ''}">`
    + `<div class="label">${impactEscape(label)}</div>`
    + `<div class="value">${impactEscape(value)}</div>`
    + `<div class="note">${impactEscape(note)}</div></div>`;
}

function renderImpactStats(districts, summary, day) {
  const host = byId('impactStats');
  if (!host) return;
  const aff = summary.series.affected_km2[day];
  const exp = summary.series.population_exposed[day];
  const wb = summary.series.max_wetbulb_c[day];
  const atRisk = districts.filter((d) => d.series.risk[day] >= 45).length;
  const areaPct = summary.land_area_km2 ? (100 * aff) / summary.land_area_km2 : 0;
  const popPct = summary.population_total ? (100 * exp) / summary.population_total : 0;
  host.innerHTML = [
    statCard('Affected crop area', fmtArea(aff),
      `${areaPct.toFixed(1)}% of ${fmtArea(summary.land_area_km2)} · day ${day + 1}`,
      aff > 200000 ? 'warn' : ''),
    statCard('Population exposed', fmtPeople(exp),
      `${popPct.toFixed(1)}% of ${fmtPeople(summary.population_total)}`, exp > 5e7 ? 'warn' : ''),
    statCard('Districts at risk', `${atRisk}/${districts.length}`,
      'composite risk ≥ 45', atRisk >= 10 ? 'warn' : ''),
    statCard('Max wet-bulb', `${wb.toFixed(1)} °C`,
      wb >= 31 ? 'danger · ≥ 31 °C' : 'danger threshold 31 °C',
      wb >= 31 ? 'bad' : ''),
  ].join('');
}

function renderImpactComparison() {
  const box = byId('impactCompare');
  if (!box) return;
  const cmp = S.impact && S.impact.comparison;
  if (!cmp || !S.shock.active) {
    box.hidden = true;
    return;
  }
  const preset = S.presets.find((p) => p.id === S.activePreset);
  const title = preset ? preset.label : 'Applied shock';
  const chips = [
    `<span class="chip">${signed(cmp.affected_km2.delta_pct, 0)}% crop area</span>`,
    `<span class="chip">${signed(cmp.population_exposed.delta / 1e6, 1)} M people</span>`,
  ];
  if (cmp.hotspots && cmp.hotspots.length) {
    const h = cmp.hotspots[0];
    chips.push(`<span class="chip">${impactEscape(h.name)} ${signed(h.delta_crop, 1)}</span>`);
  }
  box.innerHTML = `<b>${impactEscape(title)} · day ${cmp.day} shift</b>`
    + `<p>${impactEscape(cmp.narrative)}</p>`
    + `<div class="chips">${chips.join('')}</div>`;
  box.hidden = false;
}

function renderImpactTable(districts, day) {
  const table = byId('impactDistricts');
  if (!table) return;
  const deltas = impactDeltaMap(day);
  const sorted = districts.slice().sort((a, b) => b.series.risk[day] - a.series.risk[day]);
  const head = ['District', 'State', 'Risk', 'Category', 'Crop', 'CRDI',
    'Wet-bulb', 'Affected', 'People', 'Δ crop'];
  let html = '<thead><tr>' + head.map((h) => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
  for (const d of sorted) {
    const risk = d.series.risk[day];
    const cat = impactCategory(risk);
    const delta = deltas.get(d.id);
    html += '<tr>'
      + `<td class="dname">${impactEscape(d.name)}</td>`
      + `<td class="dstate">${impactEscape(d.state)}</td>`
      + `<td>${risk.toFixed(0)}</td>`
      + `<td><span class="cat ${cat}">${cat}</span></td>`
      + `<td>${d.series.crop[day].toFixed(0)}</td>`
      + `<td>${d.series.crdi[day].toFixed(0)}</td>`
      + `<td>${d.series.wetbulb[day].toFixed(1)}</td>`
      + `<td>${fmtArea(d.series.affected_km2[day])}</td>`
      + `<td>${fmtPeople(d.series.population_exposed[day])}</td>`
      + `<td>${delta === undefined ? '–' : signed(delta, 1)}</td>`
      + '</tr>';
  }
  html += '</tbody>';
  table.innerHTML = html;
}

function renderImpact() {
  if (!IMPACT.canvas) return;
  const foot = byId('impactFoot');
  const title = byId('impactTitle');
  const sub = byId('impactSub');
  const val = byId('impactVal');
  const spec = IMPACT_METRICS[S.impactMetric];

  renderImpactScenarioButtons();

  if (!S.impact) {
    if (foot) {
      foot.textContent = S.impactError
        ? `impact endpoint error: ${S.impactError}`
        : S.impactPending ? 'computing impact indices…' : 'Waiting for a live rollout…';
    }
    title.textContent = spec.label;
    sub.textContent = 'no data';
    val.textContent = '';
    const host = byId('impactStats');
    if (host) host.innerHTML = '';
    const table = byId('impactDistricts');
    if (table) table.innerHTML = '';
    const box = byId('impactCompare');
    if (box) box.hidden = true;
    byId('impactBar').style.background = rampCss(spec.stops);
    byId('impactNote').textContent = spec.note;
    paintImpact();
    drawImpactSeries();
    return;
  }

  const districts = impactDistricts();
  const summary = impactSummary();
  const cells = impactSource();
  const day = clamp(S.day, 0, cells.length - 1);

  title.textContent = spec.label;
  sub.textContent = `${S.impactScenario === 'multi' ? 'multiverse' : 'timeline prime'} · day ${day + 1}`;

  let maxVal = 0;
  for (let i = 0; i < GRID.nlat; i++) {
    for (let j = 0; j < GRID.nlon; j++) {
      if (S.impact.district_index[i][j] < 0) continue;
      const v = impactValue({
        crdi: cells[day].crdi[i][j], crop: cells[day].crop[i][j],
        heat: cells[day].heat[i][j], risk: cells[day].risk[i][j],
      }, S.impactMetric);
      if (v > maxVal) maxVal = v;
    }
  }
  val.textContent = `max ${fmt(maxVal)} / 100`;

  byId('impactBar').style.background = rampCss(spec.stops);
  byId('impactLo').textContent = '0';
  byId('impactMid').textContent = '50';
  byId('impactHi').textContent = '100';
  byId('impactNote').textContent = spec.note;

  renderImpactStats(districts, summary, day);
  renderImpactComparison();
  renderImpactTable(districts, day);

  const cmp = S.impact.comparison;
  byId('impactChartLegend').innerHTML = cmp
    ? '<span style="color:#4fc3c8">prime</span> · <span style="color:#f0a35e">multiverse</span>'
    : '<span style="color:#4fc3c8">prime</span>';
  const peakDay = summary.series.day[
    summary.series.affected_km2.indexOf(Math.max(...summary.series.affected_km2))
  ];
  byId('impactSeriesNote').textContent = cmp
    ? `Peak ${fmtArea(Math.max(...summary.series.affected_km2))} on day ${peakDay}.`
    : `Peak ${fmtArea(Math.max(...summary.series.affected_km2))} on day ${peakDay}. `
      + 'Apply a shock to see the multiverse shift.';

  const clim = S.impact.meta.climatology;
  if (foot) {
    foot.textContent = `CRDI baseline: ${clim.source}`
      + (clim.training_period ? ` (${clim.training_period[0]}–${clim.training_period[1]}). ` : '. ')
      + 'Districts are simplified agro-climatic zones; population exposure uses representative '
      + 'rural densities, not a census raster.';
  }

  paintImpact();
  drawImpactSeries();
}

/* --------------------------------------------------------------- requests */

async function refreshImpact() {
  if (!IMPACT.canvas) return;
  if (!S.primeNested || !S.initNested) {
    S.impact = null;
    renderImpact();
    return;
  }
  const seq = ++S.impactSeq;
  S.impactPending = true;
  S.impactError = null;
  renderImpact();

  const includeMulti = S.shock.active && !!S.multiNested;
  const body = JSON.stringify({
    prime: S.primeNested,
    multi: includeMulti ? S.multiNested : null,
    init_states: S.initNested,
    doy_sincos: JSON.parse(S.doyJSON),
  });

  try {
    const res = await fetch('/api/impact', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${txt.slice(0, 140)}`);
    }
    const json = await res.json();
    if (seq !== S.impactSeq) return;
    S.impact = json;
    if (S.impactScenario === 'multi' && !json.cells_multi) S.impactScenario = 'prime';
    S.impactError = null;
  } catch (err) {
    if (seq !== S.impactSeq) return;
    S.impactError = String(err.message || err);
    console.error('[impact] request failed:', err);
  } finally {
    if (seq === S.impactSeq) {
      S.impactPending = false;
      renderImpact();
    }
  }
}

/* ------------------------------------------------------------ interaction */

let impactHoverDistrict = null;

function onImpactHover(ev) {
  if (!S.impact || !IMPACT.proj) return;
  const cell = cellFromEvent({ proj: IMPACT.proj, base: IMPACT.canvas }, ev);
  const tip = byId('impactTip');
  if (!cell) { tip.hidden = true; return; }
  const id = S.impact.district_index[cell.i][cell.j];
  if (id < 0) { tip.hidden = true; return; }
  const meta = S.impact.districts[id];
  const district = impactDistricts().find((x) => x.id === meta.id);
  if (!district) { tip.hidden = true; return; }
  const day = clamp(S.day, 0, district.series.risk.length - 1);
  const delta = impactDeltaMap(day).get(district.id);
  const lines = [
    `${district.name} · ${district.state}`,
    `risk ${district.series.risk[day].toFixed(0)} · crop ${district.series.crop[day].toFixed(0)} · CRDI ${district.series.crdi[day].toFixed(0)}`,
    `heat ${district.series.heat[day].toFixed(0)} · wet-bulb ${district.series.wetbulb[day].toFixed(1)} °C`,
    `affected ${fmtArea(district.series.affected_km2[day])} · ${fmtPeople(district.series.population_exposed[day])} people`,
  ];
  if (delta !== undefined) lines.push(`crop Δ ${signed(delta, 1)} vs timeline prime`);
  tip.innerHTML = lines.join('<br>');
  tip.hidden = false;
  impactHoverDistrict = district.id;

  const rect = IMPACT.wrap.getBoundingClientRect();
  const tw = tip.offsetWidth || 190;
  const left = Math.min(Math.max(6, ev.clientX - rect.left + 12), rect.width - tw - 6);
  const top = Math.min(Math.max(6, ev.clientY - rect.top + 12), rect.height - 78);
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}

/* ------------------------------------------------------------------- init */

function initImpact() {
  if (IMPACT.ready) return;
  IMPACT.canvas = byId('impactCanvas');
  IMPACT.series = byId('impactSeries');
  if (!IMPACT.canvas) return;
  IMPACT.wrap = IMPACT.canvas.parentElement;
  IMPACT.ctx = IMPACT.canvas.getContext('2d');
  IMPACT.seriesCtx = IMPACT.series ? IMPACT.series.getContext('2d') : null;
  IMPACT.off = document.createElement('canvas');
  IMPACT.off.width = GRID.nlon;
  IMPACT.off.height = GRID.nlat;
  IMPACT.offCtx = IMPACT.off.getContext('2d');
  IMPACT.img = IMPACT.offCtx.createImageData(GRID.nlon, GRID.nlat);
  IMPACT.ready = true;

  IMPACT.wrap.addEventListener('mousemove', onImpactHover);
  IMPACT.wrap.addEventListener('mouseleave', () => { byId('impactTip').hidden = true; });

  renderImpactMetricButtons();
  renderImpactScenarioButtons();
  if (window.ResizeObserver) new ResizeObserver(resizeImpact).observe(IMPACT.wrap);
  else window.addEventListener('resize', resizeImpact);

  resizeImpact();
  renderImpact();
}
