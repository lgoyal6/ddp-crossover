// Draws results/ddp_sweep.json and nothing else.
//
// Every number here came out of a run committed to the repository. Nothing is
// modelled, fitted or extrapolated: the curve is six measured points joined up,
// which is the only kind of results page that can be checked.

const el = (id) => document.getElementById(id);
const plot = el('plot');
const ctx = plot.getContext('2d');
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// The sweep's tags, in the order someone would try them.
const VARIANTS = [
  { key: '2gpu_ddp', label: 'Plain DDP' },
  { key: 'bucket', label: 'Bigger buckets' },
  { key: 'fp16hook', label: 'fp16 gradients' },
  { key: 'accum', label: 'Grad accumulation' },
  { key: 'fsdp', label: 'FSDP' },
];

const state = { rows: [], base: {}, variant: '2gpu_ddp', step: 2 };

const compact = (n) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1e3)}k`;

// Some mitigations were swept over their own parameter as well as over width,
// so they carry several rows per width. The most favourable one is kept: the
// question is whether the mitigation can help at all, and answering it with an
// unlucky setting of its own knob would be arguing against a straw man.
function series(variant) {
  const best = new Map();
  for (const r of state.rows.filter((x) => x.tag === variant)) {
    const seen = best.get(r.width);
    if (!seen || r.samples_per_s > seen.samples_per_s) best.set(r.width, r);
  }
  return [...best.values()]
    .sort((a, b) => a.width - b.width)
    .map((r) => ({ ...r, speedup: r.samples_per_s / state.base[r.width] }));
}

// FSDP does its communication inside its own hooks rather than through the
// allreduce the harness times, so its comm_s is zero at every width. That is a
// gap in the measurement, not a free lunch, and it has to read as one.
function commMeasured(variant) {
  return state.rows.some((r) => r.tag === variant && r.comm_s > 0);
}

function draw(points, active) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w0 = plot.clientWidth || 1200;
  // Capped: past a point a wider screen only adds empty plot, not resolution.
  const h0 = Math.min(Math.round(w0 * 0.38), 430);
  plot.width = Math.round(w0 * dpr);
  plot.height = Math.round(h0 * dpr);
  plot.style.height = h0 + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w0, h0);

  const pad = { l: 58, r: 22, t: 22, b: 52 };
  const w = w0 - pad.l - pad.r;
  const h = h0 - pad.t - pad.b;
  if (points.length < 2) return;

  const top = 1.75;
  const X = (i) => pad.l + (i / (points.length - 1)) * w;
  const Y = (v) => pad.t + h - (v / top) * h;

  // comms fraction behind everything: the cause, drawn under the effect
  points.forEach((p, i) => {
    const bw = (w / points.length) * 0.5;
    const x = X(i) - bw / 2;
    const bh = p.comm_frac * h;
    ctx.fillStyle = 'rgba(58,49,112,.10)';
    ctx.fillRect(x, pad.t + h - bh, bw, bh);
  });

  ctx.strokeStyle = css('--hair');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h);
  ctx.stroke();
  ctx.font = "11px 'Courier New', monospace";
  ctx.fillStyle = css('--faint');
  ctx.textAlign = 'right';
  for (let v = 0; v <= top; v += 0.25) {
    const y = Y(v);
    ctx.fillText(`${v.toFixed(2)}x`, pad.l - 8, y + 3);
    if (v > 0) {
      ctx.strokeStyle = '#e8e3d6';
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + w, y); ctx.stroke();
    }
  }

  // break even: the only line that matters
  ctx.save();
  ctx.strokeStyle = css('--bad');
  ctx.lineWidth = 1.4;
  ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(1)); ctx.lineTo(pad.l + w, Y(1)); ctx.stroke();
  ctx.restore();
  ctx.textAlign = 'left';
  ctx.fillStyle = css('--bad');
  ctx.font = "12px 'Times New Roman', serif";
  ctx.fillText('break even: one GPU is faster below this', pad.l + 10, Y(1) - 6);

  // the curve, with each point coloured by which side of one it fell
  ctx.save();
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p.speedup)) : ctx.moveTo(X(i), Y(p.speedup))));
  ctx.strokeStyle = css('--ox');
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(X(i), Y(p.speedup), i === active ? 5 : 3, 0, Math.PI * 2);
    // Shape as well as colour: filled where it pays, hollow where it does not.
    if (p.speedup >= 1) {
      ctx.fillStyle = css('--ok');
      ctx.fill();
    } else {
      ctx.fillStyle = css('--paper');
      ctx.fill();
      ctx.strokeStyle = css('--bad');
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
  });

  if (active >= 0 && active < points.length) {
    ctx.save();
    ctx.strokeStyle = css('--ox');
    ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(active), pad.t); ctx.lineTo(X(active), pad.t + h); ctx.stroke();
    ctx.restore();
  }

  ctx.textAlign = 'center';
  ctx.fillStyle = css('--faint');
  ctx.font = "11px 'Courier New', monospace";
  points.forEach((p, i) => ctx.fillText(compact(p.params), X(i), pad.t + h + 16));
  ctx.fillText('parameters', pad.l + w / 2, h0 - 8);

  ctx.textAlign = 'left';
  ctx.font = "13px 'Times New Roman', serif";
  ctx.fillStyle = css('--sub');
  ctx.fillText('shaded bars: share of each step spent moving gradients', pad.l + 12, pad.t + h - 12);
}

function render() {
  const points = series(state.variant);
  const scrub = el('scrub');
  scrub.max = String(Math.max(points.length - 1, 0));
  state.step = Math.min(state.step, points.length - 1);
  scrub.value = String(state.step);
  draw(points, state.step);
  const p = points[state.step];
  if (!p) return;

  el('r-params').textContent = p.params.toLocaleString('en-US');
  el('r-one').textContent = `${Math.round(state.base[p.width]).toLocaleString('en-US')}/s`;
  el('r-two').textContent = `${Math.round(p.samples_per_s).toLocaleString('en-US')}/s`;
  el('r-speed').textContent = `${p.speedup.toFixed(2)}x`;
  const measured = commMeasured(state.variant);
  el('r-comm').textContent = measured ? `${(p.comm_frac * 100).toFixed(0)}%` : 'not measured';

  const b = el('banner');
  const commNote = measured
    ? `, with ${(p.comm_frac * 100).toFixed(0)}% of every step spent moving gradients`
    : ', and this configuration\'s communication is not instrumented here';
  if (p.speedup > 1.02) {
    b.className = 'banner calm';
    b.textContent =
      `The second GPU pays here: ${p.speedup.toFixed(2)}x${commNote}.`;
  } else if (p.speedup >= 0.98) {
    b.className = 'banner';
    b.textContent =
      `Break even, near enough: ${p.speedup.toFixed(2)}x. Two GPUs for what one was already doing.`;
  } else {
    b.className = 'banner alarm';
    b.textContent =
      `Two GPUs are ${((1 - p.speedup) * 100).toFixed(0)}% slower than one${commNote}.`;
  }
}

function buildTable() {
  const widths = [...new Set(state.rows.map((r) => r.width))].sort((a, b) => a - b);
  const head =
    `<tr><th>configuration</th>${widths.map((w) => {
      const any = state.rows.find((r) => r.width === w);
      return `<th>${compact(any.params)}</th>`;
    }).join('')}</tr>`;
  const body = VARIANTS.map(({ key, label }) => {
    const byWidth = Object.fromEntries(series(key).map((r) => [r.width, r.speedup]));
    const cells = widths
      .map((w) => {
        const s = byWidth[w];
        if (s === undefined) return '<td>-</td>';
        const cls = s >= 1 ? 'good' : s < 0.6 ? 'bad' : '';
        return `<td class="${cls}">${s.toFixed(2)}x</td>`;
      })
      .join('');
    return `<tr><td>${label}</td>${cells}</tr>`;
  }).join('');
  el('table-wrap').innerHTML = `<table class="data"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function picker(node, items, current, onPick) {
  node.innerHTML = '';
  items.forEach(({ key, label }) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(key === current()));
    b.addEventListener('click', () => {
      onPick(key);
      [...node.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
      render();
    });
    node.appendChild(b);
  });
}

async function main() {
  const res = await fetch('./data/ddp_sweep.json');
  if (!res.ok) {
    el('banner').textContent = `Could not load the measurements (HTTP ${res.status}).`;
    return;
  }
  state.rows = (await res.json()).rows;
  state.base = Object.fromEntries(
    state.rows.filter((r) => r.tag === '1gpu').map((r) => [r.width, r.samples_per_s]),
  );

  const present = VARIANTS.filter((v) => state.rows.some((r) => r.tag === v.key));
  picker(el('variant'), present, () => state.variant, (k) => { state.variant = k; });
  el('scrub').addEventListener('input', (e) => { state.step = Number(e.target.value); render(); });
  window.addEventListener('resize', render);

  // Open on the first size where the second GPU has stopped paying, which is
  // the point of the repository rather than its happiest reading.
  const pts = series(state.variant);
  const firstLoss = pts.findIndex((p) => p.speedup < 1);
  state.step = firstLoss >= 0 ? firstLoss : 0;
  render();
  buildTable();
}

main();
