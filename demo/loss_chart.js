/**
 * Zero-dependency responsive SVG Training & Validation Loss Chart for PlaygroundLM.
 * Reads the JSON schema emitted by `scripts/plot_loss.py`:
 *   {
 *     "points": [{ "step": number, "tokens_seen": number, "loss": number, "val_loss"?: number }],
 *     "summary": { "final_step", "final_train_loss", "best_val_loss", "best_val_step", "tokens_seen" }
 *   }
 */

export function computeEmaSeries(values, weight = 0.9) {
  if (!values.length) return [];
  const out = [];
  let running = values[0];
  for (let i = 0; i < values.length; i++) {
    running = weight * running + (1 - weight) * values[i];
    out.push(running);
  }
  return out;
}

export function renderLossChart(container, data, options = {}) {
  if (!container) return;
  const points = Array.isArray(data?.points) ? data.points : [];
  if (points.length === 0) {
    container.innerHTML =
      '<div class="empty-banner">No <code>loss_curve.json</code> points published yet. Run <code>scripts/plot_loss.py</code> to populate live training telemetry.</div>';
    return;
  }

  const width = 620;
  const height = 220;
  const pad = { top: 18, right: 20, bottom: 34, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const steps = points.map((p) => Number(p.step));
  const losses = points.map((p) => Number(p.loss));
  const emaLosses = computeEmaSeries(losses, 0.9);
  const valPoints = points.filter(
    (p) => p.val_loss !== undefined && p.val_loss !== null && Number.isFinite(Number(p.val_loss))
  );

  const allY = losses.concat(valPoints.map((p) => Number(p.val_loss)));
  if (options.gpt2Baseline && Number.isFinite(options.gpt2Baseline)) {
    allY.push(Number(options.gpt2Baseline));
  }

  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps, minStep + 1);
  const minY = Math.max(0, Math.floor(Math.min(...allY) * 10) / 10 - 0.15);
  const maxY = Math.ceil(Math.max(...allY) * 10) / 10 + 0.15;

  const xOf = (step) =>
    pad.left + ((Number(step) - minStep) / Math.max(1, maxStep - minStep)) * plotW;
  const yOf = (loss) =>
    pad.top + (1 - (Number(loss) - minY) / Math.max(1e-6, maxY - minY)) * plotH;

  const rawPath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(p.step).toFixed(1)},${yOf(p.loss).toFixed(1)}`)
    .join(' ');

  const emaPath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(p.step).toFixed(1)},${yOf(emaLosses[i]).toFixed(1)}`)
    .join(' ');

  const valPath = valPoints
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${xOf(p.step).toFixed(1)},${yOf(p.val_loss).toFixed(1)}`)
    .join(' ');

  const valCircles = valPoints
    .map(
      (p) =>
        `<circle cx="${xOf(p.step).toFixed(1)}" cy="${yOf(p.val_loss).toFixed(1)}" r="3.2" fill="#fbbf24" stroke="#0b0914" stroke-width="1.2"><title>Step ${p.step}: val_loss ${Number(p.val_loss).toFixed(4)}</title></circle>`
    )
    .join('');

  // Horizontal grid lines (4 ticks)
  const yTicks = [0, 0.33, 0.67, 1].map((t) => {
    const val = minY + t * (maxY - minY);
    const y = yOf(val).toFixed(1);
    return `
      <line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#1f1b33" stroke-dasharray="3,3" />
      <text x="${pad.left - 6}" y="${Number(y) + 3}" text-anchor="end" fill="#948fb1" font-size="10" font-family="monospace">${val.toFixed(1)}</text>
    `;
  });

  // X-axis step labels
  const xTicks = [minStep, Math.round((minStep + maxStep) / 2), maxStep].map((st) => {
    const x = xOf(st).toFixed(1);
    return `<text x="${x}" y="${height - 8}" text-anchor="middle" fill="#948fb1" font-size="10" font-family="monospace">step ${st.toLocaleString()}</text>`;
  });

  // Active checkpoint vertical cursor if within domain
  let activeCursor = '';
  if (
    options.activeStep !== undefined &&
    options.activeStep !== null &&
    Number(options.activeStep) >= minStep &&
    Number(options.activeStep) <= maxStep
  ) {
    const cx = xOf(options.activeStep).toFixed(1);
    activeCursor = `
      <line x1="${cx}" y1="${pad.top}" x2="${cx}" y2="${height - pad.bottom}" stroke="#7cf7d4" stroke-width="1.5" stroke-dasharray="4,3" />
    `;
  }

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="loss-svg" role="img" aria-label="Training and validation loss curve">
      ${yTicks.join('')}
      ${xTicks.join('')}
      ${activeCursor}
      <path d="${rawPath}" fill="none" stroke="rgba(167, 139, 250, 0.28)" stroke-width="1.1" />
      <path d="${emaPath}" fill="none" stroke="#a78bfa" stroke-width="2.1" />
      ${valPath ? `<path d="${valPath}" fill="none" stroke="#fbbf24" stroke-width="1.6" />` : ''}
      ${valCircles}
    </svg>
    <div class="chart-legend">
      <span><i class="swatch swatch-ema"></i> Train EMA (0.9)</span>
      <span><i class="swatch swatch-raw"></i> Train Raw</span>
      <span><i class="swatch swatch-val"></i> Val Loss</span>
      ${
        data.summary?.best_val_loss != null
          ? `<span class="mono-tag">best val: ${Number(data.summary.best_val_loss).toFixed(3)} @ step ${Number(data.summary.best_val_step).toLocaleString()}</span>`
          : ''
      }
    </div>
  `;
}
