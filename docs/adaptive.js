(() => {
  const input = document.querySelector('#traces');
  const status = document.querySelector('#status');
  const metrics = document.querySelector('#metrics');
  const details = document.querySelector('#details');
  const canvas = document.querySelector('#quality');
  const ctx = canvas.getContext('2d');
  const colors = ['#3c65c7', '#b45235', '#25836b', '#8352a5'];

  function parse(text) {
    const stripped = text.trim();
    if (!stripped) return [];
    try {
      const value = JSON.parse(stripped);
      return Array.isArray(value) ? value : [value];
    } catch (_) {
      return stripped.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
    }
  }

  function validateRows(fileName, records) {
    if (!records.length) throw new Error(`${fileName}: no training records found`);
    const first = records[0];
    for (const [index, row] of records.entries()) {
      const numeric = ['step', 'validation_accuracy', 'elapsed_s', 'target_accuracy', 'seed', 'world_size'];
      if (numeric.some((key) => typeof row[key] !== 'number' || !Number.isFinite(row[key]))) {
        throw new Error(`${fileName}: row ${index + 1} is missing required numeric step, accuracy, elapsed, target, seed, or world-size metadata`);
      }
      if (row.step < 0 || row.elapsed_s < 0 || row.world_size < 1 ||
          row.validation_accuracy < 0 || row.validation_accuracy > 1 ||
          row.target_accuracy < 0 || row.target_accuracy > 1 || !row.dataset || !row.arm) {
        throw new Error(`${fileName}: row ${index + 1} has invalid training metadata`);
      }
      if (row.arm !== first.arm || row.seed !== first.seed || row.dataset !== first.dataset ||
          row.world_size !== first.world_size || row.target_accuracy !== first.target_accuracy) {
        throw new Error(`${fileName}: mixed run metadata in one trace file`);
      }
      if (index && row.step <= records[index - 1].step) {
        throw new Error(`${fileName}: optimizer steps must increase strictly`);
      }
    }
  }

  function draw(runs) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 800;
    const height = canvas.clientHeight || 320;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);
    const left = 54, right = 18, top = 24, bottom = 38;
    const plotW = width - left - right, plotH = height - top - bottom;
    const maxStep = Math.max(1, ...runs.flatMap((run) => run.rows.map((row) => row.step)));
    ctx.font = '12px system-ui';
    ctx.strokeStyle = '#d5dbe3';
    ctx.fillStyle = '#576273';
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = top + plotH * tick / 4;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.fillText((1 - tick / 4).toFixed(2), 8, y + 4);
    }
    ctx.fillText('optimizer step', left + plotW / 2 - 35, height - 8);
    runs.forEach((run, index) => {
      const rows = [...run.rows].sort((a, b) => a.step - b.step);
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      rows.forEach((row, point) => {
        const x = left + plotW * row.step / maxStep;
        const y = top + plotH * (1 - Math.max(0, Math.min(1, row.validation_accuracy)));
        if (point === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  function textNode(tag, text) {
    const node = document.createElement(tag);
    node.textContent = text;
    return node;
  }

  input.addEventListener('change', async () => {
    try {
      const runs = [];
      const recovery = [];
      for (const file of input.files) {
        const records = parse(await file.text());
        if (!records.length) continue;
        const first = records[0];
        if (first.schema === 'ddp-crossover.restart-equivalence/v1' ||
            first.schema === 'ddp-crossover.nccl-gradient-fault/v1') {
          recovery.push({name: file.name, report: first});
          continue;
        }
        if (first.schema === 'ddp-crossover.final-campaign/v1') {
          recovery.push({name: file.name, report: first.restart_correctness || {status: 'not recorded'}});
          continue;
        }
        validateRows(file.name, records);
        runs.push({name: `${first.arm} / ${first.dataset} / seed ${first.seed}`, mode: first.dataset === 'synthetic' ? 'synthetic smoke' : 'recorded trace', rows: records});
      }
      if (!runs.length && !recovery.length) throw new Error('No training traces or campaign/recovery reports found.');
      if (runs.length) draw(runs);
      else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      const finalRows = runs.map((run) => run.rows[run.rows.length - 1]);
      const reached = finalRows.filter((row) => row.target_reached).length;
      const values = [
        ['Runs', String(runs.length)],
        ['Reached target', `${reached} / ${runs.length}`],
        ['World size', [...new Set(finalRows.map((row) => row.world_size))].join(', ') || 'n/a'],
        ['Recovery reports', String(recovery.length)],
      ];
      metrics.replaceChildren(...values.map(([label, value]) => {
        const card = document.createElement('div'); card.className = 'metric';
        card.append(textNode('span', label), textNode('b', value));
        return card;
      }));
      const entries = runs.map((run) => {
        const row = run.rows[run.rows.length - 1];
        const entry = document.createElement('p');
        entry.append(textNode('b', run.name), document.createElement('br'));
        entry.append(textNode('span', `${run.mode}; quality ${Number(row.validation_accuracy).toFixed(4)}, ` +
          (row.target_reached ? `target reached at ${Number(row.elapsed_s).toFixed(1)}s` : `censored at step ${row.step}`)));
        if (Number.isFinite(Number(row.resume_bucket_rebuild_s))) {
          entry.append(document.createElement('br'), textNode('span',
            `checkpoint-resume metadata recorded; reducer rebuild ${Number(row.resume_bucket_rebuild_s).toFixed(4)}s`));
        }
        const changes = new Map();
        for (const decision of row.rank_decisions || []) {
          changes.set(`${decision.step}:${decision.bucket}`, decision);
        }
        const decisions = [...changes.values()];
        if (!decisions.length) {
          entry.append(document.createElement('br'), document.createTextNode('No adaptive rank decisions recorded.'));
        }
        for (const decision of decisions) {
          entry.append(document.createElement('br'), document.createTextNode(
            `step ${decision.step}, bucket ${decision.bucket ?? 0}: rank ${decision.rank}, residual ${(decision.residual ?? 0).toFixed(4)}`));
        }
        return entry;
      });
      for (const item of recovery) {
        const report = item.report;
        const entry = document.createElement('p');
        const kind = report.schema === 'ddp-crossover.nccl-gradient-fault/v1' ? 'NCCL gradient-fault check' : 'Checkpoint restart equivalence';
        entry.append(textNode('b', `${kind} / ${item.name}`), document.createElement('br'));
        entry.append(textNode('span', `recorded report; status ${report.status || 'not recorded'}`));
        if (report.fault) entry.append(document.createElement('br'), textNode('span', report.fault));
        if (report.fields_compared) entry.append(document.createElement('br'), textNode('span',
          `restored fields: ${report.fields_compared.join(', ')}`));
        if (report.partial_checkpoint) entry.append(document.createElement('br'), textNode('span',
          `partial ${report.partial_checkpoint}; resumed ${report.resumed_checkpoint}; reference ${report.reference_checkpoint}`));
        if (Number.isFinite(Number(report.worst_replica_parameter_error))) entry.append(document.createElement('br'), textNode('span',
          `worst replica parameter error ${Number(report.worst_replica_parameter_error).toExponential(2)}`));
        entries.push(entry);
      }
      details.replaceChildren(...entries);
      status.className = 'muted';
      status.textContent = `Loaded ${runs.length} training run(s) and ${recovery.length} recorded recovery report(s). No live training session is connected.`;
    } catch (error) {
      status.className = 'error';
      status.textContent = error.message;
      metrics.replaceChildren();
      details.textContent = 'Trace could not be loaded.';
    }
  });
  window.addEventListener('resize', () => {
    if (input.files.length) input.dispatchEvent(new Event('change'));
  });
})();
