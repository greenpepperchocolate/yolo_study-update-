// Run page: poll status, draw the metric chart, show the log.
(() => {
  const fmt = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(3));
  const logBox = document.getElementById('log');
  let stickToBottom = true;
  logBox.addEventListener('scroll', () => {
    stickToBottom = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 20;
  });

  function line(rows, key, color, scale) {
    const values = rows.map((r) => r[key]).filter((v) => typeof v === 'number');
    if (values.length < 1) return '';
    const max = scale || Math.max(...values) || 1;
    const n = Math.max(rows.length - 1, 1);
    const pts = rows
      .map((r, i) => (typeof r[key] === 'number'
        ? `${(i / n) * 790 + 5},${215 - (r[key] / max) * 205}` : null))
      .filter(Boolean)
      .join(' ');
    return `<polyline fill="none" stroke="${color}" stroke-width="2.5" points="${pts}" vector-effect="non-scaling-stroke"/>`;
  }

  function drawChart(rows) {
    const svg = document.getElementById('chart');
    if (!rows.length) {
      svg.innerHTML = '<text x="400" y="110" text-anchor="middle" fill="#677084" font-size="14">1 エポック終わるとグラフが表示されます</text>';
      return;
    }
    const grid = [0.25, 0.5, 0.75, 1].map((g) =>
      `<line x1="0" x2="800" y1="${215 - g * 205}" y2="${215 - g * 205}" stroke="#e6e9ef"/>`
      + `<text x="4" y="${212 - g * 205}" fill="#9aa3b5" font-size="11">${g}</text>`).join('');
    svg.innerHTML = grid
      + line(rows, 'map50', '#2f6fde', 1)
      + line(rows, 'map', '#1f8a4c', 1)
      + line(rows, 'box_loss', '#c0362c');
  }

  function update() {
    api(RUN_API).then((r) => {
      document.getElementById('phase').textContent = r.phase_text;
      document.getElementById('percent').textContent = `${r.percent}%`;
      document.getElementById('bar').style.width = `${r.percent}%`;
      document.getElementById('status').textContent = r.status_label;
      document.getElementById('status').className = `badge ${r.status}`;
      document.getElementById('m-map50').textContent = fmt(r.metrics.mAP50);
      document.getElementById('m-map').textContent = fmt(r.metrics['mAP50-95']);
      document.getElementById('m-p').textContent = fmt(r.metrics.precision);
      document.getElementById('m-r').textContent = fmt(r.metrics.recall);
      document.getElementById('m-best').textContent = r.best_epoch || '—';
      drawChart(r.rows);
      logBox.textContent = r.log || '（ログはまだありません）';
      if (stickToBottom) logBox.scrollTop = logBox.scrollHeight;
      const exportsRunning = Object.values(r.exports || {})
        .some((e) => e.status === 'running');
      if (INITIAL_STATUS === 'running' && r.status !== 'running') {
        window.location.reload();  // show plots and download buttons
        return;
      }
      if (r.status === 'running' || exportsRunning) {
        setTimeout(update, 2000);
      } else if (window.EXPORT_WAS_RUNNING) {
        window.location.reload();
      }
      window.EXPORT_WAS_RUNNING = exportsRunning;
    }).catch(() => setTimeout(update, 5000));
  }
  update();
})();
