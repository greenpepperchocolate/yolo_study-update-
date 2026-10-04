// Shared helpers: CSRF-aware fetch and class colours.
const CSRF = document.querySelector('meta[name="csrf-token"]').content;

function api(url, options = {}) {
  const headers = Object.assign(
    {'X-CSRF-Token': CSRF, 'X-Requested-With': 'fetch'},
    options.headers || {});
  return fetch(url, Object.assign({}, options, {headers}))
    .then(async (res) => {
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `エラー（${res.status}）`);
      return data;
    });
}

function classColor(i) {
  return `hsl(${(i * 137.5) % 360}, 75%, 50%)`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

// Colour preview boxes rendered server-side with data-cls.
document.querySelectorAll('[data-cls]').forEach((el) => {
  const c = classColor(Number(el.dataset.cls));
  if (el.classList.contains('swatch')) el.style.background = c;
  else el.style.borderColor = c;
});
