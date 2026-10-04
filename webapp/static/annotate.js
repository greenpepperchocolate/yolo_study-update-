// Minimal bounding-box annotation tool for one dataset.
(() => {
  const $ = (id) => document.getElementById(id);
  const stage = $('stage');
  const img = $('img');
  let images = [];
  let view = [];
  let index = -1;
  let boxes = [];
  let selected = -1;
  let currentClass = 0;
  let dirty = false;
  let drawing = null;

  function setState(text, isError) {
    $('save-state').textContent = text;
    $('save-state').style.color = isError ? 'var(--err)' : 'var(--muted)';
  }

  function renderClasses() {
    $('classes').innerHTML = CLASSES.map((c, i) =>
      `<button data-i="${i}" class="${i === currentClass ? 'cur' : ''}">`
      + `<span class="swatch" style="background:${classColor(i)}"></span>`
      + `${i < 9 ? `<b>${i + 1}</b> ` : ''}${escapeHtml(c)}</button>`).join('');
    $('classes').querySelectorAll('button').forEach((b) => {
      b.onclick = () => chooseClass(Number(b.dataset.i));
    });
  }

  function chooseClass(i) {
    if (i < 0 || i >= CLASSES.length) return;
    currentClass = i;
    if (selected >= 0) {
      boxes[selected].cls = i;
      dirty = true;
      renderBoxes();
    }
    renderClasses();
  }

  function renderList() {
    const filter = $('filter').value;
    view = images.map((im, i) => i)
      .filter((i) => filter === 'all' || images[i].status === filter);
    const counts = {labeled: 0, unlabeled: 0, background: 0};
    images.forEach((im) => { counts[im.status] += 1; });
    $('counts').textContent = `全 ${images.length} 枚・ラベル付き ${counts.labeled}・未作成 ${counts.unlabeled}・物体なし ${counts.background}`;
    $('list').innerHTML = view.map((i) => {
      const im = images[i];
      return `<div data-i="${i}" class="${i === index ? 'cur' : ''}" title="${escapeHtml(im.file)}">`
        + `<span class="st ${im.status}"></span>${im.split === 'valid' ? '[検] ' : ''}${escapeHtml(im.file)}</div>`;
    }).join('') || '<div class="muted">該当する画像はありません</div>';
    $('list').querySelectorAll('[data-i]').forEach((el) => {
      el.onclick = () => go(Number(el.dataset.i));
    });
    const cur = $('list').querySelector('.cur');
    if (cur) cur.scrollIntoView({block: 'nearest'});
  }

  function renderBoxes() {
    stage.querySelectorAll('.b').forEach((el) => el.remove());
    boxes.forEach((b, i) => {
      const el = document.createElement('div');
      el.className = `b${i === selected ? ' sel' : ''}`;
      const color = classColor(b.cls);
      Object.assign(el.style, {
        left: `${(b.cx - b.w / 2) * 100}%`, top: `${(b.cy - b.h / 2) * 100}%`,
        width: `${b.w * 100}%`, height: `${b.h * 100}%`, borderColor: color,
      });
      el.innerHTML = `<span style="background:${color}">${escapeHtml(CLASSES[b.cls] ?? `#${b.cls}`)}</span>`;
      el.addEventListener('mousedown', (e) => {
        e.stopPropagation();
        selected = i;
        renderBoxes();
      });
      stage.appendChild(el);
    });
    $('box-count').textContent = `（${boxes.length}）`;
    $('boxes').innerHTML = boxes.map((b, i) =>
      `<div style="cursor:pointer;${i === selected ? 'font-weight:700' : ''}" data-i="${i}">`
      + `<span class="swatch" style="background:${classColor(b.cls)}"></span>`
      + `${escapeHtml(CLASSES[b.cls] ?? `#${b.cls}`)} <a href="#" data-del="${i}">削除</a></div>`).join('')
      || '<span class="muted">枠はありません</span>';
    $('boxes').querySelectorAll('[data-i]').forEach((el) => {
      el.onclick = (e) => {
        e.preventDefault();
        if (e.target.dataset.del !== undefined) {
          boxes.splice(Number(e.target.dataset.del), 1);
          selected = -1;
          dirty = true;
        } else {
          selected = Number(el.dataset.i);
        }
        renderBoxes();
      };
    });
    setState(dirty ? '未保存の変更があります' : '');
  }

  async function load(i) {
    index = i;
    const im = images[i];
    $('title').textContent = `${im.split === 'valid' ? '検証用' : '学習用'} / ${im.file}（${i + 1} / ${images.length}）`;
    img.src = URLS.image.replace('__S__', im.split).replace('__F__', encodeURIComponent(im.file));
    const q = new URLSearchParams({split: im.split, file: im.file});
    const data = await api(`${URLS.label}?${q}`);
    boxes = data.boxes;
    selected = -1;
    dirty = false;
    renderBoxes();
    renderList();
    history.replaceState(null, '', `?file=${encodeURIComponent(`${im.split}/${im.file}`)}`);
  }

  async function save() {
    if (index < 0) return true;
    const im = images[index];
    try {
      const res = await api(URLS.label, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({split: im.split, file: im.file, boxes}),
      });
      im.status = res.saved ? 'labeled' : 'background';
      dirty = false;
      setState(`保存しました（${res.saved} 個）`);
      renderList();
      return true;
    } catch (err) {
      setState(err.message, true);
      return false;
    }
  }

  async function go(i) {
    if (i < 0 || i >= images.length || i === index) return;
    if (dirty && !(await save())) return;
    load(i);
  }

  function step(delta) {
    const pos = view.indexOf(index);
    const target = pos < 0 ? view[0] : view[pos + delta];
    if (target !== undefined) go(target);
  }

  function relPoint(e) {
    const r = img.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
  }

  stage.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || index < 0) return;
    const p = relPoint(e);
    drawing = {x0: p.x, y0: p.y, el: document.createElement('div')};
    drawing.el.className = 'b';
    drawing.el.style.borderColor = classColor(currentClass);
    stage.appendChild(drawing.el);
    selected = -1;
  });
  window.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    const p = relPoint(e);
    Object.assign(drawing.el.style, {
      left: `${Math.min(p.x, drawing.x0) * 100}%`, top: `${Math.min(p.y, drawing.y0) * 100}%`,
      width: `${Math.abs(p.x - drawing.x0) * 100}%`, height: `${Math.abs(p.y - drawing.y0) * 100}%`,
    });
  });
  window.addEventListener('mouseup', (e) => {
    if (!drawing) return;
    const p = relPoint(e);
    drawing.el.remove();
    const w = Math.abs(p.x - drawing.x0);
    const h = Math.abs(p.y - drawing.y0);
    const r = img.getBoundingClientRect();
    if (w * r.width >= 4 && h * r.height >= 4) {
      boxes.push({cls: currentClass, cx: (p.x + drawing.x0) / 2, cy: (p.y + drawing.y0) / 2, w, h});
      selected = boxes.length - 1;
      dirty = true;
    }
    drawing = null;
    renderBoxes();
  });

  document.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    const k = e.key;
    if (k >= '1' && k <= '9') chooseClass(Number(k) - 1);
    else if ((k === 'Delete' || k === 'Backspace') && selected >= 0) {
      boxes.splice(selected, 1);
      selected = -1;
      dirty = true;
      renderBoxes();
      e.preventDefault();
    } else if (k === 's' || k === 'S' || (e.ctrlKey && k === 's')) { e.preventDefault(); save(); }
    else if (k === 'd' || k === 'ArrowRight') step(1);
    else if (k === 'a' || k === 'ArrowLeft') step(-1);
    else if (k === 'n' || k === 'N') { boxes = []; selected = -1; dirty = true; save().then(() => renderBoxes()); }
    else if (k === 'Escape') { selected = -1; renderBoxes(); }
  });

  $('save').onclick = save;
  $('next').onclick = () => step(1);
  $('prev').onclick = () => step(-1);
  $('filter').onchange = renderList;
  $('no-objects').onclick = () => { boxes = []; selected = -1; dirty = true; save().then(() => renderBoxes()); };
  $('delete-image').onclick = async () => {
    if (index < 0) return;
    const im = images[index];
    if (!window.confirm(`「${im.file}」とラベルを削除します。よろしいですか？`)) return;
    try {
      await api(URLS.del, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({split: im.split, file: im.file}),
      });
      images.splice(index, 1);
      dirty = false;
      if (!images.length) { window.location.reload(); return; }
      const next = Math.min(index, images.length - 1);
      index = -1;
      load(next);
    } catch (err) { setState(err.message, true); }
  };
  window.addEventListener('beforeunload', (e) => {
    if (dirty) { e.preventDefault(); e.returnValue = ''; }
  });

  renderClasses();
  api(URLS.images).then((res) => {
    images = res.images;
    if (!images.length) {
      $('title').textContent = '画像がありません。データセット画面からアップロードしてください';
      return;
    }
    let start = images.findIndex((im) => `${im.split}/${im.file}` === START);
    if (start < 0) start = Math.max(0, images.findIndex((im) => im.status === 'unlabeled'));
    load(start);
  }).catch((err) => setState(err.message, true));
})();
