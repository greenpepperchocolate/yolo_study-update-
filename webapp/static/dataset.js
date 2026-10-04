// Dataset page: drag-and-drop upload with progress, presets, preview.
(() => {
  const form = document.getElementById('upload-form');
  const zone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const folderInput = document.getElementById('folder-input');
  const picked = document.getElementById('picked');
  const button = document.getElementById('upload-btn');
  const bar = document.getElementById('upload-progress');
  const status = document.getElementById('upload-status');
  // Each item is {file, path}; path keeps folders (train/images/a.jpg)
  // so the server can tell same-named files in different folders apart.
  let files = [];
  const wrap = (list) => [...list].map((f) => ({
    file: f, path: f.webkitRelativePath || f.name}));

  function setFiles(list) {
    files = list;
    const bytes = files.reduce((n, f) => n + f.file.size, 0);
    picked.textContent = files.length
      ? `${files.length} ファイル（${(bytes / 1024 / 1024).toFixed(1)}MB）`
      : '未選択';
    button.disabled = files.length === 0;
  }

  // Walk dropped folders recursively (Chrome / Edge / Firefox).
  function readEntry(entry) {
    return new Promise((resolve) => {
      if (entry.isFile) {
        entry.file((f) => resolve([{file: f, path: entry.fullPath.replace(/^\//, '')}]),
          () => resolve([]));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const all = [];
        const next = () => reader.readEntries(async (entries) => {
          if (!entries.length) {
            const nested = await Promise.all(all.map(readEntry));
            resolve(nested.flat());
            return;
          }
          all.push(...entries);
          next();
        }, () => resolve([]));
        next();
      } else {
        resolve([]);
      }
    });
  }

  zone.addEventListener('click', () => fileInput.click());
  document.getElementById('pick-folder').addEventListener(
    'click', () => folderInput.click());
  fileInput.addEventListener('change', () => setFiles(wrap(fileInput.files)));
  folderInput.addEventListener('change', () => setFiles(wrap(folderInput.files)));
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => {
    e.preventDefault();
    zone.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, () => {
    zone.classList.remove('over');
  }));
  zone.addEventListener('drop', async (e) => {
    e.preventDefault();
    const entries = [...e.dataTransfer.items]
      .map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry())
      .filter(Boolean);
    if (entries.length) {
      status.textContent = 'ファイルを読み込み中…';
      const nested = await Promise.all(entries.map(readEntry));
      status.textContent = '';
      setFiles(nested.flat());
    } else {
      setFiles(wrap(e.dataTransfer.files));
    }
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (!files.length) return;
    const data = new FormData();
    files.forEach((f) => data.append('files', f.file, f.path));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', form.action);
    xhr.setRequestHeader('X-CSRF-Token', CSRF);
    xhr.setRequestHeader('X-Requested-With', 'fetch');
    bar.style.display = '';
    button.disabled = true;
    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      const pct = (ev.loaded / ev.total) * 100;
      bar.firstElementChild.style.width = `${pct}%`;
      status.textContent = pct < 100
        ? `送信中 ${pct.toFixed(0)}%`
        : 'サーバーで確認・保存しています…';
    };
    xhr.onload = () => {
      let res = {};
      try { res = JSON.parse(xhr.responseText); } catch (err) { /* empty */ }
      if (xhr.status === 200 && res.redirect) {
        window.location = res.redirect;
      } else {
        status.textContent = res.error || `アップロードに失敗しました（${xhr.status}）`;
        button.disabled = false;
      }
    };
    xhr.onerror = () => {
      status.textContent = '通信に失敗しました。サイズが大きすぎる場合は分けてください';
      button.disabled = false;
    };
    xhr.send(data);
  });

  // Online augmentation presets.
  document.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const values = PRESETS[btn.dataset.preset];
      Object.entries(values).forEach(([key, value]) => {
        const input = document.getElementById(`on_${key}`);
        if (input) input.value = value;
      });
    });
  });

  // Offline augmentation preview.
  const previewBtn = document.getElementById('preview-btn');
  const previewStatus = document.getElementById('preview-status');
  const previewBox = document.getElementById('preview-images');
  previewBtn.addEventListener('click', () => {
    const trainForm = document.getElementById('train-form');
    const data = new FormData();
    for (const [key, value] of new FormData(trainForm)) {
      if (key.startsWith('off_')) data.append(key, value);
    }
    previewBtn.disabled = true;
    previewStatus.textContent = '作成中…';
    api(PREVIEW_URL, {method: 'POST', body: data})
      .then((res) => {
        previewBox.innerHTML = res.images
          .map((src) => `<div class="thumb"><img src="${src}" alt=""></div>`)
          .join('');
        previewStatus.textContent = 'ランダムに選んだ学習用画像に、今の設定で 1 回ずつ変換をかけた見本です';
      })
      .catch((err) => { previewStatus.textContent = err.message; })
      .finally(() => { previewBtn.disabled = false; });
  });
})();
