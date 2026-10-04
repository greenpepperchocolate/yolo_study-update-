"""
In-process image inference for the web app.

Each request gets a folder under runs/predict/<id>/ holding the inputs,
annotated images, YOLO labels and a result.json summary, so results can
be downloaded or imported into a dataset as pre-labelled data.
"""

import io
import logging
import os
import shutil
import threading
import time
import zipfile

import numpy as np
from PIL import Image

import dataset_store
import runmeta

MAX_IMAGES_PER_REQUEST = 100
_model_lock = threading.Lock()
_model_cache = {}


def available_models():
    """List (path, label) for trained best.pt files and base models."""
    models = []
    if os.path.isdir(runmeta.RUNS_DIR):
        runs = []
        for name in os.listdir(runmeta.RUNS_DIR):
            best = os.path.join(runmeta.RUNS_DIR, name, 'weights', 'best.pt')
            if os.path.exists(best):
                runs.append((os.path.getmtime(best), name, best))
        for _, name, best in sorted(runs, reverse=True):
            models.append((best, f'学習結果: {name}'))
    for f in sorted(os.listdir(runmeta.REPO_ROOT)):
        if f.endswith('.pt'):
            models.append((os.path.join(runmeta.REPO_ROOT, f),
                           f'ベースモデル: {f}（COCO 80 クラス）'))
    return models


def resolve_model(path):
    """Accept only a model path offered by available_models()."""
    allowed = {p for p, _ in available_models()}
    if path not in allowed:
        raise dataset_store.DatasetError('モデルの指定が不正です')
    return path


def load_model(path):
    """Load a YOLO model, caching the most recently used one."""
    from ultralytics import YOLO

    key = (path, os.path.getmtime(path))
    with _model_lock:
        if key not in _model_cache:
            _model_cache.clear()
            _model_cache[key] = YOLO(path)
        return _model_cache[key]


def model_names(path):
    """Class names of a model as a list ordered by index."""
    names = load_model(path).names
    return [names[i] for i in sorted(names)]


def new_output_dir(prefix):
    """Create a fresh runs/predict/<prefix>_<time> folder."""
    os.makedirs(runmeta.PREDICT_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for i in range(100):
        name = f'{prefix}_{stamp}' + (f'_{i}' if i else '')
        path = os.path.join(runmeta.PREDICT_DIR, name)
        if not os.path.exists(path):
            os.makedirs(path)
            return name, path
    raise RuntimeError('Could not allocate an output folder')


def predict_images(model_path, uploads, conf, imgsz):
    """Run inference on uploaded images and save everything to disk."""
    model = load_model(model_path)
    names = model.names
    result_name, out_dir = new_output_dir('images')
    for sub in ('inputs', 'annotated', 'labels'):
        os.makedirs(os.path.join(out_dir, sub))
    items, skipped = [], []
    for upload in uploads:
        filename = upload.filename or ''
        stem, ext = dataset_store.sanitize_filename(filename)
        if ext not in dataset_store.IMAGE_EXTS:
            skipped.append((filename, '画像ではありません'))
            continue
        data = upload.stream.read(dataset_store.MAX_FILE_BYTES + 1)
        if len(data) > dataset_store.MAX_FILE_BYTES:
            skipped.append((filename, '50MB を超えています'))
            continue
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception:
            skipped.append((filename, '画像として読み込めません'))
            continue
        base = stem + ext
        n = 1
        while os.path.exists(os.path.join(out_dir, 'inputs', base)):
            base = f'{stem}_{n}{ext}'
            n += 1
        input_file = os.path.join(out_dir, 'inputs', base)
        with open(input_file, 'wb') as f:
            f.write(data)
        with Image.open(input_file) as img:
            rgb = img.convert('RGB')
        bgr = np.array(rgb)[:, :, ::-1].copy()
        result = model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
        annotated = Image.fromarray(result.plot()[:, :, ::-1])
        annotated_name = os.path.splitext(base)[0] + '.jpg'
        annotated.save(os.path.join(out_dir, 'annotated', annotated_name),
                       quality=90)
        detections, lines = [], []
        for cls, score, xywhn in zip(result.boxes.cls.tolist(),
                                     result.boxes.conf.tolist(),
                                     result.boxes.xywhn.tolist()):
            detections.append({'name': names[int(cls)],
                               'conf': round(score, 3)})
            lines.append(f"{int(cls)} " + ' '.join(f'{v:.6f}' for v in xywhn))
        label_name = os.path.splitext(base)[0] + '.txt'
        with open(os.path.join(out_dir, 'labels', label_name), 'w',
                  encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        items.append({'input': base, 'annotated': annotated_name,
                      'label': label_name, 'detections': detections})
    runmeta.write_json(os.path.join(out_dir, 'result.json'), {
        'kind': 'images', 'weights': model_path, 'conf': conf,
        'imgsz': imgsz, 'names': [names[i] for i in sorted(names)],
        'items': items, 'skipped': skipped, 'created': time.time()})
    logging.info(f"Predicted {len(items)} images into {out_dir}")
    return result_name


def result_dir(name):
    """Resolve a runs/predict/<name> folder safely."""
    dataset_store.validate_name(name, '結果名')
    path = os.path.join(runmeta.PREDICT_DIR, name)
    if not os.path.isdir(path):
        raise dataset_store.DatasetError('推論結果が見つかりません')
    return path


def list_results(limit=30):
    """Recent inference results, newest first."""
    if not os.path.isdir(runmeta.PREDICT_DIR):
        return []
    results = []
    for name in os.listdir(runmeta.PREDICT_DIR):
        path = os.path.join(runmeta.PREDICT_DIR, name)
        if not os.path.isdir(path) or not dataset_store.NAME_RE.match(name):
            continue
        info = runmeta.read_json(os.path.join(path, 'result.json'))
        results.append({'name': name, 'mtime': os.path.getmtime(path),
                        'kind': info.get('kind') or 'video',
                        'count': len(info.get('items', [])),
                        'done': bool(info)})
    return sorted(results, key=lambda r: r['mtime'], reverse=True)[:limit]


def zip_result(name):
    """Zip a result folder in memory for download."""
    path = result_dir(name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(path):
            for f in files:
                if f in ('job.json', 'progress.json', runmeta.LOG_FILE) or \
                        f.startswith('source'):
                    continue
                full = os.path.join(root, f)
                archive.write(full, os.path.relpath(full, path))
    buf.seek(0)
    return buf


def import_to_dataset(name, dataset_name, selected=None):
    """Copy inputs plus predicted labels into a dataset for review."""
    path = result_dir(name)
    info = runmeta.read_json(os.path.join(path, 'result.json'))
    ds_path = dataset_store.dataset_path(dataset_name)
    classes = dataset_store.load_classes(ds_path)
    names = info.get('names', [])
    if classes and classes != names:
        raise dataset_store.DatasetError(
            'データセットのクラス名がモデルのクラス名と一致しません。'
            f"モデル: {', '.join(names[:10])}{' …' if len(names) > 10 else ''}")
    if not classes:
        dataset_store.save_classes(ds_path, names)
    pairs = []
    for item in info.get('items', []):
        if selected is not None and item['input'] not in selected:
            continue
        with open(os.path.join(path, 'inputs', item['input']), 'rb') as f:
            image_bytes = f.read()
        with open(os.path.join(path, 'labels', item['label']),
                  encoding='utf-8') as f:
            label_text = f.read()
        pairs.append((item['input'], image_bytes, label_text))
    return dataset_store.ingest_pairs(dataset_name, pairs)


def delete_result(name):
    """Delete an inference result folder."""
    shutil.rmtree(result_dir(name))
