"""
Flask web app: upload data, annotate, augment, train, export, infer.

Run with:  .venv\\Scripts\\python webapp\\app.py   (or run_webapp.bat)
Then open http://127.0.0.1:5000
"""

import io
import logging
import os
import re
import secrets
import shutil
import time

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from PIL import Image

import augment
import dataset_store as ds
import jobs
import params
import predictor
import runmeta
from dataset_store import DatasetError
from jobs import JobError
from params import ParamError

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 5000
MAX_VIDEO_BYTES = 500 * 1024 * 1024
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
PLOT_FILES = [
    ('results.png', '学習曲線'),
    ('confusion_matrix_normalized.png', '混同行列（正規化）'),
    ('BoxPR_curve.png', 'PR 曲線'),
    ('BoxF1_curve.png', 'F1 曲線'),
    ('labels.jpg', 'ラベルの分布'),
    ('val_batch0_labels.jpg', '検証画像（正解）'),
    ('val_batch0_pred.jpg', '検証画像（予測）'),
    ('train_batch0.jpg', '学習画像（水増し後）'),
]
PLOT_NAMES = {f for f, _ in PLOT_FILES}
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
STATUS_LABELS = {
    'running': '実行中', 'finished': '完了', 'failed': '失敗',
    'stopped': '停止', 'interrupted': '中断', 'unknown': '不明',
}

def load_secret_key():
    """Keep the session key across restarts so open pages stay valid."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '.secret_key')
    try:
        with open(path, encoding='ascii') as f:
            key = f.read().strip()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_hex(32)
    with open(path, 'w', encoding='ascii') as f:
        f.write(key)
    return key


app = Flask(__name__)
app.config.update(
    SECRET_KEY=load_secret_key(),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    MAX_FORM_PARTS=MAX_FILES_PER_UPLOAD + 200,
)
job_manager = jobs.JobManager()


# ---------------------------------------------------------------- helpers

def wants_json():
    """True for fetch/XHR requests that expect a JSON reply."""
    return request.headers.get('X-Requested-With') == 'fetch'


@app.before_request
def check_csrf():
    """Reject state-changing requests without the session token."""
    if request.method != 'POST':
        return
    token = request.headers.get('X-CSRF-Token') or \
        request.form.get('csrf_token')
    if not token or token != session.get('csrf_token'):
        if wants_json():
            return jsonify(error='ページの有効期限が切れました。'
                           '再読み込みしてください'), 400
        flash('ページの有効期限が切れました。もう一度操作してください', 'error')
        return redirect(request.referrer or url_for('index'))


@app.context_processor
def inject_globals():
    """Values every template can use."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)
    active = job_manager.active()
    return {
        'csrf_token': session['csrf_token'],
        'active_job': active and {
            'kind': jobs.KIND_LABELS.get(active['kind'], active['kind']),
            'name': active['name'],
            'url': job_url(active)},
        'status_labels': STATUS_LABELS,
    }


def job_url(job):
    """Page that shows a job's progress."""
    if job['kind'] == 'predict_video':
        return url_for('predict_result', name=job['name'])
    return url_for('run_page', run=job['name'])


def error_reply(message, status=400, back=None):
    """Flash-and-redirect, or JSON error for fetch requests."""
    if wants_json():
        return jsonify(error=message), status
    flash(message, 'error')
    return redirect(back or request.referrer or url_for('index'))


@app.errorhandler(DatasetError)
@app.errorhandler(ParamError)
@app.errorhandler(JobError)
def handle_user_error(e):
    """Show validation errors to the user instead of a 500 page."""
    return error_reply(str(e))


@app.errorhandler(413)
def too_large(_):
    """Upload exceeded MAX_CONTENT_LENGTH."""
    return error_reply(
        f'アップロードが大きすぎます（1 回 {MAX_UPLOAD_BYTES // 1024 ** 3}GB'
        ' まで）。分けてアップロードしてください', 413)


def run_dir_for(run):
    """Resolve runs/train/<run>, validating the name."""
    run = ds.validate_name(run, '学習名')
    path = os.path.join(runmeta.RUNS_DIR, run)
    if not os.path.isdir(path):
        abort(404)
    return path


def run_status(run_dir, meta):
    """Effective status, noticing runs whose worker has gone away."""
    status = meta.get('status')
    if status == 'running':
        active = job_manager.active()
        if active and os.path.samefile(active['folder'], run_dir) and \
                active['kind'] in ('train', 'resume'):
            return 'running'
        return 'interrupted'
    if status:
        return status
    has_best = os.path.exists(os.path.join(run_dir, 'weights', 'best.pt'))
    return 'finished' if has_best else 'unknown'


def run_info(run, with_rows=False):
    """Everything the UI shows about one training run."""
    run_dir = run_dir_for(run)
    meta = runmeta.read_json(os.path.join(run_dir, runmeta.META_FILE))
    status = run_status(run_dir, meta)
    rows = runmeta.read_results(run_dir)
    epochs = meta.get('settings', {}).get('epochs') or len(rows) or 1
    progress = runmeta.read_json(os.path.join(run_dir, runmeta.PROGRESS_FILE))
    percent = min(100.0, len(rows) / epochs * 100)
    phase_text = f'{len(rows)} / {epochs} エポック'
    if status == 'running':
        phase = progress.get('phase')
        if phase == 'augment':
            total = progress.get('total') or 1
            percent = 0
            phase_text = (f"水増し画像を作成中 {progress.get('done', 0)}"
                          f" / {progress.get('total', 0)}")
        elif phase == 'train':
            batches = progress.get('batches') or 1
            done = progress.get('epoch', 1) - 1 + \
                progress.get('batch', 0) / batches
            percent = min(100.0, done / (progress.get('epochs') or epochs)
                          * 100)
            phase_text = (f"エポック {progress.get('epoch')} / "
                          f"{progress.get('epochs')}（バッチ "
                          f"{progress.get('batch', 0)} / {batches}）")
        else:
            phase_text = '準備中（モデルとデータを読み込んでいます）'
    elif status == 'finished':
        percent = 100.0
    last = rows[-1] if rows else {}
    best = max(rows, key=lambda r: r.get('metrics/mAP50-95(B)', 0)) \
        if rows else {}
    weights_dir = os.path.join(run_dir, 'weights')
    info = {
        'name': run,
        'meta': meta,
        'status': status,
        'status_label': STATUS_LABELS.get(status, status),
        'percent': round(percent, 1),
        'phase_text': phase_text,
        'epochs_done': len(rows),
        'epochs': epochs,
        'metrics': {k: last.get(f'metrics/{k}(B)') for k in
                    ('mAP50', 'mAP50-95', 'precision', 'recall')},
        'best_epoch': int(best['epoch']) if best.get('epoch') else None,
        'best_map': best.get('metrics/mAP50-95(B)'),
        'has_best': os.path.exists(os.path.join(weights_dir, 'best.pt')),
        'has_last': os.path.exists(os.path.join(weights_dir, 'last.pt')),
        'plots': [(f, label) for f, label in PLOT_FILES
                  if os.path.exists(os.path.join(run_dir, f))],
        'exports': meta.get('exports', {}),
        'error': meta.get('error') if status == 'failed' else None,
        'created': meta.get('created_at') or os.path.getmtime(run_dir),
    }
    info['can_resume'] = (status in ('stopped', 'interrupted', 'failed')
                          and info['has_last'] and len(rows) < epochs)
    if with_rows:
        info['rows'] = [{
            'epoch': int(r.get('epoch', i + 1)),
            'map50': r.get('metrics/mAP50(B)'),
            'map': r.get('metrics/mAP50-95(B)'),
            'box_loss': r.get('train/box_loss'),
            'cls_loss': r.get('train/cls_loss'),
        } for i, r in enumerate(rows)]
    return info


def list_runs():
    """Run summaries, newest first."""
    if not os.path.isdir(runmeta.RUNS_DIR):
        return []
    runs = []
    for name in os.listdir(runmeta.RUNS_DIR):
        if os.path.isdir(os.path.join(runmeta.RUNS_DIR, name)) and \
                ds.NAME_RE.match(name):
            runs.append(run_info(name))
    return sorted(runs, key=lambda r: r['created'], reverse=True)


def tail_log(path, max_lines=80):
    """Last lines of a worker log with progress-bar redraws collapsed."""
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - 64 * 1024))
        text = f.read().decode('utf-8', errors='replace')
    lines = []
    for line in text.split('\n'):
        # Keep only the last redraw of a progress bar; lines end in \r\n
        line = ANSI_RE.sub('', line.rstrip('\r').split('\r')[-1]).rstrip()
        if line:
            lines.append(line)
    return '\n'.join(lines[-max_lines:])


def quick_stats(name):
    """Image counts for the dataset list."""
    items = ds.list_images(name)
    return {
        'name': name,
        'images': len(items),
        'labeled': sum(i['status'] == 'labeled' for i in items),
        'unlabeled': sum(i['status'] == 'unlabeled' for i in items),
        'classes': len(ds.load_classes(ds.dataset_path(name))),
    }


def save_limited(stream, target, limit, chunk=1024 * 1024):
    """Copy an upload to disk, stopping as soon as it exceeds limit."""
    written = 0
    with open(target, 'wb') as f:
        while True:
            block = stream.read(chunk)
            if not block:
                return True
            written += len(block)
            if written > limit:
                return False
            f.write(block)


def suggest_run_name(dataset):
    """Default run name: <dataset>_<YYYYmmdd_HHMM>."""
    name = f"{dataset[:24]}_{time.strftime('%Y%m%d_%H%M')}"
    base, n = name, 2
    while os.path.exists(os.path.join(runmeta.RUNS_DIR, name)):
        name = f'{base}_{n}'
        n += 1
    return name


def train_form_values(dataset_path):
    """Last settings used for this dataset, else defaults."""
    values = {}
    values.update(params.defaults(params.TRAIN_PARAMS))
    values.update({'on_' + k: v for k, v in
                   params.defaults(params.ONLINE_AUG_PARAMS).items()})
    values.update({'off_' + k: v for k, v in
                   params.defaults(params.all_offline_params()).items()})
    values['model'] = 'base:yolo26n.pt'
    values['auto_split'] = True
    values.update(ds.load_meta(dataset_path).get('last_settings', {}))
    values['val_percent'] = round(ds.load_meta(dataset_path)['val_ratio']
                                  * 100)
    return values


def parse_offline(form):
    """Parse the offline augmentation section of a form."""
    values, errors = params.parse_group(form, params.all_offline_params(),
                                        prefix='off_')
    if errors:
        raise ParamError(' / '.join(errors))
    return values


# ------------------------------------------------------------------ pages

@app.route('/')
def index():
    """Home: datasets, runs and shortcuts."""
    datasets = [quick_stats(n) for n in ds.list_datasets()]
    return render_template('index.html', datasets=datasets,
                           runs=list_runs()[:20])


@app.post('/datasets/new')
def dataset_create():
    """Create a dataset, optionally with class names."""
    name = request.form.get('name', '')
    classes_text = request.form.get('classes', '')
    classes = ds.parse_class_names(classes_text) if classes_text.strip() \
        else None
    ds.create_dataset(name, classes)
    flash(f'データセット「{name.strip()}」を作成しました', 'success')
    return redirect(url_for('dataset_page', name=name.strip()))


@app.route('/datasets/<name>')
def dataset_page(name):
    """Dataset overview: upload, classes, preview and training form."""
    path = ds.dataset_path(name)
    summary = ds.summarize(name)
    finished = [r for r in list_runs() if r['has_best']]
    return render_template(
        'dataset.html', s=summary,
        warnings=ds.training_warnings(summary),
        blockers=ds.training_blockers(summary, auto_split=True),
        values=train_form_values(path), params=params,
        runs_for_weights=finished, run_name=suggest_run_name(name),
        max_files=MAX_FILES_PER_UPLOAD,
        max_upload_gb=MAX_UPLOAD_BYTES // 1024 ** 3)


@app.post('/datasets/<name>/classes')
def dataset_classes(name):
    """Save class names."""
    names = ds.parse_class_names(request.form.get('classes', ''))
    ds.update_classes(name, names)
    flash(f'クラスを {len(names)} 個登録しました', 'success')
    return redirect(url_for('dataset_page', name=name) + '#classes')


@app.post('/datasets/<name>/upload')
def dataset_upload(name):
    """Bulk upload of images, labels, classes.txt / data.yaml or zips."""
    ds.dataset_path(name)
    files = [f for f in request.files.getlist('files') if f.filename]
    if not files:
        return error_reply('ファイルが選ばれていません')
    if len(files) > MAX_FILES_PER_UPLOAD:
        return error_reply(f'1 回にアップロードできるのは {MAX_FILES_PER_UPLOAD}'
                           ' ファイルまでです。zip にまとめるか分けてください')
    report = ds.ingest_uploads(name, files)
    message = f'画像 {report.images} 枚、ラベル {report.labels} 件を追加しました'
    if report.replaced:
        message += f'（同じ名前の {report.replaced} 件は上書き）'
    if report.renamed:
        message += (f'。別フォルダに同じ名前があった {report.renamed} 件は'
                    '名前の末尾にフォルダ名を付けて保存しました')
    if report.polygons:
        message += (f'。多角形のラベル {report.polygons} 件は'
                    '矩形（物体検出用）に変換しました')
    if report.class_names:
        message += f"。クラス名 {len(report.class_names)} 個を読み込みました"
    flash(message, 'success')
    for filename, reason in report.skipped[:20]:
        flash(f'スキップ: {filename} — {reason}', 'warning')
    if len(report.skipped) > 20:
        flash(f'ほか {len(report.skipped) - 20} 件をスキップしました', 'warning')
    target = url_for('dataset_page', name=name)
    if wants_json():
        return jsonify(redirect=target)
    return redirect(target)


@app.post('/datasets/<name>/delete')
def dataset_delete(name):
    """Delete a dataset after the user types its name."""
    if request.form.get('confirm', '').strip() != name:
        return error_reply('確認のため、データセット名を正しく入力してください')
    active = job_manager.active()
    if active:
        return error_reply('処理の実行中は削除できません')
    ds.delete_dataset(name)
    flash(f'データセット「{name}」を削除しました', 'success')
    return redirect(url_for('index'))


@app.route('/datasets/<name>/image/<split>/<path:filename>')
def dataset_image(name, split, filename):
    """Serve a dataset image (thumb=1 for a small JPEG)."""
    image_file, _ = ds._image_file(name, split, filename)
    if request.args.get('thumb'):
        with Image.open(image_file) as img:
            img = img.convert('RGB')
            img.thumbnail((360, 360))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=80)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg', max_age=60)
    return send_file(image_file, max_age=60)


@app.route('/datasets/<name>/annotate')
def annotate_page(name):
    """Browser annotation tool."""
    path = ds.dataset_path(name)
    classes = ds.load_classes(path)
    if not classes:
        flash('アノテーションの前にクラス名を登録してください', 'warning')
        return redirect(url_for('dataset_page', name=name) + '#classes')
    return render_template('annotate.html', name=name, classes=classes,
                           start=request.args.get('file', ''))


@app.route('/api/datasets/<name>/images')
def api_images(name):
    """Image list with label status, for the annotation tool."""
    return jsonify(images=ds.list_images(name))


@app.route('/api/datasets/<name>/label')
def api_get_label(name):
    """Boxes for one image."""
    try:
        return jsonify(ds.get_label(name, request.args.get('split'),
                                    request.args.get('file')))
    except DatasetError as e:
        return jsonify(error=str(e)), 400


@app.post('/api/datasets/<name>/label')
def api_save_label(name):
    """Save boxes for one image."""
    data = request.get_json(silent=True) or {}
    try:
        count = ds.save_label(name, data.get('split'), data.get('file'),
                              data.get('boxes'))
    except DatasetError as e:
        return jsonify(error=str(e)), 400
    return jsonify(saved=count)


@app.post('/api/datasets/<name>/image-delete')
def api_delete_image(name):
    """Delete one image and its label."""
    data = request.get_json(silent=True) or {}
    try:
        ds.delete_image(name, data.get('split'), data.get('file'))
    except DatasetError as e:
        return jsonify(error=str(e)), 400
    return jsonify(ok=True)


@app.post('/datasets/<name>/augment-preview')
def augment_preview(name):
    """Return a few augmented samples for the current settings."""
    path = ds.dataset_path(name)
    try:
        cfg = parse_offline(request.form)
    except ParamError as e:
        return jsonify(error=str(e)), 400
    images = augment.preview(path, cfg, ds.load_classes(path))
    if not images:
        return jsonify(error='学習用（train）に画像がありません'), 400
    return jsonify(images=images)


@app.post('/datasets/<name>/train')
def dataset_train(name):
    """Validate the training form, resplit and start the worker."""
    path = ds.dataset_path(name)
    form = request.form
    errors = []
    run_name = ''
    try:
        run_name = ds.validate_name(form.get('run_name'), '学習名')
        if os.path.exists(os.path.join(runmeta.RUNS_DIR, run_name)):
            errors.append(f'学習名「{run_name}」は既に使われています')
    except DatasetError as e:
        errors.append(str(e))
    train_values, e1 = params.parse_group(form, params.TRAIN_PARAMS)
    online, e2 = params.parse_group(form, params.ONLINE_AUG_PARAMS,
                                    prefix='on_')
    offline, e3 = params.parse_group(form, params.all_offline_params(),
                                     prefix='off_')
    errors += e1 + e2 + e3
    if 'imgsz' in train_values and train_values['imgsz'] % 32:
        errors.append('「画像サイズ」は 32 の倍数にしてください')

    model_choice = form.get('model', '')
    base_model, weights = None, None
    if model_choice.startswith('base:') and \
            model_choice[5:] in params.MODEL_KEYS:
        base_model = model_choice[5:]
    elif model_choice.startswith('run:'):
        src = os.path.join(runmeta.RUNS_DIR,
                           os.path.basename(model_choice[4:]),
                           'weights', 'best.pt')
        if os.path.exists(src):
            weights = src
        else:
            errors.append('追加学習の元にするモデルが見つかりません')
    else:
        errors.append('ベースモデルを選んでください')

    auto_split = form.get('auto_split') == 'on'
    summary = ds.summarize(name)
    errors += ds.training_blockers(summary, auto_split)
    if offline.get('copies'):
        planned = summary['total_images'] * offline['copies']
        if planned > params.MAX_OFFLINE_IMAGES:
            errors.append(f'水増しで作る画像が多すぎます（約 {planned} 枚。'
                          f'{params.MAX_OFFLINE_IMAGES} 枚まで）')
    if errors:
        for message in errors:
            flash(message, 'error')
        return redirect(url_for('dataset_page', name=name) + '#train')

    if auto_split:
        ds.resplit(name, train_values['val_percent'] / 100)
        blockers = ds.training_blockers(ds.summarize(name), auto_split=False)
        if blockers:
            for message in blockers:
                flash(message, 'error')
            return redirect(url_for('dataset_page', name=name) + '#train')

    meta = ds.load_meta(path)
    last = {k: form.get(k) for k in form if k not in
            ('csrf_token', 'run_name')}
    last['auto_split'] = auto_split
    meta['last_settings'] = last
    ds.save_meta(path, meta)

    run_dir = os.path.join(runmeta.RUNS_DIR, run_name)
    os.makedirs(run_dir)
    settings = dict(train_values, model=base_model,
                    weights=weights and os.path.relpath(
                        weights, runmeta.REPO_ROOT),
                    online=online, offline=offline, auto_split=auto_split)
    runmeta.write_json(os.path.join(run_dir, runmeta.META_FILE), {
        'dataset': name, 'classes': ds.load_classes(path),
        'settings': settings, 'status': 'running',
        'created_at': time.time()})
    job = {
        'run_dir': run_dir, 'dataset_dir': path,
        'classes': ds.load_classes(path), 'model': base_model,
        'weights': weights, 'epochs': train_values['epochs'],
        'imgsz': train_values['imgsz'], 'batch': train_values['batch'],
        'patience': train_values['patience'], 'device': 'auto',
        'online': online, 'offline': offline,
    }
    try:
        job_manager.start('train', run_dir, job, run_name)
    except JobError:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    flash(f'学習「{run_name}」を開始しました', 'success')
    return redirect(url_for('run_page', run=run_name))


# ------------------------------------------------------------------- runs

@app.route('/runs/<run>')
def run_page(run):
    """Training progress and results."""
    return render_template('run.html', r=run_info(run),
                           export_formats=params.EXPORT_FORMATS,
                           predict_params=params.PREDICT_PARAMS)


@app.route('/api/runs/<run>')
def api_run(run):
    """Live status for the run page."""
    info = run_info(run, with_rows=True)
    info['log'] = tail_log(os.path.join(run_dir_for(run), runmeta.LOG_FILE))
    return jsonify(info)


@app.post('/runs/<run>/stop')
def run_stop(run):
    """Stop the running training."""
    run_dir = run_dir_for(run)
    active = job_manager.active()
    if not active or not os.path.samefile(active['folder'], run_dir):
        return error_reply('この学習は実行中ではありません')
    job_manager.stop()
    flash('学習を停止しました。「再開」で続きから学習できます', 'success')
    return redirect(url_for('run_page', run=run))


@app.post('/runs/<run>/resume')
def run_resume(run):
    """Resume an interrupted training run."""
    info = run_info(run)
    if not info['can_resume']:
        return error_reply('この学習は再開できません')
    run_dir = run_dir_for(run)
    job_manager.start('resume', run_dir, {'run_dir': run_dir}, run)
    flash('学習を再開しました', 'success')
    return redirect(url_for('run_page', run=run))


@app.post('/runs/<run>/export')
def run_export(run):
    """Export best.pt to another format in the background."""
    run_dir = run_dir_for(run)
    fmt = request.form.get('format')
    if fmt not in params.EXPORT_KEYS:
        return error_reply('書き出し形式が不正です')
    if not os.path.exists(os.path.join(run_dir, 'weights', 'best.pt')):
        return error_reply('best.pt がまだありません')
    meta = runmeta.read_json(os.path.join(run_dir, runmeta.META_FILE))
    imgsz = meta.get('settings', {}).get('imgsz', 640)
    exports = meta.get('exports', {})
    job_manager.start('export', run_dir,
                      {'run_dir': run_dir, 'format': fmt, 'imgsz': imgsz},
                      run, extra={'format': fmt})
    exports[fmt] = {'status': 'running'}
    runmeta.update_meta(run_dir, exports=exports)
    flash(f'{fmt.upper()} 形式への書き出しを開始しました', 'success')
    return redirect(url_for('run_page', run=run) + '#export')


@app.post('/runs/<run>/delete')
def run_delete(run):
    """Delete a run folder after the user types its name."""
    run_dir = run_dir_for(run)
    if request.form.get('confirm', '').strip() != run:
        return error_reply('確認のため、学習名を正しく入力してください')
    active = job_manager.active()
    if active and os.path.samefile(active['folder'], run_dir):
        return error_reply('実行中の学習は削除できません。先に停止してください')
    shutil.rmtree(run_dir)
    flash(f'学習「{run}」を削除しました', 'success')
    return redirect(url_for('index'))


@app.route('/runs/<run>/weights/<kind>.pt')
def run_weights(run, kind):
    """Download best.pt or last.pt."""
    if kind not in ('best', 'last'):
        abort(404)
    path = os.path.join(run_dir_for(run), 'weights', f'{kind}.pt')
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f'{run}_{kind}.pt')


@app.route('/runs/<run>/exports/<filename>')
def run_export_file(run, filename):
    """Download an exported model."""
    folder = os.path.join(run_dir_for(run), 'exports')
    path = os.path.join(folder, os.path.basename(filename))
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=f'{run}_{os.path.basename(filename)}')


@app.route('/runs/<run>/plot/<filename>')
def run_plot(run, filename):
    """Serve a whitelisted result image."""
    if filename not in PLOT_NAMES:
        abort(404)
    path = os.path.join(run_dir_for(run), filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, max_age=0)


# ---------------------------------------------------------------- predict

@app.route('/predict')
def predict_page():
    """Inference form and recent results."""
    selected = request.args.get('model', '')
    if selected and ds.NAME_RE.match(selected):
        selected = os.path.join(runmeta.RUNS_DIR, selected, 'weights',
                                'best.pt')
    return render_template(
        'predict.html', models=predictor.available_models(),
        selected=selected,
        predict_params=params.PREDICT_PARAMS,
        results=predictor.list_results(),
        max_images=predictor.MAX_IMAGES_PER_REQUEST,
        max_video_mb=MAX_VIDEO_BYTES // 1024 // 1024)


@app.post('/predict')
def predict_run():
    """Run inference on images (now) or a video (in the background)."""
    model_path = predictor.resolve_model(request.form.get('model', ''))
    values, errors = params.parse_group(request.form, params.PREDICT_PARAMS)
    if 'imgsz' in values and values['imgsz'] % 32:
        errors.append('「画像サイズ」は 32 の倍数にしてください')
    if errors:
        return error_reply(' / '.join(errors))
    files = [f for f in request.files.getlist('files') if f.filename]
    if not files:
        return error_reply('画像か動画を選んでください')
    videos = [f for f in files
              if os.path.splitext(f.filename)[1].lower() in VIDEO_EXTS]
    if videos:
        if len(files) > 1:
            return error_reply('動画は 1 本ずつ推論してください')
        if job_manager.active():
            return error_reply('ほかの処理が実行中です。終わってから実行してください')
        video = videos[0]
        result_name, out_dir = predictor.new_output_dir('video')
        ext = os.path.splitext(video.filename)[1].lower()
        source = os.path.join(out_dir, 'source' + ext)
        if not save_limited(video.stream, source, MAX_VIDEO_BYTES):
            shutil.rmtree(out_dir)
            return error_reply(f'動画は {MAX_VIDEO_BYTES // 1024 // 1024}MB'
                               ' までです')
        job_manager.start('predict_video', out_dir, {
            'out_dir': out_dir, 'weights': model_path, 'input': source,
            'conf': values['conf'], 'imgsz': values['imgsz'],
            'device': 'auto'}, result_name)
        return redirect(url_for('predict_result', name=result_name))
    if len(files) > predictor.MAX_IMAGES_PER_REQUEST:
        return error_reply(f'画像は 1 回 {predictor.MAX_IMAGES_PER_REQUEST}'
                           ' 枚までです')
    result_name = predictor.predict_images(model_path, files,
                                           values['conf'], values['imgsz'])
    return redirect(url_for('predict_result', name=result_name))


@app.route('/predict/<name>')
def predict_result(name):
    """Show an inference result."""
    path = predictor.result_dir(name)
    info = runmeta.read_json(os.path.join(path, 'result.json'))
    return render_template('predict_result.html', name=name, info=info,
                           datasets=ds.list_datasets())


@app.route('/api/predict/<name>')
def api_predict(name):
    """Progress of a video inference."""
    path = predictor.result_dir(name)
    progress = runmeta.read_json(os.path.join(path, runmeta.PROGRESS_FILE))
    done = os.path.exists(os.path.join(path, 'result.json'))
    return jsonify(progress=progress, done=done,
                   log=tail_log(os.path.join(path, runmeta.LOG_FILE), 20))


@app.route('/predict/<name>/file/<kind>/<path:filename>')
def predict_file(name, kind, filename):
    """Serve an input/annotated image or the result video."""
    path = predictor.result_dir(name)
    if kind == 'video' and filename == 'result.mp4':
        target = os.path.join(path, 'result.mp4')
    elif kind in ('inputs', 'annotated'):
        target = os.path.join(path, kind, os.path.basename(filename))
    else:
        abort(404)
    if not os.path.isfile(target):
        abort(404)
    return send_file(target, max_age=60,
                     as_attachment=request.args.get('download') == '1')


@app.route('/predict/<name>/download')
def predict_download(name):
    """Download a result as a zip."""
    return send_file(predictor.zip_result(name), mimetype='application/zip',
                     as_attachment=True, download_name=f'{name}.zip')


@app.post('/predict/<name>/import')
def predict_import(name):
    """Add predicted images and labels to a dataset for correction."""
    dataset_name = request.form.get('dataset', '')
    selected = request.form.getlist('selected') or None
    report = predictor.import_to_dataset(name, dataset_name, selected)
    flash(f'画像 {report.images} 枚とラベルを「{dataset_name}」に追加しました。'
          'アノテーション画面で確認・修正してください', 'success')
    for filename, reason in report.skipped[:10]:
        flash(f'スキップ: {filename} — {reason}', 'warning')
    return redirect(url_for('annotate_page', name=dataset_name))


@app.post('/predict/<name>/delete')
def predict_delete(name):
    """Delete an inference result."""
    active = job_manager.active()
    if active and active['name'] == name:
        return error_reply('実行中の推論は削除できません')
    predictor.delete_result(name)
    flash('推論結果を削除しました', 'success')
    return redirect(url_for('predict_page'))


@app.post('/jobs/stop')
def job_stop():
    """Stop whatever job is running (used for video inference)."""
    job_manager.stop()
    flash('処理を停止しました', 'success')
    return redirect(request.referrer or url_for('index'))


if __name__ == '__main__':
    os.makedirs(ds.DATASETS_DIR, exist_ok=True)
    port = int(os.environ.get('PORT', '5000'))
    print(f'\n  ブラウザで http://127.0.0.1:{port} を開いてください\n')
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
