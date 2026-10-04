"""End-to-end check of the Flask app: upload, validation, training,
inference, export, video, stop/resume. Trains on CPU (a few minutes).

Run: .venv\Scripts\python tests\test_webapp_e2e.py
"""
import io
import os
import sys
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'webapp'))
os.chdir(REPO)

import app as webapp  # noqa: E402
import dataset_store as ds  # noqa: E402
import runmeta  # noqa: E402

DS = 'e2eテスト'
RUN = 'e2e_run'
SAMPLE = os.path.join(REPO, 'datasets', 'sample_coco8')
client = webapp.app.test_client()
failures = []


def check(cond, label):
    print(('OK   ' if cond else 'FAIL ') + label)
    if not cond:
        failures.append(label)


def token():
    client.get('/')
    with client.session_transaction() as s:
        return s['csrf_token']


def post(url, data=None, **kw):
    data = dict(data or {})
    data['csrf_token'] = TOKEN
    return client.post(url, data=data, **kw)


def flashes():
    with client.session_transaction() as s:
        msgs = [m for _, m in s.get('_flashes', [])]
        s.pop('_flashes', None)
    return msgs


def wait_job(timeout=900):
    t0 = time.time()
    while webapp.job_manager.active() and time.time() - t0 < timeout:
        time.sleep(1)
    time.sleep(1.5)  # let the watcher thread record the result


TOKEN = token()

# cleanup from previous attempts
for p in [os.path.join(ds.DATASETS_DIR, DS), os.path.join(ds.DATASETS_DIR, DS + '_取込'), os.path.join(runmeta.RUNS_DIR, RUN), os.path.join(runmeta.RUNS_DIR, RUN + '_stop')]:
    if os.path.exists(p):
        import shutil
        shutil.rmtree(p)

# --- CSRF
r = client.post('/datasets/new', data={'name': 'x'})
check(r.status_code == 302 and not os.path.exists(os.path.join(ds.DATASETS_DIR, 'x')),
      'POST without CSRF token is rejected')
flashes()

# --- dataset name validation
for bad in ['', '../evil', 'a' * 41, 'CON', 'has space']:
    post('/datasets/new', {'name': bad})
    msgs = flashes()
    check(any('名' in m for m in msgs) and not os.path.exists(
        os.path.join(ds.DATASETS_DIR, bad or '_')), f'rejects dataset name {bad!r}')

post('/datasets/new', {'name': DS, 'classes': 'a\na'})
check(any('重複' in m for m in flashes()), 'rejects duplicate class names')

r = post('/datasets/new', {'name': DS})
check(r.status_code == 302 and os.path.isdir(os.path.join(ds.DATASETS_DIR, DS)),
      'creates dataset with Japanese name')
flashes()

# --- upload a zip (images + labels + classes.txt), plus junk
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w') as z:
    for split in ('train', 'valid'):
        for kind in ('images', 'labels'):
            d = os.path.join(SAMPLE, split, kind)
            for f in os.listdir(d):
                z.write(os.path.join(d, f), f'{split}/{kind}/{f}')
    z.write(os.path.join(SAMPLE, 'classes.txt'), 'classes.txt')
    z.writestr('README.roboflow.txt', 'not a label')
    z.writestr('notes.docx', 'x')
buf.seek(0)
bad_label = (io.BytesIO(b'0 0.5 0.5 1.5 0.2\n'), 'broken.txt')
fake_img = (io.BytesIO(b'not an image'), 'fake.jpg')
r = post(f'/datasets/{DS}/upload',
         {'files': [(buf, 'coco8.zip'), bad_label, fake_img]},
         content_type='multipart/form-data',
         headers={'X-Requested-With': 'fetch', 'X-CSRF-Token': TOKEN})
msgs = flashes()
print('   ', msgs[:6])
check(r.status_code == 200 and 'redirect' in r.get_json(), 'zip upload returns redirect JSON')
check(any('画像 8 枚、ラベル 8 件' in m for m in msgs), 'zip: 8 images + 8 labels stored')
check(any('broken.txt' in m and '0〜1' in m for m in msgs), 'out-of-range label rejected')
check(any('fake.jpg' in m for m in msgs), 'corrupt image rejected')
check(any('notes.docx' in m for m in msgs), 'unsupported file reported')
s = ds.summarize(DS)
check(len(s['classes']) == 80, 'classes.txt from zip adopted (80 classes)')
check(s['total_labeled'] == 8, 'summary counts 8 labeled')

# --- pages render
for url in ['/', f'/datasets/{DS}', f'/datasets/{DS}/annotate', '/predict']:
    r = client.get(url)
    check(r.status_code == 200, f'GET {url} renders')
items = ds.list_images(DS)
first = items[0]
r = client.get(f"/datasets/{DS}/image/{first['split']}/{first['file']}?thumb=1")
check(r.status_code == 200 and r.mimetype == 'image/jpeg', 'thumbnail served')
r = client.get(f"/datasets/{DS}/image/train/..%2F..%2Fclasses.txt")
check(r.status_code in (302, 400, 404), 'path traversal on image route blocked')

# --- class shrink protection
post(f'/datasets/{DS}/classes', {'classes': 'only_one'})
check(any('少なく' in m for m in flashes()), 'refuses to drop classes in use')

# --- label API
hdr = {'X-CSRF-Token': TOKEN, 'X-Requested-With': 'fetch'}
r = client.get(f"/api/datasets/{DS}/label?split={first['split']}&file={first['file']}")
boxes = r.get_json()['boxes']
check(r.status_code == 200 and boxes, 'label API returns boxes')
r = client.post(f'/api/datasets/{DS}/label', headers=hdr, json={
    'split': first['split'], 'file': first['file'],
    'boxes': boxes + [{'cls': 999, 'cx': .5, 'cy': .5, 'w': .1, 'h': .1}]})
check(r.status_code == 400 and '999' in r.get_json()['error'], 'label API rejects unknown class id')
r = client.post(f'/api/datasets/{DS}/label', headers=hdr, json={
    'split': first['split'], 'file': first['file'],
    'boxes': boxes + [{'cls': 0, 'cx': .5, 'cy': .5, 'w': .2, 'h': .2}]})
check(r.status_code == 200 and r.get_json()['saved'] == len(boxes) + 1, 'label API saves boxes')
r = client.post(f'/api/datasets/{DS}/label', headers=hdr, json={
    'split': 'train', 'file': '../../classes.txt', 'boxes': []})
check(r.status_code == 400, 'label API blocks path traversal')

# --- augmentation preview
form = {f'off_{k}': v for k, v in webapp.params.defaults(webapp.params.all_offline_params()).items()}
form['off_gray_p'] = 100
r = post(f'/datasets/{DS}/augment-preview', form, headers=hdr)
check(r.status_code == 200 and len(r.get_json()['images']) >= 1, 'augmentation preview images')
form['off_gray_p'] = 150
r = post(f'/datasets/{DS}/augment-preview', form, headers=hdr)
check(r.status_code == 400 and 'グレースケール' in r.get_json()['error'], 'preview rejects 150%')

# --- training form validation
def train_form(**over):
    f = {}
    f.update(webapp.params.defaults(webapp.params.TRAIN_PARAMS))
    f.update({'on_' + k: v for k, v in webapp.params.defaults(webapp.params.ONLINE_AUG_PARAMS).items()})
    f.update({'off_' + k: v for k, v in webapp.params.defaults(webapp.params.all_offline_params()).items()})
    f.update(run_name=RUN, model='base:yolo26n.pt', auto_split='on',
             epochs=2, imgsz=320, batch=4, off_copies=1, off_gray_p=100)
    f.update(over)
    return f

for over, expect in [({'epochs': 0}, 'エポック数'), ({'imgsz': 333}, '32 の倍数'),
                     ({'batch': 'abc'}, '数値'), ({'on_fliplr': 2}, '左右反転'),
                     ({'model': 'base:evil.pt'}, 'ベースモデル'),
                     ({'run_name': '../x'}, '学習名'), ({'off_copies': 11}, '水増し枚数')]:
    post(f'/datasets/{DS}/train', train_form(**over))
    msgs = flashes()
    check(any(expect in m for m in msgs) and not os.path.exists(os.path.join(runmeta.RUNS_DIR, RUN)),
          f'train form rejects {over}')

# --- real training
r = post(f'/datasets/{DS}/train', train_form())
print('   ', flashes())
check(r.status_code == 302 and '/runs/' in r.headers['Location'], 'training started')
time.sleep(3)
r = client.get(f'/api/runs/{RUN}')
check(r.get_json()['status'] == 'running', 'run status is running')
post(f'/datasets/{DS}/train', train_form(run_name='second'))
check(any('実行中' in m for m in flashes()), 'second job refused while one runs')
wait_job()
info = client.get(f'/api/runs/{RUN}').get_json()
print('   status', info['status'], info.get('error'), info['phase_text'])
check(info['status'] == 'finished' and info['has_best'], 'training finished with best.pt')
aug_dir = os.path.join(runmeta.RUNS_DIR, RUN, 'augmented', 'images')
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
aug_files = os.listdir(aug_dir)
arr = np.array(Image.open(os.path.join(aug_dir, aug_files[0])).convert('RGB')).astype(int)
check(len(aug_files) == s['splits']['train']['images'] and
      abs(arr[..., 0] - arr[..., 1]).max() <= 2, 'offline augmentation wrote grayscale copies (train only)')
import yaml  # noqa: E402
data_yaml = yaml.safe_load(open(os.path.join(runmeta.RUNS_DIR, RUN, 'data.yaml'), encoding='utf-8'))
check(len(data_yaml['train']) == 2 and 'augmented' not in data_yaml['val'], 'augmented images only in train')
r = client.get(f'/runs/{RUN}')
check(r.status_code == 200 and '結果のモデル' in r.get_data(as_text=True), 'run page renders')
r = client.get(f'/runs/{RUN}/weights/best.pt')
check(r.status_code == 200 and len(r.data) > 1000, 'best.pt downloadable')
r.close()

# --- inference on images
best = os.path.join(runmeta.RUNS_DIR, RUN, 'weights', 'best.pt')
imgs = [(open(os.path.join(SAMPLE, 'valid', 'images', f), 'rb'), f)
        for f in os.listdir(os.path.join(SAMPLE, 'valid', 'images'))]
post('/predict', {'model': best, 'conf': 5, 'imgsz': 320, 'files': imgs[:1]},
     content_type='multipart/form-data')
check(any('信頼度' in m for m in flashes()), 'predict rejects conf=5')
post('/predict', {'model': r'C:\Windows\evil.pt', 'conf': .25, 'imgsz': 320,
                  'files': [(io.BytesIO(b'x'), 'a.jpg')]}, content_type='multipart/form-data')
check(any('モデル' in m for m in flashes()), 'predict rejects unknown model path')
imgs = [(open(os.path.join(SAMPLE, 'valid', 'images', f), 'rb'), f)
        for f in os.listdir(os.path.join(SAMPLE, 'valid', 'images'))]
r = post('/predict', {'model': best, 'conf': 0.05, 'imgsz': 320, 'files': imgs},
         content_type='multipart/form-data')
check(r.status_code == 302 and '/predict/images_' in r.headers['Location'], 'image inference ran')
result_name = r.headers['Location'].rsplit('/', 1)[-1]
r = client.get(f'/predict/{result_name}')
check(r.status_code == 200, 'predict result page renders')
r = client.get(f'/predict/{result_name}/download')
check(r.status_code == 200 and zipfile.ZipFile(io.BytesIO(r.data)).namelist(), 'result zip downloadable')

# --- import predictions into a fresh dataset
post('/datasets/new', {'name': DS + '_取込'})
flashes()
r = post(f'/predict/{result_name}/import', {'dataset': DS + '_取込'})
msgs = flashes()
print('   ', msgs[:2])
s2 = ds.summarize(DS + '_取込')
check(s2['total_images'] == 4 and len(s2['classes']) == 80, 'predictions imported as labelled data')

# --- export
r = post(f'/runs/{RUN}/export', {'format': 'torchscript'})
wait_job(600)
meta = runmeta.read_json(os.path.join(runmeta.RUNS_DIR, RUN, runmeta.META_FILE))
print('   exports', meta.get('exports'))
check(meta.get('exports', {}).get('torchscript', {}).get('status') == 'finished', 'TorchScript export')
r = post(f'/runs/{RUN}/export', {'format': 'onnx'})
wait_job(900)
meta = runmeta.read_json(os.path.join(runmeta.RUNS_DIR, RUN, runmeta.META_FILE))
print('   exports', meta.get('exports'))
check(meta.get('exports', {}).get('onnx', {}).get('status') == 'finished', 'ONNX export')

# --- video inference
import cv2  # noqa: E402
import tempfile  # noqa: E402
tmp = os.path.join(tempfile.gettempdir(), 'e2e_video.mp4')
w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 5, (640, 480))
for f in sorted(os.listdir(os.path.join(SAMPLE, 'valid', 'images'))) * 3:
    frame = cv2.resize(cv2.imdecode(np.fromfile(os.path.join(SAMPLE, 'valid', 'images', f), np.uint8), 1), (640, 480))
    w.write(frame)
w.release()
r = post('/predict', {'model': best, 'conf': 0.05, 'imgsz': 320,
                      'files': [(open(tmp, 'rb'), '動画テスト.mp4')]},
         content_type='multipart/form-data')
check(r.status_code == 302 and '/predict/video_' in r.headers['Location'], 'video inference started')
video_name = r.headers['Location'].rsplit('/', 1)[-1]
wait_job(600)
info = runmeta.read_json(os.path.join(runmeta.PREDICT_DIR, video_name, 'result.json'))
print('   video', info)
check(info.get('frames') == 12, 'video processed all 12 frames')
r = client.get(f'/predict/{video_name}/file/video/result.mp4')
check(r.status_code == 200 and len(r.data) > 1000, 'result video served')
r.close()

# --- stop and resume
post(f'/datasets/{DS}/train', train_form(run_name=RUN + '_stop', epochs=30, off_copies=0))
flashes()
t0 = time.time()
while time.time() - t0 < 300:
    if len(runmeta.read_results(os.path.join(runmeta.RUNS_DIR, RUN + '_stop'))) >= 2:
        break
    time.sleep(1)
post(f'/runs/{RUN}_stop/stop')
flashes()
time.sleep(3)
info = client.get(f'/api/runs/{RUN}_stop').get_json()
check(info['status'] == 'stopped' and info['can_resume'], 'stopped run can be resumed')
post(f'/runs/{RUN}_stop/resume')
flashes()
time.sleep(25)
info = client.get(f'/api/runs/{RUN}_stop').get_json()
print('   resume', info['status'], info['phase_text'])
check(info['status'] == 'running', 'resumed run is running')
post(f'/runs/{RUN}_stop/stop')
flashes()
time.sleep(3)

print('\nFAILURES:', failures if failures else 'none')

# Remove everything this test created
import shutil  # noqa: E402
for p in [os.path.join(ds.DATASETS_DIR, DS), os.path.join(ds.DATASETS_DIR, DS + '_取込'),
          os.path.join(runmeta.RUNS_DIR, RUN), os.path.join(runmeta.RUNS_DIR, RUN + '_stop'),
          os.path.join(runmeta.PREDICT_DIR, result_name), os.path.join(runmeta.PREDICT_DIR, video_name)]:
    shutil.rmtree(p, ignore_errors=True)
sys.exit(1 if failures else 0)
