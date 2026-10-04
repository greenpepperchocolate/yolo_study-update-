"""Upload handling: duplicate names across folders, polygon labels,
re-upload overwrite and the video size limit. Runs in seconds.

Run: .venv\Scripts\python tests\test_ingest.py
"""
import hashlib
import io
import os
import shutil
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'webapp'))
os.chdir(REPO)
import app as webapp  # noqa: E402
import dataset_store as ds  # noqa: E402

SAMPLE = os.path.join(REPO, 'datasets', 'sample_coco8')
img_a = open(os.path.join(SAMPLE, 'train', 'images', '000000000009.jpg'), 'rb').read()
img_b = open(os.path.join(SAMPLE, 'valid', 'images', '000000000036.jpg'), 'rb').read()
fails = []


def check(cond, label):
    print(('OK   ' if cond else 'FAIL ') + label)
    if not cond:
        fails.append(label)


def stored(name):
    out = {}
    for item in ds.list_images(name):
        path = os.path.join(ds.DATASETS_DIR, name, item['split'], 'images', item['file'])
        stem = os.path.splitext(item['file'])[0]
        label = os.path.join(ds.DATASETS_DIR, name, item['split'], 'labels', stem + '.txt')
        out[item['file']] = (hashlib.md5(open(path, 'rb').read()).hexdigest(),
                             open(label, encoding='utf-8').read().strip())
    return out


client = webapp.app.test_client()
client.get('/')
with client.session_transaction() as s:
    token = s['csrf_token']
hdr = {'X-CSRF-Token': token, 'X-Requested-With': 'fetch'}

for layout, entries in {
    'roboflow': [('train/images/a.jpg', img_a), ('train/labels/a.txt', b'0 0.5 0.5 0.2 0.2\n'),
                 ('valid/labels/a.txt', b'1 0.3 0.3 0.1 0.1\n'), ('valid/images/a.jpg', img_b)],
    'ultralytics': [('images/train/a.jpg', img_a), ('images/val/a.jpg', img_b),
                    ('labels/val/a.txt', b'1 0.3 0.3 0.1 0.1\n'), ('labels/train/a.txt', b'0 0.5 0.5 0.2 0.2\n')],
}.items():
    name = f'取込テスト_{layout}'
    shutil.rmtree(os.path.join(ds.DATASETS_DIR, name), ignore_errors=True)
    ds.create_dataset(name, ['c0', 'c1'])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for path, data in entries:
            z.writestr(path, data)
    buf.seek(0)
    r = client.post(f'/datasets/{name}/upload', headers=hdr,
                    data={'files': [(buf, 'set.zip')]}, content_type='multipart/form-data')
    files = stored(name)
    print('   ', layout, {k: v[1] for k, v in files.items()})
    md5_a, md5_b = hashlib.md5(img_a).hexdigest(), hashlib.md5(img_b).hexdigest()
    by_hash = {v[0]: v[1] for v in files.values()}
    check(len(files) == 2, f'{layout}: both same-named images kept')
    check(by_hash.get(md5_a, '').startswith('0 ') and by_hash.get(md5_b, '').startswith('1 '),
          f'{layout}: each image kept its own label')
    shutil.rmtree(os.path.join(ds.DATASETS_DIR, name))

# Folder upload from the browser sends relative paths as the filename
name = '取込テスト_folder'
shutil.rmtree(os.path.join(ds.DATASETS_DIR, name), ignore_errors=True)
ds.create_dataset(name, ['c0', 'c1'])
client.post(f'/datasets/{name}/upload', headers=hdr, content_type='multipart/form-data', data={'files': [
    (io.BytesIO(img_a), 'ds/train/images/a.jpg'), (io.BytesIO(b'0 0.5 0.5 0.2 0.2\n'), 'ds/train/labels/a.txt'),
    (io.BytesIO(img_b), 'ds/valid/images/a.jpg'), (io.BytesIO(b'1 0.3 0.3 0.1 0.1\n'), 'ds/valid/labels/a.txt'),
    # polygon label: triangle -> bbox (0.2..0.6, 0.1..0.5)
    (io.BytesIO(img_a), 'ds/train/images/poly.jpg'),
    (io.BytesIO(b'1 0.2 0.1 0.6 0.1 0.4 0.5\n'), 'ds/train/labels/poly.txt')]})
files = stored(name)
check(len(files) == 3, 'folder upload: duplicates kept apart')
poly = files.get('poly.jpg', ('', ''))[1]
check(poly == '1 0.400000 0.300000 0.400000 0.400000', f'polygon converted to box ({poly!r})')
with client.session_transaction() as s:
    msgs = [m for _, m in s.get('_flashes', [])]
print('   ', msgs)
check(any('矩形' in m for m in msgs) and any('フォルダ名' in m for m in msgs), 'user is told about rename/conversion')
shutil.rmtree(os.path.join(ds.DATASETS_DIR, name))

# Re-uploading the same file still overwrites (documented behaviour)
name = '取込テスト_again'
shutil.rmtree(os.path.join(ds.DATASETS_DIR, name), ignore_errors=True)
ds.create_dataset(name, ['c0'])
for _ in range(2):
    client.post(f'/datasets/{name}/upload', headers=hdr, content_type='multipart/form-data',
                data={'files': [(io.BytesIO(img_a), 'x.jpg')]})
check(len(stored_list := ds.list_images(name)) == 1, 'separate uploads of the same name overwrite')
shutil.rmtree(os.path.join(ds.DATASETS_DIR, name))

# Video limit is enforced while streaming
webapp.MAX_VIDEO_BYTES = 1024 * 1024
best = os.path.join(REPO, 'yolo26n.pt')
r = client.post('/predict', headers={'X-CSRF-Token': token}, content_type='multipart/form-data', data={
    'model': best, 'conf': 0.25, 'imgsz': 320, 'files': [(io.BytesIO(b'0' * (3 * 1024 * 1024)), 'big.mp4')]})
with client.session_transaction() as s:
    msgs = [m for _, m in s.get('_flashes', [])]
leftovers = [d for d in os.listdir(os.path.join(REPO, 'runs', 'predict')) if d.startswith('video_')] \
    if os.path.isdir(os.path.join(REPO, 'runs', 'predict')) else []
check(any('1MB' in m for m in msgs) and not leftovers, 'oversized video rejected and cleaned up')
print('\nFAILURES:', fails or 'none')
sys.exit(1 if fails else 0)
