"""
Local YOLO dataset storage for the web app.

A dataset lives in datasets/<name>/ with the layout scripts/main.py
expects (train/ and valid/, each with images/ and labels/) plus
classes.txt. Each image is assigned to train or valid by a hash of its
file stem, so an image and its label always land in the same split no
matter which one is uploaded first.
"""

import hashlib
import io
import json
import logging
import os
import re
import shutil
import zipfile

import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(REPO_ROOT, 'datasets')

SPLITS = ('train', 'valid')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
LABEL_EXT = '.txt'

MAX_CLASSES = 100
MAX_CLASS_NAME_LEN = 50
MAX_NAME_LEN = 40
MAX_STEM_LEN = 100
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_ZIP_ENTRIES = 20000
MAX_ZIP_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MIN_VAL_RATIO = 0.05
MAX_VAL_RATIO = 0.5
DEFAULT_VAL_RATIO = 0.2

NAME_RE = re.compile(r'^\w[\w\-]*$')
WINDOWS_RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {
    f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}
UNSAFE_CHARS_RE = re.compile(r'[\x00-\x1f<>:"/\\|?*]')
META_FILE = 'dataset.json'
CLASSES_FILE = 'classes.txt'


class DatasetError(ValueError):
    """Raised for invalid user input about a dataset."""


def validate_name(name, label='名前'):
    """Validate a dataset or run name and return it stripped."""
    name = (name or '').strip()
    if not name:
        raise DatasetError(f'{label}を入力してください')
    if len(name) > MAX_NAME_LEN:
        raise DatasetError(f'{label}は {MAX_NAME_LEN} 文字以内にしてください')
    if not NAME_RE.match(name):
        raise DatasetError(
            f'{label}に使えるのは文字・数字・「_」「-」だけです'
            '（先頭は「-」以外）')
    if name.upper() in WINDOWS_RESERVED:
        raise DatasetError(f'「{name}」は Windows の予約名なので使えません')
    return name


def parse_class_names(text):
    """Parse class names (one per line) and validate them."""
    names = [line.strip() for line in (text or '').splitlines()]
    names = [n for n in names if n]
    if not names:
        raise DatasetError('クラス名を 1 つ以上入力してください')
    if len(names) > MAX_CLASSES:
        raise DatasetError(f'クラスは {MAX_CLASSES} 個までです')
    seen = set()
    for n in names:
        if len(n) > MAX_CLASS_NAME_LEN:
            raise DatasetError(
                f'クラス名「{n[:20]}…」が長すぎます'
                f'（{MAX_CLASS_NAME_LEN} 文字以内）')
        if UNSAFE_CHARS_RE.search(n) or ',' in n:
            raise DatasetError(
                f'クラス名「{n}」に使えない文字が含まれています'
                '（, < > : " / \\ | ? * など）')
        if n in seen:
            raise DatasetError(f'クラス名「{n}」が重複しています')
        seen.add(n)
    return names


def parse_val_ratio(value):
    """Parse the validation split percentage (e.g. "20") to a ratio."""
    try:
        percent = float(value)
    except (TypeError, ValueError):
        raise DatasetError('検証用の割合は数値で入力してください') from None
    ratio = percent / 100
    if not MIN_VAL_RATIO <= ratio <= MAX_VAL_RATIO:
        raise DatasetError(
            f'検証用の割合は {MIN_VAL_RATIO * 100:.0f}〜'
            f'{MAX_VAL_RATIO * 100:.0f}% にしてください')
    return ratio


def sanitize_filename(filename):
    """Return (stem, ext) safe to store, keeping non-ASCII characters."""
    base = os.path.basename((filename or '').replace('\\', '/'))
    stem, ext = os.path.splitext(base)
    stem = UNSAFE_CHARS_RE.sub('_', stem).strip(' .')[:MAX_STEM_LEN]
    return stem or 'file', ext.lower()


def split_for_stem(stem, val_ratio):
    """Pick train or valid for a file stem, stable across uploads."""
    digest = hashlib.md5(stem.encode('utf-8')).hexdigest()
    return 'valid' if int(digest[:8], 16) / 0xFFFFFFFF < val_ratio \
        else 'train'


def _listdir(path):
    """os.listdir that treats a missing folder as empty."""
    try:
        return os.listdir(path)
    except (FileNotFoundError, NotADirectoryError):
        return []


def list_datasets():
    """Return the names of all datasets under datasets/."""
    if not os.path.isdir(DATASETS_DIR):
        return []
    return sorted(
        d for d in _listdir(DATASETS_DIR)
        if os.path.isdir(os.path.join(DATASETS_DIR, d))
        and NAME_RE.match(d))


def dataset_path(name):
    """Return the folder of an existing dataset, validating the name."""
    name = validate_name(name, 'データセット名')
    path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(path):
        raise DatasetError(f'データセット「{name}」が見つかりません')
    return path


def create_dataset(name, class_names=None):
    """Create an empty dataset folder and return its path."""
    name = validate_name(name, 'データセット名')
    path = os.path.join(DATASETS_DIR, name)
    if os.path.exists(path):
        raise DatasetError(f'データセット「{name}」は既にあります')
    for split in SPLITS:
        for kind in ('images', 'labels'):
            os.makedirs(os.path.join(path, split, kind))
    save_meta(path, {'val_ratio': DEFAULT_VAL_RATIO})
    if class_names:
        save_classes(path, class_names)
    logging.info(f"Created dataset: {path}")
    return path


def delete_dataset(name):
    """Delete a dataset folder and everything in it."""
    path = dataset_path(name)
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(
            DATASETS_DIR):
        raise DatasetError('削除できない場所です')
    shutil.rmtree(path)
    logging.info(f"Deleted dataset: {path}")


def load_meta(path):
    """Load dataset.json, falling back to defaults."""
    meta = {'val_ratio': DEFAULT_VAL_RATIO}
    meta_file = os.path.join(path, META_FILE)
    if os.path.exists(meta_file):
        try:
            with open(meta_file, encoding='utf-8') as f:
                meta.update(json.load(f))
        except (OSError, ValueError) as e:
            logging.warning(f"Ignoring broken {meta_file}: {e}")
    return meta


def save_meta(path, meta):
    """Write dataset.json."""
    with open(os.path.join(path, META_FILE), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_classes(path):
    """Return the class names in classes.txt (empty list if none)."""
    classes_file = os.path.join(path, CLASSES_FILE)
    if not os.path.exists(classes_file):
        return []
    with open(classes_file, encoding='utf-8-sig') as f:
        return [line.strip() for line in f if line.strip()]


def save_classes(path, names):
    """Write classes.txt, one class per line."""
    with open(os.path.join(path, CLASSES_FILE), 'w', encoding='utf-8') as f:
        f.write('\n'.join(names) + '\n')


def parse_label_text(text):
    """Parse YOLO label text into [(class_id, coords)].

    Raises DatasetError with the offending line on malformed input.
    Accepts boxes (class cx cy w h) and polygons (class x1 y1 x2 y2 ...).
    """
    objects = []
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 and (len(parts) < 7 or len(parts) % 2 == 0):
            raise DatasetError(
                f'{line_no} 行目の列数が不正です（{len(parts)} 列）')
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            raise DatasetError(f'{line_no} 行目に数値でない値があります') \
                from None
        if class_id < 0:
            raise DatasetError(f'{line_no} 行目のクラス番号が負の値です')
        if any(not 0 <= v <= 1 for v in coords):
            raise DatasetError(
                f'{line_no} 行目の座標が 0〜1 の範囲外です'
                '（YOLO 形式は画像サイズで割った値です）')
        if len(coords) == 4 and (coords[2] <= 0 or coords[3] <= 0):
            raise DatasetError(f'{line_no} 行目の幅か高さが 0 です')
        objects.append((class_id, coords))
    return objects


def read_label_file(label_file):
    """Parse a stored label file, returning [] when it is missing."""
    if not os.path.exists(label_file):
        return []
    with open(label_file, encoding='utf-8-sig') as f:
        return parse_label_text(f.read())


def _find_stem(path, stem):
    """Return (split, image_file, label_file) already stored for a stem."""
    for split in SPLITS:
        images_dir = os.path.join(path, split, 'images')
        image_files = [
            os.path.join(images_dir, f) for f in _listdir(images_dir)
            if os.path.splitext(f)[0] == stem
            and os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        label_file = os.path.join(path, split, 'labels', stem + LABEL_EXT)
        if image_files or os.path.exists(label_file):
            return split, image_files, label_file
    return None, [], None


def _class_names_from_yaml(data):
    """Extract class names from a data.yaml payload."""
    try:
        names = (yaml.safe_load(data.decode('utf-8-sig')) or {}).get('names')
    except (yaml.YAMLError, UnicodeDecodeError, AttributeError):
        return None
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names] if names else None


class IngestReport:
    """Summary of one upload, shown back to the user."""

    def __init__(self):
        self.images = 0
        self.labels = 0
        self.replaced = 0
        self.skipped = []
        self.class_names = None
        self.renamed = 0
        self.polygons = 0
        # Same file name from different folders in one upload (e.g.
        # train/images/a.jpg and valid/images/a.jpg) must not collide
        self._stem_owner = {}
        self._stem_map = {}

    def resolve_stem(self, filename, stem):
        """Stored stem for a file, renaming cross-folder duplicates."""
        group = _source_group(filename)
        key = (group, stem)
        if key in self._stem_map:
            return self._stem_map[key]
        new = stem
        owner = self._stem_owner.get(stem)
        if owner is not None and owner != group:
            suffix = UNSAFE_CHARS_RE.sub('_', group.split('/')[-1]) or 'root'
            new = f'{stem}_{suffix}'[:MAX_STEM_LEN]
            n = 2
            while new in self._stem_owner:
                new = f'{stem}_{suffix}{n}'[:MAX_STEM_LEN]
                n += 1
            self.renamed += 1
        self._stem_owner[new] = group
        self._stem_map[key] = new
        return new

    def skip(self, filename, reason):
        """Record a file that was not stored."""
        self.skipped.append((filename, reason))


def _source_group(filename):
    """Folder a file came from, ignoring images/ and labels/ segments.

    train/images/a.jpg and train/labels/a.txt (or images/train/a.jpg and
    labels/train/a.txt) share the group "train", so pairs stay together.
    """
    parts = filename.replace('\\', '/').split('/')[:-1]
    parts = [x for x in parts if x and x not in ('.', '..')
             and x.lower() not in ('images', 'labels')]
    return '/'.join(parts)


def _polygons_to_boxes(objects):
    """Rewrite label objects as box lines (polygons become their bbox)."""
    lines = []
    for class_id, coords in objects:
        if len(coords) == 4:
            cx, cy, w, h = coords
        else:
            xs, ys = coords[0::2], coords[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
        lines.append(f'{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return '\n'.join(lines) + ('\n' if lines else '')


def _store_image(path, stem, ext, data, val_ratio, report, filename):
    """Validate and store one image."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception:
        report.skip(filename, '画像として読み込めません（壊れている可能性）')
        return
    split, old_images, _ = _find_stem(path, stem)
    split = split or split_for_stem(stem, val_ratio)
    for old in old_images:
        os.remove(old)
    if old_images:
        report.replaced += 1
    os.makedirs(os.path.join(path, split, 'images'), exist_ok=True)
    with open(os.path.join(path, split, 'images', stem + ext), 'wb') as f:
        f.write(data)
    report.images += 1


def _store_label(path, stem, data, val_ratio, report, filename):
    """Validate and store one YOLO label file."""
    try:
        text = data.decode('utf-8-sig')
        objects = parse_label_text(text)
    except UnicodeDecodeError:
        report.skip(filename, 'テキストとして読み込めません')
        return
    except DatasetError as e:
        report.skip(filename, f'YOLO 形式のラベルではありません: {e}')
        return
    if any(len(coords) != 4 for _, coords in objects):
        # This app trains detection models only, so store boxes
        text = _polygons_to_boxes(objects)
        report.polygons += 1
    split, _, old_label = _find_stem(path, stem)
    if split and os.path.exists(old_label):
        report.replaced += 1
    split = split or split_for_stem(stem, val_ratio)
    os.makedirs(os.path.join(path, split, 'labels'), exist_ok=True)
    label_file = os.path.join(path, split, 'labels', stem + LABEL_EXT)
    with open(label_file, 'w', encoding='utf-8') as f:
        f.write(text)
    report.labels += 1


def _ingest_one(path, filename, data, val_ratio, report):
    """Route one uploaded file to the right handler."""
    base = os.path.basename(filename.replace('\\', '/'))
    stem, ext = sanitize_filename(base)
    if ext in IMAGE_EXTS or ext == LABEL_EXT:
        stem = report.resolve_stem(filename, stem)
    if len(data) > MAX_FILE_BYTES:
        report.skip(filename, f'{MAX_FILE_BYTES // 1024 // 1024}MB を超えています')
    elif base.lower() == CLASSES_FILE:
        try:
            report.class_names = parse_class_names(data.decode('utf-8-sig'))
        except (DatasetError, UnicodeDecodeError) as e:
            report.skip(filename, f'クラス名を読み込めません: {e}')
    elif base.lower() in ('data.yaml', 'data.yml'):
        names = _class_names_from_yaml(data)
        if names:
            report.class_names = names
        else:
            report.skip(filename, 'names（クラス名）が見つかりません')
    elif ext in IMAGE_EXTS:
        _store_image(path, stem, ext, data, val_ratio, report, filename)
    elif ext == LABEL_EXT:
        _store_label(path, stem, data, val_ratio, report, filename)
    else:
        report.skip(filename, '対象外の形式です')


def _zip_entry_name(info):
    """Decode a zip entry name, handling cp932 names made on Windows."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode('cp437').decode('cp932')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def _ingest_zip(path, filename, stream, val_ratio, report):
    """Extract supported files from an uploaded zip archive."""
    try:
        archive = zipfile.ZipFile(stream)
    except zipfile.BadZipFile:
        report.skip(filename, 'zip ファイルとして開けません')
        return
    with archive:
        infos = [i for i in archive.infolist() if not i.is_dir()]
        if len(infos) > MAX_ZIP_ENTRIES:
            report.skip(filename,
                        f'zip 内のファイルが多すぎます（{MAX_ZIP_ENTRIES} 個まで）')
            return
        if sum(i.file_size for i in infos) > MAX_ZIP_TOTAL_BYTES:
            report.skip(filename, '展開後のサイズが大きすぎます（4GB まで）')
            return
        for info in infos:
            name = _zip_entry_name(info)
            base = os.path.basename(name.replace('\\', '/'))
            if '__MACOSX' in name or base.startswith('.'):
                continue
            if info.file_size > MAX_FILE_BYTES:
                report.skip(name, f'{MAX_FILE_BYTES // 1024 // 1024}MB を超えています')
                continue
            _ingest_one(path, name, archive.read(info), val_ratio, report)


def ingest_uploads(name, uploads):
    """Store uploaded files (FileStorage-like objects) into a dataset."""
    path = dataset_path(name)
    val_ratio = load_meta(path)['val_ratio']
    report = IngestReport()
    for upload in uploads:
        filename = upload.filename or ''
        if not filename:
            continue
        if filename.lower().endswith('.zip'):
            _ingest_zip(path, filename, upload.stream, val_ratio, report)
        else:
            data = upload.stream.read(MAX_FILE_BYTES + 1)
            _ingest_one(path, filename, data, val_ratio, report)
    if report.class_names and not load_classes(path):
        save_classes(path, report.class_names)
    logging.info(
        f"Ingested into {name}: {report.images} images, "
        f"{report.labels} labels, {len(report.skipped)} skipped")
    return report


def resplit(name, val_ratio):
    """Move image/label pairs so the split matches val_ratio.

    Guarantees at least one validation image when there are 2+ images.
    Returns the number of moved pairs.
    """
    path = dataset_path(name)
    entries = []
    for split in SPLITS:
        images_dir = os.path.join(path, split, 'images')
        for f in _listdir(images_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTS:
                entries.append((split, stem, f))
    targets = {stem: split_for_stem(stem, val_ratio)
               for _, stem, _ in entries}
    if len(entries) >= 2 and 'valid' not in targets.values():
        stem = min(targets, key=lambda s: hashlib.md5(
            s.encode('utf-8')).hexdigest())
        targets[stem] = 'valid'
    if len(entries) >= 2 and 'train' not in targets.values():
        stem = max(targets, key=lambda s: hashlib.md5(
            s.encode('utf-8')).hexdigest())
        targets[stem] = 'train'
    moved = 0
    for split, stem, f in entries:
        target = targets[stem]
        if target == split:
            continue
        for kind in ('images', 'labels'):
            os.makedirs(os.path.join(path, target, kind), exist_ok=True)
        os.replace(os.path.join(path, split, 'images', f),
                   os.path.join(path, target, 'images', f))
        label = os.path.join(path, split, 'labels', stem + LABEL_EXT)
        if os.path.exists(label):
            os.replace(label, os.path.join(
                path, target, 'labels', stem + LABEL_EXT))
        moved += 1
    for split in SPLITS:
        cache = os.path.join(path, split, 'labels.cache')
        if os.path.exists(cache):
            os.remove(cache)
    meta = load_meta(path)
    meta['val_ratio'] = val_ratio
    save_meta(path, meta)
    logging.info(f"Resplit {name} at {val_ratio:.2f}: moved {moved}")
    return moved


def summarize(name, preview_limit=24):
    """Collect counts, class usage, problems and preview items."""
    path = dataset_path(name)
    classes = load_classes(path)
    summary = {
        'name': name,
        'classes': classes,
        'val_ratio': load_meta(path)['val_ratio'],
        'splits': {},
        'class_counts': [0] * len(classes),
        'problems': [],
        'out_of_range': [],
        'preview': [],
    }
    for split in SPLITS:
        images_dir = os.path.join(path, split, 'images')
        labels_dir = os.path.join(path, split, 'labels')
        images = sorted(
            f for f in _listdir(images_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        stems = {os.path.splitext(f)[0] for f in images}
        labels = {os.path.splitext(f)[0] for f in _listdir(labels_dir)
                  if f.endswith(LABEL_EXT)}
        stats = {'images': len(images), 'labeled': 0, 'background': 0,
                 'unlabeled': 0, 'boxes': 0,
                 'orphan_labels': len(labels - stems)}
        for f in images:
            stem = os.path.splitext(f)[0]
            if stem not in labels:
                stats['unlabeled'] += 1
                if len(summary['preview']) < preview_limit:
                    summary['preview'].append(
                        {'split': split, 'file': f, 'boxes': [],
                         'unlabeled': True})
                continue
            try:
                objects = read_label_file(
                    os.path.join(labels_dir, stem + LABEL_EXT))
            except (DatasetError, UnicodeDecodeError) as e:
                summary['problems'].append(f'{split}/{stem}.txt: {e}')
                continue
            if objects:
                stats['labeled'] += 1
            else:
                stats['background'] += 1
            stats['boxes'] += len(objects)
            for class_id, _ in objects:
                if class_id < len(classes):
                    summary['class_counts'][class_id] += 1
                else:
                    summary['out_of_range'].append(
                        f'{split}/{stem}.txt（クラス番号 {class_id}）')
            if len(summary['preview']) < preview_limit:
                summary['preview'].append(
                    {'split': split, 'file': f,
                     'boxes': [_box_percent(c, xy) for c, xy in objects]})
        summary['splits'][split] = stats
    summary['total_images'] = sum(
        s['images'] for s in summary['splits'].values())
    summary['total_labeled'] = sum(
        s['labeled'] for s in summary['splits'].values())
    summary['total_unlabeled'] = sum(
        s['unlabeled'] for s in summary['splits'].values())
    return summary


def _box_percent(class_id, coords):
    """Convert a YOLO box or polygon to CSS percentages for preview."""
    if len(coords) == 4:
        cx, cy, w, h = coords
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        xs, ys = coords[0::2], coords[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return {'cls': class_id, 'left': x0 * 100, 'top': y0 * 100,
            'width': (x1 - x0) * 100, 'height': (y1 - y0) * 100}


def training_warnings(summary):
    """Return non-blocking issues worth showing before training."""
    warnings = []
    if summary['total_unlabeled']:
        warnings.append(
            f"ラベル未作成の画像が {summary['total_unlabeled']} 枚あります。"
            '「物体なし」の画像として学習されます')
    if summary['total_labeled'] and summary['total_labeled'] < 20:
        warnings.append('ラベル付きの画像が 20 枚未満です。'
                        '精度を出すには数十〜数百枚あると安心です')
    unused = [name for name, count in
              zip(summary['classes'], summary['class_counts']) if count == 0]
    if unused and summary['total_labeled']:
        shown = '、'.join(unused[:8])
        more = f' ほか {len(unused) - 8} 個' if len(unused) > 8 else ''
        warnings.append(f'ラベルが 1 つもないクラスが {len(unused)} 個あります'
                        f'（{shown}{more}）。そのクラスは検出できません')
    return warnings


def training_blockers(summary, auto_split):
    """Return reasons the dataset cannot be trained yet."""
    blockers = []
    if not summary['classes']:
        blockers.append('クラス名が登録されていません')
    if summary['total_labeled'] == 0:
        blockers.append('ラベル付きの画像がありません')
    if summary['total_images'] < 2:
        blockers.append('画像が 2 枚以上必要です（学習用と検証用）')
    if not auto_split:
        if summary['splits']['train']['labeled'] == 0:
            blockers.append('学習用（train）にラベル付きの画像がありません')
        if summary['splits']['valid']['images'] == 0:
            blockers.append('検証用（valid）に画像がありません')
    if summary['out_of_range']:
        blockers.append(
            f"クラス数（{len(summary['classes'])}）を超えるクラス番号を使う"
            f"ラベルが {len(summary['out_of_range'])} 件あります。"
            'クラス名を追加するか、ラベルを直してください')
    if summary['problems']:
        blockers.append(f"読み込めないラベルが {len(summary['problems'])} 件あります")
    return blockers


def _image_file(name, split, filename):
    """Resolve an image inside a dataset, rejecting path tricks."""
    if split not in SPLITS:
        raise DatasetError('分割の指定が不正です')
    path = dataset_path(name)
    base = os.path.basename(filename or '')
    if not base or base != filename or \
            os.path.splitext(base)[1].lower() not in IMAGE_EXTS:
        raise DatasetError('画像の指定が不正です')
    image_file = os.path.join(path, split, 'images', base)
    if not os.path.isfile(image_file):
        raise DatasetError(f'画像「{base}」が見つかりません')
    label_file = os.path.join(
        path, split, 'labels', os.path.splitext(base)[0] + LABEL_EXT)
    return image_file, label_file


def list_images(name):
    """List every image with its split and label status."""
    path = dataset_path(name)
    items = []
    for split in SPLITS:
        images_dir = os.path.join(path, split, 'images')
        labels_dir = os.path.join(path, split, 'labels')
        for f in sorted(_listdir(images_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMAGE_EXTS:
                continue
            label_file = os.path.join(labels_dir, stem + LABEL_EXT)
            if not os.path.exists(label_file):
                status = 'unlabeled'
            elif os.path.getsize(label_file) == 0:
                status = 'background'
            else:
                status = 'labeled'
            items.append({'split': split, 'file': f, 'status': status})
    return items


def get_label(name, split, filename):
    """Return the boxes of one image as dicts (polygons become boxes)."""
    image_file, label_file = _image_file(name, split, filename)
    with Image.open(image_file) as img:
        width, height = img.size
    boxes = []
    for class_id, coords in read_label_file(label_file):
        b = _box_percent(class_id, coords)
        boxes.append({'cls': class_id,
                      'cx': (b['left'] + b['width'] / 2) / 100,
                      'cy': (b['top'] + b['height'] / 2) / 100,
                      'w': b['width'] / 100, 'h': b['height'] / 100})
    return {'boxes': boxes, 'width': width, 'height': height,
            'exists': os.path.exists(label_file)}


def save_label(name, split, filename, boxes):
    """Validate and write boxes for one image. Empty means no objects."""
    _, label_file = _image_file(name, split, filename)
    num_classes = len(load_classes(dataset_path(name)))
    if not num_classes:
        raise DatasetError('先にクラス名を登録してください')
    if not isinstance(boxes, list) or len(boxes) > 1000:
        raise DatasetError('枠の指定が不正です（1 枚あたり 1000 個まで）')
    lines = []
    for box in boxes:
        try:
            class_id = int(box['cls'])
            values = [float(box[k]) for k in ('cx', 'cy', 'w', 'h')]
        except (KeyError, TypeError, ValueError):
            raise DatasetError('枠の値が不正です') from None
        if not 0 <= class_id < num_classes:
            raise DatasetError(f'クラス番号 {class_id} は登録されていません')
        cx, cy, w, h = values
        x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x1 - x0 <= 0.001 or y1 - y0 <= 0.001:
            continue
        lines.append(f'{class_id} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} '
                     f'{x1 - x0:.6f} {y1 - y0:.6f}')
    with open(label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    return len(lines)


def delete_image(name, split, filename):
    """Delete one image and its label."""
    image_file, label_file = _image_file(name, split, filename)
    os.remove(image_file)
    if os.path.exists(label_file):
        os.remove(label_file)


def ingest_pairs(name, pairs):
    """Store (filename, image_bytes, label_text) triples into a dataset."""
    path = dataset_path(name)
    val_ratio = load_meta(path)['val_ratio']
    report = IngestReport()
    for filename, image_bytes, label_text in pairs:
        stem, ext = sanitize_filename(filename)
        _store_image(path, stem, ext, image_bytes, val_ratio, report,
                     filename)
        _store_label(path, stem, label_text.encode('utf-8'), val_ratio,
                     report, stem + LABEL_EXT)
    return report


def update_classes(name, names):
    """Replace class names, refusing to drop classes still in use."""
    path = dataset_path(name)
    max_used = -1
    for split in SPLITS:
        labels_dir = os.path.join(path, split, 'labels')
        for f in _listdir(labels_dir):
            if not f.endswith(LABEL_EXT):
                continue
            try:
                objects = read_label_file(os.path.join(labels_dir, f))
            except (DatasetError, UnicodeDecodeError):
                continue
            for class_id, _ in objects:
                max_used = max(max_used, class_id)
    if max_used >= len(names):
        raise DatasetError(
            f'クラス番号 {max_used} を使っているラベルがあるため、'
            f'クラスを {max_used + 1} 個より少なくできません')
    save_classes(path, names)
