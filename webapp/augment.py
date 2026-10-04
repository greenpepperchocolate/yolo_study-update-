"""
Pre-generated (offline) augmentation for YOLO detection datasets.

Augmented copies are written to a separate folder so they never mix
into the validation split. Polygon labels are converted to boxes.
"""

import base64
import io
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw

import dataset_store

FILL = (114, 114, 114)


def build_pipeline(cfg):
    """Build an albumentations pipeline from parsed offline settings."""
    def prob(key):
        return cfg[f'{key}_p'] / 100

    transforms = []
    if prob('hflip'):
        transforms.append(A.HorizontalFlip(p=prob('hflip')))
    if prob('vflip'):
        transforms.append(A.VerticalFlip(p=prob('vflip')))
    if prob('rotate'):
        transforms.append(A.Rotate(
            limit=cfg['rotate_limit'], border_mode=cv2.BORDER_CONSTANT,
            fill=FILL, p=prob('rotate')))
    if prob('scale'):
        r = cfg['scale_range']
        transforms.append(A.Affine(
            scale=(1 - r, 1 + r), border_mode=cv2.BORDER_CONSTANT,
            fill=FILL, p=prob('scale')))
    if prob('brightness'):
        limit = cfg['brightness_limit']
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=limit, contrast_limit=limit,
            p=prob('brightness')))
    if prob('hsv'):
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=cfg['hue_shift'],
            sat_shift_limit=cfg['sat_shift'], val_shift_limit=0,
            p=prob('hsv')))
    if prob('blur'):
        k = max(3, int(cfg['blur_max']) | 1)
        transforms.append(A.GaussianBlur(blur_limit=(3, k), p=prob('blur')))
    if prob('noise'):
        transforms.append(A.GaussNoise(
            std_range=(0.01, max(0.01, cfg['noise_std'])), p=prob('noise')))
    if prob('jpeg'):
        transforms.append(A.ImageCompression(
            quality_range=(int(cfg['jpeg_quality']), 95), p=prob('jpeg')))
    if prob('cutout'):
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, int(cfg['cutout_holes'])),
            hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15),
            fill=FILL, p=prob('cutout')))
    # Grayscale last so colour jitter does not re-colour it
    if prob('gray'):
        transforms.append(A.ToGray(num_output_channels=3, p=prob('gray')))
    return A.Compose(transforms, bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'], min_visibility=0.3,
        clip=True))


def _to_box(coords):
    """Convert YOLO box or polygon coords to a clipped (cx, cy, w, h)."""
    if len(coords) == 4:
        cx, cy, w, h = coords
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        xs, ys = coords[0::2], coords[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def _load(image_file, label_file):
    """Load an RGB image array and its boxes."""
    with Image.open(image_file) as img:
        image = np.array(img.convert('RGB'))
    boxes, classes = [], []
    for class_id, coords in dataset_store.read_label_file(label_file):
        box = _to_box(coords)
        if box[2] > 0 and box[3] > 0:
            boxes.append(box)
            classes.append(class_id)
    return image, boxes, classes


def _save_image(path, array):
    """Save an RGB array, choosing quality for JPEG."""
    img = Image.fromarray(array)
    if os.path.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
        img.save(path, quality=95)
    else:
        img.save(path)


def train_items(dataset_dir):
    """List (image_file, label_file) pairs in the train split."""
    images_dir = os.path.join(dataset_dir, 'train', 'images')
    labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    items = []
    for f in sorted(dataset_store._listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in dataset_store.IMAGE_EXTS:
            items.append((os.path.join(images_dir, f),
                          os.path.join(labels_dir, stem + '.txt')))
    return items


def generate(dataset_dir, out_dir, cfg, progress=None):
    """Write cfg['copies'] augmented copies of every train image.

    Returns the number of images written to out_dir/images.
    """
    copies = int(cfg['copies'])
    images_out = os.path.join(out_dir, 'images')
    labels_out = os.path.join(out_dir, 'labels')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    items = train_items(dataset_dir)
    pipeline = build_pipeline(cfg)
    total = len(items) * copies
    written = 0
    for index, (image_file, label_file) in enumerate(items):
        stem, ext = os.path.splitext(os.path.basename(image_file))
        try:
            image, boxes, classes = _load(image_file, label_file)
        except Exception as e:
            logging.warning(f"Skipping {image_file}: {e}")
            continue
        for i in range(copies):
            try:
                out = pipeline(image=image, bboxes=boxes,
                               class_labels=classes)
            except Exception as e:
                logging.warning(f"Augmentation failed for {image_file}: {e}")
                continue
            name = f'{stem}_aug{i}'
            _save_image(os.path.join(images_out, name + ext), out['image'])
            with open(os.path.join(labels_out, name + '.txt'), 'w',
                      encoding='utf-8') as f:
                for class_id, box in zip(out['class_labels'], out['bboxes']):
                    f.write(f"{int(class_id)} "
                            + ' '.join(f'{v:.6f}' for v in box) + '\n')
            written += 1
        if progress:
            progress((index + 1) * copies, total)
    logging.info(f"Generated {written} augmented images in {out_dir}")
    return written


def _color(class_id):
    """Stable distinct colour for a class index."""
    rng = random.Random(class_id * 7919)
    return tuple(rng.randint(60, 255) for _ in range(3))


def draw_boxes(array, boxes, classes, names):
    """Draw YOLO boxes on an RGB array and return a PIL image."""
    img = Image.fromarray(array)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line = max(2, round(min(w, h) / 200))
    for class_id, (cx, cy, bw, bh) in zip(classes, boxes):
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
        color = _color(int(class_id))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line)
        label = names[int(class_id)] if int(class_id) < len(names) \
            else str(class_id)
        draw.text((x0 + line, max(0, y0 - 12)), label, fill=color)
    return img


def preview(dataset_dir, cfg, names, count=6, max_side=480):
    """Augment a few random train images and return them as data URLs."""
    items = train_items(dataset_dir)
    random.shuffle(items)
    pipeline = build_pipeline(cfg)
    urls = []
    for image_file, label_file in items[:count]:
        try:
            image, boxes, classes = _load(image_file, label_file)
            out = pipeline(image=image, bboxes=boxes, class_labels=classes)
        except Exception as e:
            logging.warning(f"Preview failed for {image_file}: {e}")
            continue
        img = draw_boxes(out['image'], out['bboxes'], out['class_labels'],
                         names)
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        urls.append('data:image/jpeg;base64,'
                    + base64.b64encode(buf.getvalue()).decode('ascii'))
    return urls
