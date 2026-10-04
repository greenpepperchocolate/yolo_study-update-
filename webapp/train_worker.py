"""
Background worker started by the web app as a separate process.

Usage: python webapp/train_worker.py <job.json>

Modes (job["mode"]):
  train          generate offline augmentation, write data.yaml, train
  resume         resume an interrupted run from weights/last.pt
  export         export best.pt to ONNX / NCNN / TorchScript
  predict_video  run a model over a video and save an annotated copy
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import augment  # noqa: E402
import runmeta  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def resolve_device(device):
    """Return '0' when CUDA is available for 'auto', else 'cpu'."""
    if device != 'auto':
        return device
    import torch
    return '0' if torch.cuda.is_available() else 'cpu'


class ProgressWriter:
    """Throttled writer for progress.json."""

    def __init__(self, folder):
        self.path = os.path.join(folder, runmeta.PROGRESS_FILE)
        self.state = {}
        self.last_write = 0

    def update(self, force=False, **fields):
        """Merge fields and write at most once per second."""
        self.state.update(fields, updated=time.time())
        if force or time.time() - self.last_write >= 1:
            runmeta.write_json(self.path, self.state)
            self.last_write = time.time()


def add_progress_callbacks(model, progress):
    """Report epoch and batch progress from Ultralytics callbacks."""
    counter = {'batch': 0}

    def on_epoch_start(trainer):
        counter['batch'] = 0
        progress.update(force=True, phase='train', epoch=trainer.epoch + 1,
                        epochs=trainer.epochs, batch=0,
                        batches=len(trainer.train_loader))

    def on_batch_end(trainer):
        counter['batch'] += 1
        progress.update(batch=counter['batch'])

    def on_fit_epoch_end(trainer):
        progress.update(force=True, phase='train', epoch=trainer.epoch + 1,
                        epochs=trainer.epochs, batch=0)

    model.add_callback('on_train_epoch_start', on_epoch_start)
    model.add_callback('on_train_batch_end', on_batch_end)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)


def write_data_yaml(job, aug_images):
    """Write the run's data.yaml (augmented images join train only)."""
    dataset_dir = job['dataset_dir']
    train = [os.path.join(dataset_dir, 'train', 'images')]
    if aug_images:
        train.append(aug_images)
    data = {
        'path': dataset_dir,
        'train': train,
        'val': os.path.join(dataset_dir, 'valid', 'images'),
        'names': dict(enumerate(job['classes'])),
    }
    path = os.path.join(job['run_dir'], 'data.yaml')
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def run_train(job, progress):
    """Offline augmentation followed by Ultralytics training."""
    from ultralytics import YOLO

    run_dir = job['run_dir']
    offline = job['offline']
    aug_images = None
    if offline.get('copies', 0) > 0:
        out_dir = os.path.join(run_dir, 'augmented')
        shutil.rmtree(out_dir, ignore_errors=True)
        progress.update(force=True, phase='augment', done=0, total=0)
        written = augment.generate(
            job['dataset_dir'], out_dir, offline,
            lambda done, total: progress.update(
                phase='augment', done=done, total=total))
        runmeta.update_meta(run_dir, augmented_images=written)
        if written:
            aug_images = os.path.join(out_dir, 'images')
    data_yaml = write_data_yaml(job, aug_images)

    progress.update(force=True, phase='prepare')
    model = YOLO(job.get('weights') or job['model'])
    add_progress_callbacks(model, progress)
    model.train(
        data=data_yaml,
        epochs=job['epochs'],
        imgsz=job['imgsz'],
        batch=job['batch'],
        patience=job['patience'],
        device=resolve_device(job.get('device', 'auto')),
        project=os.path.dirname(run_dir),
        name=os.path.basename(run_dir),
        exist_ok=True,
        plots=True,
        **job['online'],
    )


def run_resume(job, progress):
    """Resume an interrupted run."""
    from ultralytics import YOLO

    progress.update(force=True, phase='prepare')
    model = YOLO(os.path.join(job['run_dir'], 'weights', 'last.pt'))
    add_progress_callbacks(model, progress)
    model.train(resume=True)


def run_export(job, progress):
    """Export best.pt and place a downloadable file in exports/."""
    from ultralytics import YOLO

    run_dir = job['run_dir']
    fmt = job['format']
    exports = runmeta.read_json(
        os.path.join(run_dir, runmeta.META_FILE)).get('exports', {})
    try:
        out_dir = os.path.join(run_dir, 'exports')
        os.makedirs(out_dir, exist_ok=True)
        # TorchScript/NCNN writers fail on non-ASCII paths, so use TEMP
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'best.pt')
            shutil.copy2(os.path.join(run_dir, 'weights', 'best.pt'), src)
            result = str(YOLO(src).export(format=fmt, imgsz=job['imgsz']))
            if os.path.isdir(result):
                target = os.path.join(out_dir, f'best_{fmt}')
                shutil.make_archive(target, 'zip', result)
                target += '.zip'
            else:
                target = os.path.join(out_dir, os.path.basename(result))
                shutil.copy2(result, target)
        exports[fmt] = {'status': 'finished', 'file': os.path.basename(target)}
    except Exception as e:
        exports[fmt] = {'status': 'failed', 'error': str(e)}
        runmeta.update_meta(run_dir, exports=exports)
        raise
    runmeta.update_meta(run_dir, exports=exports)


def run_predict_video(job, progress):
    """Annotate a video frame by frame, re-encoding to H.264 if possible."""
    import cv2
    from ultralytics import YOLO

    out_dir = job['out_dir']
    model = YOLO(job['weights'])
    device = resolve_device(job.get('device', 'auto'))
    counts = {}
    # OpenCV cannot open non-ASCII paths on Windows, so work in TEMP
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'input' + os.path.splitext(job['input'])[1])
        shutil.copy2(job['input'], src)
        raw = os.path.join(tmp, 'raw.mp4')
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError('動画を開けません。形式を確認してください')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=job['conf'],
                                   imgsz=job['imgsz'], device=device,
                                   verbose=False)[0]
            for c in result.boxes.cls.tolist():
                name = result.names[int(c)]
                counts[name] = counts.get(name, 0) + 1
            writer.write(result.plot())
            frame_no += 1
            progress.update(phase='video', done=frame_no, total=total)
        cap.release()
        writer.release()
        final = os.path.join(out_dir, 'result.mp4')
        ffmpeg = shutil.which('ffmpeg')
        encoded = False
        if ffmpeg:
            encoded = subprocess.run(
                [ffmpeg, '-y', '-loglevel', 'error', '-i', raw,
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', final]).returncode == 0
        if not encoded:
            shutil.copy2(raw, final)
    runmeta.write_json(os.path.join(out_dir, 'result.json'), {
        'kind': 'video', 'frames': frame_no, 'counts': counts,
        'browser_playable': encoded, 'weights': job['weights']})


MODES = {
    'train': run_train,
    'resume': run_resume,
    'export': run_export,
    'predict_video': run_predict_video,
}


def main():
    """Load the job file and run the requested mode."""
    job = runmeta.read_json(sys.argv[1])
    folder = job.get('run_dir') or job.get('out_dir')
    progress = ProgressWriter(folder)
    is_run = job['mode'] in ('train', 'resume')
    try:
        MODES[job['mode']](job, progress)
    except Exception as e:
        traceback.print_exc()
        progress.update(force=True, phase='failed', error=str(e))
        if is_run:
            runmeta.update_meta(folder, status='failed', error=str(e),
                                finished_at=time.time())
        sys.exit(1)
    progress.update(force=True, phase='finished')
    if is_run:
        runmeta.update_meta(folder, status='finished',
                            finished_at=time.time())


if __name__ == '__main__':
    main()
