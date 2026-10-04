"""
Runs one background worker process at a time and tracks its state.

Only one heavy job (training, export, video inference) runs at once,
which keeps a CPU-only PC responsive. A worker that outlives a web app
restart is re-attached by its pid.
"""

import logging
import os
import subprocess
import sys
import threading
import time

import psutil

import runmeta

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'train_worker.py')
JOB_FILE = 'job.json'
KIND_LABELS = {
    'train': '学習', 'resume': '学習の再開', 'export': 'モデルの書き出し',
    'predict_video': '動画の推論',
}


class JobError(RuntimeError):
    """Raised when a job cannot be started or stopped."""


def _is_worker(pid):
    """True if pid is a live train_worker.py process."""
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and any(
            'train_worker.py' in part for part in proc.cmdline())
    except (psutil.Error, OSError):
        return False


class JobManager:
    """Start, watch and stop the single background worker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = None
        self._adopt_running()

    def _adopt_running(self):
        """Re-attach to a training worker left running by a restart."""
        if not os.path.isdir(runmeta.RUNS_DIR):
            return
        for name in os.listdir(runmeta.RUNS_DIR):
            run_dir = os.path.join(runmeta.RUNS_DIR, name)
            meta = runmeta.read_json(os.path.join(run_dir, runmeta.META_FILE))
            pid = meta.get('pid')
            if meta.get('status') == 'running' and pid and _is_worker(pid):
                self.current = {
                    'kind': meta.get('job_kind', 'train'), 'folder': run_dir,
                    'name': name, 'proc': None, 'pid': pid,
                    'started': meta.get('started_at', time.time()),
                    'stop_requested': False, 'extra': {}}
                threading.Thread(target=self._watch, args=(self.current,),
                                 daemon=True).start()
                logging.info(f"Re-attached to running worker {pid} ({name})")
                return

    def active(self):
        """Return the running job dict, or None."""
        cur = self.current
        if not cur:
            return None
        if cur['proc'] is not None:
            alive = cur['proc'].poll() is None
        else:
            alive = _is_worker(cur['pid'])
        return cur if alive else None

    def start(self, kind, folder, job, name, extra=None):
        """Write job.json into folder and launch the worker."""
        with self.lock:
            cur = self.active()
            if cur:
                raise JobError(
                    f"{KIND_LABELS.get(cur['kind'], cur['kind'])}"
                    f"（{cur['name']}）が実行中です。終わってから実行してください")
            os.makedirs(folder, exist_ok=True)
            job = dict(job, mode=kind)
            job_path = os.path.join(folder, JOB_FILE)
            runmeta.write_json(job_path, job)
            progress = os.path.join(folder, runmeta.PROGRESS_FILE)
            if os.path.exists(progress):
                os.remove(progress)
            env = dict(os.environ, PYTHONIOENCODING='utf-8',
                       PYTHONUNBUFFERED='1')
            log = open(os.path.join(folder, runmeta.LOG_FILE), 'a',
                       encoding='utf-8')
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"{KIND_LABELS.get(kind, kind)} =====\n")
            log.flush()
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            proc = subprocess.Popen(
                [sys.executable, WORKER, job_path], cwd=runmeta.REPO_ROOT,
                stdout=log, stderr=subprocess.STDOUT, env=env,
                creationflags=flags)
            self.current = {
                'kind': kind, 'folder': folder, 'name': name, 'proc': proc,
                'pid': proc.pid, 'started': time.time(), 'log': log,
                'stop_requested': False, 'extra': extra or {}}
            if kind in ('train', 'resume'):
                runmeta.update_meta(folder, status='running', pid=proc.pid,
                                    job_kind=kind, error=None,
                                    started_at=time.time())
            threading.Thread(target=self._watch, args=(self.current,),
                             daemon=True).start()
            logging.info(f"Started {kind} worker {proc.pid} for {name}")
            return self.current

    def _watch(self, cur):
        """Wait for the worker and record how it ended."""
        if cur['proc'] is not None:
            code = cur['proc'].wait()
            cur['log'].close()
        else:
            while _is_worker(cur['pid']):
                time.sleep(2)
            code = None
        if cur['kind'] in ('train', 'resume'):
            meta = runmeta.read_json(
                os.path.join(cur['folder'], runmeta.META_FILE))
            if cur['stop_requested']:
                runmeta.update_meta(cur['folder'], status='stopped',
                                    finished_at=time.time())
            elif meta.get('status') == 'running':
                runmeta.update_meta(
                    cur['folder'], status='failed', finished_at=time.time(),
                    error=f'処理が異常終了しました（終了コード {code}）。'
                          'メモリ不足の可能性があります')
        elif cur['kind'] == 'export':
            fmt = cur['extra'].get('format')
            meta = runmeta.read_json(
                os.path.join(cur['folder'], runmeta.META_FILE))
            exports = meta.get('exports', {})
            if exports.get(fmt, {}).get('status') == 'running':
                exports[fmt] = {'status': 'stopped' if cur['stop_requested']
                                else 'failed',
                                'error': '処理が途中で終了しました'}
                runmeta.update_meta(cur['folder'], exports=exports)
        elif cur['kind'] == 'predict_video' and cur['stop_requested']:
            runmeta.write_json(
                os.path.join(cur['folder'], runmeta.PROGRESS_FILE),
                {'phase': 'stopped'})
        logging.info(f"Worker {cur['pid']} ended with code {code}")

    def stop(self):
        """Kill the running worker and its child processes."""
        cur = self.active()
        if not cur:
            raise JobError('実行中の処理はありません')
        cur['stop_requested'] = True
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(cur['pid']), '/T', '/F'],
                           capture_output=True)
        else:
            try:
                parent = psutil.Process(cur['pid'])
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.Error:
                pass
        return cur
