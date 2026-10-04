"""
Small JSON state files shared by the web app and the worker process.

web_run.json  - what was requested and the final status of a run
progress.json - live progress written by the worker
"""

import csv
import json
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, 'runs', 'train')
PREDICT_DIR = os.path.join(REPO_ROOT, 'runs', 'predict')
META_FILE = 'web_run.json'
PROGRESS_FILE = 'progress.json'
LOG_FILE = 'web_job.log'


def read_json(path, default=None):
    """Read a JSON file, returning default when missing or broken."""
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, data):
    """Write JSON atomically so readers never see half a file."""
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows refuses while another process has the file open
            time.sleep(0.1)
    os.replace(tmp, path)


def update_meta(run_dir, **fields):
    """Merge fields into a run's web_run.json."""
    path = os.path.join(run_dir, META_FILE)
    meta = read_json(path)
    meta.update(fields)
    write_json(path, meta)
    return meta


def read_results(run_dir):
    """Return results.csv rows as dicts with stripped keys and floats."""
    path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                parsed = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    try:
                        parsed[key.strip()] = float(value)
                    except (TypeError, ValueError):
                        parsed[key.strip()] = value
                rows.append(parsed)
    except OSError:
        return []
    return rows
