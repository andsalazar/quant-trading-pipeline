"""
Overnight Experiment Runner
============================
Chains multiple experiments sequentially to maximize overnight GPU time.
Waits for any existing experiment process to finish, then runs the rest.

Usage:
  python overnight_runner.py

All output logged to #_dl_experiments/logs/overnight_<timestamp>.log
"""

import subprocess
import sys
import os
import time
import json
import psutil
from datetime import datetime

PYTHON = r"<project_root>niconda3\envs\dl_experiments\python.exe"
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dl_experiment.py")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT QUEUE — runs in order, skips if results already exist
# ════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # 1. MLP experiment (may already be running from earlier launch)
    {
        'name': 'MLP_200ep',
        'args': ['--mode', 'mlp', '--epochs', '200'],
        'check_prefix': 'MLP_xxlarge_',  # last model in MLP run
    },
    # 2. Model-size sweep — the key double-descent test
    {
        'name': 'Sweep_300ep',
        'args': ['--mode', 'sweep', '--epochs', '300'],
        'check_prefix': 'sweep_mlp_d3_',
    },
    # 3. Longer MLP with bigger epochs to really push into double descent
    {
        'name': 'MLP_500ep',
        'args': ['--mode', 'mlp', '--epochs', '500'],
        'check_prefix': None,  # always run
    },
]


def log(msg, f=None):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + '\n')
        f.flush()


def check_existing_results(prefix):
    """Check if results with this prefix already exist."""
    if prefix is None:
        return False
    if not os.path.exists(RESULTS_DIR):
        return False
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith(prefix) and fname.endswith('_history.csv'):
            return True
    return False


def wait_for_existing_experiment(pid, logf):
    """Wait for an existing experiment process to finish."""
    try:
        proc = psutil.Process(pid)
        if proc.is_running() and proc.name().lower().startswith('python'):
            log(f"Waiting for existing experiment PID {pid} to finish...", logf)
            while proc.is_running():
                try:
                    proc.wait(timeout=60)
                except psutil.TimeoutExpired:
                    try:
                        cpu = proc.cpu_percent(interval=1)
                        mem_gb = proc.memory_info().rss / (1024**3)
                        log(f"  PID {pid} still running | CPU: {cpu:.0f}% | RAM: {mem_gb:.1f} GB", logf)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
            log(f"PID {pid} finished.", logf)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        log(f"PID {pid} not found or already finished.", logf)


def run_experiment(name, args, logf):
    """Run a single experiment, streaming output to log."""
    cmd = [PYTHON, '-u', RUNNER] + args
    log(f"{'='*60}", logf)
    log(f"STARTING: {name}", logf)
    log(f"Command: {' '.join(cmd)}", logf)
    log(f"{'='*60}", logf)

    t0 = time.time()
    result = subprocess.run(
        cmd, stdout=logf, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(RUNNER),
    )
    elapsed = time.time() - t0

    log(f"", logf)
    log(f"FINISHED: {name} | Exit code: {result.returncode} | "
        f"Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)", logf)
    log(f"{'='*60}\n", logf)

    return result.returncode


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f'overnight_{timestamp}.log')

    with open(log_path, 'w') as logf:
        log(f"OVERNIGHT EXPERIMENT RUNNER", logf)
        log(f"Started: {datetime.now()}", logf)
        log(f"Experiments queued: {len(EXPERIMENTS)}", logf)
        log(f"Log file: {log_path}", logf)
        log(f"", logf)

        t_start = time.time()

        # Wait for any existing MLP experiment (PID 76724)
        wait_for_existing_experiment(76724, logf)

        completed = 0
        skipped = 0
        failed = 0

        for i, exp in enumerate(EXPERIMENTS):
            log(f"\n{'#'*60}", logf)
            log(f"QUEUE [{i+1}/{len(EXPERIMENTS)}]: {exp['name']}", logf)
            log(f"{'#'*60}", logf)

            # Skip if results already exist
            if check_existing_results(exp.get('check_prefix')):
                log(f"SKIPPING {exp['name']} — results already exist", logf)
                skipped += 1
                continue

            rc = run_experiment(exp['name'], exp['args'], logf)
            if rc == 0:
                completed += 1
            else:
                failed += 1
                log(f"WARNING: {exp['name']} failed with exit code {rc}", logf)

        total_time = time.time() - t_start
        log(f"\n{'='*60}", logf)
        log(f"OVERNIGHT RUN COMPLETE", logf)
        log(f"{'='*60}", logf)
        log(f"Total time: {total_time/60:.0f} min ({total_time/3600:.1f} hr)", logf)
        log(f"Completed: {completed} | Skipped: {skipped} | Failed: {failed}", logf)
        log(f"Results in: {RESULTS_DIR}", logf)
        log(f"Finished: {datetime.now()}", logf)

        # Save summary JSON
        summary = {
            'started': timestamp,
            'finished': datetime.now().isoformat(),
            'total_hours': total_time / 3600,
            'completed': completed,
            'skipped': skipped,
            'failed': failed,
            'experiments': [e['name'] for e in EXPERIMENTS],
        }
        summary_path = os.path.join(LOG_DIR, f'overnight_{timestamp}_summary.json')
        with open(summary_path, 'w') as sf:
            json.dump(summary, sf, indent=2)
        log(f"Summary saved: {summary_path}", logf)


if __name__ == '__main__':
    main()
