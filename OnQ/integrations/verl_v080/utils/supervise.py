"""Launch training from a resolved configuration and preserve logs on exit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .method_config import validate_method
from .process_groups import _terminate_group


def wait_owned(child, cancelled, state, grace_seconds=30):
    while child.poll() is None and not cancelled:
        time.sleep(0.1)
    code = child.returncode
    # Confirm the complete owned group is gone even after a zero exit; a
    # leader can otherwise leave descendants holding log files or GPUs.
    state['cleanup'] = _terminate_group(child, grace_seconds=grace_seconds)
    if cancelled:
        return 128 + cancelled[0]
    return 128 - code if code < 0 else code


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_snapshot(root):
    files, symlinks = {}, {}
    for path in sorted(root.rglob('*')):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            symlinks[relative] = os.readlink(path)
        elif path.is_file():
            files[relative] = dict(sha256=digest(path), bytes=path.stat().st_size)
    return dict(files=files, symlinks=symlinks)


def archive_closed(active, destination, settle_seconds=1):
    # The owned child process group has already been proven stopped. A stable
    # double snapshot catches residual or unrelated writers without requiring
    # access to every system process's protected /proc/<pid>/fd directory.
    first = tree_snapshot(active)
    time.sleep(settle_seconds)
    stable = tree_snapshot(active)
    if first != stable:
        raise RuntimeError('Active directory changed after the owned process group stopped')
    destination.mkdir(exist_ok=False)
    import shutil
    files = []
    for relative, metadata in stable['files'].items():
        source = active / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        current = dict(sha256=digest(source), bytes=source.stat().st_size)
        if metadata['sha256'] != digest(target) or metadata != current:
            raise RuntimeError(f'Archive checksum mismatch or source changed: {relative}')
        files.append(dict(path=relative, **metadata))
    if tree_snapshot(active) != stable:
        raise RuntimeError('Active directory changed during archive')
    return dict(files=files, skipped_symlinks=stable['symlinks'],
                owned_process_group_stopped=True, stable_snapshots=True,
                local_originals_retained=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source', required=True, help='Path to the installed, patched veRL checkout')
    parser.add_argument('--repo', default=str(Path(__file__).resolve().parents[4]))
    parser.add_argument('--checkpoint-parent', type=Path, required=True,
                        help='Explicit reviewed physical storage parent on non-OSS hosts')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7',
                        help='Explicit reviewed physical CUDA device IDs')
    args = parser.parse_args()
    # The supervisor itself imports shared cleanup helpers after the child
    # exits; make its import path match the child runtime explicitly.
    sys.path[:0] = [args.repo, args.source]
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    root = Path(config.trainer.default_local_dir).resolve()
    expected = args.checkpoint_parent.resolve()
    if root.parent != expected or not expected.is_dir():
        raise ValueError('Checkpoint root must be a direct reviewed storage run directory')
    runtime = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars)
    active = Path(runtime['WANDB_DIR']).resolve()
    if active.is_relative_to(root) or active.name != root.name:
        raise ValueError('Local active directory must match the OSS run basename')
    if runtime['WANDB_MODE'] != 'offline':
        raise ValueError('Expected offline W&B')
    gpu_ids = args.gpus.split(',')
    if (any(not x.isdecimal() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)
            or len(gpu_ids) not in (1, 4, 8) or config.trainer.n_gpus_per_node != len(gpu_ids)
            or config.data.train_batch_size % len(gpu_ids)
            or config.actor_rollout_ref.rollout.tensor_model_parallel_size != 1):
        raise ValueError('Expected one/four/eight-GPU TP1 topology and divisible batch')
    validate_method(config, 'dynamic_kl', require_external=True)
    if not config.get('qat_r8_plan'):
        raise ValueError('Use OnQ opd prepare to create the training configuration')
    from .resume import validate_resume
    validate_resume(config, 'dynamic_kl')
    active.mkdir(parents=True, exist_ok=True)
    active.chmod(0o700)
    with (root / 'launch.lock').open('x') as stream:
        stream.write(str(os.getpid()))
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in runtime.items()})
    env.update(PYTHONPATH=args.source + os.pathsep + args.repo,
               PATH=str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', ''),
               CUDA_VISIBLE_DEVICES=args.gpus, PYTHONUNBUFFERED='1')
    env.pop('RAY_ADDRESS', None)
    command = [sys.executable, '-m', 'OnQ.integrations.verl_v080.utils.trainer',
               '--config', str(Path(args.config).resolve())]
    state = dict(supervisor_pid=os.getpid(), active_dir=str(active), command=command,
                 status='STARTING', checkpoint_root=str(root))
    cancelled = []
    def request_cancel(signum, frame):
        if not cancelled:
            cancelled.append(signum)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_cancel)
    code = 1
    child = None
    try:
        with (active / 'train.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=args.repo, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            state.update(driver_pid=child.pid, status='RUNNING')
            (root / 'driver.pid').write_text(str(child.pid))
            (root / 'supervisor_status.json').write_text(json.dumps(state, indent=2))
            code = wait_owned(child, cancelled, state)
    except Exception as exc:
        state['supervisor_error'] = repr(exc)
        if child is not None:
            try:
                state['cleanup'] = _terminate_group(child)
            except Exception as cleanup_error:
                state['cleanup'] = dict(status='FAILED', error=repr(cleanup_error))
    finally:
        state.update(train_exit=code, status='TRAINING_EXITED')
        (active / 'train.exit').write_text(str(code) + '\n')
        try:
            (root / 'train.exit').write_text(str(code) + '\n')
            receipt = archive_closed(active, root / 'active_archive')
            receipt.update(status='PASS', active_dir=str(active), train_exit=code)
        except Exception as exc:
            receipt = dict(status='FAILED', error=repr(exc), active_dir=str(active),
                           train_exit=code, local_originals_retained=True)
        (active / 'archive_status.json').write_text(json.dumps(receipt, indent=2))
        state.update(status='FINISHED', archive_status=receipt['status'])
        (active / 'supervisor_status.json').write_text(json.dumps(state, indent=2))
        try:
            (root / 'archive_status.json').write_text(json.dumps(receipt, indent=2))
            (root / 'supervisor_status.json').write_text(json.dumps(state, indent=2))
        except OSError as exc:
            state.update(status='OSS_STATUS_WRITE_FAILED', error=repr(exc))
            (active / 'supervisor_status.json').write_text(json.dumps(state, indent=2))
            code = 1
    return code if code else (0 if receipt['status'] == 'PASS' else 1)


if __name__ == '__main__':
    sys.exit(main())
