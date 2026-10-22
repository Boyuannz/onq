"""Cleanup of owned new-session process groups, including live descendants."""
import os
from pathlib import Path
import signal
import sys
import time


def _group_alive(pgid):
    if sys.platform.startswith('linux'):
        # Zombies cannot hold GPUs or respond to signals; their reaping belongs
        # to their parent/init. Check live group members, not killpg(0) alone.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc/'stat').read_text(errors='replace').rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_group(child, grace_seconds=30, kill_seconds=5):
    pgid = child.pid  # Popen(start_new_session=True) creates this owned group.
    if pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError('Refusing to signal an unowned process group')
    sent = []
    for signum, timeout in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, kill_seconds)):
        try:
            os.killpg(pgid, signum)
            sent.append(signum.name)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while True:
            child.poll()
            if not _group_alive(pgid):
                child.wait(timeout=kill_seconds)
                return dict(status='STOPPED', pgid=pgid, signals=sent)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    raise RuntimeError(f'Owned process group {pgid} still has live members after SIGKILL')
