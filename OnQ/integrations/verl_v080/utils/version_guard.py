"""Reject legacy namespace shadowing or an unpinned native veRL checkout."""
import importlib.util
from pathlib import Path
import subprocess

from .. import UPSTREAM_COMMIT, DEPENDENCY_COMMIT


def verify_source(root):
    root = Path(root).resolve()
    sha = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    if sha not in (UPSTREAM_COMMIT, DEPENDENCY_COMMIT):
        raise RuntimeError(f'Wrong veRL source: {root} at {sha}; expected {DEPENDENCY_COMMIT} or {UPSTREAM_COMMIT}')
    untracked = subprocess.run(['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard', '--', 'verl'],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    if untracked:
        raise RuntimeError('Native veRL source contains untracked package files')
    diff = subprocess.run(['git', '-C', str(root), 'diff', '--name-only', 'HEAD', '--', 'verl'],
                          capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    if sha == DEPENDENCY_COMMIT:
        # The release fork already includes the admitted patch. Reject further edits.
        dirty = subprocess.run(['git', '-C', str(root), 'diff', '--name-only', 'HEAD'],
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if dirty:
            raise RuntimeError('Pinned veRL dependency has local modifications')
        from .backports import manifest, verify_reviewed_edits
        verify_reviewed_edits(root, manifest()['files'])
    elif diff:
        from .backports import verify_reviewed_edits
        verify_reviewed_edits(root, diff.splitlines())
    return sha


def assert_pinned_upstream():
    spec = importlib.util.find_spec('verl')
    if spec is None or spec.origin is None:
        raise RuntimeError('Native veRL is not installed')
    root = Path(spec.origin).resolve().parent.parent
    verify_source(root)
    return root
