"""Apply or verify the exact reviewed backport on an isolated pinned checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .. import UPSTREAM_COMMIT


BUNDLE = Path(__file__).resolve().parents[1] / 'backports'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest():
    data = json.loads((BUNDLE / 'manifest.json').read_text())
    if data['base_commit'] != UPSTREAM_COMMIT or _sha(BUNDLE / data['patch']) != data['patch_sha256']:
        raise RuntimeError('Invalid reviewed veRL backport bundle')
    return data


def verify_reviewed_edits(root, changed_files):
    data = manifest()
    if set(changed_files) != set(data['files']):
        raise RuntimeError('Native veRL source has unreviewed edits: unexpected file set')
    for name, hashes in data['files'].items():
        path = Path(root) / name
        if path.is_symlink() or not path.is_file() or _sha(path) != hashes['after']:
            raise RuntimeError(f'Native veRL source has unreviewed edits: {name}')
    return data


def apply_backport(source, *, apply=False):
    root = Path(source).resolve()
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    if git('rev-parse', 'HEAD') != UPSTREAM_COMMIT:
        raise RuntimeError('Backport requires the exact pinned veRL commit')
    data = manifest()
    changed = git('diff', '--name-only', 'HEAD', '--', 'verl').splitlines()
    if changed:
        verify_reviewed_edits(root, changed)
        return {'status': 'APPLIED_VERIFIED', 'source': str(root), 'upstream_pr': data['upstream_pr']}
    for name, hashes in data['files'].items():
        if _sha(root / name) != hashes['before']:
            raise RuntimeError(f'Unexpected original veRL source: {name}')
    patch = str(BUNDLE / data['patch'])
    git('apply', '--check', patch)
    if apply:
        git('apply', patch)
        verify_reviewed_edits(root, git('diff', '--name-only', 'HEAD', '--', 'verl').splitlines())
    return {'status': 'APPLIED_VERIFIED' if apply else 'READY_NOT_APPLIED',
            'source': str(root), 'upstream_pr': data['upstream_pr']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--apply', action='store_true', help='Modify this checkout; use an isolated copy')
    args = parser.parse_args()
    print(json.dumps(apply_backport(args.source, apply=args.apply), indent=2))


if __name__ == '__main__':
    main()
