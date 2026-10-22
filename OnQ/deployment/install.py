"""Create a new isolated Linux environment from a resolved profile."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from OnQ.integrations.verl_v080 import UPSTREAM_COMMIT as VERL_COMMIT, DEPENDENCY_REPOSITORY


def commands(profile, root, repo, python):
    env = root / profile
    exe = str(env / 'bin/python')
    result = [[python, '-m', 'venv', str(env)],
              [exe, '-m', 'pip', 'install', 'pip==25.3', 'setuptools==80.9.0',
               'wheel==0.45.1', 'packaging==25.0', 'ninja==1.13.0'],
              [exe, '-m', 'pip', 'install', '-r', str(repo / f'OnQ/deployment/{profile}.lock')]]
    if profile in ('train', 'calibration'):
        version = '2.8.3' if profile == 'train' else '2.7.1.post4'
        result.append([exe, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps',
                       f'flash-attn=={version}'])
    result.append([exe, '-m', 'pip', 'install', '--no-deps', '-e', str(repo)])
    if profile == 'eval':
        result.append([exe, '-m', 'OnQ.evaluation.evalscope.resources', '--download'])
    if profile == 'train':
        source = root / 'verl'
        result.extend([['git', 'clone', DEPENDENCY_REPOSITORY, str(source)],
                       ['git', '-C', str(source), 'checkout', '--detach', VERL_COMMIT],
                       [exe, '-m', 'OnQ.integrations.verl_v080.utils.backports',
                        '--source', str(source), '--apply'],
                       [exe, '-m', 'pip', 'install', '--no-deps', '-e', str(source)]])
    result.extend([[exe, '-m', 'pip', 'check'],
                   [exe, '-m', 'OnQ.deployment.doctor', '--profile', profile]])
    return result


def source_revision(repo):
    # Anonymous source archives intentionally omit Git history.
    if not (repo / '.git').exists():
        return None
    return subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', required=True, choices=('train', 'calibration', 'eval'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', default='python3.11')
    p.add_argument('--plan', action='store_true', help='Print commands without creating anything')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = a.root.expanduser().absolute()
    plan = commands(a.profile, root, repo, a.python)
    if a.plan:
        print('\n'.join(shlex.join(cmd) for cmd in plan))
        return
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        p.error('These profiles target Linux x86_64; use --plan to inspect elsewhere')
    version = subprocess.check_output([a.python, '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
    if version != '3.11':
        p.error('The resolved profiles require Python 3.11')
    env = root / a.profile
    if env.exists() or (a.profile == 'train' and (root / 'verl').exists()):
        p.error('Choose a fresh profile/source location; existing environments are never upgraded')
    if a.profile != 'eval':
        cuda_home = os.environ.get('CUDA_HOME', '')
        nvcc = Path(cuda_home) / 'bin/nvcc'
        expected = '12.8' if a.profile == 'train' else '12.1'
        if not cuda_home or not nvcc.is_file():
            p.error(f'Set CUDA_HOME to a CUDA {expected} development toolkit for FlashAttention')
        if f'release {expected},' not in subprocess.check_output([str(nvcc), '--version'], text=True):
            p.error(f'CUDA_HOME must select CUDA {expected}')
    root.mkdir(parents=True, exist_ok=True)
    for cmd in plan:
        print(shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    exe = str(env / 'bin/python')
    (env / 'installed.txt').write_text(subprocess.check_output([exe, '-m', 'pip', 'freeze', '--all'], text=True))
    (env / 'INSTALL_PASS.json').write_text(json.dumps(dict(profile=a.profile, python=exe,
        revision=source_revision(repo),
        acceptance='dependency, import and version checks'), indent=2))
    print(f'PASS: installed {a.profile}: {exe}')


if __name__ == '__main__':
    main()
