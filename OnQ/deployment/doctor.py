"""Check the selected environment dependencies, imports and versions."""
import argparse
import importlib
from importlib.metadata import version
import json
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', required=True, choices=('train', 'calibration', 'eval'))
    a = p.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use Python 3.11 for these resolved profiles')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    modules = {
        'train': ['torch', 'vllm', 'vllm.vllm_flash_attn.flash_attn_interface', 'transformers', 'pyarrow', 'datasets', 'ray', 'tensordict', 'flash_attn',
                  'OnQ.integrations.verl_v080.utils.engine'],
        'calibration': ['torch', 'transformers', 'accelerate', 'datasets', 'triton', 'flash_attn'],
        'eval': ['evalscope', 'evalplus', 'openai'],
    }
    for name in modules[a.profile]:
        importlib.import_module(name)
    if a.profile == 'eval':
        from OnQ.evaluation.evalscope.resources import check_ifeval_resources
        check_ifeval_resources()
    if a.profile == 'train':
        from OnQ.integrations.verl_v080.utils.version_guard import assert_pinned_upstream
        assert_pinned_upstream()
        if version('vllm') != '0.12.0':
            raise RuntimeError('The exact-candidate teacher requires vLLM 0.12.0')
    if a.profile != 'eval':
        import torch
        expected = ('2.9.0', '12.8') if a.profile == 'train' else ('2.3.1', '12.1')
        if (torch.__version__.split('+')[0], torch.version.cuda) != expected:
            raise RuntimeError(f'Expected torch/CUDA {expected}, got {torch.__version__}/{torch.version.cuda}')
    print(json.dumps(dict(status='PASS', profile=a.profile, python=sys.executable)))


if __name__ == '__main__':
    main()
