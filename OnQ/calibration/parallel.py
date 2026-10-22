"""Two-device data parallelism with unchanged global batch and optimizer.

The block outputs are gathered before the original global mean MSE. PyTorch
autograd sums replica gradients onto the original parameters; no extra loss
scaling or optimizer step is introduced. Shared rotary tensors are expanded
before scatter so that both samples receive their matching position data.
"""
import os

import torch
from torch import nn


class TensorBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, _amp_dtype=torch.float16, _amp_enabled=False, **kwargs):
        # DataParallel's worker autocast defaults to FP16 in torch2.3. Preserve
        # the caller's BF16 training / FP16 activation-update context explicitly.
        with torch.autocast('cuda', dtype=_amp_dtype, enabled=_amp_enabled):
            output = self.block(x, **kwargs)
        return output[0] if isinstance(output, tuple) else output


class TwoGPUBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        assert torch.cuda.device_count() == 2
        self.parallel = nn.DataParallel(TensorBlock(block), device_ids=[0, 1], output_device=0)

    def forward(self, x, attention_mask=None, position_ids=None, position_embeddings=None):
        assert x.shape[0] == 2, 'This run preserves global batch2: one sample per GPU'

        def expand(t):
            if t is None:
                return None
            assert t.shape[0] in (1, 2), t.shape
            return t.expand(2, *t.shape[1:]) if t.shape[0] == 1 else t

        return self.parallel(x, _amp_dtype=torch.get_autocast_gpu_dtype(),
                             _amp_enabled=torch.is_autocast_enabled(),
                             attention_mask=expand(attention_mask), position_ids=expand(position_ids),
                             position_embeddings=tuple(expand(t) for t in position_embeddings))


def make_forward(block):
    mode = os.environ.get('ONQ_CALIBRATION_GPUS', '1')
    assert mode in ('1', '2')
    return TwoGPUBlock(block) if mode == '2' else block
