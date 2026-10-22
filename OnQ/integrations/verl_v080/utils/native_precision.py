"""Inspect saved native shards before FSDP can cast them into FP32 parameters."""
from pathlib import Path

import torch


def check_model_state_fp32(state):
    if not isinstance(state, dict) or not state:
        raise ValueError('Native model shard must contain a nonempty state dict')
    floating = 0
    for name, value in state.items():
        # ShardedTensor and DTensor expose their original logical dtype. Check
        # both metadata and local storage, including ranks with empty shards.
        dtype = getattr(value, 'dtype', None)
        if not isinstance(dtype, torch.dtype) or getattr(value, 'is_meta', False):
            raise ValueError(f'Unsupported native model tensor: {name}')
        if dtype.is_floating_point:
            floating += 1
            if dtype != torch.float32:
                raise ValueError(f'Native resume requires raw FP32: {name} is {dtype}')
        elif dtype.is_complex:
            raise ValueError(f'Unsupported complex native model tensor: {name}')
        if hasattr(value, 'local_shards'):
            parts = [shard.tensor for shard in value.local_shards()]
        elif hasattr(value, 'to_local'):
            parts = [value.to_local()]
        elif isinstance(value, torch.Tensor):
            parts = [value]
        else:
            raise ValueError(f'Unsupported native model storage: {name}')
        for part in parts:
            if part.is_meta or part.dtype != dtype:
                raise ValueError(f'Native shard storage differs from metadata: {name}')
    if not floating:
        raise ValueError('Native model shard has no floating-point model state')


def require_native_fp32_resume(local_path, adapter, *, rank, world_size):
    """All ranks must pass before any rank enters the native mutating loader.

    veRL's checkpoint manager uses global rank/world size for shard names.
    Load trusted native checkpoint pickles on CPU without dtype conversion;
    release this inspection copy before the parent's normal restore.
    """
    from .resume import checkpoint_objective

    error = None
    try:
        checkpoint_objective(local_path, adapter)
        source = Path(local_path) / f'model_world_size_{world_size}_rank_{rank}.pt'
        state = torch.load(source, map_location='cpu', weights_only=False)
        try:
            check_model_state_fp32(state)
        finally:
            del state
    except Exception as exc:
        error = f'rank {rank}: {type(exc).__name__}: {exc}'
    errors = [error]
    if torch.distributed.is_initialized():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
    elif world_size != 1:
        raise ValueError('Native multi-rank admission requires an initialized process group')
    failures = [item for item in errors if item is not None]
    if failures:
        raise ValueError('Native FP32 resume admission failed; ' + '; '.join(failures))
