"""Reusable BF16 execution copies; FP32 frozen parameters remain authoritative.

Copies are ordinary tensors, not buffers: FSDP's FP32 buffer policy must not
upcast them, and they must not enter state_dict or optimizer state. FSDP views
change identity/version during unsharding, so the engine explicitly marks those
caches and invalidates them at checkpoint load. Out-of-band edits to frozen
FSDP parameters require clear_frozen_weight_caches().
"""
from types import MethodType

import torch
from torch.nn import functional as F


class _FrozenBF16Copy:
    def __init__(self, parameter):
        self.shape = tuple(parameter.shape)
        self.fsdp_managed = False
        self.clear()

    def clear(self):
        self.value = None
        self.key = None

    def get(self, parameter):
        if parameter.is_meta or tuple(parameter.shape) != self.shape:
            raise RuntimeError('Frozen execution copy requires a full materialized parameter')
        if parameter.dtype != torch.float32:
            raise RuntimeError('Frozen execution copy requires FP32 source storage')
        key = (parameter.device, parameter.dtype)
        if not self.fsdp_managed:
            if parameter.requires_grad:
                raise RuntimeError('Cannot cache a trainable parameter')
            key += (id(parameter), parameter.data_ptr(), parameter._version)
        if self.value is None or self.key != key:
            # A rollout/logprob pass may run in inference_mode first. Its copy
            # must still be saveable by autograd for later input gradients.
            with torch.inference_mode(False), torch.no_grad():
                self.value = parameter.detach().to(torch.bfloat16)
            self.key = key
        return self.value


def frozen_bf16(module, name='weight'):
    return module._qat_frozen_bf16_copies[name].get(getattr(module, name))


def clear_frozen_weight_caches(model):
    for module in model.modules():
        for cache in getattr(module, '_qat_frozen_bf16_copies', {}).values():
            cache.clear()


def mark_frozen_caches_fsdp_managed(model):
    """Call only after FSDP wrapping/sync, with the frozen scope validated."""
    clear_frozen_weight_caches(model)
    for module in model.modules():
        for cache in getattr(module, '_qat_frozen_bf16_copies', {}).values():
            cache.fsdp_managed = True


def _after_load(module, incompatible_keys):
    for cache in module._qat_frozen_bf16_copies.values():
        cache.clear()


def _apply(module, fn, recurse=True):
    # Module.to()/to_empty() do not visit plain tensor attributes themselves.
    for cache in module._qat_frozen_bf16_copies.values():
        cache.clear()
    return type(module)._apply(module, fn, recurse=recurse)


def attach_frozen_copies(module, shared, names=('weight',)):
    copies = {}
    for name in names:
        parameter = getattr(module, name)
        if parameter is None:
            continue
        if parameter.requires_grad:
            raise ValueError('BF16 execution copies require frozen parameters')
        copies[name] = shared.setdefault(id(parameter), _FrozenBF16Copy(parameter))
    module._qat_frozen_bf16_copies = copies
    module.register_load_state_dict_post_hook(_after_load)
    module._apply = MethodType(_apply, module)


def embedding_forward(module, input):
    # max_norm mutates the weight, so installation excludes that mode.
    return F.embedding(input, frozen_bf16(module), module.padding_idx, None,
                       module.norm_type, module.scale_grad_by_freq, module.sparse)


def linear_forward(module, input):
    # Preserve ordinary nn.Linear behavior outside the engine's BF16 autocast.
    device = input.device.type
    if torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device) == torch.bfloat16:
        bias = None if module.bias is None else frozen_bf16(module, 'bias')
        return F.linear(input, frozen_bf16(module), bias)
    return F.linear(input, module.weight, module.bias)
