"""QAT-only operations; no GRPO, reward, sampler or old veRL imports.

Call installation before FSDP wrapping. Distributed loading/broadcast and
summon-full-parameters remain the engine adapter's responsibility.
"""
import math
from collections.abc import Mapping
from types import MethodType

import torch

from OnQ.quantization.efficientqat_helpers import (
    build_param_groups,
    freeze_non_quant_params,
    iter_eqat_linears,
)
from OnQ.quantization.qlinear import convertModelToQuant
from .frozen_bf16 import attach_frozen_copies, embedding_forward, frozen_bf16, linear_forward


def _bf16_activation_output(module, inputs, output):
    """Cast activations only; never replace or round FP32 stored parameters."""
    if not torch.is_tensor(output) or not output.is_floating_point():
        raise TypeError('Expected a floating embedding/RMSNorm activation')
    return output.to(torch.bfloat16)


def _qwen3_rmsnorm_bf16_forward(module, hidden_states):
    """Match HF Qwen3 RMSNorm with BF16 weights, without rounding stored masters."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
    weight = (frozen_bf16(module) if hasattr(module, '_qat_frozen_bf16_copies')
              else module.weight.to(torch.bfloat16))
    return weight * hidden_states.to(input_dtype)


def install_qlinears(model, *, bits: int, group_size: int = 128,
                     scale_floor: float = 1e-4,
                     cache_frozen_weights=True):
    if bits not in (2, 3, 4):
        raise ValueError("This adapter supports W2/W3/W4 only")
    if group_size <= 0 or not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("group_size and scale_floor must be positive")
    if any(iter_eqat_linears(model)):
        raise ValueError("QLinear installation must happen exactly once")
    if type(cache_frozen_weights) is not bool:
        raise ValueError('cache_frozen_weights must be a boolean')
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise ValueError('Training parameters must already be FP32: ' + name)
    model, converted = convertModelToQuant(
        model, quant_type=f"efficientqat-int{bits}", q_group_size=group_size,
        compute_dtype=torch.bfloat16, quant_math_dtype=torch.float32,
        scale_floor=scale_floor,
    )
    if not converted:
        raise ValueError("No quantized linear modules were installed")
    freeze_non_quant_params(model, train_mode="weight_scale")
    activation_modules = []
    norm_weight_modules = []
    shared_copies = {}
    for name, module in model.named_modules():
        if cache_frozen_weights and name == 'lm_head' and type(module) is torch.nn.Linear:
            attach_frozen_copies(module, shared_copies, ('weight', 'bias'))
            module.forward = MethodType(linear_forward, module)
        if isinstance(module, torch.nn.Embedding) or type(module).__name__.endswith('RMSNorm'):
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                raise ValueError('BF16 activation contract expects frozen embedding/Norm parameters')
            if type(module).__name__ == 'Qwen3RMSNorm':
                from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

                if type(module) is Qwen3RMSNorm:
                    if cache_frozen_weights:
                        attach_frozen_copies(module, shared_copies)
                    module.forward = MethodType(_qwen3_rmsnorm_bf16_forward, module)
                    norm_weight_modules.append(name)
            if (cache_frozen_weights and type(module) is torch.nn.Embedding
                    and module.max_norm is None):
                attach_frozen_copies(module, shared_copies)
                module.forward = MethodType(embedding_forward, module)
            module.register_forward_hook(_bf16_activation_output)
            activation_modules.append(name)
    model._qat_bf16_activation_modules = tuple(activation_modules)
    model._qat_bf16_norm_weight_modules = tuple(norm_weight_modules)
    if hasattr(model, 'config'):
        model.config.qat_activation_dtype = 'bfloat16'
        model.config.qat_quant_input_dtype = 'float32'
        model.config.qat_frozen_weight_cache = cache_frozen_weights
        if norm_weight_modules:
            model.config.qat_norm_weight_dtype = 'bfloat16'
    return model


def restore_quantizer_state(model, state: Mapping[str, torch.Tensor], *, legacy_metadata=None):
    """Strictly restore quantizer tensors after the engine loads latent weights.

This does not certify checkpoint provenance or restore optimizer state. It must
not be used to claim inference-grid exports retain the original latent weights.
"""
    expected = {k: v for k, v in model.state_dict().items()
                if ".weight_quantizer." in k}
    supplied = {k: v for k, v in state.items() if ".weight_quantizer." in k}
    synthesized = []
    if legacy_metadata is not None:
        if legacy_metadata.get('format') != 'official_fakequant':
            raise ValueError('Legacy state must be verified official_fakequant')
        for name, q in iter_eqat_linears(model):
            if (legacy_metadata.get('bits') != q.weight_quantizer.wbits
                    or legacy_metadata.get('group_size') != q.weight_quantizer.group_size):
                raise ValueError('Legacy quantizer configuration mismatch')
            for tail in ('wbits_buf', 'fixed_codes_buf'):
                key = name + '.weight_quantizer.' + tail
                if key not in supplied:
                    supplied[key] = expected[key].clone()
                    synthesized.append(key)
    if not expected or supplied.keys() != expected.keys():
        raise ValueError("Incomplete or unexpected quantizer checkpoint keys")
    for key, value in supplied.items():
        target = expected[key]
        if target.is_meta:
            raise ValueError("Restore quantizer state on a materialized source rank")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError('Quantizer handoff must be FP32: ' + key)
        if value.shape != target.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid quantizer tensor: {key}")
        if not key.endswith((".scale", ".zero_point")) and not torch.equal(value.to(target), target):
            raise ValueError(f"Quantizer configuration mismatch: {key}")
    # copy_ preserves FP32 parameter identity and the frozen zero-point scope.
    with torch.no_grad():
        for key, value in supplied.items():
            expected[key].copy_(value.to(expected[key]))
    return {'restored_keys': sorted(set(supplied) - set(synthesized)),
            'metadata_initialized_buffers': synthesized}


def parameter_groups(model, *, lr: float, weight_decay: float, quant_lr=None):
    """AdamW groups: QLinear weights at ``lr``; quantizer scales at ``quant_lr``.

    ``quant_lr=None`` keeps the historical shared LR for both groups.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if quant_lr is None:
        quant_lr = lr
    if isinstance(quant_lr, bool) or not math.isfinite(quant_lr) or quant_lr <= 0:
        raise ValueError("Quantizer learning rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Weight decay must be finite and nonnegative")
    allowed = {id(p) for _, q in iter_eqat_linears(model)
               for p in (q.weight, q.weight_quantizer.scale)}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if not allowed or actual != allowed:
        raise ValueError("Expected exactly QLinear weight+scale trainable")
    # Preserve the native optimizer's configured decay for both parameter groups.
    return build_param_groups(model, weight_lr=lr, quant_lr=quant_lr,
                              weight_decay=weight_decay, quant_weight_decay=weight_decay)


def summoned_full_state(model):
    """FULL_STATE_DICT-equivalent mapping of detached views, without cloning.

    Only valid inside ``FSDP.summon_full_params(model, writeback=False,
    rank0_only=False)`` with ``use_orig_params=True``, where every original
    parameter is a view of the unsharded flat parameter. The returned keys and
    their order equal ``model.state_dict()`` under FULL_STATE_DICT: the
    ``nn.Module.state_dict`` traversal (per module: parameters, then persistent
    buffers, then children in registration order) with wrapper prefixes removed
    by torch's own ``clean_tensor_name``. FSDP's post-state-dict hook moves each
    wrapped subtree's already-contiguous trailing block, so key order is
    unchanged by its prefix replacement. Unlike that hook, full parameters are
    not cloned: the caller must consume the views inside the summon context.
    Unreviewed state-dict customizations fail closed.
    """
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp._common_utils import clean_tensor_name
    from torch.distributed.fsdp.fully_sharded_data_parallel import FLAT_PARAM

    state = {}

    def visit(module, prefix):
        wrapper = isinstance(module, FSDP)
        if not wrapper and (module._state_dict_hooks or module._state_dict_pre_hooks):
            raise ValueError(f'Unreviewed state_dict hook on {type(module).__name__}')
        for method in ('state_dict', '_save_to_state_dict', 'get_extra_state'):
            if not wrapper and getattr(type(module), method) is not getattr(nn.Module, method):
                raise ValueError(f'Unreviewed {method} override on {type(module).__name__}')
        for name, param in module._parameters.items():
            if param is None:
                continue
            if name == FLAT_PARAM:
                raise ValueError('Flat parameter is registered; expected use_orig_params=True')
            state[clean_tensor_name(prefix + name)] = param.detach()
        for name, buffer in module._buffers.items():
            if buffer is None or name in module._non_persistent_buffers_set:
                continue
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                # FSDP recasts floating buffers to their original dtype; that is
                # the identity only for the reviewed all-FP32 configuration.
                raise ValueError(f'Unreviewed floating buffer dtype {buffer.dtype}: {prefix}{name}')
            state[clean_tensor_name(prefix + name)] = buffer.detach()
        for name, child in module._modules.items():
            if child is not None:
                visit(child, prefix + name + '.')

    visit(model, '')
    return state


@torch.no_grad()
def dense_full_state_weights(model, full_state):
    """Export canonical FSDP full-state keys despite wrapper-prefixed module names."""
    quant = {name.replace('_fsdp_wrapped_module.', ''): q
             for name, q in iter_eqat_linears(model)}
    emitted = set()
    for name, q in quant.items():
        key = name + '.weight'
        if key not in full_state:
            raise ValueError(f'Missing canonical quantized weight: {key}')
        for p in (q.weight, q.weight_quantizer.scale, q.weight_quantizer.zero_point):
            if p.is_meta or p.dtype != torch.float32:
                raise ValueError('FSDP export requires full FP32 quantizer parameters')
        if q.weight.shape != full_state[key].shape:
            raise ValueError(f'FSDP weight is not fully materialized: {key}')
        yield key, q.quantized_weight().detach().bfloat16()
        emitted.add(key)
    for key, value in full_state.items():
        if key in emitted or any(key.startswith(name + '.weight_quantizer.') for name in quant):
            continue
        yield key, value.detach().bfloat16() if value.is_floating_point() else value.detach()
