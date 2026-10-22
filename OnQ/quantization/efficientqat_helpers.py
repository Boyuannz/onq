# Source lineage: EfficientQAT/BitDistiller-derived quantization and local QAT integration.
# Exact upstream import revision is not recorded; see OnQ/UPSTREAMS.md.
"""
Helpers for training an EfficientQAT-style quantized actor with on-policy distillation.

Three entry points consumed by the verl FSDP worker (see
``verl/workers/fsdp_workers.py``):

    freeze_non_quant_params(model)          -- requires_grad discipline
    build_param_groups(model, ...)          -- two-LR AdamW groups
    materialize_dense_state_dict(model, ..) -- HF-style dict for vLLM sync

All three are specialized for modules produced by
``quantization/qlinear.py::EfficientQATQLinear``.
"""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from OnQ.quantization.qlinear import EfficientQATQLinear
from OnQ.quantization.quantizer import efficientqat_weight_forward


def iter_eqat_linears(model: nn.Module) -> Iterable[Tuple[str, EfficientQATQLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, EfficientQATQLinear):
            yield name, module


def freeze_non_quant_params(
    model: nn.Module,
    train_mode: str = 'all',
    also_train_lm_head: bool = False,
    also_train_embed_tokens: bool = False,
    also_train_norms: bool = False,
) -> dict:
    """Freeze every parameter outside an EfficientQATQLinear; enable
    weight / scale / zero_point inside one.

    train_mode controls which QuantLinear parameters are trainable:
      - 'all': weight + scale + zero_point (Exp 4/5, full E2E training)
      - 'weight_scale': weight + scale; zero_point remains frozen
      - 'scale_only': only scale (Exp 2/3, matching EfficientQAT E2E-QP phase)
      - 'quant_only': scale + zero_point; weight remains frozen

    Optional flags re-enable specific non-quant Parameters:
      - also_train_lm_head: unfreezes lm_head (needed when tie_word_embeddings=False)
      - also_train_embed_tokens: unfreezes input embedding.
      - also_train_norms: unfreezes all RMSNorm / LayerNorm weights.

    Returns counts so the caller can sanity-log the trainable param set.
    """
    assert train_mode in (
        'all', 'weight_scale', 'scale_only', 'quant_only'
    ), f"Unknown train_mode: {train_mode}"

    fixed_modules = [
        name
        for name, qmod in iter_eqat_linears(model)
        if qmod.weight_quantizer.fixed_integer_codes
    ]
    if fixed_modules and train_mode != 'scale_only':
        raise RuntimeError(
            "fixed_integer_codes requires train_mode=scale_only so the "
            "integer weight codes and zero points remain fixed; "
            f"got train_mode={train_mode!r}"
        )
    if fixed_modules and (
        also_train_lm_head or also_train_embed_tokens or also_train_norms
    ):
        raise RuntimeError(
            "fixed_integer_codes is the canonical EfficientQAT E2E-QP "
            "control and only permits quantizer scales to train"
        )

    for p in model.parameters():
        p.requires_grad = False

    trainable_modules = 0
    trainable_numel = 0
    for _, qmod in iter_eqat_linears(model):
        if train_mode == 'all':
            qmod.weight.requires_grad = True
            qmod.weight_quantizer.scale.requires_grad = True
            qmod.weight_quantizer.zero_point.requires_grad = True
            trainable_numel += (
                qmod.weight.numel()
                + qmod.weight_quantizer.scale.numel()
                + qmod.weight_quantizer.zero_point.numel()
            )
        elif train_mode == 'weight_scale':
            qmod.weight.requires_grad = True
            qmod.weight_quantizer.scale.requires_grad = True
            trainable_numel += (
                qmod.weight.numel()
                + qmod.weight_quantizer.scale.numel()
            )
        elif train_mode == 'scale_only':
            qmod.weight_quantizer.scale.requires_grad = True
            trainable_numel += qmod.weight_quantizer.scale.numel()
        elif train_mode == 'quant_only':
            # Freeze weights; only train scale + zero_point
            qmod.weight_quantizer.scale.requires_grad = True
            qmod.weight_quantizer.zero_point.requires_grad = True
            trainable_numel += (
                qmod.weight_quantizer.scale.numel()
                + qmod.weight_quantizer.zero_point.numel()
            )
        trainable_modules += 1

    extra_unfrozen_numel = 0
    extra_unfrozen_modules = []
    # quant_only is a HARD invariant: only scale + zero_point may be trained.
    # The launcher's `also_train_lm_head/embed_tokens/norms` flags are
    # silently overridden so a W2/W3 quant_only sweep cannot hide grid
    # failures behind dense head/embed/norm tuning.
    if train_mode == 'quant_only' and (also_train_lm_head or also_train_embed_tokens or also_train_norms):
        print(f"[freeze_non_quant_params] WARN: train_mode=quant_only overrides "
              f"also_train_lm_head={also_train_lm_head}, "
              f"also_train_embed_tokens={also_train_embed_tokens}, "
              f"also_train_norms={also_train_norms} → all forced False")
        also_train_lm_head = False
        also_train_embed_tokens = False
        also_train_norms = False
    if also_train_lm_head:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            # some HF models nest lm_head deeper; fall back to attribute search
            for name, m in model.named_modules():
                if name.endswith("lm_head"):
                    lm_head = m
                    break
        if lm_head is not None:
            for p in lm_head.parameters():
                p.requires_grad = True
                extra_unfrozen_numel += p.numel()
            extra_unfrozen_modules.append("lm_head")

    if also_train_embed_tokens:
        for name, m in model.named_modules():
            if name.endswith("embed_tokens"):
                for p in m.parameters():
                    p.requires_grad = True
                    extra_unfrozen_numel += p.numel()
                extra_unfrozen_modules.append(name)
                break

    if also_train_norms:
        # Match precise normalization class names (Qwen3RMSNorm / RMSNorm / LayerNorm /
        # GroupNorm etc.) by suffix rather than substring. Substring match
        # would over-include unrelated classes that happen to contain "norm".
        for name, m in model.named_modules():
            mname = type(m).__name__.lower()
            if mname.endswith("norm"):
                for p in m.parameters(recurse=False):
                    p.requires_grad = True
                    extra_unfrozen_numel += p.numel()

    frozen_numel = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable_modules": trainable_modules,
        "trainable_numel": trainable_numel,
        "extra_unfrozen_modules": extra_unfrozen_modules,
        "extra_unfrozen_numel": extra_unfrozen_numel,
        "frozen_numel": frozen_numel,
    }


@torch.no_grad()
def materialize_fixed_integer_codes(model: nn.Module) -> dict:
    """Convert dense QLinear weights into fixed integer-code storage in place."""
    enabled = 0
    materialized = 0
    for _, qmod in iter_eqat_linears(model):
        quantizer = qmod.weight_quantizer
        if not quantizer.fixed_integer_codes:
            continue
        enabled += 1
        materialized += int(quantizer.materialize_fixed_integer_codes_(qmod.weight))
    if enabled == 0:
        raise RuntimeError("no fixed-code EfficientQATQLinear modules found")
    return {"enabled_modules": enabled, "materialized_modules": materialized}


def mark_fixed_integer_codes_ready(model: nn.Module) -> int:
    """Mark post-FSDP modules ready after rank-0 code state was broadcast."""
    count = 0
    for _, qmod in iter_eqat_linears(model):
        quantizer = qmod.weight_quantizer
        if quantizer.fixed_integer_codes:
            quantizer.mark_fixed_integer_codes_ready_()
            count += 1
    return count


@torch.no_grad()
def dequantize_efficientqat_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    wbits: int,
    group_size: int,
    fixed_integer_codes: bool = False,
    scale_floor: float = 1e-4,
    scale_ceil: float = 1e4,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize one raw EfficientQAT state-dict weight for HF export."""
    if wbits < 2 or wbits > 16:
        raise ValueError(f"unsupported bit-width: {wbits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} is not divisible by group_size={group_size}"
        )
    expected_groups = weight.numel() // group_size
    if scale.numel() != expected_groups or zero_point.numel() != expected_groups:
        raise ValueError(
            "quantizer shape mismatch: "
            f"expected {expected_groups} scale/zero groups for weight "
            f"shape={tuple(weight.shape)} group_size={group_size}, got "
            f"scale={tuple(scale.shape)} zero_point={tuple(zero_point.shape)}"
        )
    if scale_floor <= 0 or scale_ceil < scale_floor:
        raise ValueError(
            f"invalid scale bounds: floor={scale_floor}, ceil={scale_ceil}"
        )
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported quantization compute dtype: {compute_dtype}")
    qmin = 0
    qmax = 2 ** wbits - 1
    values = weight.reshape(-1, group_size).to(torch.float32)
    if fixed_integer_codes:
        if not torch.equal(values, values.round()):
            raise RuntimeError("fixed-code checkpoint contains non-integer weight codes")
        if values.min() < qmin or values.max() > qmax:
            raise RuntimeError(
                f"fixed-code checkpoint has codes outside [{qmin}, {qmax}]"
            )
    return efficientqat_weight_forward(
        weight, scale, zero_point, wbits=wbits, group_size=group_size,
        compute_dtype=compute_dtype, scale_floor=scale_floor,
        scale_ceil=scale_ceil, fixed_integer_codes=fixed_integer_codes,
    )


def build_param_groups(
    model: nn.Module,
    weight_lr: float,
    quant_lr: float,
    weight_decay: float = 0.0,
    quant_weight_decay: float = 0.0,
    extra_lr: Optional[float] = None,
) -> List[dict]:
    """Return AdamW param groups keyed by topology rather than name patterns:
      * EfficientQATQLinear.weight           -> group 0 (lr=weight_lr)
      * weight_quantizer.scale / .zero_point -> group 1 (lr=quant_lr, wd=0)
      * any other still-trainable parameter  -> group 2 (lr=extra_lr or weight_lr)

    Each Parameter is deduped by ``id`` so tied weights (embed / lm_head) can
    never be double-updated. Run ``freeze_non_quant_params`` first.
    """
    seen: set = set()
    weight_params: List[nn.Parameter] = []
    quant_params: List[nn.Parameter] = []
    extra_params: List[nn.Parameter] = []

    def _add(group: List[nn.Parameter], p: nn.Parameter) -> None:
        if id(p) in seen or not p.requires_grad:
            return
        seen.add(id(p))
        group.append(p)

    for _, qmod in iter_eqat_linears(model):
        _add(weight_params, qmod.weight)
        _add(quant_params, qmod.weight_quantizer.scale)
        _add(quant_params, qmod.weight_quantizer.zero_point)

    for _, p in model.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        extra_params.append(p)

    if not weight_params and not quant_params and not extra_params:
        raise RuntimeError(
            "No trainable parameters found. Did you call freeze_non_quant_params()?"
        )

    groups: List[dict] = [
        {"params": weight_params, "lr": weight_lr, "weight_decay": weight_decay},
        {"params": quant_params, "lr": quant_lr, "weight_decay": quant_weight_decay},
    ]
    if extra_params:
        groups.append(
            {
                "params": extra_params,
                "lr": extra_lr if extra_lr is not None else weight_lr,
                "weight_decay": weight_decay,
            }
        )
    return groups


def _storage_key(tensor: torch.Tensor):
    try:
        storage_ptr = tensor.untyped_storage().data_ptr()
    except AttributeError:
        try:
            storage_ptr = tensor.storage().data_ptr()
        except Exception:
            storage_ptr = tensor.data_ptr()
    return (
        tensor.device, storage_ptr, tensor.storage_offset(), tensor.dtype,
        tuple(tensor.shape), tensor.stride(),
    )


@torch.no_grad()
def iter_dense_state_dict(
    model: nn.Module,
    dtype: Optional[torch.dtype] = torch.float16,
    offload_device: Optional[torch.device] = None,
    dedupe_storage: bool = True,
):
    """Yield (name, tensor) pairs as a streaming HF-style state dict.

    Each EfficientQATQLinear contributes ``{name}.weight`` (fake-quantized
    dense tensor) and optionally ``{name}.bias``. Every other parameter/buffer
    contributes unchanged. scale / zero_point / quantizer internals are
    dropped so the downstream loader sees a plain HF dict.
    """
    quant_module_names = {name for name, _ in iter_eqat_linears(model)}
    quant_owned_keys: set = set()
    storage_cache: dict = {}

    def _finalize(tensor: torch.Tensor, *, cacheable: bool = True) -> torch.Tensor:
        use_cache = dedupe_storage and cacheable
        if use_cache:
            key = _storage_key(tensor)
            cached = storage_cache.get(key)
            if cached is not None:
                return cached
        out = tensor
        if dtype is not None and out.is_floating_point():
            out = out.to(dtype)
        out = out.detach()
        if offload_device is not None:
            out = out.to(offload_device)
        if use_cache:
            storage_cache[key] = out
        return out

    for name, qmod in iter_eqat_linears(model):
        dense_w = qmod.weight_quantizer(qmod.weight)
        # Quantized temporaries can release/reuse their source allocation after
        # dtype conversion or offload; pointer identity is not a stable cache key.
        yield f"{name}.weight", _finalize(dense_w, cacheable=False)
        quant_owned_keys.add(f"{name}.weight")
        if qmod.bias is not None:
            yield f"{name}.bias", _finalize(qmod.bias)
            quant_owned_keys.add(f"{name}.bias")

    def _is_inside_quant_module(key: str) -> bool:
        for qname in quant_module_names:
            if key == f"{qname}.weight" or key.startswith(f"{qname}."):
                return True
        return False

    for key, tensor in model.state_dict().items():
        if key in quant_owned_keys:
            continue
        if _is_inside_quant_module(key):
            continue
        yield key, _finalize(tensor)


_DEFAULT_OFFLOAD = torch.device("cpu")


@torch.no_grad()
def materialize_dense_state_dict(
    model: nn.Module,
    dtype: Optional[torch.dtype] = torch.float16,
    offload_device: Optional[torch.device] = _DEFAULT_OFFLOAD,
    dedupe_storage: bool = True,
) -> dict:
    """Collect the streaming dict into a plain dictionary.

    Default offload is CPU: an 8B actor dense fp16 is ~16 GB, which would
    nearly double peak GPU memory alongside the FSDP-summoned source. Pass
    ``offload_device=None`` or a cuda device when the consumer expects GPU
    tensors and the budget is safe.
    """
    return dict(
        iter_dense_state_dict(
            model,
            dtype=dtype,
            offload_device=offload_device,
            dedupe_storage=dedupe_storage,
        )
    )


# NOTE: load_blockap_checkpoint() was removed 2026-04-30 along with the
# Phase-2 scale-only launchers (train_grpo_blockap_e2e.sh,
# train_opd_grpo_blockap_e2e.sh). The canonical workflow loads Block-AP
# output as `actor_rollout_ref.model.path` directly, so the separate
# blockap_ckpt_path loader was no longer needed.


def inject_blockap_quant_state(model: nn.Module, local_path: str) -> int:
    """Inject Block-AP-calibrated scale/zero_point into freshly-constructed
    EfficientQATWeightQuantizer modules by reading weight_quantizer.* keys
    directly from safetensors at ``local_path`` and ``load_state_dict``-ing
    them with strict=False.

    Must be called AFTER ``convertModelToQuant`` (so the modules exist) but
    before FSDP wrapping (so load_state_dict sees the right parameter tree).

    Raises if ``local_path`` has no safetensors or no weight_quantizer.* keys
    — caller is responsible for only invoking when init_mode='blockap' is set.
    """
    import glob
    import os
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(local_path, "model*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"init_mode=blockap: no safetensors found under {local_path}; "
            f"use the current calibration FP32 latent checkpoint"
        )
    # Load to CUDA if available so the inject tensors end up on the same
    # device the FSDP wrap will use. safe_open loads to CPU by default; with
    # assign=True the loaded tensor REPLACES the model Parameter, so its
    # device becomes the parameter's device. We want GPU.
    import torch as _torch
    target_device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
    inject_sd = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if "weight_quantizer." in k:
                    inject_sd[k] = f.get_tensor(k).to(target_device)
    if not inject_sd:
        raise RuntimeError(
            f"init_mode=blockap: no weight_quantizer.* keys in {local_path}. "
            f"Use the current calibration FP32 latent checkpoint, "
            f"not a plain HF model or packed inference export."
        )
    # Sanity: compute the model's expected weight_quantizer.* key set.
    # Every EfficientQATQLinear contributes scale, zero_point, wbits_buf, and
    # fixed_codes_buf. Legacy Block-AP checkpoints may omit both buffers.
    # Some fakequant Block-AP checkpoints store only the trained scale/zero
    # tensors; wbits_buf is already fixed by the constructed quantizer
    # (efficientqat-int2/int3/etc.), so it is allowed to be absent.
    expected_keys = {
        k for k in model.state_dict().keys() if "weight_quantizer." in k
    }
    optional_suffixes = (".wbits_buf", ".fixed_codes_buf")
    required_keys = {
        k for k in expected_keys if not k.endswith(optional_suffixes)
    }
    inject_keys = set(inject_sd.keys())
    missing_in_ckpt = required_keys - inject_keys
    extra_in_ckpt = inject_keys - expected_keys
    if missing_in_ckpt or extra_in_ckpt:
        raise RuntimeError(
            f"[Block-AP inject] key set mismatch — refuses to silently mix "
            f"Block-AP and RTN init. "
            f"missing_in_ckpt={len(missing_in_ckpt)} "
            f"(sample: {sorted(missing_in_ckpt)[:3]}); "
            f"extra_in_ckpt={len(extra_in_ckpt)} "
            f"(sample: {sorted(extra_in_ckpt)[:3]}). "
            f"Did the converter fail partially, or is wbits mismatched?"
        )
    missing_wbits = {k for k in expected_keys if k.endswith(".wbits_buf")} - inject_keys
    if missing_wbits:
        print(
            f"[Block-AP inject] WARN: checkpoint missing {len(missing_wbits)} "
            "weight_quantizer.wbits_buf keys; bit-width mismatch cannot be "
            "verified from checkpoint. Make sure QUANT_TYPE matches the "
            f"checkpoint. sample={sorted(missing_wbits)[:3]}"
        )

    # assign=True is REQUIRED: under init_empty_weights, the model parameters
    # are on meta device. load_state_dict(... assign=False) would silently
    # NO-OP for meta target parameters (PyTorch >= 2.1 emits a warning to that
    # effect: "copying from a non-meta parameter in the checkpoint to a meta
    # parameter in the current model, which is a no-op"). With assign=True,
    # the loaded tensors REPLACE the meta Parameters with actual real ones,
    # materializing the model with our injected Block-AP scale/zero values.
    #
    # Verify a sample target IS materialized after the load — fail fast if not
    # (catches PyTorch version regressions or workflow changes that re-meta
    # the params).
    missing, unexpected = model.load_state_dict(inject_sd, strict=False, assign=True)
    n_loaded = len(inject_sd) - len(unexpected)
    if unexpected:
        raise RuntimeError(
            f"[Block-AP inject] load_state_dict returned {len(unexpected)} "
            f"unexpected keys after key-set check passed (likely a shape "
            f"mismatch); sample: {unexpected[:3]}"
        )
    # Sanity: verify a sample key is actually a real tensor now (not meta).
    sample_key = next(iter(inject_sd))
    sample_param = model
    for part in sample_key.split('.'):
        sample_param = getattr(sample_param, part)
    if sample_param.is_meta:
        raise RuntimeError(
            f"[Block-AP inject] After load_state_dict(assign=True), the "
            f"target parameter '{sample_key}' is still on meta device. "
            f"This means injection silently failed. Check PyTorch version "
            f"(>=2.1 required for assign=True) and that init_empty_weights "
            f"context is properly exited before injection."
        )
    if os.environ.get("EQAT_BLOCKAP_MATCH_PARAM_DTYPE", "").lower() in {"1", "true", "yes"}:
        n_cast = 0
        for _, qmod in iter_eqat_linears(model):
            target_dtype = qmod.weight.dtype
            q = qmod.weight_quantizer
            if q.scale.dtype != target_dtype:
                q.scale.data = q.scale.data.to(target_dtype)
                n_cast += 1
            if q.zero_point.dtype != target_dtype:
                q.zero_point.data = q.zero_point.data.to(target_dtype)
                n_cast += 1
        print(
            f"[Block-AP inject] cast {n_cast} scale/zero tensors to matching "
            f"QLinear weight dtype because EQAT_BLOCKAP_MATCH_PARAM_DTYPE is enabled"
        )
    print(
        f"[Block-AP inject] {n_loaded}/{len(inject_sd)} weight_quantizer.* keys "
        f"loaded from {local_path} (assign=True, materialized; scale/zero coverage verified)"
    )
    return n_loaded


__all__ = [
    "freeze_non_quant_params",
    "build_param_groups",
    "materialize_dense_state_dict",
    "materialize_fixed_integer_codes",
    "mark_fixed_integer_codes_ready",
    "dequantize_efficientqat_weight",
    "iter_dense_state_dict",
    "iter_eqat_linears",
    "inject_blockap_quant_state",
]
