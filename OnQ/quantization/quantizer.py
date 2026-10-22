# Source lineage: EfficientQAT/BitDistiller-derived quantization and local QAT integration.
# Exact upstream import revision is not recorded; see OnQ/UPSTREAMS.md.
import torch
import torch.nn as nn
from torch.autograd import Function

class Round(Function):
    @staticmethod
    def forward(ctx, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


DEFAULT_PSEUDO_SCALE_FLOOR = 1e-5
DEFAULT_STE_SCALE_FLOOR = 1e-5

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False,
                           scale_floor=DEFAULT_PSEUDO_SCALE_FLOOR
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    elif q_group_size == -1:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    assert zero_point, "pseudo_quantize_tensor only supports zero_point=True (asymmetric)."
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=scale_floor) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w






class SteInt4AsymQuantizer(nn.Module):
    def __init__(self, q_group_size=128, scale_floor=DEFAULT_STE_SCALE_FLOOR):
        super().__init__()
        self.q_group_size = q_group_size
        self.bit = 4
        self.scale_floor = scale_floor
    def forward(self, x):
        org_w_shape = x.shape

        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            x = x.reshape(-1, self.q_group_size)
        assert x.dim() == 2

        # Handle NaN/Inf: warn and clamp to finite range (don't use arbitrary constants)
        has_nan = torch.isnan(x).any()
        has_inf = torch.isinf(x).any()
        if has_nan or has_inf:
            import warnings
            warnings.warn(f"SteInt4AsymQuantizer: input has NaN={has_nan.item()}, Inf={has_inf.item()}, clamping")
            finite_vals = x[torch.isfinite(x)]
            fill_val = 0.0 if finite_vals.numel() == 0 else finite_vals.mean().item()
            x = torch.where(torch.isfinite(x), x, torch.full_like(x, fill_val))

        max_val = x.amax(dim=1, keepdim=True).detach()
        min_val = x.amin(dim=1, keepdim=True).detach()
        max_int = 2 ** self.bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=self.scale_floor) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        x = (torch.clamp(Round.apply(x / scales) +
                         zeros, min_int, max_int) - zeros) * scales

        x = x.reshape(org_w_shape)

        return x

class SteInt2AsymQuantizer(nn.Module):
    def __init__(self, q_group_size=64, scale_floor=DEFAULT_STE_SCALE_FLOOR):
        super().__init__()
        self.q_group_size = q_group_size
        self.bit = 2
        self.scale_floor = scale_floor
    def forward(self, x):
        org_w_shape = x.shape

        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            x = x.reshape(-1, self.q_group_size)
        assert x.dim() == 2

        max_val = x.amax(dim=1, keepdim=True).detach()
        min_val = x.amin(dim=1, keepdim=True).detach()
        max_int = 2 ** self.bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=self.scale_floor) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(x).sum() == 0

        x = (torch.clamp(Round.apply(x / scales) +
                         zeros, min_int, max_int) - zeros) * scales
        assert torch.isnan(x).sum() == 0

        x = x.reshape(org_w_shape)

        return x

class SteN2F3Quantizer(nn.Module):
    def __init__(self, q_group_size=128):
        super().__init__()
        self.q_group_size = q_group_size
    
    def forward(self, x):
        org_w_shape = x.shape

        # reshape to groupsize
        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            qx = x.reshape(-1, self.q_group_size)
        elif self.q_group_size == -1:
            qx = x.reshape(-1, x.shape[-1])
        assert qx.dim() == 2

        # Get the Min Max (detach to avoid sparse gradients through amax/amin)
        max_val = qx.amax(dim=1, keepdim=True).detach()
        min_val = qx.amin(dim=1, keepdim=True).detach()

        
        scale_pos = torch.abs(max_val)
        scale_neg = torch.abs(min_val)

        dev = qx.device
        x_pos = torch.zeros_like(qx)
        x_neg = torch.zeros_like(qx)
        x_pos = torch.where(qx >= 0, qx, x_pos)
        x_neg = torch.where(qx < 0, qx, x_neg)
        q_pos = x_pos / scale_pos
        q_neg = x_neg / scale_neg

        q_pos, q_neg = self.round_pass(q_pos, q_neg, dev)

        qx = q_pos * scale_pos + q_neg * scale_neg

        qx = qx.reshape(org_w_shape)

        return qx
    
    def round_n2f3(self, q_pos, q_neg, dev):
        q_pos = torch.where(q_pos >= 0.8114928305149078,                                        torch.tensor(1.0).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.8114928305149078)    & (q_pos >= 0.5024898052215576),    torch.tensor(0.6229856610298157).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.5024898052215576)    & (q_pos >= 0.2826657369732857),    torch.tensor(0.3819939494132996).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.2826657369732857)    & (q_pos >= 0.0916687622666359),    torch.tensor(0.1833375245332718).to(dev), q_pos)
        q_pos = torch.where(q_pos < 0.0916687622666359,                                        torch.tensor(0).to(dev), q_pos)

        q_neg = torch.where(q_neg >= -0.1234657019376755,                                     torch.tensor(0).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.1234657019376755)   & (q_neg >= -0.39097706973552704),   torch.tensor(-0.2469314038753510).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.39097706973552704)   & (q_neg >= -0.7675113677978516),   torch.tensor(-0.5350227355957031).to(dev), q_neg)
        q_neg = torch.where(q_neg < -0.7675113677978516,                                        torch.tensor(-1.0).to(dev), q_neg)

        return q_pos, q_neg

    def round_pass(self, q_pos, q_neg, dev):
        y_grad_pos, y_grad_neg = q_pos, q_neg
        y_pos, y_neg = self.round_n2f3(q_pos, q_neg, dev)

        return (y_pos - y_grad_pos).detach() + y_grad_pos, (y_neg - y_grad_neg).detach() + y_grad_neg


# ---------------------------------------------------------------------------
# EfficientQAT-style trainable quantizer
# ---------------------------------------------------------------------------
# Adapted from EfficientQAT UniformAffineQuantizer; see UPSTREAMS.md.
# The OPD-BitDistiller lineage of quantizers above keeps `scale` / `zero_point`
# as detached on-the-fly tensors recomputed from weight min/max every forward.
# This one stores them as `nn.Parameter`s, initialized once from per-group
# min/max. OnQ restores their calibrated values, trains scale with AdamW,
# and keeps zero_point frozen.


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Functional form of the existing Round.apply — same STE semantics."""
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, lo, hi) -> torch.Tensor:
    return (x.clamp(lo, hi) - x).detach() + x


DEFAULT_EQAT_SCALE_FLOOR = 1e-4
DEFAULT_EQAT_SCALE_CEIL = 1e4


def efficientqat_weight_forward(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    wbits: int,
    group_size: int,
    compute_dtype: torch.dtype,
    scale_floor: float,
    scale_ceil: float,
    fixed_integer_codes: bool = False,
) -> torch.Tensor:
    """Shared differentiable forward for training and no-grad materialization.

    Preserve the input storage dtype on return; callers select the final vLLM
    or export dtype afterwards. STE order is part of the numerical contract.
    """
    values = weight.reshape(-1, group_size).to(compute_dtype)
    scales = clamp_ste(scale.reshape(-1, 1).to(compute_dtype).abs(), scale_floor, scale_ceil)
    qmin, qmax = 0, 2 ** wbits - 1
    if fixed_integer_codes:
        zeros = zero_point.detach().reshape(-1, 1).to(compute_dtype).round().clamp(qmin, qmax)
        dense = (values - zeros) * scales
    else:
        zeros = clamp_ste(round_ste(zero_point.reshape(-1, 1).to(compute_dtype)), qmin, qmax)
        codes = (round_ste(values / scales) + zeros).clamp(qmin, qmax)
        # Zero point cancels on the unsaturated path; its gradient comes from
        # clipping, exactly as in the previous EfficientQAT forward.
        dense = (codes - zeros) * scales
    return dense.reshape(weight.shape).to(weight.dtype)


class EfficientQATWeightQuantizer(nn.Module):
    """Trainable per-group W2/W3/W4 asymmetric weight quantizer.

    scale and zero_point are nn.Parameters. scale is initialized from per-group
    (max - min) / (2**wbits - 1), zero_point from -round(min / scale). Both are
    handled separately: scale uses abs() and an STE clamp; zero_point uses
    STE rounding and clamping without abs(). The forward pass stays on the
    configured W2/W3/W4 grid, with the surrogate gradients defined above.
    The raw scale Parameter is intentionally
    unconstrained to match ReasoningQAT and may be negative or zero.

    Caller must provide `weight` at construction so scale/zero can be seeded
    from the actual master weight distribution; this is why we cannot build
    the quantizer before the weight has been assigned to the QLinear wrapper.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        wbits: int = 4,
        q_group_size: int = 128,
        scale_floor: float = DEFAULT_EQAT_SCALE_FLOOR,
        scale_ceil: float = DEFAULT_EQAT_SCALE_CEIL,
        compute_dtype: torch.dtype = torch.float32,
        fixed_integer_codes: bool = False,
    ):
        super().__init__()
        assert 2 <= wbits <= 16, f"unsupported bitwidth {wbits}"
        assert weight is not None, "EfficientQATWeightQuantizer requires a reference weight"
        # Register wbits as a buffer so it survives state_dict save/load
        # under FSDP. Plain Python attrs would NOT — a W2 ckpt loaded by a
        # launcher that builds the module with wbits=4 would silently keep
        # qmax=15 on the actor, blowing the quant grid.
        self.register_buffer("wbits_buf", torch.tensor(wbits, dtype=torch.int32), persistent=True)
        self.register_buffer(
            "fixed_codes_buf",
            torch.tensor(int(fixed_integer_codes), dtype=torch.int8),
            persistent=True,
        )
        self.wbits = wbits
        self.fixed_integer_codes = bool(fixed_integer_codes)
        self._fixed_codes_loaded_from_state = False
        self._fixed_codes_ready = False
        self.qmin = 0
        self.qmax = 2 ** wbits - 1
        self.group_size = q_group_size if q_group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % self.group_size == 0, (
            f"weight shape {tuple(weight.shape)} not divisible by group_size={self.group_size}"
        )
        self.scale_floor = scale_floor
        self.scale_ceil = scale_ceil
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "EfficientQAT quantizer compute_dtype must be fp32 or bf16, "
                f"got {compute_dtype}"
            )
        self.compute_dtype = compute_dtype

        with torch.no_grad():
            x = weight.detach().reshape(-1, self.group_size).to(torch.float32)
            xmin = x.amin(dim=-1, keepdim=True)
            xmax = x.amax(dim=-1, keepdim=True)
            rng = (xmax - xmin).clamp(min=scale_floor)
            scale = (rng / self.qmax).clamp(min=scale_floor, max=scale_ceil)
            # Init zero_point in the same [qmin, qmax] integer band the
            # quantizer actually uses at forward. The previous wide clamp
            # [-scale_ceil, +scale_ceil] (=[-1e4, +1e4]) was a no-op for
            # the W4 grid (zp ~ 7-8 naturally) but at W2 (qmax=3) the
            # init can land far above qmax; then clamp_ste in fake_quant
            # freezes its grad via plain clamp and the optimizer is stuck.
            zero_point = (-(xmin / scale)).clamp(min=self.qmin, max=self.qmax).round()

        # Keep scale and zero_point in fp32 regardless of master weight dtype.
        # Reason: under FSDP MP `param_dtype=bf16`, casting these tiny tensors
        # to bf16 loses ~1% relative precision at scale~1e-4, which perturbs
        # the int-rounding decision in fake_quant. Matching EfficientQAT
        # reference (UniformAffineQuantizer keeps scale fp32 throughout).
        # Combined with FSDP `mixed_precision.param_dtype=fp32` for the
        # actor (set in launcher), this preserves quant grid precision.
        self.scale = nn.Parameter(scale.to(torch.float32))
        self.zero_point = nn.Parameter(zero_point.to(torch.float32))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Fail fast if ckpt wbits disagrees with module construction —
        # prevents silently loading a W2 ckpt into a wbits=4 module
        # (which would keep qmax=15 on forward and corrupt the quant grid).
        saved = state_dict.get(prefix + "wbits_buf")
        if saved is not None:
            saved_wbits = int(saved.item())
            if saved_wbits != self.wbits:
                raise RuntimeError(
                    f"EfficientQATWeightQuantizer wbits mismatch: ckpt has "
                    f"wbits_buf={saved_wbits} but module was constructed "
                    f"with wbits={self.wbits}. Set QUANT_TYPE to match "
                    f"the saved ckpt (e.g. efficientqat-int{saved_wbits})."
                )
        saved_fixed = state_dict.get(prefix + "fixed_codes_buf")
        if saved_fixed is not None:
            saved_fixed = bool(int(saved_fixed.item()))
            if saved_fixed and not self.fixed_integer_codes:
                raise RuntimeError(
                    "EfficientQAT fixed-code checkpoint loaded with "
                    "fixed_integer_codes=False. Enable the fixed-code model "
                    "path or load a dense/fake-quant checkpoint instead."
                )
            self._fixed_codes_loaded_from_state = saved_fixed
            self._fixed_codes_ready = saved_fixed
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def materialize_fixed_integer_codes_(self, weight: torch.Tensor) -> bool:
        """Replace one dense master weight with its fixed asymmetric codes.

        The integer codes are represented in the existing floating-point
        Parameter storage, so this adds no full-size buffer. The Parameter is
        frozen by ``train_mode=scale_only``; only ``scale`` remains trainable.
        """
        if not self.fixed_integer_codes:
            raise RuntimeError("fixed integer codes were not enabled at construction")
        if self._fixed_codes_loaded_from_state:
            self._fixed_codes_ready = True
            return False

        org_shape = weight.shape
        weight_fp32 = weight.detach().reshape(-1, self.group_size).to(torch.float32)
        scale = self.scale.detach().to(torch.float32).abs().clamp(
            self.scale_floor, self.scale_ceil
        )
        zero = self.zero_point.detach().to(torch.float32).round().clamp(
            self.qmin, self.qmax
        )
        codes = (weight_fp32 / scale).round().add(zero).clamp(
            self.qmin, self.qmax
        )
        weight.copy_(codes.reshape(org_shape).to(weight.dtype))
        self.fixed_codes_buf.fill_(1)
        self._fixed_codes_loaded_from_state = True
        self._fixed_codes_ready = True
        return True

    def mark_fixed_integer_codes_ready_(self) -> None:
        if self.fixed_integer_codes:
            self._fixed_codes_ready = True

    def fixed_code_dequant(self, codes: torch.Tensor) -> torch.Tensor:
        if not self._fixed_codes_ready:
            raise RuntimeError(
                "fixed_integer_codes=True but integer codes were not materialized "
                "before the first forward"
            )
        return efficientqat_weight_forward(
            codes, self.scale, self.zero_point,
            wbits=self.wbits, group_size=self.group_size,
            compute_dtype=self.compute_dtype, scale_floor=self.scale_floor,
            scale_ceil=self.scale_ceil, fixed_integer_codes=True,
        )

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        # FP32 remains the stable default. BF16 is an explicit experiment mode
        # that keeps the entire fake-quant forward in the QLinear compute dtype.
        return efficientqat_weight_forward(
            x, self.scale, self.zero_point,
            wbits=self.wbits, group_size=self.group_size,
            compute_dtype=self.compute_dtype, scale_floor=self.scale_floor,
            scale_ceil=self.scale_ceil,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_integer_codes:
            return self.fixed_code_dequant(x)
        return self.fake_quant(x)
