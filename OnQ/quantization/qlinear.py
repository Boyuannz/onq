# Source lineage: EfficientQAT/BitDistiller-derived quantization and local QAT integration.
# Exact upstream import revision is not recorded; see OnQ/UPSTREAMS.md.
import torch
import torch.nn.functional as F
from torch import Tensor, device, dtype, nn
from OnQ.quantization.quantizer import *


EQAT_WBITS = {
    "efficientqat-int4": 4,
    "efficientqat-int3": 3,
    "efficientqat-int2": 2,
}


def resolve_quantizer_scale_floor(quant_type, scale_floor):
    """Return the scale floor actually used by the selected quantizer."""
    if quant_type in EQAT_WBITS and scale_floor == DEFAULT_STE_SCALE_FLOOR:
        return DEFAULT_EQAT_SCALE_FLOOR
    return scale_floor

def convertModelToQuant(model,
                        modules_to_not_convert=["lm_head"],
                        current_key_name=None,
                        has_been_replaced=False,
                        compute_dtype=torch.bfloat16,
                        quant_math_dtype=torch.float32,
                        quant_type="clsq-n2f3",
                        q_group_size=128,
                        scale_floor=DEFAULT_STE_SCALE_FLOOR,
                        fixed_integer_codes=False):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            # EQAT-style quantizers parametrized by bit-width: 4 / 3 / 2.
            # Underlying EfficientQATWeightQuantizer supports wbits in [2, 16].
            if quant_type in EQAT_WBITS:
                # EfficientQAT-style: pass org_module so the quantizer can
                # initialize scale/zero_point from the real weight distribution.
                # Translate the legacy STE scale_floor (default 1e-5) into the
                # EQAT default if the caller did not override it explicitly:
                # EQAT's quantizer historically uses 1e-4 to keep the grid stable.
                eqat_scale_floor = resolve_quantizer_scale_floor(
                    quant_type, scale_floor
                )
                model._modules[name] = EfficientQATQLinear(
                    org_module=module,
                    wbits=EQAT_WBITS[quant_type],
                    q_group_size=q_group_size,
                    scale_floor=eqat_scale_floor,
                    compute_dtype=compute_dtype,
                    quant_math_dtype=quant_math_dtype,
                    fixed_integer_codes=fixed_integer_codes,
                )
            else:
                in_features = module.in_features
                out_features = module.out_features
                weight = module.weight
                bias = module.bias

                model._modules[name] = QLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    q_group_size=q_group_size,
                    scale_floor=scale_floor,
                )

                model._modules[name].weight = weight
                model._modules[name].bias = bias
            has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            model._modules[name].source_cls = type(module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = convertModelToQuant(
                module,
                modules_to_not_convert,
                current_key_name,
                has_been_replaced,
                compute_dtype,
                quant_math_dtype,
                quant_type,
                q_group_size,
                scale_floor,
                fixed_integer_codes,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

class QLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=torch.bfloat16, quant_type="ste-n2f3", q_group_size=128, scale_floor=DEFAULT_STE_SCALE_FLOOR, device=None):
        super().__init__(input_features, output_features, bias, device)

        if quant_type == "ste-n2f3":
            self.weight_quantizer = SteN2F3Quantizer(q_group_size=q_group_size)
        elif quant_type == "int4-asym":
            self.weight_quantizer = SteInt4AsymQuantizer(q_group_size=q_group_size, scale_floor=scale_floor)
        elif quant_type == "int2-asym":
            self.weight_quantizer = SteInt2AsymQuantizer(q_group_size=q_group_size, scale_floor=scale_floor)
        else:
            raise ValueError(
                f"Unsupported quant_type={quant_type!r}. "
                f"Valid: ste-n2f3, int4-asym, int2-asym (legacy QLinear path); "
                f"efficientqat-int4, efficientqat-int3, efficientqat-int2 "
                f"(EfficientQATQLinear path, dispatched in convertModelToQuant)."
            )
        # self.quant_type = quant_type
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        inp_dtype = x.dtype

        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = None

        quantize_weight = self.weight_quantizer(self.weight.to(self.compute_dtype))
        out = F.linear(x, quantize_weight, bias).to(inp_dtype)

        return out


class EfficientQATQLinear(nn.Linear):
    """nn.Linear replacement using EfficientQAT's trainable scale/zero quantizer.

    Unlike the legacy ``QLinear`` which recomputes scale/zero from weight
    min/max each forward (and detaches them from the grad graph), this module
    stores scale and zero_point as ``nn.Parameter``s. OnQ trains the latent
    weight and scale with AdamW while keeping zero_point frozen; scale may use
    a separate learning rate.

    Requires the original ``nn.Linear`` at construction so the quantizer can
    be seeded with per-group min/max of the real weight distribution.
    """

    def __init__(
        self,
        org_module: nn.Linear,
        wbits: int = 4,
        q_group_size: int = 128,
        scale_floor: float = DEFAULT_EQAT_SCALE_FLOOR,
        scale_ceil: float = DEFAULT_EQAT_SCALE_CEIL,
        compute_dtype=torch.bfloat16,
        quant_math_dtype=torch.float32,
        fixed_integer_codes: bool = False,
    ):
        super().__init__(
            org_module.in_features,
            org_module.out_features,
            org_module.bias is not None,
            device=org_module.weight.device,
            dtype=org_module.weight.dtype,
        )
        # Reuse the original weight / bias storage so we do not double-allocate.
        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias

        self.weight_quantizer = EfficientQATWeightQuantizer(
            weight=self.weight,
            wbits=wbits,
            q_group_size=q_group_size,
            scale_floor=scale_floor,
            scale_ceil=scale_ceil,
            compute_dtype=quant_math_dtype,
            fixed_integer_codes=fixed_integer_codes,
        )
        self.compute_dtype = compute_dtype

    def quantized_weight(self):
        """Quantize the master directly, then cast the effective linear weight.

        The quantizer owns arithmetic precision; compute_dtype selects only
        the final linear dtype. Neither operation overwrites the master.
        """
        return self.weight_quantizer(self.weight).to(self.compute_dtype)

    def forward(self, x: torch.Tensor):
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        w = self.quantized_weight()
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        return F.linear(x, w, bias).to(inp_dtype)
