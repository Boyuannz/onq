# Source lineage: yasu0001/ReasoningQAT
# Upstream revision: c36060fefb05aa8c29dcad47b52feacf95201d37
# Original file: quantize/int_linear_fake.py; local latent-checkpoint reload support.
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import math
from logging import getLogger
import importlib

import numpy as np
import torch
import torch.nn as nn
import transformers

import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc  
import os

logger = getLogger(__name__)

# 循環インポートを避けるため、必要な関数を直接定義
def get_named_linears(module, type):
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)



class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
        quantizer_class="UniformAffineQuantizer",
        **quantizer_kwargs
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        # initialize quantizer dynamically
        self.weight_quantizer = self._create_quantizer(quantizer_class, wbits, group_size, org_module.weight, **quantizer_kwargs)
        self.use_temporary_parameter = False
        # self.output_scale = nn.Parameter([2.0], dtype=self.weight.dtype, device=self.weight.device)

    def _create_quantizer(self, quantizer_class, wbits, group_size, weight, **kwargs):
        """動的にquantizerクラスを作成する"""
        if isinstance(quantizer_class, str):
            # 文字列の場合は、quantize.quantizerモジュールからインポート
            try:
                module = importlib.import_module("quantize.quantizer")
                quantizer_cls = getattr(module, quantizer_class)
                return quantizer_cls(wbits, group_size, weight=weight, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to import {quantizer_class}, falling back to UniformAffineQuantizer: {e}")
                return UniformAffineQuantizer(wbits, group_size, weight=weight, **kwargs)
        elif hasattr(quantizer_class, '__call__'):
            # クラスオブジェクトの場合は直接インスタンス化
            return quantizer_class(wbits, group_size, weight=weight, **kwargs)
        else:
            logger.warning(f"Invalid quantizer_class: {quantizer_class}, falling back to UniformAffineQuantizer")
            return UniformAffineQuantizer(wbits, group_size, weight=weight, **kwargs)

    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            if not torch.isfinite(self.weight).all():
                print(f"weight is not finite: {self.weight.max().item()} {self.weight.min().item()} non finite {torch.sum(~torch.isfinite(self.weight))} / {self.weight.numel()}")

            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

def load_quantized_model_init(model_path, wbits, group_size, quantizer_class, use_flash_attn, scale=1.0, grad_scale=False):
    print(f"Initializing quantized model from {model_path}, {wbits}, {group_size}, {quantizer_class}")
    attn_impl = "flash_attention_2" if use_flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation=attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(module, wbits, group_size, quantizer_class, scale=scale, grad_scale=grad_scale)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    return model, tokenizer


def load_quantized_model(model_path, wbits, group_size, replace=False, strict=False, quantizer_class="UniformAffineQuantizer", scale=1.0, grad_scale=False, use_flash_attn=True, device=None, inference_only=False):
    print(f"Loading quantized model from {model_path}")
    if inference_only and not os.path.isdir(model_path):
        raise ValueError("Inference requires an existing local checkpoint")
    if replace and not inference_only:
        from OnQ.quantization.latent_checkpoint import require_fp32_latent_source
        require_fp32_latent_source(model_path, bits=wbits, group_size=group_size)
    if not os.path.exists(model_path):
        model, tokenizer = load_quantized_model_init(model_path, wbits, group_size, quantizer_class, use_flash_attn)
        return model, tokenizer

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    from OnQ.quantization.latent_checkpoint import read_latent_metadata
    latent = read_latent_metadata(model_path, bits=wbits, group_size=group_size)
    declared_latent = latent is not None or any(
        getattr(config, name, {}).get('latent_weights_preserved') is True
        for name in ('qat_latent_checkpoint', 'qat_blockwise'))
    if declared_latent and not replace:
        raise ValueError("Latent checkpoints require quantizer modules (replace=True)")
    if inference_only and declared_latent:
        from OnQ.quantization.latent_checkpoint import require_fp32_latent_source
        require_fp32_latent_source(model_path, bits=wbits, group_size=group_size)
    # FP32 only for continuous latent masters. A materialized export was rounded
    # to FP16 by Block-AP; evaluate it in FP16 as upstream does, never recast it.
    latent_masters = replace and (declared_latent or not inference_only)
    with init_empty_weights():
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float32 if latent_masters else torch.float16, trust_remote_code=True, attn_implementation=attn_impl)
    layers = model.model.layers
    print(f"Loading models with {wbits} {group_size}")
    if replace:
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            named_linears = get_named_linears(layer, torch.nn.Linear)
            for name, module in named_linears.items():
                q_linear = QuantLinear(module, wbits, group_size, quantizer_class, scale=scale, grad_scale=grad_scale)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
    else:
        pass
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    # if device is None:
    if device is not None:
        device_map = {"": "cpu"}
    else:
        device_map = infer_auto_device_map(model)
    # else:
    # device_map = {"": device}
    # device_map = infer_auto_device_map(model)
    print(f"initialize model: {model}")
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=False, strict=strict)
    if replace:
        for module in model.modules():
            if isinstance(module, QuantLinear):
                # Materialized evaluation weights have already been quantized.
                # Only latent masters should pass through fake quantization.
                module.set_quant_state(weight_quant=not inference_only or declared_latent)
    if inference_only:
        model.eval()
        model.requires_grad_(False)
    # load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True, strict=strict)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

__all__ = ["QuantLinear"]
