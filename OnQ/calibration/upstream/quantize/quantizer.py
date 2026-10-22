# Source lineage: yasu0001/ReasoningQAT
# Upstream revision: c36060fefb05aa8c29dcad47b52feacf95201d37
# Original file: quantize/quantizer.py; identical shadowed helper removed; see UPSTREAM_PROVENANCE.md.
import torch
import torch.nn as nn

import pdb
import math

CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def gradscale(x, scale_factor):
    """
    疑似コードに基づくgradscale実装
    Forward: xをそのまま返す
    Backward: 勾配をscale_factorで乗算
    """
    y_out = x
    y_grad = x * scale_factor
    y = (y_out - y_grad).detach() + y_grad
    return y

class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        scale=1.0,
        grad_scale=False,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True

        if grad_scale:
            self.grad_scale = 1 / (math.sqrt(self.group_size) * (2**(n_bits -1) - 1))
        else:
            self.grad_scale = 1

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True) * scale
                xmax =  x.amax([-1], keepdim=True) * scale
                range = xmax - xmin
                # Add small epsilon to prevent division by zero
                range = range + 1e-8
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
            
                # x = weight.reshape(-1,self.group_size)
                # xmax = x.abs().mean([-1], keepdim=True)
                # xmin = torch.zeros_like(xmax)
                # range = xmax * 6
                # scale = range / (2**self.n_bits-1)
                # scale = scale.clamp(min=1e-4, max=1e4)
                # zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                # self.scale = nn.Parameter(scale)
                # self.zero_point = nn.Parameter(zero_point.round())


    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):
        # print(f"[DEBUG] Input x: min={x.min().item():.6f}, max={x.max().item():.6f}")

        scale = gradscale(self.scale, self.grad_scale)
        # print(f"[DEBUG] After gradscale: scale min={scale.min().item():.6f}, max={scale.max().item():.6f}")

        # ensure to scale become larger than zero
        scale = scale.abs()
        scale = clamp_ste(scale,1e-4, 1e4)
        # print(f"[DEBUG] After clamp_ste: scale min={scale.min().item():.6f}, max={scale.max().item():.6f}")

        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        # print(f"[DEBUG] round_zero_point: min={round_zero_point.min().item():.6f}, max={round_zero_point.max().item():.6f}")

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        # print(f"[DEBUG] After reshape: x min={x.min().item():.6f}, max={x.max().item():.6f}")

        x_int = round_ste(x / (scale + 1e-6))
        # print(f"[DEBUG] After x/scale: x_int min={x_int.min().item():.6f}, max={x_int.max().item():.6f}")

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
            # print(f"[DEBUG] After add zero_point: x_int min={x_int.min().item():.6f}, max={x_int.max().item():.6f}")

        x_int = x_int.clamp(self.qmin, self.qmax)
        # print(f"[DEBUG] After clamp: x_int min={x_int.min().item():.6f}, max={x_int.max().item():.6f}")

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
            # print(f"[DEBUG] After sub zero_point: x_dequant min={x_dequant.min().item():.6f}, max={x_dequant.max().item():.6f}")

        x_dequant = x_dequant.mul(scale)
        # print(f"[DEBUG] Final result: min={x_dequant.min().item():.6f}, max={x_dequant.max().item():.6f}")

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            if self.n_bits >= 16 or not self.enable:
                return x

            x_dequant = self.fake_quant(x.float())
            # print(f"Debug info: x min, max: {x.min().item()}, {x.max().item()}, x_dequant min, max: {x_dequant.min().item()}, {x_dequant.max().item()}, dtype: {x_dequant.dtype}, {x.dtype}")
        return x_dequant.to(dtype)

class UniformAffineQuantizerOnline(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        scale=1.0,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        self.initial_scale = scale
        
        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True) * self.initial_scale
                xmax =  x.amax([-1], keepdim=True) * self.initial_scale
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                # self.scale = nn.Parameter(scale)
                self.register_buffer('scale', scale)
                # self.zero_point = nn.Parameter(zero_point.round())
                self.register_buffer('zero_point', zero_point.round())

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
    
    def update_scale_and_zero_point(self, x):
        with torch.no_grad():
            x = x.reshape(-1,self.group_size)
            xmin = x.amin([-1], keepdim=True) * self.initial_scale
            xmax =  x.amax([-1], keepdim=True) * self.initial_scale
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            scale = scale.clamp(min=1e-4, max=1e4)
            zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
            # self.scale = nn.Parameter(scale)
            self.scale = scale
            # self.zero_point = nn.Parameter(zero_point.round())
            self.zero_point = zero_point.round()

    def fake_quant(self, x):
        # update scale and zero_point
        self.update_scale_and_zero_point(x)
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant


class SLTQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        scale=1.0,
        grad_scale=False,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        random_sign = torch.randn_like(weight).sign()
        self.register_buffer('random_sign', random_sign)

        if grad_scale:
            self.grad_scale = 1 / (math.sqrt(self.group_size) * (2**(n_bits -1) - 1))
        else:
            self.grad_scale = 1

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = (weight * self.random_sign).reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True) * scale
                xmax =  x.amax([-1], keepdim=True) * scale
                range = xmax - xmin
                # Add small epsilon to prevent division by zero
                range = range + 1e-8
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                # zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.register_buffer('zero_point', torch.ones_like(scale))

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):
        scale = gradscale(self.scale, self.grad_scale)
        scale = scale.abs()
        scale = clamp_ste(scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = (x * self.random_sign).reshape(-1, self.group_size)
        x_int = round_ste(x / (scale + 1e-6))
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant * self.random_sign
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant

class SLTQuantizerZP(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        scale=1.0,
        grad_scale=False,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        random_sign = torch.randn_like(weight).sign()
        self.register_buffer('random_sign', random_sign)

        if grad_scale:
            self.grad_scale = 1 / (math.sqrt(self.group_size) * (2**(n_bits -1) - 1))
        else:
            self.grad_scale = 1

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1,self.group_size)
                xmin = x.amin([-1], keepdim=True) * scale
                xmax =  x.amax([-1], keepdim=True) * scale
                range = xmax - xmin
                # Add small epsilon to prevent division by zero
                range = range + 1e-8
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)

    def fake_quant(self, x):
        scale = gradscale(self.scale, self.grad_scale)
        scale = clamp_ste(scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        dim1, dim2 = x.shape
        x = (x * self.random_sign).reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant * self.random_sign
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant

class OddQuantizer(nn.Module):
    """
    Odd-bit quantizer that uses odd number of bits for quantization.
    This can be useful for certain hardware implementations that prefer odd bit-widths.
    """
    def __init__(
        self,
        n_bits: int = 3,
        group_size=None,
        weight=None,
    ):
        super().__init__()
        # Ensure n_bits is odd
        self.n_bits = n_bits
        self.qmin = - 2 ** (n_bits - 1) # if 2-bits -> -2
        self.qmax = 2 ** (n_bits - 1) - 1 # if 2-bits -> 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True

        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                x = weight.reshape(-1, self.group_size)
                xmax = x.abs().amax([-1], keepdim=True)
                # because of odd quantization, {-3, -1, 1, 3}
                scale = xmax * 0.95 / (2**self.n_bits - 1)
                scale = scale.clamp(min=1e-4, max=1e4)
                # zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
                self.scale = nn.Parameter(scale)
                # odd quantization does not optimize zero point

    # def change_n_bits(self, n_bits):
    #     # Ensure n_bits is odd
    #     if n_bits % 2 == 0:
    #         n_bits += 1
    #     self.n_bits = n_bits
    #     self.qmin = 0
    #     self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):
        scale = clamp_ste(self.scale, 1e-4, 1e4)

        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / (scale * 2) - 0.5)
        x_int = x_int.clamp(self.qmin, self.qmax) + 0.5
        x_dequant = x_int
        x_dequant = x_dequant.mul(scale * 2)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        x_dequant = self.fake_quant(x)
        return x_dequant


# class AsymSLTQuantizer(nn.Module):
#     """
#     Asymmetric Straight-Through Estimator Quantizer with learnable scale and zero point.
#     Uses asymmetric quantization with improved gradient flow through STE.
#     """
#     def __init__(
#         self,
#         n_bits: int = 4,
#         group_size=None,
#         weight=None,
#     ):
#         super().__init__()
#         assert 2 <= n_bits <= 16, "bitwidth not supported"
#         self.n_bits = n_bits
#         self.qmin = 0
#         self.qmax = 2 ** (n_bits) - 1
#         self.group_size = group_size if group_size != -1 else weight.shape[-1]
#         assert weight.shape[-1] % group_size == 0
#         self.enable = True
        
#         # init scale and zero point through Max-Min quantization
#         with torch.no_grad():
#             if weight is not None:
#                 x = weight.reshape(-1, self.group_size)
#                 xmin = x.amin([-1], keepdim=True)
#                 xmax = x.amax([-1], keepdim=True)
#                 range = xmax - xmin
#                 scale = range / (2**self.n_bits - 1)
#                 scale = scale.clamp(min=1e-4, max=1e4)
#                 zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
#                 self.scale = nn.Parameter(scale)
#                 self.zero_point = nn.Parameter(zero_point)

#     def change_n_bits(self, n_bits):
#         self.n_bits = n_bits
#         self.qmin = 0
#         self.qmax = int(2 ** (n_bits) - 1)
        
#     def fake_quant(self, x):
#         scale = clamp_ste(self.scale, 1e-4, 1e4)
#         round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

#         dim1, dim2 = x.shape
#         x = x.reshape(-1, self.group_size)
        
#         # Asymmetric quantization with improved STE
#         x_scaled = x / scale
#         x_int = round_ste(x_scaled)
#         if round_zero_point is not None:
#             x_int = x_int.add(round_zero_point)
#         x_int = x_int.clamp(self.qmin, self.qmax)
        
#         # Dequantization
#         x_dequant = x_int
#         if round_zero_point is not None:
#             x_dequant = x_dequant.sub(round_zero_point)
#         x_dequant = x_dequant.mul(scale)
        
#         if self.group_size:
#             x_dequant = x_dequant.reshape(dim1, dim2)
#         return x_dequant

#     def forward(self, x: torch.Tensor):
#         if self.n_bits >= 16 or not self.enable:
#             return x
#         x_dequant = self.fake_quant(x)
#         return x_dequant


__all__ = ["UniformAffineQuantizer", "OddQuantizer", "SLTQuantizer"]