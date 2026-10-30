"""Simulated (fake) quantization for mixed-precision KV retention.

Values are quantized and immediately dequantized, so accuracy reflects the
quantization error while tensors keep their original dtype. Memory savings are
reported analytically via ``quantized_bytes``. Keys use per-channel groups and
values per-token groups, following KIVI.
"""

import torch


def fake_quant(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Asymmetric min-max quantization along ``dim``."""
    if bits >= 16:
        return x
    xf = x.float()
    lo = xf.amin(dim=dim, keepdim=True)
    hi = xf.amax(dim=dim, keepdim=True)
    scale = (hi - lo).clamp(min=1e-6) / (2**bits - 1)
    q = ((xf - lo) / scale).round().clamp(0, 2**bits - 1)
    return (q * scale + lo).to(x.dtype)


def quant_keys(k: torch.Tensor, bits: int) -> torch.Tensor:
    # k: (B, H, T, D); per-channel => statistics over the token axis
    return fake_quant(k, bits, dim=-2)


def quant_values(v: torch.Tensor, bits: int) -> torch.Tensor:
    # per-token => statistics over the channel axis
    return fake_quant(v, bits, dim=-1)


def quantized_bytes(numel: int, bits: int, group: int = 128, elem_bytes: int = 2) -> float:
    """Storage for ``numel`` elements at ``bits`` with an fp16 scale+zero per group."""
    if bits >= 16:
        return numel * elem_bytes
    return numel * bits / 8 + (numel / group) * 2 * elem_bytes
