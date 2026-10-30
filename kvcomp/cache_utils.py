"""Version-tolerant accessors for HuggingFace DynamicCache.

transformers >= 4.54 stores per-layer tensors in ``cache.layers[i].keys/.values``;
older versions use ``cache.key_cache[i]`` / ``cache.value_cache[i]``.
Tensors have shape (batch, num_kv_heads, seq_len, head_dim) and keys already
carry RoPE, so dropping entries never requires re-rotating anything.
"""

import torch


def num_layers(cache) -> int:
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)


def get_kv(cache, i: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[i]
        return layer.keys, layer.values
    return cache.key_cache[i], cache.value_cache[i]


def set_kv(cache, i: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[i].keys = keys
        cache.layers[i].values = values
    else:
        cache.key_cache[i] = keys
        cache.value_cache[i] = values


def cache_len(cache) -> int:
    if num_layers(cache) == 0:  # layers are created lazily on the first update
        return 0
    k = get_kv(cache, 0)[0]
    return 0 if k is None else k.shape[-2]


def cache_bytes(cache) -> int:
    total = 0
    for i in range(num_layers(cache)):
        k, v = get_kv(cache, i)
        if k is None:
            continue
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total
