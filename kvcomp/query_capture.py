"""Capture post-RoPE query states of the most recent tokens in every attention layer.

Compression methods score cached keys with the attention that recent queries pay
to them (SnapKV / R-KV style). Fused attention kernels never materialise the
attention matrix, so we recompute the queries from the attention input with a
forward pre-hook and keep a small ring buffer of the last ``window`` queries.
The extra cost is one q_proj per layer per decoded token.
"""

from functools import partial

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (B, T, D)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def project(module, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, which: str):
    """Recompute q or k (post-norm, post-RoPE) of an HF Llama/Qwen2/Qwen3 attention module."""
    bsz, seqlen, _ = hidden.shape
    proj = module.q_proj if which == "q" else module.k_proj
    x = proj(hidden).view(bsz, seqlen, -1, module.head_dim)
    norm = getattr(module, f"{which}_norm", None)  # Qwen3 has q_norm / k_norm
    if norm is not None:
        x = norm(x)
    return apply_rope(x.transpose(1, 2), cos, sin)


def _attention_inputs(args, kwargs):
    hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    cos, sin = kwargs["position_embeddings"] if "position_embeddings" in kwargs else args[1]
    return hidden, cos, sin


def attention_modules(model):
    return [layer.self_attn for layer in model.model.layers]


class QueryCapture:
    def __init__(self, model, window: int):
        self.window = window
        self.buf: dict[int, torch.Tensor] = {}
        self.handles = [
            attn.register_forward_pre_hook(partial(self._hook, i), with_kwargs=True)
            for i, attn in enumerate(attention_modules(model))
        ]

    @torch.no_grad()
    def _hook(self, layer_idx, module, args, kwargs):
        hidden, cos, sin = _attention_inputs(args, kwargs)
        w = self.window
        q = project(module, hidden[:, -w:], cos[:, -w:], sin[:, -w:], "q")
        prev = self.buf.get(layer_idx)
        if prev is not None:
            q = torch.cat([prev, q], dim=2)
        self.buf[layer_idx] = q[:, :, -w:]

    def reset(self):
        self.buf.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def window_attention(q: torch.Tensor, k: torch.Tensor, scaling: float) -> torch.Tensor:
    """Attention the window queries pay to every cached key, summed per KV head.

    q: (B, Hq, w, D) for the last w tokens, which are also the last w cache entries.
    k: (B, Hkv, T, D).
    Returns (B, Hkv, T) in float32: softmax probabilities summed over the window and
    over the query heads of each GQA group.
    """
    bsz, hq, w, d = q.shape
    hkv, t = k.shape[1], k.shape[2]
    n_rep = hq // hkv
    qg = q.float().reshape(bsz, hkv, n_rep * w, d)
    logits = torch.matmul(qg, k.float().transpose(-1, -2)) * scaling  # (B, Hkv, n_rep*w, T)
    # Causal mask inside the window: query j may not see the keys of later window tokens.
    qpos = torch.arange(w, device=k.device).repeat(n_rep)
    kpos = torch.arange(t, device=k.device) - (t - w)
    logits = logits.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
    return logits.softmax(-1).sum(2)


def joint_attention(q: torch.Tensor, k: torch.Tensor, k_ev: torch.Tensor, scaling: float):
    """Attention the window queries would pay to resident keys ``k`` and evicted keys ``k_ev``
    together (one softmax), averaged over the query heads of each KV group and the window
    positions: returns (B, Hkv, T) and (B, Hkv, E), summing to 1 per (B, Hkv)."""
    bsz, hq, w, d = q.shape
    hkv, t = k.shape[1], k.shape[2]
    qg = q.float().reshape(bsz, hkv, (hq // hkv) * w, d)
    lk = torch.matmul(qg, k.float().transpose(-1, -2)) * scaling
    qpos = torch.arange(w, device=k.device).repeat(hq // hkv)
    kpos = torch.arange(t, device=k.device) - (t - w)
    lk = lk.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
    le = torch.matmul(qg, k_ev.float().transpose(-1, -2)) * scaling
    p = torch.cat([lk, le], -1).softmax(-1).mean(2)
    return p[..., :t], p[..., t:]


def evicted_attention_share(q: torch.Tensor, k: torch.Tensor, k_ev: torch.Tensor, scaling: float) -> float:
    """Share of softmax mass the window queries would put on evicted keys ``k_ev`` (B, Hkv, E, D)
    if they were attended together with the resident keys ``k`` (same layout as ``window_attention``).
    Mean over batch, KV heads, query heads of each group and window positions."""
    bsz, hq, w, d = q.shape
    hkv, t = k.shape[1], k.shape[2]
    qg = q.float().reshape(bsz, hkv, (hq // hkv) * w, d)
    lk = torch.matmul(qg, k.float().transpose(-1, -2)) * scaling
    qpos = torch.arange(w, device=k.device).repeat(hq // hkv)
    kpos = torch.arange(t, device=k.device) - (t - w)
    lk = lk.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
    le = torch.matmul(qg, k_ev.float().transpose(-1, -2)) * scaling  # evicted entries are all older
    return float(torch.sigmoid(le.logsumexp(-1) - lk.logsumexp(-1)).mean())
