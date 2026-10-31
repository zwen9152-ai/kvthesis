"""Token-selection policies. Every method sees the decoded part of one layer's cache
and returns, per KV head, the indices of the ``budget`` entries to keep.

All methods share the same harness (trigger schedule, protected prompt, recent
window), so differences in results come from the selection rule alone.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

FORCE = 1e9  # score offset that guarantees an entry is kept
ANCHOR = 1e6  # offset that ranks anchors above ordinary tokens but below forced ones


@dataclass
class SelectCtx:
    scores: torch.Tensor  # (B, H, T) window attention to each decoded entry
    keys: torch.Tensor  # (B, H, T, D) decoded keys
    steps: torch.Tensor  # (B, H, T) reasoning-step id of each entry
    budget: int  # entries to keep per head
    recent: int  # most recent entries that are always kept
    state: dict  # per-layer mutable state, gathered along with the cache
    cfg: dict = field(default_factory=dict)


def _force_recent(score: torch.Tensor, recent: int) -> torch.Tensor:
    score = score.clone()
    score[..., -recent:] += FORCE
    return score


def _topk(score: torch.Tensor, budget: int) -> torch.Tensor:
    return score.topk(budget, dim=-1).indices.sort(dim=-1).values


def _pool(scores: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return scores
    b, h, t = scores.shape
    pooled = F.max_pool1d(scores.reshape(b * h, 1, t), kernel, stride=1, padding=kernel // 2)
    return pooled.reshape(b, h, t)


def select_streaming(ctx: SelectCtx) -> torch.Tensor:
    """StreamingLLM: attention sinks (the protected prompt) + the most recent tokens."""
    b, h, t = ctx.scores.shape
    idx = torch.arange(t - ctx.budget, t, device=ctx.scores.device)
    return idx.expand(b, h, ctx.budget)


def select_snapkv(ctx: SelectCtx) -> torch.Tensor:
    """SnapKV applied at decode time: top-k of max-pooled window attention."""
    score = _pool(ctx.scores, ctx.cfg.get("pool", 7))
    return _topk(_force_recent(score, ctx.recent), ctx.budget)


def select_h2o(ctx: SelectCtx) -> torch.Tensor:
    """H2O heavy hitters: attention accumulated over every compression round."""
    acc = ctx.state.get("acc")
    if acc is None:
        acc = ctx.scores
    else:  # entries appended since the last round start from zero
        acc = F.pad(acc, (0, ctx.scores.shape[-1] - acc.shape[-1])) + ctx.scores
    ctx.state["acc"] = acc
    return _topk(_force_recent(acc, ctx.recent), ctx.budget)


def select_rkv(ctx: SelectCtx) -> torch.Tensor:
    """R-KV (re-implementation of the official ``R1KV``): importance mixed with key redundancy.

    Redundancy follows ``cal_similarity``: cosine similarity between keys, where each key's
    most recent near-duplicate (similarity > ``rkv_threshold``) is exempted so the newest
    copy of repeated content survives; mean over the other keys, then softmax.
    """
    lam = ctx.cfg.get("rkv_lambda", 0.07)
    imp = _pool(ctx.scores, ctx.cfg.get("pool", 7))
    imp = imp / imp.sum(-1, keepdim=True).clamp(min=1e-9)
    kn = F.normalize(ctx.keys.float(), dim=-1)
    sim = torch.matmul(kn, kn.transpose(-1, -2))  # (B, H, T, T)
    sim.diagonal(dim1=-2, dim2=-1).zero_()
    t = sim.shape[-1]
    idx = torch.arange(t, device=sim.device)
    last = torch.where(sim > ctx.cfg.get("rkv_threshold", 0.5), idx, torch.zeros_like(idx)).amax(-1)
    sim.scatter_(-1, last.unsqueeze(-1), 0.0)
    redundancy = sim.mean(-2).softmax(-1)
    score = lam * imp - (1 - lam) * redundancy
    return _topk(_force_recent(score, ctx.recent), ctx.budget)


def _rank_within_step(score: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """1-based rank of each entry among the entries of its own step (by descending score)."""
    _, inv = torch.unique(steps, return_inverse=True)
    n_steps = int(inv.max()) + 1
    order = score.argsort(dim=-1, descending=True)
    seg_sorted = inv.gather(-1, order)
    onehot = F.one_hot(seg_sorted, n_steps)
    rank_sorted = onehot.cumsum(-2).gather(-1, seg_sorted.unsqueeze(-1)).squeeze(-1)
    return torch.empty_like(rank_sorted).scatter_(-1, order, rank_sorted)


def select_stepkv(ctx: SelectCtx) -> torch.Tensor:
    """StepKV: step-level importance re-weighting + per-step anchor tokens.

    1. step score = aggregate window attention received by the step's entries;
    2. token score = pooled token attention * (step score / max step score) ** beta;
    3. the top ``anchors`` tokens of every completed step are kept preferentially,
       so each step retains a compact "summary" instead of vanishing entirely.
    """
    beta = ctx.cfg.get("beta", 1.0)
    n_anchor = ctx.cfg.get("anchors", 2)
    agg = ctx.cfg.get("step_agg", "mean")
    score = _pool(ctx.scores, ctx.cfg.get("pool", 7))
    steps = ctx.steps

    _, inv = torch.unique(steps, return_inverse=True)
    n_steps = int(inv.max()) + 1
    b, h, t = score.shape
    step_score = torch.zeros(b, h, n_steps, device=score.device)
    step_score.scatter_reduce_(-1, inv, ctx.scores, reduce="amax" if agg == "max" else "mean", include_self=False)
    step_score = step_score / step_score.amax(-1, keepdim=True).clamp(min=1e-9)
    weight = step_score.gather(-1, inv).pow(beta)
    weight = torch.where(inv == n_steps - 1, torch.ones_like(weight), weight)  # current (open) step
    adjusted = score * weight

    if n_anchor > 0:
        rank = _rank_within_step(ctx.scores, steps)
        is_anchor = rank <= n_anchor
        # anchors beat ordinary tokens but rank among themselves by their own score
        adjusted = torch.where(is_anchor, ANCHOR + score, adjusted)
    return _topk(_force_recent(adjusted, ctx.recent), ctx.budget)


METHODS = {
    "streaming": select_streaming,
    "snapkv": select_snapkv,
    "h2o": select_h2o,
    "rkv": select_rkv,
    "stepkv": select_stepkv,
}
