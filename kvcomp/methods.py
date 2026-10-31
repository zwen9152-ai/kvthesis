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
    values: torch.Tensor | None = None  # (B, H, T, D) decoded values (value-aware methods)
    rng: torch.Generator | None = None  # for stochastic methods


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


def select_lazy(ctx: SelectCtx) -> torch.Tensor:
    """LazyEviction (Zhang et al., ACL 2026; port of the official ``Window_LAZYKVCluster``).

    A token is "re-attended" at a decoding step when its attention weight reaches
    ``alpha``. Per token the batched decoder tracks the last such step and the maximum
    recurrence interval (MRI) between two of them; the score keeps tokens whose next
    recurrence is due soon. Needs per-step attention, so it only runs in
    ``kvcomp.batch.generate_batch`` (which fills ``lazy_last``/``lazy_mp``/``lazy_step``).
    Adaptation: official code evicts per query head (KV repeated); we share one decision
    per KV head and mark a token re-attended if any query head of its group attends to it.
    """
    try:
        last, mp, step = ctx.state["lazy_last"], ctx.state["lazy_mp"], ctx.state["lazy_step"]
    except KeyError as e:
        raise RuntimeError("lazy needs the per-step observer of kvcomp.batch.generate_batch") from e
    s = torch.where(mp == 0, torch.zeros_like(mp), 2 / (1 + torch.exp(mp - 1)))
    score = 2 / (1 + torch.exp(1 + (step - last) / (mp + 1e-5))) + s
    return _topk(_force_recent(score, ctx.recent), ctx.budget)


def select_vase(ctx: SelectCtx) -> torch.Tensor:
    """VaSE-AttnV (arXiv 2606.03928; official mode ``range_sink_sample_attn``).

    Of the non-recent candidates, the ``n_large`` (default a quarter of the free slots)
    with the largest value range max(v) - min(v) are kept unconditionally; the remaining
    slots are drawn without replacement with probability proportional to the
    average-pooled window attention. The prompt (incl. sink tokens) is never evicted here.
    """
    if ctx.values is None:
        raise RuntimeError("vase needs the decoded values")
    t = ctx.scores.shape[-1]
    n_cand = t - ctx.recent
    k = ctx.budget - ctx.recent
    n_large = min(ctx.cfg.get("n_large", k // 4), k)
    vmag = (ctx.values.amax(-1) - ctx.values.amin(-1)).float()[..., :n_cand]
    ids_large = vmag.topk(n_large, dim=-1).indices
    kernel = ctx.cfg.get("pool", 5)
    b, h, _ = ctx.scores.shape
    w = F.avg_pool1d(ctx.scores.reshape(b * h, 1, t), kernel, stride=1, padding=kernel // 2,
                     count_include_pad=False).reshape(b, h, t)[..., :n_cand].clamp_min(1e-12)
    w = w.scatter(-1, ids_large, 0.0)
    ids_sample = torch.multinomial(w.reshape(b * h, n_cand), k - n_large, replacement=False,
                                   generator=ctx.rng).reshape(b, h, -1)
    recent = torch.arange(n_cand, t, device=w.device).expand(b, h, -1)
    return torch.cat([ids_large, ids_sample, recent], dim=-1).sort(dim=-1).values


METHODS = {
    "streaming": select_streaming,
    "snapkv": select_snapkv,
    "h2o": select_h2o,
    "rkv": select_rkv,
    "stepkv": select_stepkv,
    "lazy": select_lazy,
    "vase": select_vase,
}
