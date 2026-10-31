"""Single-sequence decoding loop with explicit positions and optional rollback.

After eviction the cache holds fewer entries than the number of tokens seen, so
HF's default ``position_ids = cache length`` would be wrong. We pass the true
absolute position for RoPE and a cache-relative ``cache_position`` for masking.

Rollback (detector + offloading compressor): when the detector reports a
repetition loop, generation is cut back to the start of the reasoning step where
the loop began, every evicted entry before that point is recalled from CPU, the
last ``refeed`` tokens are re-encoded to obtain fresh queries, and the compressor
re-selects its budget with them. Decoding then resumes (with sampling, a new draw).
"""

import time

import torch
from transformers import DynamicCache

from .cache_utils import cache_bytes, cache_len
from .segment import StepTracker


def sample_next(logits: torch.Tensor, temperature: float, top_p: float, generator=None) -> int:
    if temperature <= 0:
        return int(logits.argmax(-1))
    probs = (logits.float() / temperature).softmax(-1)
    if top_p < 1.0:
        sorted_p, sorted_i = probs.sort(descending=True)
        drop = sorted_p.cumsum(-1) - sorted_p > top_p
        sorted_p[drop] = 0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_i, sorted_p)
    return int(torch.multinomial(probs, 1, generator=generator))


def _forward(model, cache, tokens: list[int], start_pos: int):
    device = model.device
    n, clen = len(tokens), cache_len(cache)
    return model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache, use_cache=True,
                 position_ids=torch.arange(start_pos, start_pos + n, device=device)[None],
                 cache_position=torch.arange(clen, clen + n, device=device))


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, *, compressor=None, detector=None, delimiter_ids=frozenset(),
             max_new_tokens: int = 4096, temperature: float = 0.0, top_p: float = 1.0,
             eos_ids=(), seed: int | None = None, max_rollbacks: int = 4, refeed: int = 16,
             max_rewind: int = 512, recall: bool = True, escalate: float = 1.0,
             max_budget: int = 4096) -> dict:
    """Generate from ``prompt_ids`` (shape (1, P)); returns token ids and statistics.

    With ``escalate > 1`` every rollback multiplies the compressor budget (capped at
    ``max_budget``) before re-selection, so problems that keep looping get more memory.
    """
    device = model.device
    P = prompt_ids.shape[1]
    generator = None
    if seed is not None and temperature > 0:
        generator = torch.Generator(device=device).manual_seed(seed)
    if detector is not None and (compressor is None or not compressor.offload):
        raise ValueError("rollback needs a compressor with offload=True")

    delimiter_ids = set(delimiter_ids)
    tracker = StepTracker(delimiter_ids, P + max_new_tokens + 1, device)
    tracker.add_prompt(P)
    cache = DynamicCache()
    if compressor is not None:
        compressor.reset()
    if detector is not None:
        detector.reset()

    t0 = time.perf_counter()
    out = _forward(model, cache, prompt_ids[0].tolist(), 0)
    if compressor is not None:
        compressor.on_prefill(cache, P)

    generated: list[int] = []
    rollbacks: list[dict] = []
    computed = 0  # tokens decoded including those discarded by rollbacks
    max_cache = cache_len(cache)
    max_bytes = cache_bytes(cache)
    max_offload = 0
    eos_ids = set(eos_ids)
    while len(generated) < max_new_tokens:
        tok = sample_next(out.logits[0, -1], temperature, top_p, generator)
        generated.append(tok)
        computed += 1
        if tok in eos_ids:
            break
        at_boundary = tracker.add_token(tok)
        pos = P + len(generated) - 1
        out = _forward(model, cache, [tok], pos)
        if compressor is not None:
            compressor.on_tokens(cache, [pos], tracker.steps, at_boundary)
            max_offload = max(max_offload, compressor.offloaded_bytes())
        max_cache = max(max_cache, cache_len(cache))
        max_bytes = max(max_bytes, cache_bytes(cache))

        if detector is None or len(rollbacks) >= max_rollbacks:
            continue
        loop_start = detector.update(generated, at_boundary)
        # A loop before any eviction is the model's own; recall cannot change it.
        if loop_start is None or compressor.n_compress == 0:
            continue
        cut = tracker.step_start(P + loop_start)
        if P + loop_start - cut > max_rewind:  # unusually long step: rewind to the loop itself
            cut = P + loop_start
        cut = max(cut, P + 1)  # keep >= 1 token to re-feed
        rollbacks.append({"at": len(generated), "loop_start": loop_start, "cut": cut - P})
        generated = generated[: cut - P]
        tracker.truncate(cut, generated)
        detector.truncate(generated, lambda t: t in delimiter_ids)
        c0 = max(P, cut - refeed)
        compressor.rollback(cache, c0, recall=recall)
        compressor.budget = min(max_budget, max(compressor.budget, int(compressor.budget * escalate)))
        rollbacks[-1]["budget"] = compressor.budget
        chunk = generated[c0 - P:]
        out = _forward(model, cache, chunk, c0)
        # recalled history is resident until the re-selection below: a transient peak
        rollbacks[-1]["transient_cache_len"] = cache_len(cache)
        compressor.on_tokens(cache, list(range(c0, cut)), tracker.steps, at_boundary=True)
        max_cache = max(max_cache, cache_len(cache))
        max_bytes = max(max_bytes, cache_bytes(cache))

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "gen_ids": generated,
        "num_tokens": len(generated),
        "computed_tokens": computed,
        "finished": bool(generated) and generated[-1] in eos_ids,
        "num_steps": tracker.cur_step,
        "max_cache_len": max_cache,
        "max_cache_bytes": max_bytes,
        "max_offload_bytes": max_offload,
        "n_compress": compressor.n_compress if compressor is not None else 0,
        "n_recalled": compressor.n_recalled if compressor is not None else 0,
        "final_budget": compressor.budget if compressor is not None else None,
        "rollbacks": rollbacks,
        "seconds": elapsed,
        "tok_per_s": computed / max(elapsed, 1e-9),
    }
