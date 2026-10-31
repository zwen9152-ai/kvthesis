"""KV-Rescue-lite: a full-KV helper model takes over the reasoning step that degenerated.

Simplified re-implementation of the recovery idea of KV-Rescue (arXiv 2608.15797) for a
head-to-head comparison with RewindKV under the *same* trigger: when the degeneration
detector fires, generation is cut back to the start of the looping reasoning step (as in
RewindKV), but instead of recalling the base model's own evicted KV, a smaller helper
model with a full, never-evicted KV cache writes that step. The base model then ingests
the helper's tokens into its compressed cache and continues. Unlike the paper there is
no candidate sampling and no process reward model; the paper interleaves at every step.

The helper shares the tokenizer of the base model (e.g. R1-Distill-Qwen 1.5B for 7B),
so token ids are passed through unchanged.
"""

import torch
from transformers import DynamicCache

from .cache_utils import cache_len
from .decode import _forward, sample_next


class HelperState:
    """The helper's full KV cache of one problem, synced lazily to the shared prefix."""

    def __init__(self, helper):
        self.helper = helper
        self.cache = DynamicCache()
        self.tokens: list[int] = []  # tokens whose KV is in ``cache``
        self.generated = 0  # tokens written by the helper
        self.max_cache = 0

    def write_step(self, context: list[int], *, delimiter_ids, eos_ids, max_step: int, temperature: float,
                   top_p: float, generator=None) -> list[int]:
        """Sync the helper cache to ``context`` and let it write one reasoning step."""
        keep = 0  # longest common prefix with what the helper has already encoded
        for a, b in zip(self.tokens, context[:-1]):  # always re-feed >= 1 token to get logits
            if a != b:
                break
            keep += 1
        if keep < len(self.tokens):
            self.cache.crop(keep)
            self.tokens = self.tokens[:keep]
        out = _forward(self.helper, self.cache, context[keep:], keep)
        self.tokens = list(context)
        new: list[int] = []
        while len(new) < max_step:
            tok = sample_next(out.logits[0, -1], temperature, top_p, generator)
            new.append(tok)
            if tok in eos_ids:
                break
            out = _forward(self.helper, self.cache, [tok], len(self.tokens))
            self.tokens.append(tok)
            if tok in delimiter_ids:  # the step is complete
                break
        self.generated += len(new)
        self.max_cache = max(self.max_cache, cache_len(self.cache))
        return new


def helper_bytes(helper, n_entries: int) -> int:
    cfg = helper.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return n_entries * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * helper.dtype.itemsize


@torch.no_grad()
def takeover(model, cache, compressor, detector, tracker, helper_state, generated, P, prompt, cut, *,
             delimiter_ids, eos_ids, max_step, temperature, top_p, generator, forward=None):
    """Replace the looping step by a helper-written one. Returns (generated, logits or None if EOS)."""
    generated = generated[: cut - P]
    tracker.truncate(cut, generated)
    compressor.rollback(cache, cut, recall=False)  # drop everything from the looping step on
    new = helper_state.write_step(prompt + generated, delimiter_ids=delimiter_ids, eos_ids=eos_ids,
                                  max_step=max_step, temperature=temperature, top_p=top_p, generator=generator)
    for t in new:
        tracker.add_token(t)
    generated = generated + new
    detector.truncate(generated, lambda t: t in delimiter_ids)
    if new[-1] in eos_ids:
        return generated, new, None
    out = (forward or (lambda c, t, s: _forward(model, c, t, s)))(cache, new, cut)
    compressor.on_tokens(cache, list(range(cut, cut + len(new))), tracker.steps, at_boundary=True)
    return generated, new, out.logits[0, -1]
