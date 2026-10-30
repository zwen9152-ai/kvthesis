"""Decode-time compression harness shared by every method.

Schedule (same for all methods, following R-KV): the prompt is never evicted;
once the decoded part of the cache reaches ``budget + buffer`` entries it is
compressed back to ``budget`` entries per KV head. With ``step_trigger`` the
compression waits for the next reasoning-step boundary (up to ``2 * buffer``),
so the scoring queries come from a complete step.

With ``offload`` the evicted entries are moved to CPU memory instead of being
dropped, so ``rollback`` can restore the exact uncompressed history.
"""

import torch

from .cache_utils import get_kv, num_layers, set_kv
from .methods import METHODS, SelectCtx
from .query_capture import QueryCapture, attention_modules, window_attention


class KVCompressor:
    def __init__(self, model, method: str, budget: int, buffer: int = 128, window: int = 16,
                 step_trigger: bool = False, offload: bool = False, **cfg):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
        if budget <= window:
            raise ValueError("budget must exceed the recent window")
        self.select = METHODS[method]
        self.method = method
        self.budget = budget
        self.buffer = buffer
        self.window = window
        self.step_trigger = step_trigger
        self.offload = offload
        self.cfg = cfg
        self.modules = attention_modules(model)
        self.capture = QueryCapture(model, window)
        self.reset()

    def reset(self):
        self.capture.reset()
        self.prompt_len = 0
        self.pos: list[torch.Tensor] = []  # per layer (B, H, T_decoded) absolute positions
        self.state: list[dict] = []
        self.store: list = []  # per layer (pos, keys, values) evicted to CPU, or None
        self.n_compress = 0
        self.n_recalled = 0

    def close(self):
        self.capture.remove()

    def on_prefill(self, cache, prompt_len: int):
        self.prompt_len = prompt_len
        k, _ = get_kv(cache, 0)
        b, h = k.shape[:2]
        empty = torch.empty(b, h, 0, dtype=torch.long, device=k.device)
        n = num_layers(cache)
        self.pos = [empty for _ in range(n)]
        self.state = [{} for _ in range(n)]
        self.store = [None] * n

    def on_tokens(self, cache, positions: list[int], steps_fn, at_boundary: bool) -> bool:
        """Register newly cached tokens and compress if the schedule says so."""
        new = torch.tensor(positions, device=self.pos[0].device)
        for i, p in enumerate(self.pos):
            self.pos[i] = torch.cat([p, new.expand(*p.shape[:2], -1)], dim=-1)
        n = self.pos[0].shape[-1]
        if n < self.budget + self.buffer:
            return False
        if self.step_trigger and not at_boundary and n < self.budget + 2 * self.buffer:
            return False
        self.compress(cache, steps_fn)
        return True

    @torch.no_grad()
    def compress(self, cache, steps_fn):
        P = self.prompt_len
        for i in range(num_layers(cache)):
            k, v = get_kv(cache, i)
            kd, vd = k[:, :, P:], v[:, :, P:]
            scores = window_attention(self.capture.buf[i], k, self.modules[i].scaling)[..., P:]
            pos = self.pos[i]
            ctx = SelectCtx(scores=scores, keys=kd, steps=steps_fn(pos), budget=self.budget,
                            recent=self.window, state=self.state[i], cfg=self.cfg)
            idx = self.select(ctx)
            if self.offload:
                self._offload(i, kd, vd, pos, idx)
            gidx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            set_kv(cache, i, torch.cat([k[:, :, :P], kd.gather(2, gidx)], dim=2),
                   torch.cat([v[:, :, :P], vd.gather(2, gidx)], dim=2))
            self.pos[i] = pos.gather(-1, idx)
            for key, val in self.state[i].items():
                if torch.is_tensor(val) and val.shape == pos.shape:
                    self.state[i][key] = val.gather(-1, idx)
        self.n_compress += 1

    def _offload(self, i, kd, vd, pos, keep_idx):
        keep = torch.zeros_like(pos, dtype=torch.bool).scatter_(-1, keep_idx, True)
        n_evict = pos.shape[-1] - keep_idx.shape[-1]
        ev = keep.to(torch.int8).argsort(dim=-1, stable=True)[..., :n_evict]  # evicted first, in order
        gev = ev.unsqueeze(-1).expand(-1, -1, -1, kd.shape[-1])
        new = (pos.gather(-1, ev).cpu(), kd.gather(2, gev).cpu(), vd.gather(2, gev).cpu())
        old = self.store[i]
        self.store[i] = new if old is None else tuple(torch.cat([a, b], dim=-2 if a.dim() == 4 else -1)
                                                      for a, b in zip(old, new))

    def offloaded_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self.store if s is not None for t in s[1:])

    @torch.no_grad()
    def rollback(self, cache, cut: int, recall: bool = True) -> None:
        """Drop every entry at position >= cut and restore all offloaded entries below it.

        With ``recall=False`` (ablation) the offloaded entries are discarded instead; heads
        may then keep different numbers of entries below ``cut``, so every head is trimmed
        to the smallest count, keeping its most recent ones.
        """
        P = self.prompt_len
        merged = []
        for i in range(num_layers(cache)):
            k, v = get_kv(cache, i)
            pos, kd, vd = self.pos[i], k[:, :, P:], v[:, :, P:]
            if recall and self.store[i] is not None:
                sp, sk, sv = (t.to(k.device) for t in self.store[i])
                self.n_recalled += int((sp < cut).sum())
                pos, kd, vd = torch.cat([pos, sp], -1), torch.cat([kd, sk], 2), torch.cat([vd, sv], 2)
            merged.append((pos, kd, vd))
        # all layers must end up with the same length, so trim to the global minimum
        n = min(int((pos < cut).sum(-1).min()) for pos, _, _ in merged)
        for i, (pos, kd, vd) in enumerate(merged):
            k, v = get_kv(cache, i)
            valid = pos < cut
            key = torch.where(valid, pos, torch.full_like(pos, torch.iinfo(pos.dtype).max))
            order = key.argsort(dim=-1)
            counts = valid.sum(-1, keepdim=True)  # (B, H, 1)
            idx = order.gather(-1, counts - n + torch.arange(n, device=k.device))
            gidx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            set_kv(cache, i, torch.cat([k[:, :, :P], kd.gather(2, gidx)], dim=2),
                   torch.cat([v[:, :, :P], vd.gather(2, gidx)], dim=2))
            self.pos[i] = pos.gather(-1, idx)
            self.store[i] = None
            self.state[i] = {}
        self.capture.reset()
