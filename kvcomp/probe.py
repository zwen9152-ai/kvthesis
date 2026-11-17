"""Chapter 3 instrumentation: what the compressed cache keeps and what the model looks back at.

A ``Probe`` rides on a ``KVCompressor`` (option ``probe=1``) and is ticked by the decoding
loops after every token. Every ``every`` tokens it records

* age retention: for the tokens that are now between ``age`` and ``age + every`` tokens
  old, the share of (layer, KV head) slots that still hold them (summaries excluded);
* lookback (``lookback=1``): the attention the last ``window`` queries pay to every cached
  token at least ``min_age`` tokens old, averaged over layers and query heads, accumulated
  per token (run with an effectively infinite budget this is the Full-KV reference).

On every detection event (``on_detect``) it stores the retention of every completed
reasoning step, so the analysis can compare the step that is being repeated with other
steps of the same age (forgetting hypothesis).
Retention and lookback are stored as integers (per mille / 1e-6 units) to keep jsonl small.
"""

import torch

from .cache_utils import num_layers
from .query_capture import window_attention


class Probe:
    def __init__(self, every: int = 256, age: int = 1024, lookback: bool = False, min_age: int = 256,
                 max_events: int = 16):
        self.every, self.age, self.lookback, self.min_age = every, age, lookback, min_age
        self.max_events = max_events
        self.reset()

    def reset(self):
        self.ret: dict[int, int] = {}  # generated index -> retention (per mille) at age ~ ``age``
        self.look_sum: dict[int, float] = {}
        self.look_n: dict[int, int] = {}
        self.events: list[dict] = []

    def truncate(self, n_gen: int):
        """Forget measurements of tokens removed by a rollback."""
        for d in (self.ret, self.look_sum, self.look_n):
            for j in [j for j in d if j >= n_gen]:
                del d[j]

    @torch.no_grad()
    def retention(self, comp, lo: int, hi: int) -> torch.Tensor:
        """Share of (layer, head) slots holding each absolute position in [lo, hi)."""
        cnt, total = None, 0
        for i in range(len(comp.pos)):
            p = comp.normal_pos(i)
            sel = (p >= lo) & (p < hi)
            c = torch.bincount((p[sel] - lo).flatten(), minlength=hi - lo).float()
            cnt = c if cnt is None else cnt + c
            total += p.shape[0] * p.shape[1]
        return cnt / max(total, 1)

    @torch.no_grad()
    def tick(self, comp, n_gen: int, P: int, keys_fn):
        if n_gen % self.every:
            return
        hi = n_gen - self.age
        lo = max(0, hi - self.every)
        if hi > lo:
            r = self.retention(comp, P + lo, P + hi)
            for j, x in zip(range(lo, hi), (r * 1000).round().int().tolist()):
                self.ret[j] = x
        if self.lookback:
            self._lookback(comp, n_gen, P, keys_fn)

    def _lookback(self, comp, n_gen, P, keys_fn):
        cut = P + n_gen - self.min_age
        acc = None
        L = len(comp.pos)
        for i in range(L):
            k = keys_fn(i)
            q = comp.capture.buf.get(i)
            if q is None:
                return
            a = window_attention(q, k, comp.modules[i].scaling)[..., P:]  # (B, Hkv, T) summed over window x group
            a = a / (q.shape[1] * q.shape[2])  # mean attention per query
            pos = comp.pos[i]
            sel = pos < cut
            idx = (pos[sel] - P).flatten()
            s = torch.zeros(n_gen, device=a.device).scatter_add_(0, idx, a[sel].flatten())
            acc = s if acc is None else acc + s
        if acc is None:
            return
        acc = (acc / L).tolist()
        for j in range(max(0, n_gen - self.min_age)):
            self.look_sum[j] = self.look_sum.get(j, 0.0) + acc[j]
            self.look_n[j] = self.look_n.get(j, 0) + 1

    @torch.no_grad()
    def on_detect(self, comp, n_gen: int, P: int, step_ids: list[int], event: dict):
        """Record per-step retention at a detection event (``event`` is stored with it)."""
        if len(self.events) >= self.max_events or n_gen == 0:
            return
        r = self.retention(comp, P, P + n_gen).tolist()
        spans, start = [], 0
        ids = step_ids[P:P + n_gen]
        for j in range(1, n_gen + 1):
            if j == n_gen or ids[j] != ids[start]:
                seg = r[start:j]
                spans.append([start, j, round(1000 * sum(seg) / len(seg))])
                start = j
        self.events.append({**event, "steps": spans})

    def result(self, n_gen: int) -> dict:
        ret = [self.ret.get(j, -1) for j in range(n_gen)]
        out = {"probe_ret": ret if any(x >= 0 for x in ret) else [], "probe_events": self.events}
        if self.lookback:
            out["probe_look"] = [round(1e6 * self.look_sum[j] / self.look_n[j]) if j in self.look_n else -1
                                 for j in range(n_gen)]
        return out
