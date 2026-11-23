"""Decode-time compression harness shared by every method.

Schedule (same for all methods, following R-KV): the prompt is never evicted;
once the decoded part of the cache reaches ``budget + buffer`` entries it is
compressed back to ``budget`` entries per KV head. With ``step_trigger`` the
compression waits for the next reasoning-step boundary (up to ``2 * buffer``),
so the scoring queries come from a complete step.

With ``offload`` the evicted entries are moved to CPU memory instead of being
dropped, so ``rollback`` can restore the exact uncompressed history.
``generate`` may raise ``budget`` after a rollback (escalation); ``reset`` restores it.

Step-summary merging (``merge=1``, chapter 3 prevention): instead of dropping the evicted
entries of a reasoning step, each step is folded into one summary entry per KV head:
key = the key of the step's most-attended evicted entry (keys carry RoPE, so they are
not averaged), value = attention-weighted mean of the step's evicted values. Summaries
sit at the front of the decoded region, use a reserved share (``merge_frac``) of the
budget, and the oldest ones are folded together once that share is full.

Pinned entries (``state[i]["pin"]``, chapter 4 aligned recall / chapter 3 intervention)
are never evicted; ``restore`` brings chosen offloaded positions back and pins them.
"""

import time

import torch

from .cache_utils import get_kv, num_layers, set_kv
from .methods import METHODS, SelectCtx
from .query_capture import QueryCapture, attention_modules, window_attention


class KVCompressor:
    def __init__(self, model, method: str, budget: int, buffer: int = 128, window: int = 16,
                 step_trigger: bool = False, offload: bool = False, capture=None, **cfg):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
        if budget <= window:
            raise ValueError("budget must exceed the recent window")
        self.select = METHODS[method]
        self.method = method
        self.base_budget = budget
        self.budget = budget
        self.buffer = buffer
        self.window = window
        self.step_trigger = step_trigger
        self.offload = offload
        self.profile = bool(cfg.pop("profile", False))
        self.merge = bool(cfg.pop("merge", False))
        self.merge_frac = float(cfg.pop("merge_frac", 0.25))
        self.merge_w = cfg.pop("merge_w", "attn")  # attn | uniform (value weights)
        self.merge_key = cfg.pop("merge_key", "top")  # top | recent (which key represents the step)
        self.pin_frac = float(cfg.pop("pin_frac", 0.25))  # cap on pinned entries (share of budget)
        self.forget_every = int(cfg.pop("forget_every", 0))  # forgetting signal period (tokens); needs offload
        self.forget_thr = cfg.pop("forget_thr", None)  # alarm threshold on the evicted-attention share
        self.forget_thr = None if self.forget_thr is None else float(self.forget_thr)
        self.forget_stride = int(cfg.pop("forget_stride", 4))  # measure every k-th layer
        # recall="query": at a rollback restore the evicted entries the current queries want most
        self.query_frac = float(cfg.pop("query_frac", 0.25))  # how many (share of the budget)
        # size_cover > 0: an alarm sets the budget to the number of decoded entries (resident or
        # evicted) that cover this share of the window's attention, instead of a fixed multiplier
        self.size_cover = float(cfg.pop("size_cover", 0))
        self.probe = None
        if cfg.pop("probe", 0):
            from .probe import Probe
            self.probe = Probe(every=int(cfg.pop("probe_every", 256)), age=int(cfg.pop("probe_age", 1024)),
                               lookback=bool(cfg.pop("lookback", 0)), min_age=int(cfg.pop("probe_min_age", 256)))
        self.cfg = cfg
        self.modules = attention_modules(model)
        # batched decoding passes a per-sequence view of one shared capture
        self.own_capture = capture is None
        self.capture = QueryCapture(model, window) if capture is None else capture
        self.reset()

    def reset(self):
        self.capture.reset()
        self.budget = self.base_budget  # rollback may have escalated it
        self.prompt_len = 0
        self._pos: list[torch.Tensor] = []  # per layer (B, H, T_decoded) absolute positions
        self._pending: list[int] = []  # positions appended since the last flush (same for all layers)
        self.state: list[dict] = []
        self.store: list = []  # per layer: list of (pos, keys, values) chunks evicted to CPU, or None
        self.n_compress = 0
        self.n_recalled = 0
        self.n_sum = 0  # summary entries at the front of the decoded region (same for all layers)
        self._sum_cnt: list = []  # per layer (B, H, n_sum) number of tokens each summary stands for
        self.n_merged = 0  # evicted entries folded into summaries (per head, summed over compressions)
        self.n_empty_groups = 0  # (layer, head, step) groups with nothing evicted in that head (fallback used)
        self.n_restored = 0
        self.forget_trace: list = []  # [generated index, evicted-attention share in per mille]
        self._ev_keys: dict = {}  # layer -> (chunk count, last chunk id, evicted keys on device)
        if getattr(self, "probe", None) is not None:
            self.probe.reset()
        self.rng = None  # stochastic methods: re-seeded per problem, created on the cache device
        # efficiency accounting (chapter 5); timings only when ``profile`` is set (they sync the GPU)
        self.bytes_offloaded = 0  # GPU -> CPU
        self.bytes_recalled = 0  # CPU -> GPU
        self.t_compress = 0.0
        self.t_rollback = 0.0

    def close(self):
        if self.own_capture:
            self.capture.remove()

    def on_prefill(self, cache, prompt_len: int):
        self.prompt_len = prompt_len
        k, _ = get_kv(cache, 0)
        b, h = k.shape[:2]
        empty = torch.empty(b, h, 0, dtype=torch.long, device=k.device)
        n = num_layers(cache)
        self._pos = [empty for _ in range(n)]
        self._pending = []
        self.state = [{} for _ in range(n)]
        self.store = [None] * n
        self.n_sum = 0
        self._sum_cnt = [None] * n

    def on_tokens(self, cache, positions: list[int], steps_fn, at_boundary: bool) -> bool:
        """Register newly cached tokens and compress if the schedule says so."""
        self.append(positions)
        if not self.due(at_boundary):
            return False
        self.compress(cache, steps_fn)
        return True

    @property
    def pos(self) -> list[torch.Tensor]:
        """Per-layer kept positions; pending appends are materialised on first access."""
        if self._pending:
            new = torch.tensor(self._pending, device=self._pos[0].device)
            self._pos = [torch.cat([p, new.expand(*p.shape[:2], -1)], dim=-1) for p in self._pos]
            self._pending = []
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos, self._pending = value, []

    def append(self, positions: list[int]) -> None:
        """Register newly cached positions (cheap: kept in a Python list until needed)."""
        self._pending.extend(positions)

    def n_decoded(self) -> int:
        return (self._pos[0].shape[-1] if self._pos else 0) + len(self._pending)

    def due(self, at_boundary: bool) -> bool:
        """Whether the schedule asks for a compression now (``append`` already called)."""
        n = self.n_decoded()
        if n < self.budget + self.buffer:
            return False
        return not (self.step_trigger and not at_boundary and n < self.budget + 2 * self.buffer)

    def sum_cap(self) -> int:
        return max(1, int(self.budget * self.merge_frac)) if self.merge else 0

    def normal_pos(self, i: int) -> torch.Tensor:
        """Kept positions of layer ``i`` without the summary entries."""
        return self.pos[i][..., self.n_sum:]

    @torch.no_grad()
    def compress(self, cache, steps_fn):
        t0 = self._tick()
        P, ns = self.prompt_len, self.n_sum
        L = num_layers(cache)
        budget = self.budget - self.sum_cap()
        plans = []
        for i in range(L):
            k, v = get_kv(cache, i)
            kd, vd = k[:, :, P + ns:], v[:, :, P + ns:]
            scores = window_attention(self.capture.buf[i], k, self.modules[i].scaling)[..., P + ns:]
            pos = self.pos[i][..., ns:]
            if self.rng is None:
                self.rng = torch.Generator(device=k.device).manual_seed(int(self.cfg.get("seed", 0)))
            steps = steps_fn(pos)
            idx = self._select(i, scores, kd, vd, pos, steps, budget)
            plans.append((idx, scores, steps))
            if self.offload:
                self._offload(i, kd, vd, pos, idx)
        groups = self._evicted_steps(plans) if self.merge else None
        for i in range(L):
            k, v = get_kv(cache, i)
            idx, scores, steps = plans[i]
            kd, vd = k[:, :, P + ns:], v[:, :, P + ns:]
            pos = self.pos[i][..., ns:]
            gidx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            sk, sv, sp = k[:, :, P:P + ns], v[:, :, P:P + ns], self.pos[i][..., :ns]
            if groups is not None and len(groups):
                sk, sv, sp = self._merge(i, kd, vd, pos, scores, steps, idx, groups, sk, sv, sp)
            set_kv(cache, i, torch.cat([k[:, :, :P], sk, kd.gather(2, gidx)], dim=2),
                   torch.cat([v[:, :, :P], sv, vd.gather(2, gidx)], dim=2))
            self.pos[i] = torch.cat([sp, pos.gather(-1, idx)], dim=-1)
            for key, val in self.state[i].items():
                if torch.is_tensor(val) and val.shape == pos.shape:
                    self.state[i][key] = val.gather(-1, idx)
        if groups is not None:
            self.n_sum = int(self.pos[0].shape[-1] - plans[0][0].shape[-1])
        self.n_compress += 1
        self.t_compress += self._tick() - t0 if self.profile else 0.0

    def _select(self, i, scores, kd, vd, pos, steps, budget):
        """Run the base method; pinned entries are kept and excluded from its choice."""
        st = self.state[i]
        pin = st.get("pin")
        if pin is not None and pin.shape[-1] < pos.shape[-1]:  # entries appended since: not pinned
            pin = st["pin"] = torch.nn.functional.pad(pin, (0, pos.shape[-1] - pin.shape[-1]))
        if pin is not None:
            cnt = pin.sum(-1)
            if int(cnt.max()) != int(cnt.min()):  # heads disagree: keep the newest min-count pins
                rank = pin.flip(-1).long().cumsum(-1).flip(-1)  # 1 = newest pinned entry
                pin = st["pin"] = pin & (rank <= cnt.min())
        npin = int(pin[0, 0].sum()) if pin is not None else 0
        if npin == 0:
            ctx = SelectCtx(scores=scores, keys=kd, steps=steps, budget=budget, recent=self.window,
                            state=st, cfg=self.cfg, values=vd, rng=self.rng)
            return self.select(ctx)
        t = pos.shape[-1]
        order = pin.to(torch.int8).argsort(dim=-1, stable=True)  # unpinned first, in position order
        free, pinned = order[..., :t - npin], order[..., t - npin:]
        g = lambda x: x.gather(-1, free) if x.dim() == 3 else x.gather(2, free.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))  # noqa: E731
        sub = {key: (g(val) if torch.is_tensor(val) and val.shape == pos.shape else val) for key, val in st.items()}
        ctx = SelectCtx(scores=g(scores), keys=g(kd), steps=g(steps), budget=budget - npin, recent=self.window,
                        state=sub, cfg=self.cfg, values=g(vd), rng=self.rng)
        sub_idx = self.select(ctx)
        for key, val in sub.items():  # write per-entry state of the free entries back in place
            if key == "pin":
                continue
            if torch.is_tensor(val) and val.shape == free.shape:
                full = st.get(key)
                if not (torch.is_tensor(full) and full.shape == pos.shape):
                    full = torch.zeros(pos.shape, dtype=val.dtype, device=val.device)
                st[key] = full.scatter(-1, free, val)
            elif not (torch.is_tensor(val) and val.dim() == 3):
                st[key] = val
        return torch.cat([free.gather(-1, sub_idx), pinned], dim=-1).sort(dim=-1).values

    def _evicted_steps(self, plans) -> torch.Tensor:
        """Reasoning steps that lost entries in any layer/head (one summary each, all heads)."""
        out = []
        for idx, _, steps in plans:
            keep = torch.zeros_like(steps, dtype=torch.bool).scatter_(-1, idx, True)
            out.append(steps[~keep])
        return torch.unique(torch.cat(out))

    def _merge(self, i, kd, vd, pos, scores, steps, idx, groups, sk, sv, sp):
        """Fold the evicted entries of every step in ``groups`` into one summary per head."""
        b, h, t, d = kd.shape
        m = len(groups)
        ev = torch.ones_like(pos, dtype=torch.bool).scatter_(-1, idx, False)
        gi = torch.searchsorted(groups, steps.contiguous()).clamp(max=m - 1)
        ev = ev & (groups[gi] == steps)
        w = (scores + 1e-6) if self.merge_w == "attn" else torch.ones_like(scores)
        w = torch.where(ev, w, torch.zeros_like(w))
        cnt = torch.zeros(b, h, m, device=kd.device).scatter_add_(-1, gi, ev.float())
        wsum = torch.zeros(b, h, m, device=kd.device).scatter_add_(-1, gi, w)
        vsum = torch.zeros(b, h, m, d, device=kd.device).scatter_add_(
            2, gi.unsqueeze(-1).expand(-1, -1, -1, d), w.unsqueeze(-1) * vd.float())
        # representative key: the most attended (or the latest) evicted entry of the group
        rank = scores if self.merge_key == "top" else pos.float()
        rank = torch.where(ev, rank, torch.full_like(rank, -float("inf")))
        best = torch.full((b, h, m), -float("inf"), device=kd.device).scatter_reduce_(-1, gi, rank, "amax")
        ar = torch.arange(t, device=kd.device).expand(b, h, t)
        hit = ev & (rank == best.gather(-1, gi))
        rep = torch.full((b, h, m), -1, dtype=torch.long, device=kd.device).scatter_reduce_(
            -1, gi, torch.where(hit, ar, torch.full_like(ar, -1)), "amax")
        empty = cnt == 0
        if empty.any():  # this head kept the whole step: stand in its overall top evicted entry
            fb = rank.argmax(-1, keepdim=True).expand(-1, -1, m)
            rep = torch.where(empty, fb, rep)
            self.n_empty_groups += int(empty.sum())
        val = vsum / wsum.clamp_min(1e-12).unsqueeze(-1)
        val = torch.where(empty.unsqueeze(-1), vd.float().gather(2, rep.unsqueeze(-1).expand(-1, -1, -1, d)), val)
        cnt = torch.where(empty, torch.ones_like(cnt), cnt)
        nk = kd.gather(2, rep.unsqueeze(-1).expand(-1, -1, -1, d))
        nv, np_ = val.to(vd.dtype), pos.gather(-1, rep)
        if i == 0:  # per head of layer 0, so the count reads as "tokens folded"
            self.n_merged += int(ev[0, 0].sum())
        old = self._sum_cnt[i] if self._sum_cnt[i] is not None else cnt.new_zeros(b, h, 0)
        sk, sv, sp = torch.cat([sk, nk], 2), torch.cat([sv, nv], 2), torch.cat([sp, np_], -1)
        sc = torch.cat([old, cnt], -1)
        cap = self.sum_cap()
        if sc.shape[-1] > cap:  # fold the oldest summaries into one
            e = sc.shape[-1] - cap + 1
            c = sc[..., :e]
            fv = (c.unsqueeze(-1) * sv[:, :, :e].float()).sum(2, keepdim=True) / c.sum(-1, keepdim=True).unsqueeze(-1)
            j = c.argmax(-1, keepdim=True)
            fk = sk[:, :, :e].gather(2, j.unsqueeze(-1).expand(-1, -1, -1, d))
            sk = torch.cat([fk, sk[:, :, e:]], 2)
            sv = torch.cat([fv.to(sv.dtype), sv[:, :, e:]], 2)
            sp = torch.cat([sp[..., :e].gather(-1, j), sp[..., e:]], -1)
            sc = torch.cat([c.sum(-1, keepdim=True), sc[..., e:]], -1)
        self._sum_cnt[i] = sc
        return sk, sv, sp

    @torch.no_grad()
    def restore(self, cache, positions, pin: bool = True) -> int:
        """Bring the offloaded entries at ``positions`` back (and pin them), at constant cache size.

        Each head first regains every listed position it had evicted; then it drops as many
        unpinned, non-recent entries with the lowest window attention, so all heads keep the
        same length. Returns the mean number of entries restored per head.
        """
        P, ns = self.prompt_len, self.n_sum
        want = torch.as_tensor(sorted(set(positions)), dtype=torch.long)
        if not len(want):
            return 0
        n_new = 0
        for i in range(num_layers(cache)):
            k, v = get_kv(cache, i)
            dev = k.device
            w = want.to(dev)
            pos = self.pos[i][..., ns:]
            kd, vd = k[:, :, P + ns:], v[:, :, P + ns:]
            st = self.state[i]
            pinv = st.get("pin")
            if pinv is None or pinv.shape != pos.shape:
                base = torch.zeros(pos.shape, dtype=torch.bool, device=dev)
                if pinv is not None:
                    base[..., :pinv.shape[-1]] = pinv
                pinv = base
            if self.store[i] is None:
                sp = pos.new_zeros(*pos.shape[:2], 0)
                sk = kd.new_zeros(*kd.shape[:2], 0, kd.shape[-1])
                sv = sk
            else:
                chunks = self.store[i]
                sp, sk, sv = (torch.cat([c[j] for c in chunks], dim=-2 if j else -1).to(dev) for j in range(3))
            sel = torch.isin(sp, w) & ~_dup_of(sp, pos)
            c = sel.sum(-1)  # (B, H) entries to bring back per head
            cmax = int(c.max())
            if cmax == 0:
                continue
            order = (~sel).to(torch.int8).argsort(dim=-1, stable=True)[..., :cmax]
            valid = torch.arange(cmax, device=dev) < c.unsqueeze(-1)
            rp = sp.gather(-1, order)
            rk = sk.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, sk.shape[-1]))
            rv = sv.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, sv.shape[-1]))
            score = window_attention(self.capture.buf[i], k, self.modules[i].scaling)[..., P + ns:]
            already = torch.isin(pos, w)
            score = torch.where(pinv | (already & pin), score + 1e9, score)
            score[..., -self.window:] += 1e9
            rscore = torch.where(valid, torch.full_like(rp, 1e9, dtype=score.dtype), torch.full_like(rp, -float("inf"), dtype=score.dtype))
            cp = torch.cat([pos, rp], -1)
            keep_n = pos.shape[-1]
            idx = torch.cat([score, rscore], -1).topk(keep_n, dim=-1).indices
            idx = idx.gather(-1, cp.gather(-1, idx).argsort(-1))
            ck, cv = torch.cat([kd, rk], 2), torch.cat([vd, rv], 2)
            gidx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            set_kv(cache, i, torch.cat([k[:, :, :P + ns], ck.gather(2, gidx)], 2),
                   torch.cat([v[:, :, :P + ns], cv.gather(2, gidx)], 2))
            newpin = torch.cat([pinv | (already & pin), valid & pin], -1).gather(-1, idx)
            for key, val in list(st.items()):
                if torch.is_tensor(val) and val.shape == pos.shape and key != "pin":
                    st[key] = torch.cat([val, val.new_zeros(*val.shape[:2], cmax)], -1).gather(-1, idx)
            st["pin"] = newpin
            self.pos[i] = torch.cat([self.pos[i][..., :ns], cp.gather(-1, idx)], -1)
            n_new += float(c.float().mean())
            self.bytes_recalled += 2 * int(c.sum()) * kd.shape[-1] * kd.element_size()
        self._cap_pins()
        self.n_restored += int(round(n_new / max(1, num_layers(cache))))
        return int(round(n_new / max(1, num_layers(cache))))

    @torch.no_grad()
    def evicted_share(self, keys_fn) -> float | None:
        """Forgetting signal: share of attention the recent window queries would pay to the
        evicted (offloaded) entries if they were still resident, mean over heads, queries and
        every ``forget_stride``-th layer. None when nothing has been offloaded."""
        from .query_capture import evicted_attention_share
        shares = []
        for i in range(0, len(self.store), self.forget_stride):
            chunks = self.store[i]
            q = self.capture.buf.get(i)
            if not chunks or q is None:
                continue
            k = keys_fn(i)
            key = (len(chunks), id(chunks[-1]))
            hit = self._ev_keys.get(i)
            if hit is None or hit[0] != key:
                ev = torch.cat([c[1] for c in chunks], dim=-2).to(k.device, non_blocking=True)
                self._ev_keys[i] = hit = (key, ev)
            shares.append(evicted_attention_share(q, k, hit[1], self.modules[i].scaling))
        return float(sum(shares) / len(shares)) if shares else None

    def _joint(self, keys_fn):
        """Per measured layer: (resident attention (B,Hkv,T), evicted positions (B,Hkv,E),
        evicted attention (B,Hkv,E)) of the current window queries."""
        from .query_capture import joint_attention
        for i in range(0, len(self.store), self.forget_stride):
            chunks = self.store[i]
            q = self.capture.buf.get(i)
            if not chunks or q is None:
                continue
            k = keys_fn(i)
            sp = torch.cat([c[0] for c in chunks], dim=-1).to(k.device)
            ek = torch.cat([c[1] for c in chunks], dim=-2).to(k.device, non_blocking=True)
            pr, pe = joint_attention(q, k, ek, self.modules[i].scaling)
            yield i, pr, sp, pe

    @torch.no_grad()
    def query_positions(self, keys_fn, cut: int, n: int) -> list[int]:
        """The ``n`` evicted positions below ``cut`` with the most attention from the current
        window queries (summed over measured layers and KV heads)."""
        pos, mass = [], []
        for _, _, sp, pe in self._joint(keys_fn):
            pos.append(sp.flatten())
            mass.append(pe.flatten())
        if not pos or n <= 0:
            return []
        pos, mass = torch.cat(pos), torch.cat(mass)
        ok = pos < cut
        pos, mass = pos[ok], mass[ok]
        if not len(pos):
            return []
        u, inv = torch.unique(pos, return_inverse=True)
        tot = torch.zeros(len(u), device=mass.device).index_add_(0, inv, mass)
        top = tot.topk(min(n, len(u))).indices
        return sorted(u[top].tolist())

    @torch.no_grad()
    def attention_demand(self, keys_fn) -> int | None:
        """Number of decoded entries (resident or evicted) covering ``size_cover`` of the attention
        the window queries pay to decoded entries; mean over measured layers and KV heads."""
        P = self.prompt_len
        need = []
        for i, pr, _, pe in self._joint(keys_fn):
            m = torch.cat([pr[..., P:], pe], -1)  # decoded entries only (incl. summaries)
            m = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
            cs = m.sort(-1, descending=True).values.cumsum(-1)
            need.append((cs < self.size_cover).sum(-1).float().add(1).mean())
        return int(torch.stack(need).mean()) if need else None

    def _cap_pins(self):
        """Unpin the oldest pinned positions beyond ``pin_frac * budget`` (same choice in every head)."""
        cap = max(1, int(self.budget * self.pin_frac))
        for i, st in enumerate(self.state):
            pinv = st.get("pin")
            if pinv is None:
                continue
            n = pinv.sum(-1)  # (B, H)
            excess = int((n - cap).clamp_min(0).max())
            if excess == 0:
                continue
            rank = pinv.long().cumsum(-1)  # 1-based rank of each pinned entry by position
            st["pin"] = pinv & (rank > (n - cap).clamp_min(0).unsqueeze(-1))

    def _tick(self) -> float:
        if not self.profile:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _offload(self, i, kd, vd, pos, keep_idx):
        keep = torch.zeros_like(pos, dtype=torch.bool).scatter_(-1, keep_idx, True)
        n_evict = pos.shape[-1] - keep_idx.shape[-1]
        ev = keep.to(torch.int8).argsort(dim=-1, stable=True)[..., :n_evict]  # evicted first, in order
        gev = ev.unsqueeze(-1).expand(-1, -1, -1, kd.shape[-1])
        new = tuple(_to_host(t) for t in (pos.gather(-1, ev), kd.gather(2, gev), vd.gather(2, gev)))
        self.bytes_offloaded += 2 * new[1].numel() * new[1].element_size()
        # chunks are concatenated only on recall: no per-compression copy of the whole store
        self.store[i] = [new] if self.store[i] is None else self.store[i] + [new]

    def offloaded_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self.store if s is not None for c in s for t in c[1:])

    @torch.no_grad()
    def rollback(self, cache, cut: int, recall: bool = True) -> None:
        """Drop every entry at position >= cut and restore all offloaded entries below it.

        With ``recall=False`` (ablation) the offloaded entries are discarded instead; heads
        may then keep different numbers of entries below ``cut``, so every head is trimmed
        to the smallest count, keeping its most recent ones.
        """
        t0 = self._tick()
        P, ns = self.prompt_len, self.n_sum
        L = num_layers(cache)
        # summaries: dropped when the exact history comes back, otherwise the leading ones
        # whose representative position precedes the cut in every layer and head survive
        ks = 0
        if ns and not recall:
            ok = torch.stack([(self.pos[i][..., :ns] < cut).all(0).all(0) for i in range(L)]).all(0)
            ks = int(ok.long().cumprod(0).sum())
        sums = []
        merged = []
        pinned = []  # pinned positions (pins survive a rollback; they are kept first when trimming)
        for i in range(L):
            pv = self.state[i].get("pin") if self.state else None
            p_n = self.pos[i][..., ns:]
            pinned.append(p_n[..., :pv.shape[-1]][pv].unique() if pv is not None and pv.any() else None)
        for i in range(L):
            k, v = get_kv(cache, i)
            sums.append((k[:, :, P:P + ks], v[:, :, P:P + ks], self.pos[i][..., :ks],
                         None if self._sum_cnt[i] is None else self._sum_cnt[i][..., :ks]))
            pos, kd, vd = self.pos[i][..., ns:], k[:, :, P + ns:], v[:, :, P + ns:]
            if recall and self.store[i] is not None:
                chunks = self.store[i]
                sp, sk, sv = (torch.cat([c[j] for c in chunks], dim=-2 if j else -1).to(k.device, non_blocking=True)
                              for j in range(3))
                self.n_recalled += int((sp < cut).sum())
                self.bytes_recalled += sk.numel() * sk.element_size() + sv.numel() * sv.element_size()
                pos, kd, vd = torch.cat([pos, sp], -1), torch.cat([kd, sk], 2), torch.cat([vd, sv], 2)
            merged.append((pos, kd, vd))
        # all layers must end up with the same length, so trim to the global minimum
        # restore() may have brought back entries that are still in the store: count each once
        merged = [(pos, kd, vd, (pos < cut) & ~_dup_mask(pos)) for pos, kd, vd in merged]
        n = min(int(valid.sum(-1).min()) for _, _, _, valid in merged)
        for i, (pos, kd, vd, valid) in enumerate(merged):
            k, v = get_kv(cache, i)
            pin = torch.isin(pos, pinned[i]) & valid if pinned[i] is not None else None
            key = pos if pin is None else pos + pin.long() * (1 << 40)  # pinned sort last = kept first
            key = torch.where(valid, key, torch.full_like(pos, torch.iinfo(pos.dtype).max))
            order = key.argsort(dim=-1)
            counts = valid.sum(-1, keepdim=True)  # (B, H, 1)
            idx = order.gather(-1, counts - n + torch.arange(n, device=k.device))
            idx = idx.gather(-1, pos.gather(-1, idx).argsort(-1))  # back to position order
            gidx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            s_k, s_v, s_p, s_c = sums[i]
            set_kv(cache, i, torch.cat([k[:, :, :P], s_k, kd.gather(2, gidx)], dim=2),
                   torch.cat([v[:, :, :P], s_v, vd.gather(2, gidx)], dim=2))
            self.pos[i] = torch.cat([s_p, pos.gather(-1, idx)], -1)
            self._sum_cnt[i] = s_c
            # without recall the evicted history before the cut stays evicted: keep those chunks
            # (forgetting signal, aligned restore); with recall it is back in the cache
            keep_chunks = [c for c in self.store[i] if bool((c[0] < cut).all())] if (not recall and self.store[i]) else []
            self.store[i] = keep_chunks or None
            self.state[i] = {}
            if pin is not None:
                self.state[i]["pin"] = pin.gather(-1, idx)
        self.n_sum = ks
        self.capture.reset()
        self.t_rollback += self._tick() - t0 if self.profile else 0.0


def _dup_mask(pos: torch.Tensor) -> torch.Tensor:
    """True for every entry whose position already occurs earlier in the same head."""
    order = pos.argsort(dim=-1, stable=True)
    sp = pos.gather(-1, order)
    dup_sorted = torch.zeros_like(sp, dtype=torch.bool)
    dup_sorted[..., 1:] = sp[..., 1:] == sp[..., :-1]
    return torch.zeros_like(dup_sorted).scatter_(-1, order, dup_sorted)


def _dup_of(cand: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Per head: which ``cand`` positions are already present in ``pos`` (or repeat within cand)."""
    both = torch.cat([pos, cand], -1)  # stable sort: an entry already in ``pos`` comes first
    return _dup_mask(both)[..., pos.shape[-1]:]


def _to_host(t: torch.Tensor) -> torch.Tensor:
    """GPU -> pinned CPU copy without a host sync (stream order makes later reads safe)."""
    if not t.is_cuda:
        return t.clone()
    out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    out.copy_(t, non_blocking=True)
    return out
