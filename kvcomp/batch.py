"""Continuous-batching version of ``decode.generate``.

Batch-1 decoding of a 1.5B model is bound by Python and kernel-launch overhead, so
the GPU sits mostly idle. Here up to ``batch_size`` problems decode together:

* ``SlotCache`` keeps one padded K/V buffer per layer, shape (B, H, capacity, D),
  plus the number of valid entries of every slot. A decode step writes each new
  entry at its slot's own length and attends through an explicit padding mask, so
  slots with different prompt lengths, eviction histories and rollbacks coexist.
* Everything that touches one sequence only (prefill, compression, rollback with
  re-feeding) runs on a batch-1 ``DynamicCache`` built from that slot and is written
  back afterwards. The per-sequence logic (compressor, detector, step tracker,
  rollback rule) is therefore the same code path as ``decode.generate``.
* A finished slot is refilled with the next problem at once (continuous batching).

The query ring buffer for scoring is shared: one ``QueryCapture`` hooks the model and
every compressor reads its own row through ``_CaptureRow``.
"""

import time
from types import SimpleNamespace

import torch
from transformers import DynamicCache

from .cache_utils import get_kv, num_layers
from .decode import _forward, sample_rows
from .query_capture import QueryCapture
from .segment import StepTracker


class SlotCache:
    """Padded per-slot KV storage that HF attention layers can append to."""

    def __init__(self, n_layers: int, batch: int, capacity: int):
        self.n_layers, self.batch, self.capacity = n_layers, batch, capacity
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self.lens = [0] * batch
        self.write_at: torch.Tensor | None = None  # (B,) slot lengths during a decode step
        self.span = 0  # number of entries visible to the current step
        # optional per-step attention observer (LazyEviction): last re-attended step and
        # maximum recurrence interval of every entry, aligned with k/v
        self.obs_alpha: float | None = None
        self.obs_query = None  # layer -> (B, Hq, D) query of the token being decoded
        self.obs_scaling = 1.0
        self.obs_step: torch.Tensor | None = None  # (B,) decoding step of every slot
        self.obs_live: torch.Tensor | None = None  # (B,) slots that decode this step
        self.last: list[torch.Tensor | None] = [None] * n_layers
        self.mp: list[torch.Tensor | None] = [None] * n_layers

    def _alloc(self, like: torch.Tensor):
        _, h, _, d = like.shape
        for i in range(self.n_layers):
            if self.k[i] is None:
                self.k[i] = like.new_zeros(self.batch, h, self.capacity, d)
                self.v[i] = like.new_zeros(self.batch, h, self.capacity, d)
                if self.obs_alpha is not None:
                    self.last[i] = torch.zeros(self.batch, h, self.capacity, device=like.device)
                    self.mp[i] = torch.zeros(self.batch, h, self.capacity, device=like.device)

    def ensure(self, need: int):
        """Grow every buffer to hold at least ``need`` entries per slot."""
        if need <= self.capacity:
            return
        grow = max(need, self.capacity + 1024) - self.capacity
        for i in range(self.n_layers):
            if self.k[i] is not None:
                pad = self.k[i].new_zeros(*self.k[i].shape[:2], grow, self.k[i].shape[-1])
                self.k[i] = torch.cat([self.k[i], pad], 2)
                self.v[i] = torch.cat([self.v[i], pad.clone()], 2)
                if self.last[i] is not None:
                    z = self.last[i].new_zeros(*self.last[i].shape[:2], grow)
                    self.last[i] = torch.cat([self.last[i], z], 2)
                    self.mp[i] = torch.cat([self.mp[i], z.clone()], 2)
        self.capacity += grow

    # --- HF Cache protocol (decode step only) ---
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        rows = torch.arange(self.batch, device=key_states.device)
        self.k[layer_idx][rows, :, self.write_at] = key_states[:, :, 0]
        self.v[layer_idx][rows, :, self.write_at] = value_states[:, :, 0]
        keys = self.k[layer_idx][:, :, : self.span]
        if self.obs_alpha is not None:
            self._observe(layer_idx, keys, rows)
        return keys, self.v[layer_idx][:, :, : self.span]

    def _observe(self, i: int, keys: torch.Tensor, rows: torch.Tensor):
        """Record which entries the current token re-attends to (LazyEviction recurrence tracking)."""
        q = self.obs_query(i)  # (B, Hq, D)
        b, hkv, span, d = keys.shape
        qg = q.float().reshape(b, hkv, -1, d)  # GQA groups
        logits = torch.matmul(qg, keys.float().transpose(-1, -2)) * self.obs_scaling  # (B, Hkv, rep, span)
        valid = torch.arange(span, device=keys.device)[None, :] <= self.write_at[:, None]  # (B, span)
        probs = logits.masked_fill(~valid[:, None, None, :], float("-inf")).softmax(-1).amax(2)
        step = self.obs_step[:, None, None]
        last, mp = self.last[i][:, :, :span], self.mp[i][:, :, :span]
        # a new entry starts as "last seen one step ago" (official TempCache padding)
        last[rows, :, self.write_at] = (self.obs_step - 1)[:, None].expand(-1, hkv)
        mp[rows, :, self.write_at] = 0
        hit = (probs >= self.obs_alpha) & valid[:, None, :] & self.obs_live[:, None, None]
        new_last = torch.where(hit, step, last)
        torch.maximum(mp, new_last - last, out=mp)
        last.copy_(new_last)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.span

    # --- per-slot transfer ---
    def to_dynamic(self, b: int) -> DynamicCache:
        cache = DynamicCache()
        n = self.lens[b]
        for i in range(self.n_layers):
            cache.update(self.k[i][b: b + 1, :, :n].clone(), self.v[i][b: b + 1, :, :n].clone(), i)
        return cache

    def from_dynamic(self, b: int, cache) -> None:
        k0, _ = get_kv(cache, 0)
        n = k0.shape[-2]
        self._alloc(k0)
        self.ensure(n + 1)
        for i in range(num_layers(cache)):
            k, v = get_kv(cache, i)
            self.k[i][b, :, :n] = k[0]
            self.v[i][b, :, :n] = v[0]
        self.lens[b] = n


class _CaptureRow:
    """The view of a shared ``QueryCapture`` that one slot's compressor sees."""

    def __init__(self, shared: QueryCapture, row: int):
        self.shared, self.row = shared, row

    @property
    def buf(self):
        return {i: q[self.row: self.row + 1] for i, q in self.shared.buf.items()}

    def reset(self):  # the slot's rows are rewritten by the next single-sequence forward
        pass


class _Seq:
    """Per-problem decoding state (mirrors the locals of ``decode.generate``)."""

    def __init__(self, idx, prompt_ids, compressor, detector, delimiter_ids, max_new_tokens, seed,
                 temperature, device):
        self.idx = idx
        self.prompt = prompt_ids[0].tolist()
        self.P = len(self.prompt)
        self.compressor, self.detector = compressor, detector
        self.tracker = StepTracker(delimiter_ids, self.P + max_new_tokens + 1, device)
        self.tracker.add_prompt(self.P)
        self.generator = None
        if seed is not None and temperature > 0:
            self.generator = torch.Generator(device=device).manual_seed(seed)
        self.generated: list[int] = []
        self.rollbacks: list[dict] = []
        self.computed = 0
        self.max_cache = self.max_offload = 0
        self.logits = None  # next-token logits waiting to be sampled
        self.done = False  # finished by a helper-written EOS
        self.helper = None
        self.t0 = time.perf_counter()


@torch.no_grad()
def generate_batch(model, prompts, *, make_compressor=None, make_detector=None, delimiter_ids=frozenset(),
                   batch_size: int = 16, max_new_tokens: int = 4096, temperature: float = 0.0,
                   top_p: float = 1.0, eos_ids=(), seeds=None, max_rollbacks: int = 4, refeed: int = 16,
                   max_rewind: int = 512, recall: bool = True, escalate: float = 1.0, max_budget: int = 4096,
                   window: int = 16, observe_alpha: float | None = None, debug: dict | None = None,
                   make_helper=None, helper_max_step: int = 256):
    """Decode every prompt in ``prompts`` (each (1, P)) and yield ``(index, result)`` as each finishes.

    ``make_compressor(capture)`` / ``make_detector()`` build fresh per-problem objects
    (or are None for full KV); ``make_helper()`` returns a ``rescue.HelperState`` for the
    KV-Rescue-lite baseline. Results have the same fields as ``decode.generate``.
    """
    device = model.device
    delimiter_ids = set(delimiter_ids)
    eos_ids = set(eos_ids)
    n_layers = model.config.num_hidden_layers
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    bytes_per_entry = n_layers * 2 * cfg.num_key_value_heads * head_dim * model.dtype.itemsize
    B = min(batch_size, len(prompts))
    shared = QueryCapture(model, window) if make_compressor is not None else None
    cap0 = max(p.shape[1] for p in prompts) + 1024
    slots = SlotCache(n_layers, B, cap0)
    if debug is not None:
        debug["slots"] = slots
    if observe_alpha is not None:
        slots.obs_alpha = observe_alpha
        slots.obs_query = lambda i: shared.buf[i][:, :, -1]
        slots.obs_scaling = model.model.layers[0].self_attn.scaling
        slots.obs_step = torch.zeros(B, device=device)
    seqs: list[_Seq | None] = [None] * B
    queue = list(range(len(prompts)))

    def single_forward(b, cache, tokens, start):
        """Batch-1 forward for slot ``b``; its new window queries go into the shared buffer row."""
        saved = None
        if shared is not None:
            saved, shared.buf = shared.buf, {}
        out = _forward(model, cache, tokens, start)
        if shared is not None:
            fresh = shared.buf
            if not saved:  # first forward of the run: allocate the shared buffer
                saved = {i: q.new_zeros(B, *q.shape[1:2], window, q.shape[-1]) for i, q in fresh.items()}
            for i, q in fresh.items():
                n = q.shape[2]
                saved[i][b, :, window - n:] = q[0]
                saved[i][b, :, : window - n] = 0
            shared.buf = saved
        return out

    def start(b):
        idx = queue.pop(0)
        comp = make_compressor(_CaptureRow(shared, b)) if make_compressor is not None else None
        det = make_detector() if make_detector is not None else None
        if det is not None and comp is None:
            raise ValueError("degeneration handling needs a compressor")
        if det is not None and make_helper is None and not comp.offload:
            raise ValueError("rollback needs a compressor with offload=True")
        s = _Seq(idx, prompts[idx], comp, det, delimiter_ids, max_new_tokens,
                 None if seeds is None else seeds[idx], temperature, device)
        if make_helper is not None:
            s.helper = make_helper()
        if comp is not None:
            comp.reset()
        if det is not None:
            det.reset()
        cache = DynamicCache()
        out = single_forward(b, cache, s.prompt, 0)
        if comp is not None:
            comp.on_prefill(cache, s.P)
        slots.from_dynamic(b, cache)
        if observe_alpha is not None:  # official: counters restart at prefill, state starts at zero
            slots.obs_step[b] = 0
            for i in range(n_layers):
                slots.last[i][b] = 0
                slots.mp[i][b] = 0
        s.max_cache = s.P
        s.logits = out.logits[0, -1]
        seqs[b] = s

    def finish(b):
        s = seqs[b]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - s.t0
        c = s.compressor
        res = {
            "gen_ids": s.generated, "num_tokens": len(s.generated), "computed_tokens": s.computed,
            "finished": bool(s.generated) and s.generated[-1] in eos_ids, "num_steps": s.tracker.cur_step,
            "max_cache_len": s.max_cache, "max_cache_bytes": s.max_cache * bytes_per_entry,
            "max_offload_bytes": s.max_offload,
            "n_compress": c.n_compress if c is not None else 0,
            "n_recalled": c.n_recalled if c is not None else 0,
        **({"bytes_offloaded": c.bytes_offloaded, "bytes_recalled": c.bytes_recalled,
            "t_compress": c.t_compress, "t_rollback": c.t_rollback} if c is not None else {}),
            "final_budget": c.budget if c is not None else None,
            "helper_tokens": s.helper.generated if s.helper is not None else 0,
            "helper_max_cache_len": s.helper.max_cache if s.helper is not None else 0,
            "rollbacks": s.rollbacks, "seconds": elapsed, "tok_per_s": s.computed / max(elapsed, 1e-9),
            "batch_size": B,
        }
        seqs[b] = None
        return s.idx, res

    def lazy_to_state(b, s):
        """Hand the observer's per-entry history of slot ``b`` to its compressor (decoded part only)."""
        if observe_alpha is None:
            return
        n = slots.lens[b]
        for i in range(n_layers):
            st = s.compressor.state[i]
            st["lazy_last"] = slots.last[i][b: b + 1, :, s.P:n].clone()
            st["lazy_mp"] = slots.mp[i][b: b + 1, :, s.P:n].clone()
            st["lazy_step"] = float(slots.obs_step[b])

    def lazy_from_state(b, s):
        """Write the (re-selected) history back next to the entries it belongs to."""
        if observe_alpha is None:
            return
        n = slots.lens[b]
        for i in range(n_layers):
            st = s.compressor.state[i]
            if "lazy_last" in st and st["lazy_last"].shape[-1] == n - s.P:
                slots.last[i][b, :, s.P:n] = st["lazy_last"][0]
                slots.mp[i][b, :, s.P:n] = st["lazy_mp"][0]

    def after_token(b, s, tok, pos, at_boundary):
        """Compression / detection / rollback for slot ``b`` once ``tok`` is cached (cf. decode.generate)."""
        comp, det = s.compressor, s.detector
        if comp is not None:
            comp.append([pos])
            if comp.due(at_boundary):
                cache = slots.to_dynamic(b)
                lazy_to_state(b, s)
                comp.compress(cache, s.tracker.steps)
                slots.from_dynamic(b, cache)
                lazy_from_state(b, s)
                if comp.offload:
                    s.max_offload = max(s.max_offload, comp.offloaded_bytes())
        s.max_cache = max(s.max_cache, slots.lens[b])
        if det is None or len(s.rollbacks) >= max_rollbacks:
            return
        loop_start = det.update(s.generated, at_boundary)
        if loop_start is None or comp.n_compress == 0:
            return
        P = s.P
        cut = s.tracker.step_start(P + loop_start)
        if P + loop_start - cut > max_rewind:
            cut = P + loop_start
        cut = max(cut, P + 1)
        s.rollbacks.append({"at": len(s.generated), "loop_start": loop_start, "cut": cut - P})
        if s.helper is not None:
            from .rescue import takeover
            cache = slots.to_dynamic(b)
            s.generated, new, logits = takeover(
                model, cache, comp, det, s.tracker, s.helper, s.generated, P, s.prompt, cut,
                delimiter_ids=delimiter_ids, eos_ids=eos_ids, max_step=helper_max_step,
                temperature=temperature, top_p=top_p, generator=s.generator,
                forward=lambda c, t, st: single_forward(b, c, t, st))
            slots.from_dynamic(b, cache)
            s.rollbacks[-1]["helper_tokens"] = len(new)
            s.max_cache = max(s.max_cache, slots.lens[b])
            s.logits, s.done = logits, logits is None
            return
        s.generated = s.generated[: cut - P]
        s.tracker.truncate(cut, s.generated)
        det.truncate(s.generated, lambda t: t in delimiter_ids)
        c0 = max(P, cut - refeed)
        cache = slots.to_dynamic(b)
        comp.rollback(cache, c0, recall=recall)
        comp.budget = min(max_budget, max(comp.budget, int(comp.budget * escalate)))
        s.rollbacks[-1]["budget"] = comp.budget
        out = single_forward(b, cache, s.generated[c0 - P:], c0)
        s.rollbacks[-1]["transient_cache_len"] = get_kv(cache, 0)[0].shape[-2]
        comp.append(list(range(c0, cut)))
        if observe_alpha is not None:  # recalled entries have no history: restart them as just seen
            for i in range(n_layers):
                comp.state[i]["lazy_last"] = torch.full(comp.pos[i].shape, float(slots.obs_step[b]), device=device)
                comp.state[i]["lazy_mp"] = torch.zeros(comp.pos[i].shape, device=device)
                comp.state[i]["lazy_step"] = float(slots.obs_step[b])
        if comp.due(at_boundary=True):
            comp.compress(cache, s.tracker.steps)
        slots.from_dynamic(b, cache)
        lazy_from_state(b, s)
        s.max_cache = max(s.max_cache, slots.lens[b])
        s.logits = out.logits[0, -1]

    for b in range(B):
        if queue:
            start(b)

    while any(s is not None for s in seqs):
        # 1. retire finished slots and refill them; then sample every slot with pending logits
        step_tok = [None] * B
        for b in range(B):
            while seqs[b] is not None and (seqs[b].done or len(seqs[b].generated) >= max_new_tokens):
                yield finish(b)
                if queue:
                    start(b)
        ready = [b for b in range(B) if seqs[b] is not None and seqs[b].logits is not None]
        if ready:
            toks = sample_rows(torch.stack([seqs[b].logits for b in ready]), temperature, top_p,
                               [seqs[b].generator for b in ready])
        for b, tok in zip(ready, toks if ready else []):
            s = seqs[b]
            s.logits = None
            s.generated.append(tok)
            s.computed += 1
            if tok in eos_ids:
                yield finish(b)
                if queue:
                    start(b)  # the new slot has logits; it joins the next step
                continue
            step_tok[b] = tok
        live = [b for b in range(B) if step_tok[b] is not None]
        if not live:
            continue

        # 2. one batched decode step for the slots that sampled a token
        boundary = {}
        for b in live:
            boundary[b] = seqs[b].tracker.add_token(step_tok[b])
        lens = torch.tensor(slots.lens, device=device)
        slots.ensure(int(lens.max()) + 1)
        slots.write_at, slots.span = lens, int(lens.max()) + 1
        mask = torch.arange(slots.span, device=device)[None, :] <= lens[:, None]  # (B, span)
        pos = [seqs[b].P + len(seqs[b].generated) - 1 if step_tok[b] is not None else 0 for b in range(B)]
        ids = torch.tensor([[t if t is not None else 0] for t in step_tok], device=device)
        if observe_alpha is not None:
            live_mask = torch.zeros(B, dtype=torch.bool, device=device)
            live_mask[live] = True
            slots.obs_step += live_mask.float()
            slots.obs_live = live_mask
        prev_buf = dict(shared.buf) if shared is not None else None
        out = model(input_ids=ids, past_key_values=slots, use_cache=True,
                    position_ids=torch.tensor(pos, device=device)[:, None],
                    cache_position=lens.clone()[:1],
                    attention_mask={"full_attention": mask[:, None, None, :]})
        idle = [b for b in range(B) if step_tok[b] is None]
        if shared is not None and idle and prev_buf:
            rows = torch.tensor(idle, device=device)
            for i, q in shared.buf.items():  # idle slots must not shift their query window
                q[rows] = prev_buf[i][rows]
        for b in live:
            slots.lens[b] += 1
            s = seqs[b]
            s.logits = out.logits[b, -1]
            after_token(b, s, step_tok[b], pos[b], boundary[b])

    if shared is not None:
        shared.remove()
