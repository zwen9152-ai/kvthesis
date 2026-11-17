"""Online detection of degenerate decoding (repetition loops) in a chain of thought.

Cheap signals, all computed from token ids only:
  * periodic loop: the last ``span`` tokens repeat with some period p <= max_period;
  * step loop: a reasoning step (>= ``min_step_len`` tokens) repeats an earlier step for
    the ``step_repeats``-th time. With ``fuzzy=0`` a repeat must be an exact copy; with
    ``fuzzy=θ`` (RewindKV v2) a step repeats the earlier step whose 4-gram set has the
    highest Jaccard similarity, if that similarity is >= θ, so paraphrased re-derivations
    are caught too;
  * hesitation (v2, ``hesitate=k``, needs ``cue_ids``): k of the last ``hesitate_window``
    steps open with a reflection cue ("Wait", "Hmm", ...), i.e. the model keeps doubting
    itself without settling. This catches non-convergence that never repeats a step.
On detection, ``update`` returns the index (into the generated tokens) where the
loop began, so the caller can roll back to the start of that reasoning step;
``last_origin`` then holds the (start, end) generated-token span of the first copy of
the repeated step (None for periodic and hesitation triggers) and ``last_kind`` the signal.
"""

from collections import Counter, deque


def _ngrams(step: tuple, n: int = 4) -> frozenset:
    if len(step) < n:
        return frozenset([step])
    return frozenset(step[i:i + n] for i in range(len(step) - n + 1))


class DegenerationDetector:
    def __init__(self, max_period: int = 64, min_repeats: int = 3, min_span: int = 96,
                 check_every: int = 16, min_step_len: int = 8, step_repeats: int = 3,
                 fuzzy: float = 0.0, hesitate: int = 0, hesitate_window: int = 8, cue_ids=()):
        self.max_period = max_period
        self.min_repeats = min_repeats
        self.min_span = min_span
        self.check_every = check_every
        self.min_step_len = min_step_len
        self.step_repeats = step_repeats
        self.fuzzy = float(fuzzy)
        self.hesitate = int(hesitate)
        self.hesitate_window = hesitate_window
        self.cue_ids = set(cue_ids)
        self.reset()

    def reset(self):
        self.step_start = 0  # index of the first token of the current step
        self.step_counts: Counter = Counter()  # cluster id -> occurrences
        self.step_first: dict = {}  # cluster id -> index of its 2nd occurrence
        self.origin: dict = {}  # cluster id -> (start, end) of its first occurrence
        self.exact: dict = {}  # step content -> cluster id
        self.grams: list = []  # (cluster id, 4-gram set) of every long step (fuzzy mode)
        self.recent = deque(maxlen=self.hesitate_window)  # (start, opens with a cue) of recent steps
        self.since_check = 0
        self.last_origin = None
        self.last_kind = None

    def truncate(self, tokens: list[int], is_boundary) -> None:
        """Rebuild state after the generated tokens were cut back by a rollback."""
        self.reset()
        start = 0
        for i, t in enumerate(tokens):
            if is_boundary(t):
                self._close_step(tokens, start, i + 1)
                start = i + 1
        self.step_start = start
        self.recent.clear()  # the hesitation window restarts after a rollback

    def _cluster(self, step: tuple) -> int:
        cid = self.exact.get(step)
        if cid is not None or not self.fuzzy:
            if cid is None:
                cid = self.exact[step] = len(self.exact)
            return cid
        g = _ngrams(step)
        best, best_sim = None, self.fuzzy
        for c, h in self.grams:
            inter = len(g & h)
            if inter == 0:
                continue
            sim = inter / (len(g) + len(h) - inter)
            if sim >= best_sim:
                best, best_sim = c, sim
        cid = best if best is not None else len(self.exact) + len(self.grams) + 1_000_000
        self.exact[step] = cid
        self.grams.append((cid, g))
        return cid

    def _close_step(self, tokens, start, end):
        step = tuple(tokens[start:end])
        if self.hesitate:
            self.recent.append((start, bool(step) and step[0] in self.cue_ids))
        if len(step) < self.min_step_len:
            return None
        cid = self._cluster(step)
        self.step_counts[cid] += 1
        if self.step_counts[cid] == 1:
            self.origin[cid] = (start, end)
        if self.step_counts[cid] == 2:
            self.step_first[cid] = start
        if self.step_counts[cid] >= self.step_repeats:
            self.last_origin, self.last_kind = self.origin[cid], "step"
            return self.step_first[cid]
        return None

    def periodic_loop_start(self, tokens: list[int]) -> int | None:
        n = len(tokens)
        for p in range(1, self.max_period + 1):
            span = max(p * self.min_repeats, self.min_span)
            if n < span:
                break
            tail = tokens[n - span:]
            if tail[:-p] == tail[p:]:
                i = n - span - 1
                while i >= 0 and tokens[i] == tokens[i + p]:
                    i -= 1
                self.last_origin, self.last_kind = None, "periodic"
                return i + 1  # first token of the first copy of the loop body
        return None

    def _hesitation_start(self) -> int | None:
        if not self.hesitate or len(self.recent) < self.hesitate_window:
            return None
        cues = [s for s, cue in self.recent if cue]
        if len(cues) < self.hesitate:
            return None
        self.recent.clear()  # one trigger per window
        self.last_origin, self.last_kind = None, "hesitate"
        return cues[0]  # roll back to the first doubtful step of the window

    def update(self, tokens: list[int], at_boundary: bool) -> int | None:
        """Call after appending a token; returns the loop start index or None."""
        if at_boundary:
            start = self._close_step(tokens, self.step_start, len(tokens))
            self.step_start = len(tokens)
            if start is not None:
                return start
            start = self._hesitation_start()
            if start is not None:
                return start
        self.since_check += 1
        if self.since_check >= self.check_every:
            self.since_check = 0
            return self.periodic_loop_start(tokens)
        return None
