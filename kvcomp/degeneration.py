"""Online detection of degenerate decoding (repetition loops) in a chain of thought.

Two cheap signals, both computed from token ids only:
  * periodic loop: the last ``span`` tokens repeat with some period p <= max_period;
  * step loop: a reasoning step (>= ``min_step_len`` tokens) is an exact copy of an
    earlier step, for the ``step_repeats``-th time.
On detection, ``update`` returns the index (into the generated tokens) where the
loop began, so the caller can roll back to the start of that reasoning step.
"""

from collections import Counter


class DegenerationDetector:
    def __init__(self, max_period: int = 64, min_repeats: int = 3, min_span: int = 96,
                 check_every: int = 16, min_step_len: int = 8, step_repeats: int = 3):
        self.max_period = max_period
        self.min_repeats = min_repeats
        self.min_span = min_span
        self.check_every = check_every
        self.min_step_len = min_step_len
        self.step_repeats = step_repeats
        self.reset()

    def reset(self):
        self.step_start = 0  # index of the first token of the current step
        self.step_counts: Counter = Counter()
        self.step_first: dict = {}  # step content -> index of its 2nd occurrence
        self.since_check = 0

    def truncate(self, tokens: list[int], is_boundary) -> None:
        """Rebuild state after the generated tokens were cut back by a rollback."""
        self.reset()
        start = 0
        for i, t in enumerate(tokens):
            if is_boundary(t):
                self._close_step(tokens, start, i + 1)
                start = i + 1
        self.step_start = start

    def _close_step(self, tokens, start, end):
        step = tuple(tokens[start:end])
        if len(step) < self.min_step_len:
            return None
        self.step_counts[step] += 1
        if self.step_counts[step] == 2:
            self.step_first[step] = start
        if self.step_counts[step] >= self.step_repeats:
            return self.step_first[step]
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
                return i + 1  # first token of the first copy of the loop body
        return None

    def update(self, tokens: list[int], at_boundary: bool) -> int | None:
        """Call after appending a token; returns the loop start index or None."""
        if at_boundary:
            start = self._close_step(tokens, self.step_start, len(tokens))
            self.step_start = len(tokens)
            if start is not None:
                return start
        self.since_check += 1
        if self.since_check >= self.check_every:
            self.since_check = 0
            return self.periodic_loop_start(tokens)
        return None
