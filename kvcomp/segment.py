"""Online segmentation of a chain of thought into reasoning steps.

R1-style models separate reasoning steps with blank lines ("\\n\\n") and often open
a reflection step with a cue word ("Wait", "Alternatively", ...). A token whose
text contains a delimiter closes the current step; the next token opens a new one.
"""

import torch

DEFAULT_DELIMITERS = ("\n\n",)
REFLECTION_CUES = ("Wait", "But wait", "Alternatively", "Hmm", "Hold on", "Let me check", "Let me double")


def delimiter_token_ids(tokenizer, delimiters=DEFAULT_DELIMITERS) -> set[int]:
    ids = set()
    for tid in range(len(tokenizer)):
        text = tokenizer.decode([tid])
        if any(d in text for d in delimiters):
            ids.add(tid)
    return ids


class StepTracker:
    """Assigns a step id to every absolute position. Prompt positions get step 0.

    Step ids live in a Python list (one append per decoded token, no GPU work); the
    device copy needed by ``steps`` is rebuilt only when it is stale, i.e. at most once
    per compression.
    """

    def __init__(self, delimiter_ids: set[int], max_len: int, device):
        self.delimiter_ids = delimiter_ids
        self.device = device
        self.ids: list[int] = []
        self.cur_step = 1
        self.boundary = False  # True right after a step was closed
        self._dev = None  # cached device tensor of ``ids``
        self._dev_len = -1
        self._version = 0
        self._dev_version = -1

    @property
    def len(self) -> int:
        return len(self.ids)

    def add_prompt(self, n: int):
        self.ids = [0] * n
        self._version += 1

    def add_token(self, token_id: int) -> bool:
        self.ids.append(self.cur_step)
        self.boundary = token_id in self.delimiter_ids
        if self.boundary:
            self.cur_step += 1
        return self.boundary

    def steps(self, pos: torch.Tensor) -> torch.Tensor:
        if self._dev is None or self._dev_len != len(self.ids) or self._dev_version != self._version:
            self._dev = torch.tensor(self.ids, dtype=torch.long, device=pos.device)
            self._dev_len, self._dev_version = len(self.ids), self._version
        return self._dev[pos]

    def step_start(self, pos: int) -> int:
        """First absolute position of the step that contains ``pos``."""
        s, i = self.ids[pos], pos
        while i > 0 and self.ids[i - 1] == s:
            i -= 1
        return i

    def truncate(self, length: int, tokens: list[int]) -> None:
        """Forget every position >= length (``tokens`` = generated tokens kept)."""
        del self.ids[length:]
        self._version += 1
        self.cur_step = (self.ids[length - 1] if tokens else 0) + 1
        if tokens and tokens[-1] not in self.delimiter_ids:
            self.cur_step -= 1
        self.boundary = False


def split_steps(text: str, delimiters=DEFAULT_DELIMITERS) -> list[str]:
    """Offline counterpart used by the analysis scripts."""
    parts = [text]
    for d in delimiters:
        parts = [p for chunk in parts for p in chunk.split(d)]
    return [p for p in parts if p.strip()]


def is_reflection(step_text: str) -> bool:
    s = step_text.lstrip()
    return any(s.startswith(c) for c in REFLECTION_CUES)
