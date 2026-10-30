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
    """Assigns a step id to every absolute position. Prompt positions get step 0."""

    def __init__(self, delimiter_ids: set[int], max_len: int, device):
        self.delimiter_ids = delimiter_ids
        self.step_of_pos = torch.zeros(max_len, dtype=torch.long, device=device)
        self.cur_step = 1
        self.len = 0
        self.boundary = False  # True right after a step was closed

    def add_prompt(self, n: int):
        self.len = n

    def add_token(self, token_id: int) -> bool:
        self.step_of_pos[self.len] = self.cur_step
        self.len += 1
        self.boundary = token_id in self.delimiter_ids
        if self.boundary:
            self.cur_step += 1
        return self.boundary

    def steps(self, pos: torch.Tensor) -> torch.Tensor:
        return self.step_of_pos[pos]

    def step_start(self, pos: int) -> int:
        """First absolute position of the step that contains ``pos``."""
        same = (self.step_of_pos[: pos + 1] == self.step_of_pos[pos]).nonzero()
        return int(same[0])

    def truncate(self, length: int, tokens: list[int]) -> None:
        """Forget every position >= length (``tokens`` = generated tokens kept)."""
        self.len = length
        self.cur_step = (int(self.step_of_pos[length - 1]) if tokens else 0) + 1
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
