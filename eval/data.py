"""Benchmark loaders returning a list of {"id", "question", "answer"} dicts."""

import random

from datasets import load_dataset

PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def _gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"id": f"gsm8k-{i}", "question": r["question"], "answer": r["answer"].split("####")[-1].strip()}
            for i, r in enumerate(ds)]


def _math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"id": r["unique_id"], "question": r["problem"], "answer": r["answer"], "level": r["level"]}
            for r in ds]


def _aime24():
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [{"id": f"aime24-{r['ID']}", "question": r["Problem"], "answer": str(r["Answer"])} for r in ds]


def _math500dev():
    """Tuning split: MATH-500 problems outside the 200-problem main sample (``load(n=200, seed=0)``),
    in a fixed shuffled order, so hyper-parameters are never chosen on the reported problems."""
    items = _math500()
    main = {r["id"] for r in random.Random(0).sample(items, 200)}
    rest = [r for r in items if r["id"] not in main]
    random.Random(1).shuffle(rest)
    return rest


LOADERS = {"gsm8k": _gsm8k, "math500": _math500, "aime24": _aime24, "math500dev": _math500dev}


def load(name: str, n: int | None = None, seed: int = 0) -> list[dict]:
    items = LOADERS[name]()
    if name.endswith("dev"):  # already a fixed shuffled order: take the first n
        return items[:n]
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def build_prompt(tokenizer, question: str):
    messages = [{"role": "user", "content": PROMPT.format(question=question)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
