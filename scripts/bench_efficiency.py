"""Chapter 5: memory and speed of every method under batched decoding.

For each (method spec, budget, batch size) decodes the same MATH-500 prompts for a fixed
number of new tokens and records: aggregate decode throughput, peak GPU memory above the
weights, peak CPU memory of the offload store, GPU<->CPU traffic of RewindKV, and (with
``profile=1`` in the spec) the time spent in compression and rollback.

    python scripts/bench_efficiency.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --specs full,snapkv,snapkv@rollback=1,escalate=2,max_budget=1024,lazy,vase,rkv \
        --budgets 128,256 --batch_sizes 1,16,32 --n 32 --max_new_tokens 4096 \
        --out results/efficiency/bench_1.5b.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import data  # noqa: E402
from eval.run_eval import parse_method, split_methods  # noqa: E402
from kvcomp import KVCompressor  # noqa: E402
from kvcomp.batch import generate_batch  # noqa: E402
from kvcomp.degeneration import DegenerationDetector  # noqa: E402
from kvcomp.segment import delimiter_token_ids  # noqa: E402


def build(model, name, cfg, budget, window, buffer):
    """Factories for generate_batch from a method spec (mirrors eval.run_eval)."""
    if name == "full":
        return None, None, {}, None, window
    opts = dict(cfg)
    observe = None
    if name == "lazy":
        w = int(opts.pop("obs", max(8, round(0.117 * budget))))
        opts.setdefault("buffer", w)
        opts.setdefault("window", w)
        observe = float(opts.pop("alpha", 1e-4))
    opts.setdefault("buffer", buffer)
    opts.setdefault("window", window)
    rb, det_opts = {}, None
    if opts.pop("rollback", 0):
        opts["offload"] = True
        rb = {k: opts.pop(k) for k in ("max_rollbacks", "refeed", "max_rewind", "recall", "escalate", "max_budget")
              if k in opts}
        det_opts = {k: opts.pop(k) for k in ("min_span", "max_period") if k in opts}

    def make_comp(capture):
        return KVCompressor(model, name, budget, capture=capture, **opts)
    make_det = (lambda: DegenerationDetector(**det_opts)) if det_opts is not None else None
    return make_comp, make_det, rb, observe, opts["window"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--budgets", default="128,256")
    ap.add_argument("--batch_sizes", default="1,16,32")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 attn_implementation="sdpa").to(device).eval()
    weights = torch.cuda.memory_allocated() if device == "cuda" else 0
    delim = delimiter_token_ids(tok)
    gen_eos = model.generation_config.eos_token_id
    eos = {tok.eos_token_id} | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    eos.discard(None)
    probs = data.load("math500", args.n, 0)
    prompts = []
    for p in probs:
        ids = data.build_prompt(tok, p["question"])
        prompts.append((ids if torch.is_tensor(ids) else ids["input_ids"]).to(device))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for spec in split_methods(args.specs):
        name, cfg = parse_method(spec)
        for budget in ([None] if name == "full" else [int(b) for b in args.budgets.split(",")]):
            for bs in [int(b) for b in args.batch_sizes.split(",")]:
                make_comp, make_det, rb, observe, window = build(model, name, cfg, budget, args.window, args.buffer)
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    res = [r for _, r in generate_batch(
                        model, prompts, make_compressor=make_comp, make_detector=make_det, delimiter_ids=delim,
                        batch_size=bs, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                        top_p=args.top_p, eos_ids=eos, seeds=list(range(len(prompts))), window=window,
                        observe_alpha=observe, **rb)]
                except torch.cuda.OutOfMemoryError:
                    print(f"{spec} budget={budget} bs={bs}: OOM", flush=True)
                    rec = {"spec": spec, "budget": budget, "batch_size": bs, "oom": True}
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    continue
                if device == "cuda":
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                computed = sum(r["computed_tokens"] for r in res)
                rec = {
                    "spec": spec, "budget": budget, "batch_size": bs, "n": len(res), "wall_s": wall,
                    "tok_per_s": computed / wall, "computed_tokens": computed,
                    "peak_gpu_bytes_over_weights": (torch.cuda.max_memory_allocated() - weights) if device == "cuda" else 0,
                    "mean_peak_kv_bytes": sum(r["max_cache_bytes"] for r in res) / len(res),
                    "max_offload_bytes": max(r["max_offload_bytes"] for r in res),
                    "bytes_offloaded": sum(r.get("bytes_offloaded", 0) for r in res),
                    "bytes_recalled": sum(r.get("bytes_recalled", 0) for r in res),
                    "t_compress": sum(r.get("t_compress", 0) for r in res),
                    "t_rollback": sum(r.get("t_rollback", 0) for r in res),
                    "rollbacks": sum(len(r["rollbacks"]) for r in res),
                    "max_transient_cache_len": max([rb_["transient_cache_len"] for r in res for rb_ in r["rollbacks"]
                                                    if "transient_cache_len" in rb_] or [0]),
                }
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{spec:50s} budget={budget} bs={bs:3d} {rec['tok_per_s']:8.1f} tok/s "
                      f"peakGPU+={rec['peak_gpu_bytes_over_weights'] / 2**30:.2f} GiB "
                      f"offload={rec['max_offload_bytes'] / 2**20:.0f} MiB recalled={rec['bytes_recalled'] / 2**20:.0f} MiB",
                      flush=True)


if __name__ == "__main__":
    main()
