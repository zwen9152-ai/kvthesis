"""Chapter-3 analysis: how reasoning models attend to their own chain of thought.

Takes full-KV generations saved by ``eval.run_eval --save_ids`` (method ``full``),
re-runs one forward pass per trace, and for sampled decoding queries measures:

  O1 step locality      attention mass by step distance (0 = current step);
                        ``recent3`` = mass on the current + two previous steps.
  O2 anchor tokens      within each completed step, share of the step's mass that
                        falls on its top-10% tokens.
  O3 abandoned branches attention later steps pay (per token) to a step that is
                        followed by a reflection step ("Wait, ...") vs. other steps.

Example:
    python -m analysis.attn_stats --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --runs results/math500_small.jsonl --n 20 --out results/attn_stats
"""

import argparse
import json
import math
import os
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvcomp.query_capture import _attention_inputs, attention_modules, project
from kvcomp.segment import delimiter_token_ids, is_reflection

MAX_DIST = 8  # distance histogram bins 0..MAX_DIST-1, plus ">=MAX_DIST" and "prompt"


def step_ids(gen_ids, delim, prompt_len):
    steps, cur = [0] * prompt_len, 1
    for t in gen_ids:
        steps.append(cur)
        if t in delim:
            cur += 1
    return torch.tensor(steps)


class LayerProbe:
    def __init__(self, model, stride):
        self.stride = stride
        self.ctx = None
        self.results = {}
        self.handles = [m.register_forward_pre_hook(partial(self._hook, i), with_kwargs=True)
                        for i, m in enumerate(attention_modules(model))]

    @torch.no_grad()
    def _hook(self, layer, module, args, kwargs):
        hidden, cos, sin = _attention_inputs(args, kwargs)
        q = project(module, hidden, cos, sin, "q")[0]  # (Hq, T, D)
        k = project(module, hidden, cos, sin, "k")[0]  # (Hkv, T, D)
        k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
        steps, P, qidx = self.ctx["steps"], self.ctx["prompt_len"], self.ctx["qidx"]
        T = steps.shape[0]
        logits = torch.matmul(q[:, qidx].float(), k.float().transpose(-1, -2)) * module.scaling
        logits = logits.masked_fill(torch.arange(T, device=q.device)[None, :] > qidx[:, None], float("-inf"))
        attn = logits.softmax(-1)  # (Hq, Nq, T)
        self.results[layer] = layer_metrics(attn, steps, P, qidx, self.ctx["reflect_next"])

    def remove(self):
        for h in self.handles:
            h.remove()


def layer_metrics(attn, steps, P, qidx, reflect_next):
    """attn: (H, Nq, T) attention of sampled queries; steps: (T,) step id per position."""
    H, Nq, T = attn.shape
    qstep = steps[qidx]  # (Nq,)
    n_steps = int(steps.max()) + 1

    # step-level mass: (H, Nq, n_steps)
    mass = torch.zeros(H, Nq, n_steps, device=attn.device).scatter_add_(
        -1, steps.expand(H, Nq, T), attn)
    mean_mass = mass.mean(0)  # (Nq, n_steps)

    # O1: distance histogram (prompt = step 0 reported separately)
    dist = qstep[:, None] - torch.arange(n_steps, device=attn.device)[None, :]  # (Nq, n_steps)
    hist = []
    for d in range(MAX_DIST):
        hist.append(float((mean_mass * ((dist == d) & (torch.arange(n_steps, device=attn.device) > 0))).sum(1).mean()))
    far = (dist >= MAX_DIST) & (torch.arange(n_steps, device=attn.device)[None, :] > 0)
    hist.append(float((mean_mass * far).sum(1).mean()))
    prompt_mass = float(mean_mass[:, 0].mean())

    # O2: top-10% concentration inside completed steps, weighted by step mass
    conc_num, conc_den = 0.0, 0.0
    for s in range(1, n_steps):
        cols = (steps == s).nonzero().squeeze(-1)
        L = cols.numel()
        rows = qstep > s  # queries in later steps only (step s completed)
        if L < 10 or rows.sum() == 0:
            continue
        a = attn[:, rows][:, :, cols]  # (H, Nr, L)
        tot = a.sum(-1)
        top = a.topk(max(1, math.ceil(0.1 * L)), dim=-1).values.sum(-1)
        conc_num += float(top.sum())
        conc_den += float(tot.sum())
    concentration = conc_num / max(conc_den, 1e-12)

    # O3: per-token attention received from queries >= 2 steps later
    reflect_vals, other_vals = [], []
    for s in range(1, n_steps):
        rows = qstep >= s + 2
        L = int((steps == s).sum())
        if rows.sum() == 0 or L == 0:
            continue
        per_tok = float(mean_mass[rows, s].mean()) / L
        (reflect_vals if reflect_next.get(s, False) else other_vals).append(per_tok)

    return {
        "dist_hist": hist,
        "prompt_mass": prompt_mass,
        "recent3": sum(hist[:3]),
        "top10_concentration": concentration,
        "later_attn_reflected": sum(reflect_vals) / len(reflect_vals) if reflect_vals else None,
        "later_attn_other": sum(other_vals) / len(other_vals) if other_vals else None,
        "n_steps": n_steps - 1,
    }


def reflection_followers(tok, gen_ids, delim):
    """Map step id -> True if the *next* step starts with a reflection cue."""
    texts, cur = [], []
    for t in gen_ids:
        cur.append(t)
        if t in delim:
            texts.append(tok.decode(cur))
            cur = []
    texts.append(tok.decode(cur))
    return {i + 1: is_reflection(texts[i + 1]) for i in range(len(texts) - 1)}


def average(dicts):
    out = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts if d[key] is not None]
        if not vals:
            out[key] = None
        elif isinstance(vals[0], list):
            out[key] = [sum(v[i] for v in vals) / len(vals) for i in range(len(vals[0]))]
        else:
            out[key] = sum(vals) / len(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", required=True, help="jsonl written by eval.run_eval --save_ids")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 attn_implementation="sdpa").to(device).eval()
    delim = delimiter_token_ids(tok)
    probe = LayerProbe(model, args.stride)

    with open(args.runs) as f:
        recs = [r for r in map(json.loads, f) if r["method"] == "full" and "gen_ids" in r][: args.n]
    if not recs:
        raise SystemExit("no full-KV records with gen_ids; run eval.run_eval with --save_ids")

    per_trace = []
    for r in recs:
        ids = (r["prompt_ids"] + r["gen_ids"])[: args.max_len]
        P = len(r["prompt_ids"])
        steps = step_ids(r["gen_ids"], delim, P)[: len(ids)].to(device)
        qidx = torch.arange(P + args.stride, len(ids), args.stride, device=device)
        probe.ctx = {"steps": steps, "prompt_len": P, "qidx": qidx,
                     "reflect_next": reflection_followers(tok, r["gen_ids"], delim)}
        probe.results = {}
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=device), use_cache=False)
        per_trace.append({"id": r["id"], "layers": dict(probe.results)})
        mid = average(list(probe.results.values()))
        print(f"{r['id']}: steps={mid['n_steps']:.0f} recent3={mid['recent3']:.2f} "
              f"top10={mid['top10_concentration']:.2f}", flush=True)
    probe.remove()

    n_layers = len(per_trace[0]["layers"])
    by_layer = [average([t["layers"][l] for t in per_trace]) for l in range(n_layers)]
    overall = average(by_layer)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "attn_stats.json"), "w") as f:
        json.dump({"overall": overall, "by_layer": by_layer, "per_trace": per_trace}, f, indent=1)

    print("\n=== overall (mean over layers and traces) ===")
    print(f"distance histogram (0..{MAX_DIST - 1}, >={MAX_DIST}):", [round(x, 3) for x in overall["dist_hist"]])
    print(f"prompt mass: {overall['prompt_mass']:.3f}")
    print(f"O1 recent-3-steps mass: {overall['recent3']:.3f}   (gate G1: > 0.70 of non-prompt mass)")
    nonprompt = 1 - overall["prompt_mass"]
    print(f"   as share of non-prompt mass: {overall['recent3'] / max(nonprompt, 1e-9):.3f}")
    print(f"O2 top-10% concentration: {overall['top10_concentration']:.3f}   (gate G1: > 0.60)")
    print(f"O3 later attention/token, reflected steps: {overall['later_attn_reflected']}, "
          f"other steps: {overall['later_attn_other']}")
    plot(by_layer, args.out)


def plot(by_layer, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    hist = average(by_layer)["dist_hist"]
    labels = [str(d) for d in range(MAX_DIST)] + [f">={MAX_DIST}"]
    axes[0].bar(labels, hist, color="#4C72B0")
    axes[0].set_xlabel("step distance (0 = current step)")
    axes[0].set_ylabel("attention mass")
    axes[1].plot([l["recent3"] for l in by_layer], label="recent 3 steps")
    axes[1].plot([l["top10_concentration"] for l in by_layer], label="top-10% in step")
    axes[1].plot([l["prompt_mass"] for l in by_layer], label="prompt")
    axes[1].set_xlabel("layer")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "attn_stats.png"), dpi=150)
    print(f"figure written to {os.path.join(out, 'attn_stats.png')}")


if __name__ == "__main__":
    main()
