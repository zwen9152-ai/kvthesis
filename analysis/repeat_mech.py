"""Model-internal repetition signals (chapter 4 candidates), offline on saved outputs.

    python -m analysis.repeat_mech subset.jsonl --runs "snapkv|128;rkv|128" --out results/analysis/repeat_mech

A. Induction ("copy") heads: a random token block repeated twice; per (layer, head) the
   attention from the second copy to the token after the earlier occurrence (Olsson et al.).
B. Teacher-forced Full-KV replay of the compressed runs' outputs. Per token the induction
   heads' attention mass on prefix matches (j with tok[j-1] == tok[i], at least ``--min_gap``
   back) = "the model is copying older text"; also how far back the copied source lies (under a
   128-token budget a source more than a few hundred tokens back is normally evicted).
C. Hidden-state step similarity: every reasoning step -> mean hidden state of a middle layer;
   a step whose cosine similarity to an earlier step exceeds theta is a semantic repeat
   (optionally only if both steps contain the same numbers). Compared with the n-gram detectors.

Caveat: B/C replay with the full cache, so they describe the text, not the compressed run's
internal state (a real detector would see the compressed cache).
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

START = 256


def load_subset(path, runs):
    rs = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        rs[r["run"]][r["id"]] = r
    full = next(v for k, v in rs.items() if k.split("|")[0].split("@")[0] == "full")
    return rs, full


@torch.no_grad()
def hidden(model, ids):
    out = model(input_ids=ids[None].to(model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
    return out.hidden_states  # tuple: [embeddings, layer1 in, ...]


@torch.no_grad()
def qk(model, hs, layer, T):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    m = model.model
    att = m.layers[layer].self_attn
    x = m.layers[layer].input_layernorm(hs[layer])
    hd = att.head_dim
    q = att.q_proj(x).view(1, T, -1, hd).transpose(1, 2)
    k = att.k_proj(x).view(1, T, -1, hd).transpose(1, 2)
    pos = torch.arange(T, device=x.device)[None]
    cos, sin = m.rotary_emb(x, pos)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q[0].float(), k[0].float(), hd


@torch.no_grad()
def attn_rows(q, k, hd, rows, chunk=512):
    """Softmax attention of query positions ``rows`` (1-D tensor) over all keys (causal)."""
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        sc = (q[r] @ k.T) / hd ** 0.5
        sc = sc.masked_fill(torch.arange(k.shape[0], device=q.device)[None] > r[:, None], float("-inf"))
        yield r, sc.softmax(-1)


def induction_heads(model, top, L=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    blk = torch.randint(1000, 20000, (L,), generator=g)
    ids = torch.cat([torch.tensor([model.config.bos_token_id or 0]), blk, blk])
    T = len(ids)
    hs = hidden(model, ids)
    nl, scores = model.config.num_hidden_layers, []
    rows = torch.arange(L + 1, T, device=model.device)
    for layer in range(nl):
        q, k, hd = qk(model, hs, layer, T)
        nkv = k.shape[0]
        rep = q.shape[0] // nkv
        for h in range(q.shape[0]):
            tot, n = 0.0, 0
            for r, a in attn_rows(q[h], k[h // rep], hd, rows):
                tgt = r - L + 1
                tot += a[torch.arange(len(r)), tgt].sum().item()
                n += len(r)
            scores.append((tot / n, layer, h))
    scores.sort(reverse=True)
    return scores[:top], scores


def copy_scores(model, ids, heads, min_gap):
    """Per token: induction heads' attention mass on prefix matches >= min_gap back, and the
    attention-weighted distance of that mass."""
    T = len(ids)
    hs = hidden(model, ids)
    tok = ids.to(model.device)
    prev = torch.cat([tok.new_tensor([-1]), tok[:-1]])
    mass = torch.zeros(T, device=model.device)
    dist = torch.zeros(T, device=model.device)
    rows = torch.arange(T, device=model.device)
    by_layer = defaultdict(list)
    for _, layer, h in heads:
        by_layer[layer].append(h)
    for layer, hh in by_layer.items():
        q, k, hd = qk(model, hs, layer, T)
        rep = q.shape[0] // k.shape[0]
        for h in hh:
            for r, a in attn_rows(q[h], k[h // rep], hd, rows):
                j = torch.arange(T, device=q.device)[None]
                m = (prev[None] == tok[r][:, None]) & (j <= r[:, None] - min_gap)
                am = (a * m).sum(-1)
                mass[r] += am
                dist[r] += (a * m * (r[:, None] - j)).sum(-1)
    n = len(heads)
    return (mass / n).cpu(), (dist / n).cpu(), hs


def steps_of(gen, delim):
    out, s = [], 0
    for i, t in enumerate(gen):
        if t in delim:
            if i + 1 - s >= 1:
                out.append((s, i + 1))
            s = i + 1
    if s < len(gen):
        out.append((s, len(gen)))
    return out


NUM = re.compile(r"\d+(?:\.\d+)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--runs", default="snapkv|128;rkv|128")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--top_heads", type=int, default=8)
    ap.add_argument("--min_gap", type=int, default=64)
    ap.add_argument("--layer", type=int, default=None, help="hidden-state layer (default 2/3 depth)")
    ap.add_argument("--max_per_group", type=int, default=60)
    ap.add_argument("--out", default="results/analysis/repeat_mech")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from analysis.detectors import fire_times
    from eval import data
    from eval.run_eval import cue_token_ids
    from kvcomp.segment import delimiter_token_ids
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    delim, cues = delimiter_token_ids(tok), cue_token_ids(tok)
    questions = {it["id"]: it["question"] for it in data.load("math500")}
    layer = args.layer if args.layer is not None else model.config.num_hidden_layers * 2 // 3
    os.makedirs(args.out, exist_ok=True)
    lines = []

    # ---- A ----
    top, allh = induction_heads(model, args.top_heads)
    lines += ["## A. Induction heads (attention to the token after the earlier occurrence, random repeated block)",
              "| rank | layer | head | score |", "|---|---|---|---|"]
    lines += [f"| {i + 1} | {l} | {h} | {s:.3f} |" for i, (s, l, h) in enumerate(top)]
    med = sorted(s for s, _, _ in allh)[len(allh) // 2]
    lines.append(f"\nmedian head score {med:.4f}; heads with score > 0.3: {sum(s > 0.3 for s, _, _ in allh)} of {len(allh)}")

    # ---- B / C ----
    rs, full = load_subset(args.file, args.runs)
    thetas = [0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99]
    off = dict(min_span=10 ** 9)
    ngram = {"exact (v1)": {}, "fuzzy 0.6": dict(fuzzy=0.6, **off)}
    res = {}  # run -> group -> list of per-problem dicts
    for run in args.runs.split(";"):
        probs = rs.get(run, {})
        groups = {"missed": [i for i, r in probs.items() if i in full and full[i]["correct"] and not r["correct"]],
                  "healthy": [i for i, r in probs.items() if r["correct"]]}
        res[run] = {}
        for g, idx in groups.items():
            out = []
            for i in idx[:args.max_per_group]:
                r = probs[i]
                gen = r["gen_ids"]
                p = data.build_prompt(tok, questions[i])[0]
                ids = torch.cat([p, torch.tensor(gen)])
                P = len(p)
                mass, dist, hs = copy_scores(model, ids, top, args.min_gap)
                mass, dist = mass[P:], dist[P:]
                h = hs[layer][0, P:].float()
                st = [s for s in steps_of(gen, delim) if s[1] - s[0] >= 8]
                vec = torch.stack([h[a:b].mean(0) for a, b in st]) if st else torch.zeros(0, h.shape[1])
                vec = F.normalize(vec, dim=-1)
                nums = [set(NUM.findall(tok.decode(gen[a:b]))) for a, b in st]
                sims, sims_num = [], []
                for k in range(len(st)):
                    if st[k][0] < START or k < 3:
                        continue
                    c = (vec[:k - 1] @ vec[k]).tolist()  # skip the immediately previous step
                    best = max(c) if c else 0.0
                    cn = [x for x, n in zip(c, nums[:k - 1]) if not (nums[k] and n and n != nums[k])]
                    sims.append((st[k][1], best))
                    sims_num.append((st[k][1], max(cn) if cn else 0.0))
                d = {"id": i, "len": len(gen), "sims": sims, "sims_num": sims_num,
                     "copy_frac": float((mass > 0.5).float().mean()),
                     "copy_far": float(((mass > 0.5) & (dist / mass.clamp(min=1e-6) > 512)).float().mean()),
                     "copy_mass": float(mass.mean()),
                     "fires": {k: fire_times(gen, delim, kw) for k, kw in ngram.items()}}
                # copy mass at n-gram detections vs elsewhere
                det = set()
                for t in d["fires"]["fuzzy 0.6"]:
                    det.update(range(max(0, t - 32), t + 1))
                m_det = [mass[t].item() for t in det if t < len(mass)]
                m_oth = [mass[t].item() for t in range(START, len(mass)) if t not in det]
                d["mass_det"] = sum(m_det) / len(m_det) if m_det else None
                d["mass_oth"] = sum(m_oth) / len(m_oth) if m_oth else None
                out.append(d)
                del hs
                torch.cuda.empty_cache()
            res[run][g] = out
            print(f"{run} {g}: {len(out)} done", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    lines += ["", "## B. Copying by induction heads (Full-KV replay of the compressed outputs)",
              "| run | group | n | tokens in copy mode % | of which source > 512 back % | mean copy mass | copy mass at fuzzy detections | elsewhere |",
              "|---|---|---|---|---|---|---|---|"]
    for run, gs in res.items():
        for g, out in gs.items():
            cf = mean(d["copy_frac"] for d in out)
            far = mean(d["copy_far"] for d in out)
            lines.append(f"| {run} | {g} | {len(out)} | {100 * cf:.1f} | {100 * far / max(cf, 1e-9):.0f} | "
                         f"{mean(d['copy_mass'] for d in out):.3f} | {mean(d['mass_det'] for d in out):.3f} | "
                         f"{mean(d['mass_oth'] for d in out):.3f} |")

    lines += ["", f"## C. Step-repeat detectors: caught missed / alarmed healthy (hidden layer {layer})",
              "| rule | " + " | ".join(res) + " | all caught | all unnecessary | alarm at (median share of output) |",
              "|" + "---|" * (len(res) + 4)]

    def first_ng(d, key):
        f = d["fires"][key]
        return f[0] if f else None

    def first_sim(d, th, key="sims"):
        return next((t for t, s in d[key] if s >= th), None)

    rules = [(k, (lambda d, k=k: first_ng(d, k))) for k in ngram]
    rules += [(f"hidden cos >= {t}", (lambda d, t=t: first_sim(d, t))) for t in thetas]
    rules += [(f"hidden cos >= {t} + same numbers", (lambda d, t=t: first_sim(d, t, "sims_num"))) for t in thetas]
    for name, fn in rules:
        cells, tc, tm, tu, th, when = [], 0, 0, 0, 0, []
        for run, gs in res.items():
            c = [fn(d) for d in gs["missed"]]
            u = [fn(d) for d in gs["healthy"]]
            nc, nu = sum(x is not None for x in c), sum(x is not None for x in u)
            cells.append(f"{nc}/{len(c)} / {nu}/{len(u)}")
            tc, tm, tu, th = tc + nc, tm + len(c), tu + nu, th + len(u)
            when += [x / d["len"] for x, d in zip(c, gs["missed"]) if x is not None]
        when.sort()
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {tc / max(tm, 1):.0%} | {tu / max(th, 1):.0%} | "
                     + (f"{when[len(when) // 2]:.0%}" if when else "-") + " |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "repeat_mech.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "repeat_mech.json"), "w") as f:
        json.dump({"heads": [(s, l, h) for s, l, h in allh], "res": {r: {g: [{k: v for k, v in d.items() if k != "fires"}
                   for d in out] for g, out in gs.items()} for r, gs in res.items()}}, f)


if __name__ == "__main__":
    main()
