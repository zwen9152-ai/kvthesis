"""Chapter 3/4 additions: step-summary merging, restore + pinning, aligned recall,
budget decay, fuzzy / hesitation detection, observe-only mode, interventions, probe."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvcomp import KVCompressor, generate
from kvcomp.batch import generate_batch
from kvcomp.cache_utils import get_kv, num_layers
from kvcomp.degeneration import DegenerationDetector

DELIM = {5, 17}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    return Qwen2ForCausalLM(cfg).eval()


def prompts(n=4):
    g = torch.Generator().manual_seed(3)
    return [torch.randint(0, 64, (1, int(L)), generator=g) for L in torch.randint(6, 20, (n,), generator=g)]


def det():
    return DegenerationDetector(min_span=24, max_period=16, check_every=4)


def step_det():  # step repeats only (the toy model's loops are mostly periodic)
    return DegenerationDetector(min_span=10_000, check_every=10_000, min_step_len=2, step_repeats=2)


def comp_kw(**kw):
    return dict(budget=32, buffer=16, window=8, **kw)


def single(model, ps, method="snapkv", detector=True, gen_kw=None, mk_det=det, **ckw):
    out = []
    for p in ps:
        c = KVCompressor(model, method, **comp_kw(**ckw))
        out.append(generate(model, p, compressor=c, detector=mk_det() if detector else None, delimiter_ids=DELIM,
                            max_new_tokens=200, temperature=0.0, **{"refeed": 8, **(gen_kw or {})}))
        c.close()
    return out


def batched(model, ps, method="snapkv", detector=True, gen_kw=None, bs=3, mk_det=det, **ckw):
    res = dict(generate_batch(
        model, ps, make_compressor=lambda cap: KVCompressor(model, method, capture=cap, **comp_kw(**ckw)),
        make_detector=mk_det if detector else None, delimiter_ids=DELIM, batch_size=bs, max_new_tokens=200,
        temperature=0.0, window=8, **{"refeed": 8, **(gen_kw or {})}))
    return [res[i] for i in range(len(ps))]


# ---------- step-summary merging ----------

def test_merge_keeps_budget_and_caps_summaries(model):
    ps = prompts()
    for r, p in zip(single(model, ps, detector=False, merge=1, merge_frac=0.25), ps):
        assert r["n_compress"] > 0
        assert r["max_cache_len"] <= p.shape[1] + 32 + 16 + 1
        assert 1 <= r["n_sum"] <= 8  # cap = 25% of 32
        assert r["n_merged"] > 0


def test_merge_layout_and_values(model):
    p = prompts(1)[0]
    c = KVCompressor(model, "snapkv", **comp_kw(merge=1))
    generate(model, p, compressor=c, delimiter_ids=DELIM, max_new_tokens=120, temperature=0.0)
    P = p.shape[1]
    assert all(c.pos[i].shape[-1] == c.pos[0].shape[-1] for i in range(len(c.pos)))
    ns = c.n_sum
    for i in range(len(c.pos)):
        pos = c.pos[i]
        assert (pos[..., ns:].diff(dim=-1) > 0).all()  # normal entries in position order
        cnt = c._sum_cnt[i]
        assert cnt.shape[-1] == ns and (cnt >= 1).all()
    c.close()


def test_merge_matches_between_single_and_batch(model):
    ps = prompts()
    a = single(model, ps, detector=False, merge=1)
    b = batched(model, ps, detector=False, merge=1)
    assert [x["gen_ids"] for x in a] == [x["gen_ids"] for x in b]


def test_merge_with_rollback_runs(model):
    ps = prompts()
    for recall in (True, False):
        a = single(model, ps, merge=1, offload=True, gen_kw={"recall": recall})
        b = batched(model, ps, merge=1, offload=True, gen_kw={"recall": recall})
        assert [x["gen_ids"] for x in a] == [x["gen_ids"] for x in b]
        assert any(x["rollbacks"] for x in a)


def test_merge_unit_weighted_mean():
    """One step evicted in full: summary value = attention-weighted mean, key = top entry's key."""
    from kvcomp.compressor import KVCompressor as K
    c = K.__new__(K)
    c.merge, c.merge_w, c.merge_key, c.merge_frac, c.budget = True, "attn", "top", 0.5, 8
    c._sum_cnt, c.n_merged, c.n_empty_groups = [None], 0, 0
    t, d = 6, 3
    kd = torch.randn(1, 1, t, d)
    vd = torch.randn(1, 1, t, d)
    pos = torch.arange(10, 10 + t).view(1, 1, t)
    steps = torch.tensor([[[1, 1, 1, 2, 2, 2]]])
    scores = torch.tensor([[[0.1, 0.5, 0.2, 0.3, 0.3, 0.3]]])
    idx = torch.tensor([[[3, 4, 5]]])  # step 1 evicted entirely
    empty = (kd[:, :, :0], vd[:, :, :0], pos[..., :0])
    sk, sv, sp = c._merge(0, kd, vd, pos, scores, steps, idx, torch.tensor([1]), *empty)
    w = scores[0, 0, :3] + 1e-6
    ref = (w[:, None] * vd[0, 0, :3]).sum(0) / w.sum()
    torch.testing.assert_close(sv[0, 0, 0], ref)
    torch.testing.assert_close(sk[0, 0, 0], kd[0, 0, 1])
    assert sp.tolist() == [[[11]]]
    assert c._sum_cnt[0].tolist() == [[[3.0]]]


# ---------- restore / pinning / aligned recall / decay ----------

def test_restore_brings_back_and_pins(model):
    p = prompts(1)[0]
    P = p.shape[1]
    c = KVCompressor(model, "snapkv", **comp_kw(offload=True))
    from kvcomp.decode import _forward
    from transformers import DynamicCache
    from kvcomp.segment import StepTracker
    cache = DynamicCache()
    tr = StepTracker(DELIM, 1000, "cpu")
    tr.add_prompt(P)
    out = _forward(model, cache, p[0].tolist(), 0)
    c.on_prefill(cache, P)
    toks = []
    for j in range(100):
        t = int(out.logits[0, -1].argmax())
        toks.append(t)
        b = tr.add_token(t)
        out = _forward(model, cache, [t], P + j)
        c.on_tokens(cache, [P + j], tr.steps, b)
    assert c.n_compress > 0
    want = list(range(P + 2, P + 8))
    before = get_kv(cache, 0)[0].shape[-2]
    n = c.restore(cache, want)
    assert n > 0
    assert get_kv(cache, 0)[0].shape[-2] == before  # constant size
    for i in range(num_layers(cache)):
        for h in range(c.pos[i].shape[1]):
            assert set(want) <= set(c.pos[i][0, h].tolist())
        assert int(c.state[i]["pin"].sum(-1).min()) == len(want)
    # pinned entries survive further compressions
    for j in range(100, 160):
        t = int(out.logits[0, -1].argmax())
        b = tr.add_token(t)
        out = _forward(model, cache, [t], P + j)
        c.on_tokens(cache, [P + j], tr.steps, b)
    for i in range(num_layers(cache)):
        for h in range(c.pos[i].shape[1]):
            assert set(want) <= set(c.pos[i][0, h].tolist())
    # rollback without recall keeps them too, and no position is duplicated
    c.rollback(cache, P + 120, recall=False)
    for i in range(num_layers(cache)):
        for h in range(c.pos[i].shape[1]):
            row = c.pos[i][0, h].tolist()
            assert set(want) <= set(row) and len(row) == len(set(row))
    # rollback with recall after a restore must not duplicate entries
    c.rollback(cache, P + 110, recall=True)
    for i in range(num_layers(cache)):
        for h in range(c.pos[i].shape[1]):
            row = c.pos[i][0, h].tolist()
            assert len(row) == len(set(row))
    c.close()


@pytest.mark.parametrize("gen_kw", [
    {"recall": "aligned", "escalate": 2, "max_budget": 64, "refeed": 1, "_det": "step"},
    {"recall": False, "escalate": 2, "max_budget": 64, "decay": 2},
])
def test_v2_batch_matches_single(model, gen_kw):
    ps = prompts()
    gen_kw = dict(gen_kw)
    mk = step_det if gen_kw.pop("_det", None) else det
    a = single(model, ps, offload=True, gen_kw=gen_kw, mk_det=mk)
    b = batched(model, ps, offload=True, gen_kw=gen_kw, mk_det=mk)
    if "aligned" in str(gen_kw.get("recall")):
        assert any("restored" in r for x in a for r in x["rollbacks"]), "aligned recall should be attempted"
    assert [x["gen_ids"] for x in a] == [x["gen_ids"] for x in b]
    assert [x["final_budget"] for x in a] == [x["final_budget"] for x in b]
    assert any(x["rollbacks"] for x in a)


def test_decay_lowers_budget(model):
    ps = prompts()
    no = single(model, ps, offload=False, gen_kw={"recall": False, "escalate": 2, "max_budget": 128})
    yes = single(model, ps, offload=False, gen_kw={"recall": False, "escalate": 2, "max_budget": 128, "decay": 1})
    assert sum(r["final_budget"] for r in yes) <= sum(r["final_budget"] for r in no)


# ---------- detector ----------

def test_fuzzy_detector_catches_paraphrase():
    base = list(range(20, 40))
    para = base[:]
    para[10] = 63  # one token changed: no exact repeat
    para2 = base[:]
    para2[3] = 62
    toks = []
    for s in (base, [41, 42, 43, 44, 45, 46, 47, 48, 49], para, para2):
        toks += s + [5]
    exact = DegenerationDetector(min_step_len=8, step_repeats=3, check_every=10_000)
    fuzzy = DegenerationDetector(min_step_len=8, step_repeats=3, check_every=10_000, fuzzy=0.5)
    hits = {}
    for name, d in (("exact", exact), ("fuzzy", fuzzy)):
        seen = []
        for j in range(len(toks)):
            r = d.update(toks[: j + 1], toks[j] in DELIM)
            if r is not None:
                seen.append((r, d.last_origin, d.last_kind))
        hits[name] = seen
    assert hits["exact"] == []
    assert hits["fuzzy"] and hits["fuzzy"][0][1] == (0, 21) and hits["fuzzy"][0][2] == "step"


def test_hesitation_signal():
    cue = 50
    toks = []
    for j in range(8):
        toks += ([cue] if j % 4 else [30 + j]) + [31 + j, 40 + j] + [5]
    d = DegenerationDetector(check_every=10_000, hesitate=6, hesitate_window=8, cue_ids={cue}, min_step_len=100)
    fired = [j for j in range(len(toks)) if d.update(toks[: j + 1], toks[j] in DELIM) is not None]
    assert fired and d.last_kind == "hesitate"


# ---------- observe mode, interventions, probe ----------

def test_observe_only_does_not_change_output(model):
    ps = prompts()
    plain = single(model, ps, detector=False, offload=True)
    obs = single(model, ps, offload=True, gen_kw={"act": False})
    none = single(model, ps, offload=True, gen_kw={"act": False, "intervene": "none"})
    assert [r["gen_ids"] for r in plain] == [r["gen_ids"] for r in obs] == [r["gen_ids"] for r in none]
    assert any(r.get("detections") for r in obs)
    assert all(not r["rollbacks"] for r in obs)


@pytest.mark.parametrize("mode", ["aligned", "random"])
def test_intervention_batch_matches_single(model, mode):
    ps = prompts()
    kw = {"act": False, "intervene": mode}
    a = single(model, ps, offload=True, gen_kw=kw, mk_det=step_det)
    b = batched(model, ps, offload=True, gen_kw=kw, mk_det=step_det)
    assert [x["gen_ids"] for x in a] == [x["gen_ids"] for x in b]
    evs = [e for r in a for e in r.get("detections", []) if "intervene" in e]
    assert evs and all(e["intervene"] == mode for e in evs)


def test_probe_full_reference(model):
    """Infinite budget: retention is 100%, lookback is recorded, events carry step spans."""
    ps = prompts(2)
    out = []
    for p in ps:
        c = KVCompressor(model, "snapkv", budget=100_000, buffer=0, window=8, probe=1, probe_every=16,
                         probe_age=32, lookback=1, probe_min_age=32)
        out.append(generate(model, p, compressor=c, detector=det(), delimiter_ids=DELIM, max_new_tokens=150,
                            temperature=0.0, act=False))
        c.close()
    for r in out:
        ret = [x for x in r["probe_ret"] if x >= 0]
        assert ret and all(x == 1000 for x in ret)
        look = [x for x in r["probe_look"] if x >= 0]
        assert look and all(x >= 0 for x in look)
    evs = [e for r in out for e in r.get("probe_events", [])]
    assert evs and all(e["steps"] and all(s[2] == 1000 for s in e["steps"]) for e in evs)


def test_probe_compressed_matches_batch(model):
    ps = prompts()
    kw = dict(offload=True, probe=1, probe_every=16, probe_age=32)
    a = single(model, ps, gen_kw={"act": False}, **kw)
    b = batched(model, ps, gen_kw={"act": False}, **kw)
    for x, y in zip(a, b):
        assert x["gen_ids"] == y["gen_ids"]
        assert x["probe_ret"] == y["probe_ret"]
    assert any(0 <= v < 1000 for r in a for v in r["probe_ret"])
