import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvcomp import KVCompressor, generate
from kvcomp.batch import generate_batch
from kvcomp.degeneration import DegenerationDetector

DELIM = {5, 17}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    return Qwen2ForCausalLM(cfg).eval()


def prompts(n=5):
    g = torch.Generator().manual_seed(3)
    return [torch.randint(0, 64, (1, int(L)), generator=g) for L in torch.randint(6, 20, (n,), generator=g)]


def run_single(model, ps, method, rollback, max_new, **kw):
    out = []
    for p in ps:
        comp = det = None
        if method:
            comp = KVCompressor(model, method, budget=32, buffer=16, window=8, offload=rollback)
        if rollback:
            det = DegenerationDetector(min_span=24, max_period=16, check_every=4)
        out.append(generate(model, p, compressor=comp, detector=det, delimiter_ids=DELIM,
                            max_new_tokens=max_new, temperature=0.0, refeed=8, **kw))
        if comp:
            comp.close()
    return out


def run_batch(model, ps, method, rollback, max_new, batch_size, **kw):
    make_comp = make_det = None
    if method:
        def make_comp(capture):
            return KVCompressor(model, method, budget=32, buffer=16, window=8, offload=rollback, capture=capture)
    if rollback:
        def make_det():
            return DegenerationDetector(min_span=24, max_period=16, check_every=4)
    res = dict(generate_batch(model, ps, make_compressor=make_comp, make_detector=make_det, delimiter_ids=DELIM,
                              batch_size=batch_size, max_new_tokens=max_new, temperature=0.0, refeed=8,
                              window=8, **kw))
    return [res[i] for i in range(len(ps))]


@pytest.mark.parametrize("method,rollback,kw", [
    (None, False, {}),
    ("snapkv", False, {}),
    ("h2o", False, {}),
    ("rkv", False, {}),
    ("snapkv", True, {}),
    ("snapkv", True, {"escalate": 2, "max_budget": 96}),
    ("snapkv", True, {"recall": False}),
])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_batch_matches_single(model, method, rollback, kw, batch_size):
    ps = prompts()
    single = run_single(model, ps, method, rollback, 200, **kw)
    batch = run_batch(model, ps, method, rollback, 200, batch_size, **kw)
    for s, b in zip(single, batch):
        assert b["gen_ids"] == s["gen_ids"]
        for key in ("num_tokens", "computed_tokens", "n_compress", "max_cache_len", "final_budget"):
            assert b[key] == s[key], key
        assert [r["cut"] for r in b["rollbacks"]] == [r["cut"] for r in s["rollbacks"]]
    if rollback:
        assert any(s["rollbacks"] for s in single), "test should exercise rollback"


def test_batch_with_eos_refills_slots(model):
    ps = prompts(6)
    free = run_single(model, ps, "snapkv", False, 120)
    eos = {free[0]["gen_ids"][30], free[3]["gen_ids"][50]}  # tokens that do occur mid-generation
    single = run_single(model, ps, "snapkv", False, 120, eos_ids=eos)
    batch = run_batch(model, ps, "snapkv", False, 120, 2, eos_ids=eos)
    assert [b["gen_ids"] for b in batch] == [s["gen_ids"] for s in single]
    assert any(s["finished"] for s in single), "some prompt should hit EOS early"


def test_batch_sampling_matches_single(model):
    ps = prompts(4)
    single = []
    for i, p in enumerate(ps):
        comp = KVCompressor(model, "snapkv", budget=32, buffer=16, window=8)
        single.append(generate(model, p, compressor=comp, delimiter_ids=DELIM, max_new_tokens=100,
                               temperature=1.0, top_p=0.9, seed=10 + i))
        comp.close()
    res = dict(generate_batch(
        model, ps, make_compressor=lambda c: KVCompressor(model, "snapkv", budget=32, buffer=16, window=8, capture=c),
        delimiter_ids=DELIM, batch_size=3, max_new_tokens=100, temperature=1.0, top_p=0.9,
        seeds=[10 + i for i in range(len(ps))], window=8))
    assert [res[i]["gen_ids"] for i in range(len(ps))] == [s["gen_ids"] for s in single]


@pytest.mark.parametrize("rollback", [False, True])
def test_lazy_batched_is_order_independent(model, rollback):
    """LazyEviction runs only batched; batch 1 and batch 3 must agree token for token."""
    ps = prompts()

    def run(bs):
        def make_comp(capture):
            return KVCompressor(model, "lazy", budget=32, buffer=8, window=8, offload=rollback, capture=capture)
        make_det = (lambda: DegenerationDetector(min_span=24, max_period=16, check_every=4)) if rollback else None
        res = dict(generate_batch(model, ps, make_compressor=make_comp, make_detector=make_det, delimiter_ids=DELIM,
                                  batch_size=bs, max_new_tokens=200, temperature=0.0, refeed=8, window=8,
                                  observe_alpha=0.05))
        return [res[i] for i in range(len(ps))]

    one, three = run(1), run(3)
    assert [r["gen_ids"] for r in one] == [r["gen_ids"] for r in three]
    assert all(r["n_compress"] > 0 for r in one)
    assert all(r["max_cache_len"] <= p.shape[1] + 40 for r, p in zip(one, ps))


def test_lazy_observer_tracks_recurrence(model):
    """Observer state must equal a brute-force replay of the official TempCache rule."""
    ps = prompts(1)
    import copy
    model = copy.deepcopy(model)
    for layer in model.model.layers:  # sharpen the near-uniform random attention so gaps occur
        layer.self_attn.q_proj.weight.data *= 8
    alpha, n_new = 0.05, 40
    dbg = {}
    res = dict(generate_batch(
        model, ps, make_compressor=lambda c: KVCompressor(model, "lazy", budget=1000, buffer=8, window=8, capture=c),
        delimiter_ids=DELIM, batch_size=1, max_new_tokens=n_new, temperature=0.0, window=8,
        observe_alpha=alpha, debug=dbg))
    assert res[0]["n_compress"] == 0  # budget 1000: nothing evicted, so positions == indices
    ids = torch.tensor([ps[0][0].tolist() + res[0]["gen_ids"]])
    P, T = ps[0].shape[1], ids.shape[1]
    eager = Qwen2ForCausalLM(model.config).eval()
    eager.load_state_dict(model.state_dict())
    eager.config._attn_implementation = "eager"
    with torch.no_grad():
        attn = eager(ids, output_attentions=True).attentions  # per layer (1, Hq, T, T)
    hkv = model.config.num_key_value_heads
    slots = dbg["slots"]
    gaps = []
    for li in range(model.config.num_hidden_layers):
        a = attn[li][0].reshape(hkv, -1, T, T).amax(1)  # any query head of the group
        last = torch.zeros(hkv, T)
        mp = torch.zeros(hkv, T)
        for step in range(1, n_new + 1):
            q = P + step - 1  # the token decoded at this step
            last[:, q] = step - 1
            hit = a[:, q, : q + 1] >= alpha
            new_last = torch.where(hit, torch.tensor(float(step)), last[:, : q + 1])
            mp[:, : q + 1] = torch.maximum(mp[:, : q + 1], new_last - last[:, : q + 1])
            last[:, : q + 1] = new_last
        torch.testing.assert_close(slots.last[li][0, :, :T], last)
        torch.testing.assert_close(slots.mp[li][0, :, :T], mp)
        gaps.append(float(mp.max()))
    assert max(gaps) > 1  # some token was re-attended after a gap


@pytest.fixture(scope="module")
def helper():
    torch.manual_seed(1)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_rescue_lite_batch_matches_single(model, helper, temperature):
    from kvcomp.rescue import HelperState
    ps = prompts()
    eos = {63}

    def det():
        return DegenerationDetector(min_span=24, max_period=16, check_every=4)
    single = []
    for i, p in enumerate(ps):
        comp = KVCompressor(model, "snapkv", budget=32, buffer=16, window=8)
        single.append(generate(model, p, compressor=comp, detector=det(), delimiter_ids=DELIM, max_new_tokens=200,
                               temperature=temperature, seed=i, refeed=8, eos_ids=eos,
                               helper_state=HelperState(helper), helper_max_step=20))
        comp.close()
    res = dict(generate_batch(
        model, ps, make_compressor=lambda c: KVCompressor(model, "snapkv", budget=32, buffer=16, window=8, capture=c),
        make_detector=det, make_helper=lambda: HelperState(helper), helper_max_step=20, delimiter_ids=DELIM,
        batch_size=3, max_new_tokens=200, temperature=temperature, seeds=list(range(len(ps))), refeed=8,
        eos_ids=eos, window=8))
    batch = [res[i] for i in range(len(ps))]
    assert [b["gen_ids"] for b in batch] == [s["gen_ids"] for s in single]
    assert [b["helper_tokens"] for b in batch] == [s["helper_tokens"] for s in single]
    if temperature == 0:  # greedy decoding of the random model loops; sampling may not
        assert sum(s["helper_tokens"] for s in single) > 0, "the helper should take over at least once"


def test_helper_state_reuses_prefix(helper):
    from kvcomp.rescue import HelperState
    h = HelperState(helper)
    ctx = list(range(1, 20))
    a = h.write_step(ctx, delimiter_ids={5}, eos_ids=set(), max_step=10, temperature=0.0, top_p=1.0)
    # rewriting from an earlier cut must equal a fresh helper on the same context
    cut_ctx = ctx[:12]
    b = h.write_step(cut_ctx, delimiter_ids={5}, eos_ids=set(), max_step=10, temperature=0.0, top_p=1.0)
    fresh = HelperState(helper).write_step(cut_ctx, delimiter_ids={5}, eos_ids=set(), max_step=10,
                                           temperature=0.0, top_p=1.0)
    assert b == fresh and len(a) > 0
