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
