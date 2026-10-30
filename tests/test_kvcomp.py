import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from kvcomp import METHODS, KVCompressor, generate
from kvcomp.cache_utils import get_kv
from kvcomp.query_capture import QueryCapture, window_attention

DELIM = {5, 17}  # pretend these token ids contain "\n\n"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture
def prompt():
    return torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(1))


def test_matches_hf_generate(model, prompt):
    ref = model.generate(prompt, max_new_tokens=60, do_sample=False, min_new_tokens=60)[0, prompt.shape[1]:]
    ours = generate(model, prompt, max_new_tokens=60)["gen_ids"]
    assert ours == ref.tolist()


def test_hooks_without_compression_are_transparent(model, prompt):
    base = generate(model, prompt, max_new_tokens=60)["gen_ids"]
    comp = KVCompressor(model, "stepkv", budget=10_000)
    with_hooks = generate(model, prompt, compressor=comp, delimiter_ids=DELIM, max_new_tokens=60)
    comp.close()
    assert with_hooks["gen_ids"] == base
    assert with_hooks["n_compress"] == 0


def test_window_attention_matches_eager(model, prompt):
    w = 4
    model.config._attn_implementation = "eager"
    cap = QueryCapture(model, w)
    cache = DynamicCache()
    out = model(prompt, past_key_values=cache, use_cache=True, output_attentions=True)
    cap.remove()
    model.config._attn_implementation = "sdpa"
    for i, attn in enumerate(out.attentions):  # (B, Hq, T, T)
        k, _ = get_kv(cache, i)
        ours = window_attention(cap.buf[i], k, model.model.layers[i].self_attn.scaling)
        ref = attn[:, :, -w:].float()
        ref = ref.reshape(1, k.shape[1], -1, w, ref.shape[-1]).sum((2, 3))
        torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("step_trigger", [False, True])
def test_methods_respect_budget(model, prompt, method, step_trigger):
    budget, buffer = 32, 16
    comp = KVCompressor(model, method, budget=budget, buffer=buffer, window=8, step_trigger=step_trigger)
    res = generate(model, prompt, compressor=comp, delimiter_ids=DELIM, max_new_tokens=200,
                   temperature=1.0, seed=0)
    comp.close()
    assert res["num_tokens"] == 200
    assert res["n_compress"] > 0
    limit = budget + (2 if step_trigger else 1) * buffer
    assert res["max_cache_len"] <= prompt.shape[1] + limit
    for p in comp.pos:  # kept positions are unique and increasing per head
        assert bool((p[..., 1:] > p[..., :-1]).all())


def test_stepkv_keeps_anchors_of_every_step():
    from kvcomp.methods import SelectCtx, select_stepkv
    T, H = 60, 1
    steps = torch.arange(T).div(10, rounding_mode="floor")[None, None] + 1  # 6 steps of 10 tokens
    scores = torch.rand(1, H, T, generator=torch.Generator().manual_seed(0))
    scores[..., :10] *= 1e-3  # step 1 is nearly ignored
    ctx = SelectCtx(scores=scores, keys=torch.randn(1, H, T, 8), steps=steps, budget=24, recent=4,
                    state={}, cfg={"anchors": 2})
    kept_steps = steps[0, 0][select_stepkv(ctx)[0, 0]]
    for s in range(1, 7):
        assert (kept_steps == s).sum() >= 2


def test_detector_finds_periodic_loop():
    from kvcomp.degeneration import DegenerationDetector
    det = DegenerationDetector(min_span=40, check_every=1)
    prefix = list(range(100, 130))
    loop = [1, 2, 3, 4, 5, 6, 7]
    toks, hit = [], None
    for t in prefix + loop * 10:
        toks.append(t)
        hit = det.update(toks, False)
        if hit is not None:
            break
    assert hit == len(prefix)


def test_detector_ignores_non_repeating():
    from kvcomp.degeneration import DegenerationDetector
    det = DegenerationDetector(min_span=40, check_every=1)
    g = torch.Generator().manual_seed(0)
    toks = []
    for t in torch.randint(0, 1000, (500,), generator=g).tolist():
        toks.append(t)
        assert det.update(toks, False) is None


def test_rollback_restores_exact_history(model, prompt):
    from transformers import DynamicCache
    from kvcomp.decode import _forward
    from kvcomp.segment import StepTracker

    P = prompt.shape[1]
    comp = KVCompressor(model, "snapkv", budget=24, buffer=8, window=8, offload=True)
    cache = DynamicCache()
    out = _forward(model, cache, prompt[0].tolist(), 0)
    comp.on_prefill(cache, P)
    tracker = StepTracker(DELIM, 400, "cpu")
    tracker.add_prompt(P)
    gen = []
    for _ in range(120):
        tok = int(out.logits[0, -1].argmax())
        gen.append(tok)
        b = tracker.add_token(tok)
        out = _forward(model, cache, [tok], P + len(gen) - 1)
        comp.on_tokens(cache, [P + len(gen) - 1], tracker.steps, b)
    assert comp.n_compress > 0
    cut = P + 70
    comp.rollback(cache, cut)
    comp.close()
    for p in comp.pos:
        assert torch.equal(p, torch.arange(P, cut).expand_as(p))
    # layer-0 keys depend only on the token and its position, so they must match a fresh pass
    ref = DynamicCache()
    _forward(model, ref, prompt[0].tolist() + gen[: cut - P], 0)
    torch.testing.assert_close(get_kv(cache, 0)[0], get_kv(ref, 0)[0])


@pytest.mark.parametrize("method", ["snapkv", "h2o", "stepkv"])
def test_generate_with_rollback(model, prompt, method):
    from kvcomp.degeneration import DegenerationDetector
    comp = KVCompressor(model, method, budget=32, buffer=16, window=8, offload=True)
    det = DegenerationDetector(min_span=24, max_period=16, check_every=4)
    res = generate(model, prompt, compressor=comp, detector=det, delimiter_ids=DELIM,
                   max_new_tokens=300, temperature=0.0, max_rollbacks=3, refeed=8)
    comp.close()
    assert res["num_tokens"] == 300
    assert res["computed_tokens"] >= res["num_tokens"]
    assert len(res["rollbacks"]) <= 3
    for p in comp.pos:
        assert bool((p[..., 1:] > p[..., :-1]).all())


def test_rollback_without_recall_keeps_heads_consistent(model, prompt):
    comp = KVCompressor(model, "snapkv", budget=24, buffer=8, window=8, offload=True)
    res = generate(model, prompt, compressor=comp, delimiter_ids=DELIM, max_new_tokens=100,
                   temperature=1.0, seed=0)
    assert res["n_compress"] > 0
    P = prompt.shape[1]
    # rebuild the cache state of that run, then roll back without recall
    from transformers import DynamicCache
    from kvcomp.decode import _forward
    cache = DynamicCache()
    _forward(model, cache, prompt[0].tolist() + res["gen_ids"][:-1], 0)
    comp.on_prefill(cache, P)
    comp.on_tokens(cache, list(range(P, P + len(res["gen_ids"]) - 1)), lambda p: p * 0, True)
    cut = P + 60
    comp.rollback(cache, cut, recall=False)
    comp.close()
    n = comp.pos[0].shape[-1]
    for i, p in enumerate(comp.pos):
        assert p.shape[-1] == n and bool((p < cut).all())
        assert get_kv(cache, i)[0].shape[-2] == P + n
