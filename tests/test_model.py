"""KV-cache generation must be token-exact vs the cacheless sliding path.

The cache only changes WHEN compute happens, not WHAT is computed: with the
same seed, greedy (top_k=1) decoding through forward_cached must emit the
identical token sequence as _generate_slow, up to the ctx limit. Beyond ctx
the cached path cannot slide, so generate() must fall back to the slow path.
"""
import torch

from neurobrain.model import GPT, GPTConfig


def tiny() -> GPT:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, n_layer=2, n_head=2, d_model=16, ctx=64,
                    dropout=0.0)
    m = GPT(cfg)
    m.eval()
    return m


def test_cached_matches_sliding_greedy():
    m = tiny()
    prompt = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)
    torch.manual_seed(123)
    a = m.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    torch.manual_seed(123)
    b = m._generate_slow(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    assert a.shape == b.shape == (1, 17)
    assert torch.equal(a[:, :5], prompt)
    assert torch.equal(a, b)


def test_cached_matches_sliding_to_ctx_edge():
    # prompt 40 + 24 new == ctx 64 exactly: still cacheable, must match
    m = tiny()
    prompt = (torch.arange(40, dtype=torch.long) % 64).unsqueeze(0)
    torch.manual_seed(7)
    a = m.generate(prompt, max_new_tokens=24, temperature=1.0, top_k=1)
    torch.manual_seed(7)
    b = m._generate_slow(prompt, max_new_tokens=24, temperature=1.0, top_k=1)
    assert torch.equal(a, b)


def test_overflow_falls_back_to_slow_path():
    m = tiny()
    prompt = (torch.arange(60, dtype=torch.long) % 64).unsqueeze(0)
    torch.manual_seed(3)
    out = m.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    torch.manual_seed(3)
    ref = m._generate_slow(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    assert out.shape == (1, 72)
    assert torch.equal(out[:, :60], prompt)
    assert torch.equal(out, ref)


def test_eos_stop_and_sampling_shape():
    m = tiny()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = m.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=10)
    assert out.shape == (1, 23)
    torch.manual_seed(5)
    first = m.generate(prompt, max_new_tokens=1, temperature=1.0, top_k=1)
    eos = int(first[0, -1])
    torch.manual_seed(5)
    out2 = m.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=1,
                      eos_id=eos)
    assert out2.shape[1] == 4  # stopped right after emitting eos
