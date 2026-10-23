import numpy as np

from kvfetch.tracing.schema import STEP_CONVENTION, load_meta, load_trace, save_trace


def _arrays(T=5, L=2, H=3, P=40, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((T, L, H, P)).astype(np.float32)
    m /= m.sum(-1, keepdims=True)
    return m.astype(np.float16), rng.random((4, L, H, P)).astype(np.float16)


def test_full_roundtrip(tmp_path):
    m, obs = _arrays()
    p = save_trace(tmp_path / "a.safetensors", meta={"model": "x", "layers": [0, 3]}, mass=m, obs_mass=obs,
                   tokens=np.arange(5), prompt_ids=np.arange(100), fmt="full")
    t = load_trace(p)
    assert np.array_equal(t.mass, m) and np.array_equal(t.obs_mass, obs)
    assert t.tokens.tolist() == list(range(5)) and t.prompt_ids.shape == (100,)
    meta = load_meta(p)
    assert meta["layers"] == [0, 3] and meta["n_pages"] == 40 and meta["step_convention"] == STEP_CONVENTION


def test_topk_roundtrip_and_residual(tmp_path):
    m, obs = _arrays()
    k = 8
    t = load_trace(save_trace(tmp_path / "b.safetensors", meta={}, mass=m, obs_mass=obs, tokens=np.arange(5),
                              prompt_ids=np.arange(10), fmt="topk", topk=k))
    assert t.topk_idx.dtype == np.uint16 and t.topk_idx.shape[-1] == k
    d = t.dense()
    # kept entries are exactly the k largest per row
    thresh = np.sort(m.astype(np.float32), -1)[..., -k][..., None]
    assert np.array_equal(d > 0, m.astype(np.float32) >= thresh)
    total = m.astype(np.float32).sum(-1)
    assert np.allclose(t.residual + d.astype(np.float32).sum(-1), total, atol=1e-3)
