import torch

from kvfetch.baselines import (arkvale_scores, attention_by_group, expand, logits_by_group, quest_scores,
                               shared_token_set, topk_mask, unit_summaries)


def _qk(seed=0, c=3, H=4, S=2, N=20, d=8):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c, H, d, generator=g), torch.randn(S, N, d, generator=g)


def test_attention_by_group_masks_future_keys_and_normalises():
    q, k = _qk()
    att = attention_by_group(q, k, torch.tensor([5, 12, 20]), 0.5)
    assert att.shape == (3, 2, 2, 20)
    assert torch.allclose(att.sum(-1), torch.ones(3, 2, 2))
    assert att[0, ..., 5:].abs().sum() == 0 and att[1, ..., 12:].abs().sum() == 0


def test_page_bounds_upper_bound_every_key_logit():
    q, k = _qk(1)
    summ = unit_summaries(k, 8)                           # 3 units of 8, the last one partial (4 keys)
    true = logits_by_group(q, k, 0.5)                     # [c, S, g, N]
    true_unit = torch.nn.functional.pad(true, (0, 4), value=float("-inf")).reshape(3, 2, 2, 3, 8).amax(-1).amax(2)
    assert (quest_scores(q, summ, 0.5) >= true_unit - 1e-5).all()
    assert (arkvale_scores(q, summ, 0.5) >= true_unit - 1e-5).all()


def test_topk_mask_per_step_counts_and_allowed():
    score = torch.tensor([[[4.0, 3.0, 2.0, 1.0]], [[4.0, 3.0, 2.0, 1.0]]])
    allowed = torch.tensor([[False, True, True, True], [True, True, True, True]])
    m = topk_mask(score, torch.tensor([2, 0]), allowed)
    assert m[0, 0].tolist() == [False, True, True, False] and not m[1].any()
    two_heads = topk_mask(torch.stack([score[:, 0], score[:, 0].flip(-1)], 1), torch.tensor([1, 1]), allowed)
    assert two_heads[0].tolist() == [[False, True, False, False], [False, False, False, True]]


def test_shared_token_set_prefers_tokens_many_heads_nominate():
    att = torch.tensor([[[0.5, 0.3, 0.2, 0.0], [0.0, 0.3, 0.2, 0.5], [0.1, 0.4, 0.0, 0.5]]])   # c=1, 3 heads
    allowed = torch.ones(1, 4, dtype=torch.bool)
    sel = shared_token_set(att, torch.tensor([2]), allowed)
    # top-2 nominations: head0 {0,1}, head1 {3,1}, head2 {3,1} -> token 1 (3 votes), token 3 (2 votes)
    assert sel[0].tolist() == [False, True, False, True]


def test_expand_units_to_tokens():
    m = torch.tensor([[[True, False, True]]])
    assert expand(m, 2, 5)[0, 0].tolist() == [True, True, False, False, True]
