"""HYP-118: exact policies, streamed ranking, EM and the majority-sign encoder."""

import math

import pytest
import torch
from test_person import orthogonal_book, random_records

import experiment3_linkage as linkage
import person
import policy as P


def with_missing_interests(records, seed):
    """Drop random interest slots too, so interest lists have 1-3 known values."""
    g = torch.Generator().manual_seed(seed)
    drop = torch.rand(len(records), 3, generator=g) < 0.3
    interests = torch.where(drop, -1, records.interests).to(torch.int8)
    return person.from_values(records.record_index, records.value, interests)


def identity_ages(r):
    """Walsh age levels are orthogonal, so the exact kernel needs ages 0 or 9 only."""
    r.value[:, 0] = torch.where(r.value[:, 0] >= 0, (r.value[:, 0] % 2) * 9, -1)
    return r


def test_normalized_bundle_cosine_is_the_cosine_policy_with_orthogonal_atoms():
    book = orthogonal_book()
    a = identity_ages(with_missing_interests(random_records(200, 21, 0.3), 1))
    b = identity_ages(with_missing_interests(random_records(200, 22, 0.3), 2))
    measured = person.cosine(
        person.encode_normalized(book, a), person.encode_normalized(book, b), paired=True
    )
    exact = torch.diagonal(P.cosine(P.compare(a, b)))
    both = (P.known(a)[0].sum(1) > 0) & (P.known(b)[0].sum(1) > 0)
    assert (measured - exact)[both].abs().max() < 1e-5


def test_policies_differ_only_in_the_denominator_for_a_complete_query():
    q = random_records(30, 23)
    c = with_missing_interests(random_records(50, 24, 0.4), 3)
    p = P.compare(q, c)
    s, pc = p.total, p.p_c
    assert torch.allclose(P.mismatch(p), s / 4)
    assert torch.allclose(P.cosine(p), s / (2 * pc.sqrt().clamp(min=1)))
    assert torch.allclose(P.gower(p), s / pc.clamp(min=1))  # both know = what the candidate knows
    assert torch.allclose(P.pivoted(1.0, 1.7)(p), P.cosine(p))


def test_expected_ranks_place_a_tied_record_uniformly():
    g, e = torch.tensor([2, 0, 0]), torch.tensor([0, 3, 1])
    assert P.expected_recall(g, e, 1).tolist() == [0.0, 0.25, 0.5]
    assert P.expected_recall(g, e, 3).tolist() == [1.0, 0.75, 1.0]
    rr = P.expected_reciprocal_rank(g, e)
    assert rr.tolist() == pytest.approx([1 / 3, (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4, 0.75], rel=1e-6)


def test_streamed_ranks_match_brute_force(monkeypatch):
    monkeypatch.setattr(P, "CHUNK", 97)
    pool = with_missing_interests(random_records(600, 25, 0.3), 4)
    queries = random_records(12, 26)
    queries.record_index += 10_000  # not in the pool
    relevant = with_missing_interests(random_records(12, 27, 0.3), 5)
    exclude = pool.record_index[:12]
    policies = {"cosine": P.cosine, "gower": P.gower}
    ranks = P.rank_relevant(queries, pool, relevant, policies, k=5, exclude=exclude)
    for name, f in policies.items():
        truth = torch.diagonal(f(P.compare(queries, relevant)))
        p = P.compare(queries, pool)
        valid = (p.p_c > 0) & (pool.record_index[None, :] != exclude[:, None])
        score = torch.where(valid, f(p), -torch.inf)
        assert (
            ranks[name].greater.tolist() == (score > truth[:, None] + P.TOLERANCE).sum(1).tolist()
        )
        assert (
            ranks[name].equal.tolist()
            == ((score - truth[:, None]).abs() <= P.TOLERANCE).sum(1).tolist()
        )
        assert torch.equal(ranks[name].top, score.topk(5, 1).values)


def test_top_k_shares_add_up_to_k(monkeypatch):
    monkeypatch.setattr(P, "CHUNK", 64)
    complete = random_records(400, 28)
    pool = with_missing_interests(
        complete.without(torch.rand(400, 4, generator=torch.Generator().manual_seed(6)) < 0.3), 7
    )
    queries = random_records(8, 29)
    policies = {"cosine": P.cosine, "gower": P.gower}
    kth = P.scan_kth(queries, pool, complete, policies, 10)
    top = P.scan_top(queries, pool, complete, policies, kth, 10)
    for t in top.values():
        assert torch.allclose(t.coverage.sum(1), torch.full((8,), 10.0))
        assert ((t.hits >= 0) & (t.hits <= 10 + 1e-5)).all()
    assert torch.allclose(top["relevance"].hits, torch.full((8,), 10.0))


def test_em_recovers_m_when_matches_are_separable():
    g = torch.Generator().manual_seed(8)
    u = [
        torch.tensor([0.1, 0.2, 0.7]),
        torch.tensor([0.2, 0.8]),
        torch.tensor([0.25, 0.75]),
        torch.tensor([0.05, 0.25, 0.7]),
    ]
    m = [
        torch.tensor([0.85, 0.05, 0.1]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.8, 0.15, 0.05]),
    ]
    n, lam = 40_000, 0.05
    match = torch.rand(n, generator=g) < lam
    levels = torch.stack(
        [
            torch.where(
                match,
                torch.multinomial(mf, n, True, generator=g),
                torch.multinomial(uf, n, True, generator=g),
            )
            for mf, uf in zip(m, u)
        ],
        1,
    )
    estimated, share = linkage.em(levels, u, (0, 1, 2, 3))
    assert share == pytest.approx(lam, abs=0.01)
    for a, b in zip(estimated, m):
        assert (a - b).abs().max() < 0.05


def test_majority_sign_keeps_integers_and_loses_partial_credit():
    book = person.codebook(2048, 31)
    base = random_records(64, 30)
    sums = person.encode_sign(book, base)
    assert torch.equal(sums, sums.round()) and sums.abs().max() <= 4
    assert torch.equal(sums.half().float(), sums)
    only = lambda r: r.without(torch.tensor([[True, True, True, False]]).expand(len(r), 4))
    one = person.from_values(
        base.record_index,
        base.value,
        torch.where(torch.arange(3) > 0, -1, base.interests).to(torch.int8),
    )
    cos = person.cosine(
        person.encode_sign(book, only(base)), person.encode_sign(book, only(one)), paired=True
    )
    assert abs(float(cos.mean()) - 0.5) < 0.03  # 1 of 3 interests: 0.5, against sqrt(1/3) with L2
    assert abs(math.sqrt(1 / 3) - 0.577) < 1e-3


def test_mcar_masks_hit_their_rates():
    signatures = person.load_fixture()
    for rate in (0.1, 0.5):
        r = person.mcar_records(signatures, rate)
        assert abs(float((r.value < 0).float().mean()) - rate) < 0.003
        assert abs(float((r.interests < 0).float().mean()) - rate) < 0.003
        all_missing = (r.interests < 0).all(1)
        assert torch.equal(r.state[:, 3] == person.MISSING, all_missing)
