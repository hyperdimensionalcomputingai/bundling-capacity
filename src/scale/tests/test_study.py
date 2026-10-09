"""Focused checks from the methodology's acceptance list. Run before the full study."""

import math

import pytest
import torch
import torchhd

import study
from study import Encoder


def test_encoder_sums_five_bound_facts_and_facts_stay_unbindable():
    encoder = Encoder.make(512, 11)
    codes = torch.tensor([[3, 1, 42, 199, 7], [0, 9, 0, 0, 999]])
    for record, row in zip(encoder.encode(codes), codes):
        facts = torch.stack(
            [
                torchhd.bind(encoder.roles[i], encoder.values[i][row[i]]).as_subclass(torch.Tensor)
                for i in range(study.FACTS)
            ]
        )
        # The raw sum of all five facts, with no sign applied.
        assert torch.equal(record.float(), torchhd.multiset(facts).as_subclass(torch.Tensor))
        assert set(record.unique().tolist()) <= {-5, -3, -1, 1, 3, 5}
        # Unbinding the employer role recovers the employer value as the best match.
        probe = record.float() * encoder.roles[4]
        assert int((encoder.values[4] @ probe).argmax()) == int(row[4])
    assert torch.equal(encoder.encode(codes), Encoder.make(512, 11).encode(codes))
    assert not torch.equal(encoder.encode(codes), Encoder.make(512, 23).encode(codes))


def test_shared_property_counts_on_hand_built_records():
    query = torch.tensor([[1, 2, 3, 4, 5]])
    candidates = torch.tensor(
        [
            [0, 0, 0, 0, 0],  # k = 0
            [1, 0, 0, 0, 0],  # k = 1
            [0, 2, 0, 4, 0],  # k = 2
            [1, 2, 3, 0, 0],  # k = 3
            [0, 2, 3, 4, 5],  # k = 4
            [1, 2, 3, 4, 5],  # k = 5
            [5, 4, 3, 2, 1],  # k = 1: equal values only count within the same property
        ]
    )
    assert study.shared_counts(query, candidates).tolist() == [[0, 1, 2, 3, 4, 5, 1]]


def test_theory():
    assert study.expected_cosine(1) == pytest.approx(0.2)
    assert study.spread(10_000) == pytest.approx(0.01)
    assert study.threshold(1, 2_048) == pytest.approx(0.2 - 3.0902 / math.sqrt(2_048), abs=1e-4)
    # The binomial tail falls as the threshold or the dimension rises.
    assert study.tail_probability(0.1, 2_048) > study.tail_probability(0.13, 2_048)
    assert study.tail_probability(0.13, 2_048) > study.tail_probability(0.13, 4_096)


def test_all_pairs_matches_brute_force():
    generator = torch.Generator().manual_seed(5)
    # Small cardinalities so every k occurs, and a low dimension so false matches occur.
    cardinalities = (2, 2, 3, 3, 4)
    codes = torch.stack(
        [torch.randint(0, c, (300,), generator=generator) for c in cardinalities], 1
    )
    encoder = Encoder.make(64, 3, cardinalities)
    sizes = (100, 300)
    result, hist = study.all_pairs(encoder, codes, sizes=sizes, tile=50, histogram=True)

    scores = study.cosines(encoder.encode(codes), encoder.encode(codes))
    k = study.shared_counts(codes, codes)
    for n, row in zip(sizes, result.iter_rows(named=True)):
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        s, kk = scores[:n, :n], k[:n, :n]
        unrelated = upper & (kk == 0)
        assert row["pairs"] == n * (n - 1) // 2
        assert row["unrelated_pairs"] == int(unrelated.sum())
        assert row["highest_unrelated"] == pytest.approx(float(s[unrelated].max()))
        for q in study.MATCHES:
            t = study.threshold(q, 64)
            false = unrelated & (s >= t)
            assert row[f"false_pairs_{q}"] == int(false.sum())
            rows, cols = torch.nonzero(false, as_tuple=True)
            assert row[f"searches_with_false_{q}"] == len(set(rows.tolist()) | set(cols.tolist()))
            true = upper & (kk >= q)
            assert row[f"true_pairs_{q}"] == int(true.sum())
            assert row[f"misses_{q}"] == int((true & (s < t)).sum())
    assert result["false_pairs_1"].max() > 0  # the low dimension makes the check meaningful
    assert int(hist["count"].sum()) == int(
        (torch.triu(torch.ones(300, 300, dtype=torch.bool), 1) & (k <= 2)).sum()
    )
