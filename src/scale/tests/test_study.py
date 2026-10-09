"""Focused checks from the methodology's acceptance list. Run before the full study."""

import math
from fractions import Fraction

import polars as pl
import pytest
import torch
import torchhd

import study
from study import Encoder


def test_raw_sum_means_are_k_over_five():
    assert [study.expected_cosine(k) for k in study.GROUPS] == [Fraction(k, 5) for k in range(6)]


def test_encoder_binds_each_value_to_its_role_and_sums_five_facts():
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
        assert torch.equal(record, torchhd.multiset(facts).as_subclass(torch.Tensor))
        assert set(record.unique().tolist()) <= {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0}
        # Unbinding the employer role recovers the employer value as the best match.
        assert int((encoder.values[4] @ (record * encoder.roles[4])).argmax()) == int(row[4])
    # Rebuilding the encoder from the same (D, seed) gives the same vectors.
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
            [1, 2, 3, 4, 5],  # k = 5, a duplicate attribute record
            [5, 4, 3, 2, 1],  # k = 1: equal values only count within the same property
        ]
    )
    assert study.shared_counts(query, candidates).tolist() == [[0, 1, 2, 3, 4, 5, 1]]


def test_scan_keeps_duplicates_and_excludes_only_the_query_itself():
    # Records 0 and 1 are identical attribute records; record 2 shares nothing with 0.
    codes = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    result = study.scan(Encoder.make(512, 11), codes, prefixes=(4,), query_count=2, batch=2)
    counts = {
        (row["query_id"], row["k"]): row["count"]
        for row in result.iter_rows(named=True)
        if row["count"]
    }
    # Each query sees the other identical record at k = 5, never itself.
    assert counts == {(0, 5): 1, (0, 0): 1, (0, 1): 1, (1, 5): 1, (1, 0): 1, (1, 1): 1}
    assert result.filter(pl.col("k") == 5)["max"].to_list()[:2] == pytest.approx([1.0, 1.0])


def test_small_end_to_end_run_matches_brute_force():
    generator = torch.Generator().manual_seed(5)
    # Small cardinalities so every k from 0 to 5 occurs among a few hundred records.
    cardinalities = (2, 2, 3, 3, 4)
    codes = torch.stack(
        [torch.randint(0, c, (600,), generator=generator) for c in cardinalities], 1
    )
    encoder = Encoder.make(512, 37, cardinalities)
    prefixes, query_count = (150, 300, 600), 7
    result = study.scan(encoder, codes, prefixes=prefixes, query_count=query_count, batch=75)

    cosine = study.cosines(encoder.encode(codes[:query_count]), encoder.encode(codes))
    k = study.shared_counts(codes[:query_count], codes)
    for n in prefixes:
        rows = result.filter(pl.col("n") == n)
        for q in range(query_count):
            for g in study.GROUPS:
                keep = (k[q, :n] == g) & (torch.arange(n) != q)
                row = rows.filter((pl.col("query_id") == q) & (pl.col("k") == g)).row(0, named=True)
                assert row["count"] == int(keep.sum())
                if keep.any():
                    scores = cosine[q, :n][keep].double()
                    assert row["mean"] == pytest.approx(float(scores.mean()), abs=1e-12)
                    assert row["std"] == pytest.approx(float(scores.std(correction=0)), abs=1e-7)
                    assert row["max"] == pytest.approx(float(scores.max()))
                if g == 0:
                    for name, s in study.exceedance_scores().items():
                        assert row[f"zero_at_or_above_{name}"] == int((scores >= s).sum())
        pooled = study.pool(rows, ["dimension", "seed", "n", "k"])
        for g in study.GROUPS:
            keep = (k[:, :n] == g) & (
                torch.arange(n)[None, :] != torch.arange(query_count)[:, None]
            )
            row = pooled.filter(pl.col("k") == g).row(0, named=True)
            assert row["count"] == int(keep.sum())
            if keep.any():
                scores = cosine[:, :n][keep].double()
                assert row["mean"] == pytest.approx(float(scores.mean()), abs=1e-12)
                assert row["std"] == pytest.approx(float(scores.std(correction=0)), abs=1e-7)
                assert row["max"] == pytest.approx(float(scores.max()))


def test_controlled_pairs_agree_with_theory():
    """Independent random pairs with exactly k shared facts, at the smallest and a large D."""
    pairs = 4_000
    for dimension in (512, 8_192):
        for k in study.GROUPS:
            facts = torchhd.random(pairs * (2 * study.FACTS - k), dimension, "MAP").reshape(
                pairs, 2 * study.FACTS - k, dimension
            )
            query = facts[:, : study.FACTS].sum(1)
            candidate = torch.cat([facts[:, :k], facts[:, study.FACTS :]], 1).sum(1)
            scores = study.cosines(query, candidate).diagonal()
            sigma = study.spread(dimension)  # zero-overlap spread; shared facts only narrow it
            # Five standard errors of the mean; the zero-overlap spread within 10% of 1/sqrt(D).
            assert abs(float(scores.mean()) - study.MU[k]) <= 5 * sigma / math.sqrt(pairs) + 1e-12
            assert float(scores.std()) <= sigma * 1.1
            if k == 0:
                assert float(scores.std()) == pytest.approx(sigma, rel=0.1)


def test_theory_respects_the_discrete_cosine_grid():
    for dimension in study.DIMENSIONS:
        for comparisons in (10**4, 10**8):
            ref = study.maximum_reference(comparisons, dimension)
            agreeing = dimension * (1 + ref) / 2
            assert agreeing == pytest.approx(round(agreeing), abs=1e-6)  # attainable
            assert study.tail_probability(ref, dimension) <= 1 / comparisons
            below = ref - 2 / dimension  # the next attainable cosine down
            assert study.tail_probability(below, dimension) > 1 / comparisons
    assert study.spread(10_000) == pytest.approx(0.01)
