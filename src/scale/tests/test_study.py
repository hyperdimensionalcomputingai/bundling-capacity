"""Focused checks from the methodology's acceptance list. Run before the full study."""

import math
from fractions import Fraction

import polars as pl
import pytest
import torch
import torchhd

import study
from study import Encoder


def test_majority_bundle_means_by_exact_enumeration():
    expected = [Fraction(0), Fraction(9, 64), Fraction(18, 64), Fraction(28, 64)]
    expected += [Fraction(40, 64), Fraction(1)]
    assert [study.expected_cosine(k) for k in study.GROUPS] == expected


def test_encoder_binds_each_value_to_its_role_and_majority_bundles_five_facts():
    encoder = Encoder.make(512, 11)
    codes = torch.tensor([[3, 1, 42, 199, 7], [0, 9, 0, 0, 999]])
    for record, row in zip(encoder.encode(codes), codes):
        facts = torch.stack(
            [
                torchhd.bind(encoder.roles[i], encoder.values[i][row[i]]).as_subclass(torch.Tensor)
                for i in range(study.FACTS)
            ]
        )
        # One majority over all five facts; with five bipolar inputs it never ties.
        reference = torch.sign(torchhd.multiset(facts).as_subclass(torch.Tensor))
        assert torch.equal(record, reference)
        assert set(record.unique().tolist()) == {-1.0, 1.0}
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
    assert result.filter(pl.col("k") == 5)["max"].to_list()[:2] == [1.0, 1.0]


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

    vectors = encoder.encode(codes)
    cosine = (vectors[:query_count] @ vectors.T) / 512
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
                    assert row["std"] == pytest.approx(float(scores.std(correction=0)), abs=1e-9)
                    assert row["max"] == float(scores.max())
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
                assert row["std"] == pytest.approx(float(scores.std(correction=0)), abs=1e-9)
                assert row["max"] == float(scores.max())


def test_controlled_pairs_agree_with_theory():
    """Independent random pairs with exactly k shared facts, at the smallest and a large D."""
    pairs = 4_000
    for dimension in (512, 8_192):
        for k in study.GROUPS:
            facts = torchhd.random(pairs * (2 * study.FACTS - k), dimension, "MAP").reshape(
                pairs, 2 * study.FACTS - k, dimension
            )
            query = torch.sign(facts[:, : study.FACTS].sum(1))
            candidate = torch.sign(torch.cat([facts[:, :k], facts[:, study.FACTS :]], 1).sum(1))
            scores = (query * candidate).sum(1) / dimension
            sigma = study.spread(k, dimension)
            # Five standard errors of the mean; spread within 10% of the prediction.
            assert abs(float(scores.mean()) - study.MU[k]) <= 5 * sigma / math.sqrt(pairs) + 1e-12
            assert float(scores.std()) == pytest.approx(sigma, rel=0.1, abs=1e-12)


def test_theory_respects_the_discrete_cosine_grid():
    for dimension in study.DIMENSIONS:
        for comparisons in (10**4, 10**8):
            ref = study.maximum_reference(comparisons, dimension)
            agreeing = dimension * (1 + ref) / 2
            assert agreeing == pytest.approx(round(agreeing), abs=1e-6)  # attainable
            assert study.tail_probability(0, ref, dimension) <= 1 / comparisons
            below = ref - 2 / dimension  # the next attainable cosine down
            assert study.tail_probability(0, below, dimension) > 1 / comparisons
    assert study.spread(0, 10_000) == pytest.approx(0.01)
    assert study.spread(2, 512) == pytest.approx(math.sqrt((1 - 0.28125**2) / 512))
