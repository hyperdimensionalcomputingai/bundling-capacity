"""Helpers for the collision study: the record encoder, the theory and the all-pairs check.

Every record bundles five bound facts, h_fact,i = h_role,i (x) h_value,i, as their raw
coordinate-wise sum, so each fact can still be unbound. Scores are cosines: the integer dot
product over the product of the two record lengths.

The study compares every record with every other record once. The same scores answer two
practical questions: does one person's search return an unrelated record ("searching for
one person"), and does any unrelated pair anywhere score like a match ("checking the whole
dataset")?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch
import torchhd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parents[2]
RECORDS_PATH = ROOT / "data" / "scale" / "records.parquet"
OUT = ROOT / "results" / "scale"

PROPERTIES = ("region", "education", "occupation", "interest_cluster", "employer")
CARDINALITIES = (20, 10, 100, 200, 1_000)
FACTS = len(PROPERTIES)

# Run settings. They describe this study's grid, not optima.
DIMENSIONS = (2_048, 4_096, 8_192)
SIZES = (4_000, 13_000, 40_000)  # nested: each population is the first N records
TRIALS = {2_048: 40, 4_096: 3, 8_192: 3}  # fresh hypervectors per trial
MATCHES = (1, 2)  # a "match" shares at least this many properties
RECALL = 0.999  # thresholds aim to keep 99.9% of true matches
TILE = 1_000  # divides every size, so each tile belongs to one size bucket

# Score histograms, recorded for the first trial at the largest size.
HISTOGRAM_GROUPS = (0, 1, 2)
HISTOGRAM_BINS = torch.linspace(-0.3, 0.7, 401)


# ----------------------------------------------------------------------------- records


def load_codes(path: Path = RECORDS_PATH) -> torch.Tensor:
    """Categorical codes as an (N, 5) int64 tensor, in record-sequence order."""
    frame = pl.read_parquet(path, columns=["record_id", *PROPERTIES])
    if frame["record_id"].to_list() != list(range(frame.height)):
        raise ValueError("records.parquet must be in record_id order")
    return frame.select(PROPERTIES).to_torch(dtype=pl.Int64)


def shared_counts(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """k for every pair: identical values counted within the same property."""
    return (a[:, None, :] == b[None, :, :]).sum(-1, dtype=torch.int8)


# ----------------------------------------------------------------------------- encoder


@dataclass(frozen=True)
class Encoder:
    """Role and value hypervectors for one (D, seed), with each property's bound facts.

    `facts[i][v]` is h_role,i (x) h_value,i,v, precomputed for every value v so a batch of
    records encodes by lookup. Facts are stored as int8 (+/-1).
    """

    dimension: int
    seed: int
    roles: torch.Tensor
    values: tuple[torch.Tensor, ...]
    facts: tuple[torch.Tensor, ...]

    @classmethod
    def make(cls, dimension: int, seed: int, cardinalities=CARDINALITIES) -> Encoder:
        generator = torch.Generator().manual_seed(seed * 1_000_003 + dimension)
        roles = torchhd.random(len(cardinalities), dimension, "MAP", generator=generator)
        values = tuple(
            torchhd.random(c, dimension, "MAP", generator=generator) for c in cardinalities
        )
        facts = tuple(
            torchhd.bind(roles[i : i + 1], values[i]).as_subclass(torch.Tensor).to(torch.int8)
            for i in range(len(cardinalities))
        )
        return cls(dimension, seed, roles, values, facts)

    def encode(self, codes: torch.Tensor) -> torch.Tensor:
        """The raw sum of each record's five bound facts, as int8 (no sign)."""
        total = self.facts[0][codes[:, 0]].clone()
        for i in range(1, len(self.facts)):
            total += self.facts[i][codes[:, i]]
        return total


def cosines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosines in float64; the dot products are exact integers (|dot| <= 25 D < 2**24)."""
    a, b = a.to(torch.float32), b.to(torch.float32)
    dots = (a @ b.T).double()
    lengths = a.double().norm(dim=1)[:, None] * b.double().norm(dim=1)[None, :]
    return dots / lengths


# ----------------------------------------------------------------------------- theory


def expected_cosine(shared: int) -> float:
    """mu_k = k / F: a shared fact adds D to the dot product; each length is about sqrt(F D)."""
    return shared / FACTS


def spread(dimension: int) -> float:
    """1 / sqrt(D): the spread of an unrelated pair's score, used for every group."""
    return 1 / math.sqrt(dimension)


def threshold(match: int, dimension: int) -> float:
    """T = mu_q - z sigma, placed so about 99.9% of pairs sharing q properties score above it.

    Using 1 / sqrt(D) for sigma slightly overstates the spread of pairs that share facts,
    so the threshold sits a little low: conservative for misses.
    """
    return expected_cosine(match) - norm.ppf(RECALL) * spread(dimension)


def tail_probability(score: float, dimension: int) -> float:
    """p(T): chance that one unrelated pair scores at least T, from the binomial model.

    The model scores a pair as (2H - D) / D with H ~ Binomial(D, 1/2) agreeing coordinates.
    It is exact for +/-1 records and approximate for raw sums, with the same mean (0) and
    spread (1 / sqrt(D)).
    """
    agreeing = math.ceil(dimension * (1 + score) / 2 - 1e-9)
    return float(binom.sf(agreeing - 1, dimension, 0.5))


# ----------------------------------------------------------------------------- all pairs


@dataclass
class _Bucket:
    """Running counts for the pairs whose larger record index falls in one size bucket."""

    pairs: int = 0
    unrelated: int = 0
    highest_unrelated: float = -math.inf
    true_pairs: dict[int, int] = field(default_factory=lambda: dict.fromkeys(MATCHES, 0))
    misses: dict[int, int] = field(default_factory=lambda: dict.fromkeys(MATCHES, 0))
    false_pairs: dict[int, list[tuple[int, int]]] = field(
        default_factory=lambda: {q: [] for q in MATCHES}
    )


def all_pairs(
    encoder: Encoder,
    codes: torch.Tensor,
    sizes=SIZES,
    tile: int = TILE,
    histogram: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Compare every record with every other record once, for each nested size N.

    Returns one row per size with pair counts, false matches (unrelated pairs at or above
    each threshold), records whose own search returned a false match, misses (pairs sharing
    at least q properties that score below the threshold) and the highest unrelated score.
    With `histogram`, also returns score histograms by shared-property count at the largest
    size.
    """
    largest = max(sizes)
    if any(n % tile for n in sizes) or largest > len(codes):
        raise ValueError("Every size must be a multiple of the tile and fit the records")
    codes = codes[:largest]
    vectors = encoder.encode(codes)
    thresholds = {q: threshold(q, encoder.dimension) for q in MATCHES}
    bucket_of = {
        end: next(i for i, n in enumerate(sizes) if end <= n)
        for end in range(tile, largest + 1, tile)
    }
    buckets = [_Bucket() for _ in sizes]
    counts = torch.zeros(len(HISTOGRAM_GROUPS), len(HISTOGRAM_BINS) - 1, dtype=torch.int64)

    for r0 in range(0, largest, tile):
        for c0 in range(r0, largest, tile):
            c1 = c0 + tile
            bucket = buckets[bucket_of[c1]]  # c1 >= r1, so the column end decides the size
            scores = cosines(vectors[r0 : r0 + tile], vectors[c0:c1])
            k = shared_counts(codes[r0 : r0 + tile], codes[c0:c1])
            keep = torch.ones_like(k, dtype=torch.bool)
            if r0 == c0:
                keep = torch.triu(keep, diagonal=1)  # each pair once, never a record with itself
            unrelated = keep & (k == 0)
            bucket.pairs += int(keep.sum())
            bucket.unrelated += int(unrelated.sum())
            if unrelated.any():
                top = float(scores[unrelated].max())
                bucket.highest_unrelated = max(bucket.highest_unrelated, top)
            for q, t in thresholds.items():
                true = keep & (k >= q)
                bucket.true_pairs[q] += int(true.sum())
                bucket.misses[q] += int((true & (scores < t)).sum())
                rows, cols = torch.nonzero(unrelated & (scores >= t), as_tuple=True)
                bucket.false_pairs[q] += list(zip((rows + r0).tolist(), (cols + c0).tolist()))
            if histogram:
                for g, group in enumerate(HISTOGRAM_GROUPS):
                    selected = scores[keep & (k == group)].float()
                    counts[g] += torch.histogram(selected, HISTOGRAM_BINS).hist.long()

    rows = []
    for i, n in enumerate(sizes):
        included = buckets[: i + 1]
        row = {
            "dimension": encoder.dimension,
            "seed": encoder.seed,
            "n": n,
            "pairs": sum(b.pairs for b in included),
            "unrelated_pairs": sum(b.unrelated for b in included),
            "highest_unrelated": max(b.highest_unrelated for b in included),
        }
        for q in MATCHES:
            false = [pair for b in included for pair in b.false_pairs[q]]
            row[f"threshold_{q}"] = thresholds[q]
            row[f"false_pairs_{q}"] = len(false)
            row[f"searches_with_false_{q}"] = len({r for pair in false for r in pair})
            row[f"true_pairs_{q}"] = sum(b.true_pairs[q] for b in included)
            row[f"misses_{q}"] = sum(b.misses[q] for b in included)
        rows.append(row)
    hist = None
    if histogram:
        centers = ((HISTOGRAM_BINS[:-1] + HISTOGRAM_BINS[1:]) / 2).tolist()
        hist = pl.DataFrame(
            [
                {"dimension": encoder.dimension, "k": group, "score": c, "count": int(n)}
                for g, group in enumerate(HISTOGRAM_GROUPS)
                for c, n in zip(centers, counts[g].tolist())
            ]
        )
    return pl.DataFrame(rows), hist


def predictions(row: dict) -> dict:
    """Binomial predictions for one size: expected false pairs and the two chances.

    Expected false pairs = unrelated pairs x p(T). The chance that the dataset holds at
    least one is 1 - exp(-expected). One person's search makes about N - 1 comparisons, so
    its expected false matches are 2 x expected / N.
    """
    d, n = row["dimension"], row["n"]
    out = {}
    for q in MATCHES:
        p = tail_probability(row[f"threshold_{q}"], d)
        expected = row["unrelated_pairs"] * p
        out[f"p_{q}"] = p
        out[f"expected_false_pairs_{q}"] = expected
        out[f"dataset_chance_{q}"] = -math.expm1(-expected)
        out[f"search_chance_{q}"] = -math.expm1(-2 * expected / n)
    return out
