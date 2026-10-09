"""Helpers for the scale study: the record encoder, the theory baseline and the batched scan.

Every record bundles five bound facts, h_fact,i = h_role,i (x) h_value,i, with one
coordinate-wise majority sign over all five. Queries and candidates use the same encoder.
Scores are exact cosines between bipolar vectors: dot / D, where the dot product is an
integer that float32 matrix multiplication represents exactly for D <= 2**24.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import polars as pl
import torch
import torchhd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
RECORDS_PATH = ROOT / "data" / "scale" / "records.parquet"
OUT = ROOT / "results" / "scale"

PROPERTIES = ("region", "education", "occupation", "interest_cluster", "employer")
CARDINALITIES = (20, 10, 100, 200, 1_000)
FACTS = len(PROPERTIES)  # odd, so the majority sign never ties

# Run settings, as fixed by the methodology. They describe this study's grid, not optima.
DIMENSIONS = (512, 2_048, 4_096, 8_192, 10_000)
PREFIXES = (10_000, 100_000, 1_000_000)
VECTOR_SEEDS = (11, 23, 37)
QUERY_COUNT = 100  # the first 100 records, compared with every other record in each prefix
GROUPS = tuple(range(FACTS + 1))  # k = 0..5 shared properties

# Candidates encoded per batch. It divides every prefix, so prefix summaries fall on
# batch boundaries, and keeps the largest batch (D = 10,000) near 200 MB in float32.
BATCH = 5_000


# ----------------------------------------------------------------------------- records


def load_codes(path: Path = RECORDS_PATH) -> torch.Tensor:
    """Categorical codes as an (N, 5) int64 tensor, in record-sequence order."""
    frame = pl.read_parquet(path, columns=["record_id", *PROPERTIES])
    if frame["record_id"].to_list() != list(range(frame.height)):
        raise ValueError("records.parquet must be in record_id order")
    return frame.select(PROPERTIES).to_torch(dtype=pl.Int64)


def shared_counts(query_codes: torch.Tensor, candidate_codes: torch.Tensor) -> torch.Tensor:
    """k for every query-candidate pair: identical values counted within the same property."""
    return (query_codes[:, None, :] == candidate_codes[None, :, :]).sum(-1)


# ----------------------------------------------------------------------------- encoder


@dataclass(frozen=True)
class Encoder:
    """Role and value vectors for one (D, vector seed), with each property's bound facts.

    `facts[i][v]` is h_role,i (x) h_value,i,v, precomputed for every value v so a batch of
    records encodes by lookup. The facts are stored as int8 (+/-1) so the five-way sum
    stays small; it is cast to float32 only for the cosine computation.
    """

    dimension: int
    seed: int
    roles: torch.Tensor
    values: tuple[torch.Tensor, ...]
    facts: tuple[torch.Tensor, ...]

    @classmethod
    def make(cls, dimension: int, seed: int, cardinalities=CARDINALITIES) -> Encoder:
        # One generator per (D, seed): every dimension draws a fresh, independent set of
        # vectors, and the draw is reproducible from the seed alone.
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
        """Majority-bundle the five facts of each record, in one operation, as float32 +/-1.

        The sum of all five bound facts is taken first and the sign applied once, which is
        the coordinate-wise majority. No sign is applied to any intermediate partial sum.
        """
        total = self.facts[0][codes[:, 0]].clone()
        for i in range(1, len(self.facts)):
            total += self.facts[i][codes[:, i]]
        return torch.sign(total).to(torch.float32)


# ----------------------------------------------------------------------------- theory


def expected_cosine(shared: int, facts: int = FACTS) -> Fraction:
    """Exact E[cosine] between two `facts`-way majority bundles that share `shared` facts.

    Enumerates every bipolar input at one coordinate: the shared facts, the query's own
    facts and the candidate's own facts. All are independent and uniform under the ideal
    random-vector model, and coordinates are identically distributed, so the per-coordinate
    expectation of the product of the two signs is the expected cosine.
    """
    own = facts - shared
    total = 0
    for bits in itertools.product((-1, 1), repeat=shared + 2 * own):
        common, mine, theirs = bits[:shared], bits[shared : shared + own], bits[shared + own :]
        query = 1 if sum(common) + sum(mine) > 0 else -1
        candidate = 1 if sum(common) + sum(theirs) > 0 else -1
        total += query * candidate
    return Fraction(total, 2 ** len(bits))


MU = {k: float(expected_cosine(k)) for k in GROUPS}


def spread(k: int, dimension: int) -> float:
    """sigma_k = sqrt((1 - mu_k^2) / D), the ideal-model standard deviation of one score."""
    return math.sqrt((1 - MU[k] ** 2) / dimension)


def agreement_threshold(score: float, dimension: int) -> int:
    """Smallest count of agreeing coordinates H whose cosine (2H - D) / D is at least `score`."""
    return math.ceil(dimension * (1 + score) / 2 - 1e-9)


def tail_probability(k: int, score: float, dimension: int) -> float:
    """p_k(s) = P(H >= ceil(D(1 + s)/2)), with H ~ Binomial(D, (1 + mu_k)/2)."""
    h = agreement_threshold(score, dimension)
    return float(binom.sf(h - 1, dimension, (1 + MU[k]) / 2))


def maximum_reference(comparisons: int, dimension: int, k: int = 0) -> float:
    """The cosine whose ideal tail probability first falls to about 1 / M.

    Returns the smallest attainable cosine (2H - D) / D with P(score >= it) <= 1 / M, an
    illustrative reference for the highest score among M group-k comparisons.
    """
    if comparisons <= 0:
        return math.nan
    p = (1 + MU[k]) / 2
    h = torch.arange(dimension + 1, dtype=torch.float64)
    survival = torch.from_numpy(binom.sf(h.numpy() - 1, dimension, p))
    first = int(torch.nonzero(survival <= 1 / comparisons)[0])
    return (2 * first - dimension) / dimension


# ----------------------------------------------------------------------------- scan


def exceedance_scores() -> dict[str, float]:
    """Reference cosines for counting high zero-overlap scores, fixed before measuring.

    The one-property mean mu_1 and the midpoint mu_1 / 2. Neither is a match threshold;
    they are score levels at which theory predicts how many zero-overlap comparisons reach.
    """
    return {"half_mu1": MU[1] / 2, "mu1": MU[1]}


def scan(
    encoder: Encoder,
    codes: torch.Tensor,
    prefixes=PREFIXES,
    query_count: int = QUERY_COUNT,
    batch: int = BATCH,
) -> pl.DataFrame:
    """Compare the query panel with every other record, accumulating per-query summaries.

    One pass through the record sequence; a snapshot is taken as the scan reaches each
    prefix. Per (query, k) the scan keeps exact integer sums of the dot products and their
    squares, so the means and standard deviations are exact up to the final float64
    division. A query's own record ID is excluded; duplicates of its attributes are kept.
    """
    dimension = encoder.dimension
    if any(n % batch for n in prefixes) or max(prefixes) > len(codes):
        raise ValueError("Every prefix must be a multiple of the batch and fit the records")
    query_codes = codes[:query_count]
    queries = encoder.encode(query_codes)
    groups = torch.tensor(GROUPS)
    shape = (query_count, len(GROUPS))
    count = torch.zeros(shape, dtype=torch.int64)
    total = torch.zeros(shape, dtype=torch.int64)
    squares = torch.zeros(shape, dtype=torch.int64)
    highest = torch.full(shape, -dimension - 1, dtype=torch.int64)
    references = exceedance_scores()
    # Score >= s  <=>  H >= h  <=>  dot >= 2h - D, an exact integer comparison.
    floors = {
        name: 2 * agreement_threshold(s, dimension) - dimension for name, s in references.items()
    }
    exceed = {name: torch.zeros(query_count, dtype=torch.int64) for name in references}
    query_ids = torch.arange(query_count)
    rows = []
    for start in range(0, max(prefixes), batch):
        stop = start + batch
        candidates = encoder.encode(codes[start:stop])
        dots = (queries @ candidates.T).round().to(torch.int64)  # exact integers
        k = shared_counts(query_codes, codes[start:stop])
        candidate_ids = torch.arange(start, stop)
        k[query_ids[:, None] == candidate_ids[None, :]] = -1  # self-exclusion
        member = k[:, :, None] == groups  # (Q, B, groups)
        dot3 = dots[:, :, None]
        count += member.sum(1)
        total += (dot3 * member).sum(1)
        squares += (dot3 * dot3 * member).sum(1)
        masked = torch.where(member, dot3, torch.full_like(dot3, -dimension - 1))
        highest = torch.maximum(highest, masked.amax(1))
        zero = k == 0
        for name, floor in floors.items():
            exceed[name] += (zero & (dots >= floor)).sum(1)
        if stop in prefixes:
            rows.append(_snapshot(encoder, stop, count, total, squares, highest, exceed))
    return pl.concat(rows)


def _snapshot(encoder, n, count, total, squares, highest, exceed) -> pl.DataFrame:
    """Per-query, per-k rows at one prefix: raw integer sums plus readable cosines."""
    query_count = count.shape[0]
    d = encoder.dimension
    frame = pl.DataFrame(
        {
            "dimension": pl.Series([d] * (query_count * len(GROUPS)), dtype=pl.Int32),
            "seed": pl.Series([encoder.seed] * (query_count * len(GROUPS)), dtype=pl.Int32),
            "n": pl.Series([n] * (query_count * len(GROUPS)), dtype=pl.Int32),
            "query_id": pl.Series([q for q in range(query_count) for _ in GROUPS], dtype=pl.Int32),
            "k": pl.Series(list(GROUPS) * query_count, dtype=pl.Int8),
            "count": count.flatten().tolist(),
            "dot_sum": total.flatten().tolist(),
            "dot_sq_sum": squares.flatten().tolist(),
            "max_dot": highest.flatten().tolist(),
        },
        schema_overrides={
            "count": pl.Int64,
            "dot_sum": pl.Int64,
            "dot_sq_sum": pl.Int64,
            "max_dot": pl.Int64,
        },
    )
    for name, values in exceed.items():
        # Exceedance counts are recorded on the k = 0 rows only; other rows hold null.
        column = [values[q].item() if k == 0 else None for q in range(query_count) for k in GROUPS]
        frame = frame.with_columns(pl.Series(f"zero_at_or_above_{name}", column, dtype=pl.Int64))
    return frame.with_columns(
        (pl.col("dot_sum") / (pl.col("count") * d)).alias("mean"),
        (
            (pl.col("dot_sq_sum") / pl.col("count") - (pl.col("dot_sum") / pl.col("count")) ** 2)
            .clip(lower_bound=0)
            .sqrt()
            / d
        ).alias("std"),
        pl.when(pl.col("count") > 0).then(pl.col("max_dot") / d).alias("max"),
    )


def pool(per_query: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Pool per-query integer sums into exact group summaries over all comparisons."""
    frame = (
        per_query.group_by(keys)
        .agg(
            pl.col("count").sum(),
            pl.col("dot_sum").sum(),
            pl.col("dot_sq_sum").sum(),
            pl.col("max_dot").max(),
            pl.col("^zero_at_or_above_.*$").sum(),
        )
        .sort(keys)
    )
    d = pl.col("dimension")
    mean_dot = pl.col("dot_sum") / pl.col("count")
    return frame.with_columns(
        (mean_dot / d).alias("mean"),
        (
            (pl.col("dot_sq_sum") / pl.col("count") - mean_dot**2).clip(lower_bound=0).sqrt() / d
        ).alias("std"),
        pl.when(pl.col("count") > 0).then(pl.col("max_dot") / d).alias("max"),
    )
