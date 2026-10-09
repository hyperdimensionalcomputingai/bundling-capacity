"""Helpers for the scale study: the record encoder, the theory baseline and the batched scan.

Every record bundles five bound facts, h_fact,i = h_role,i (x) h_value,i, as their raw
coordinate-wise sum, so each fact can still be unbound. Queries and candidates use the same
encoder. Scores are cosines: the integer dot product over the product of the two lengths.
"""

from __future__ import annotations

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
FACTS = len(PROPERTIES)

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
        """Bundle the five facts of each record as their raw sum, in float32 (no sign)."""
        total = self.facts[0][codes[:, 0]].clone()
        for i in range(1, len(self.facts)):
            total += self.facts[i][codes[:, i]]
        return total.to(torch.float32)


def cosines(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Cosines in float64; the dot products are exact integers (|dot| <= 25 D < 2**24)."""
    dots = (queries @ candidates.T).double()
    lengths = queries.double().norm(dim=1)[:, None] * candidates.double().norm(dim=1)[None, :]
    return dots / lengths


# ----------------------------------------------------------------------------- theory


def expected_cosine(shared: int, facts: int = FACTS) -> Fraction:
    """mu_k = k / F: shared facts add D to the dot product, and each length is about sqrt(F D)."""
    return Fraction(shared, facts)


MU = {k: float(expected_cosine(k)) for k in GROUPS}


def spread(dimension: int) -> float:
    """sigma_0 = 1 / sqrt(D), the ideal-model standard deviation of a zero-overlap score."""
    return 1 / math.sqrt(dimension)


def agreement_threshold(score: float, dimension: int) -> int:
    """Smallest count of agreeing coordinates H whose cosine (2H - D) / D is at least `score`."""
    return math.ceil(dimension * (1 + score) / 2 - 1e-9)


def tail_probability(score: float, dimension: int) -> float:
    """p_0(s) = P(H >= ceil(D(1 + s)/2)), with H ~ Binomial(D, 1/2).

    The binomial model scores a zero-overlap pair as (2H - D) / D, H agreeing coordinates out
    of D. It is exact for bipolar records; for raw-sum records it is an approximation with
    the same mean (0) and spread (1 / sqrt(D)).
    """
    h = agreement_threshold(score, dimension)
    return float(binom.sf(h - 1, dimension, 0.5))


def maximum_reference(comparisons: int, dimension: int) -> float:
    """The score whose binomial tail probability first falls to about 1 / M.

    Returns the smallest (2H - D) / D with P(score >= it) <= 1 / M, an illustrative
    reference for the highest score among M zero-overlap comparisons.
    """
    if comparisons <= 0:
        return math.nan
    h = torch.arange(dimension + 1, dtype=torch.float64)
    survival = torch.from_numpy(binom.sf(h.numpy() - 1, dimension, 0.5))
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
    prefix. Per (query, k) the scan keeps float64 sums of the cosines and their squares, and
    the highest cosine. A query's own record ID is excluded; duplicates are kept.
    """
    if any(n % batch for n in prefixes) or max(prefixes) > len(codes):
        raise ValueError("Every prefix must be a multiple of the batch and fit the records")
    query_codes = codes[:query_count]
    queries = encoder.encode(query_codes)
    groups = torch.tensor(GROUPS)
    shape = (query_count, len(GROUPS))
    count = torch.zeros(shape, dtype=torch.int64)
    total = torch.zeros(shape, dtype=torch.float64)
    squares = torch.zeros(shape, dtype=torch.float64)
    highest = torch.full(shape, -math.inf, dtype=torch.float64)
    references = exceedance_scores()
    exceed = {name: torch.zeros(query_count, dtype=torch.int64) for name in references}
    query_ids = torch.arange(query_count)
    rows = []
    for start in range(0, max(prefixes), batch):
        stop = start + batch
        scores = cosines(queries, encoder.encode(codes[start:stop]))
        k = shared_counts(query_codes, codes[start:stop])
        candidate_ids = torch.arange(start, stop)
        k[query_ids[:, None] == candidate_ids[None, :]] = -1  # self-exclusion
        member = k[:, :, None] == groups  # (Q, B, groups)
        score3 = scores[:, :, None]
        count += member.sum(1)
        total += (score3 * member).sum(1)
        squares += (score3 * score3 * member).sum(1)
        masked = torch.where(member, score3, torch.full_like(score3, -math.inf))
        highest = torch.maximum(highest, masked.amax(1))
        zero = k == 0
        for name, s in references.items():
            exceed[name] += (zero & (scores >= s)).sum(1)
        if stop in prefixes:
            rows.append(_snapshot(encoder, stop, count, total, squares, highest, exceed))
    return pl.concat(rows)


def _snapshot(encoder, n, count, total, squares, highest, exceed) -> pl.DataFrame:
    """Per-query, per-k rows at one prefix: running sums plus readable cosines."""
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
            "cos_sum": total.flatten().tolist(),
            "cos_sq_sum": squares.flatten().tolist(),
            "max_cos": highest.flatten().tolist(),
        },
        schema_overrides={
            "count": pl.Int64,
            "cos_sum": pl.Float64,
            "cos_sq_sum": pl.Float64,
            "max_cos": pl.Float64,
        },
    )
    for name, values in exceed.items():
        # Exceedance counts are recorded on the k = 0 rows only; other rows hold null.
        column = [values[q].item() if k == 0 else None for q in range(query_count) for k in GROUPS]
        frame = frame.with_columns(pl.Series(f"zero_at_or_above_{name}", column, dtype=pl.Int64))
    return _readable(frame)


def _readable(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean, standard deviation and maximum cosine from the running sums."""
    mean = pl.col("cos_sum") / pl.col("count")
    return frame.with_columns(
        mean.alias("mean"),
        (pl.col("cos_sq_sum") / pl.col("count") - mean**2).clip(lower_bound=0).sqrt().alias("std"),
        pl.when(pl.col("count") > 0).then(pl.col("max_cos")).alias("max"),
    )


def pool(per_query: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Pool per-query running sums into group summaries over all comparisons."""
    frame = (
        per_query.group_by(keys)
        .agg(
            pl.col("count").sum(),
            pl.col("cos_sum").sum(),
            pl.col("cos_sq_sum").sum(),
            pl.col("max_cos").max(),
            pl.col("^zero_at_or_above_.*$").sum(),
        )
        .sort(keys)
    )
    return _readable(frame)
