"""Exact streamed cosine search over all 920,000 PERSON records (Experiment 1).

One pass over the candidates serves every codebook. Per chunk:

1. The exact similarity S_exact is computed once from the integer records. It
   does not depend on the codebook.
2. For each (D, seed) codebook, read back from LanceDB, the chunk's bundles are
   encoded in float32 and scored against the query bundles.

Scores come from exact integer dot products, so records with identical vectors
tie exactly. Bookkeeping is float32; the only float64 values are running sums
over more than 1e8 pairs, where float32 would lose digits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import pairwise

import polars as pl
import torch

import person
import store
from person import Records

DIMENSIONS = (512, 1024, 2048, 4096, 8192)
SEEDS = (11, 23, 37)
PREFIXES = (100, 1_000, 10_000, 100_000, 920_000)
CALIBRATION_CANDIDATES = 100_000  # a balanced prefix; a pair's cosine does not depend on N
CHUNK = 8192

# Exact-similarity tiers, declared before any MAP score is inspected.
TIER_NAMES = ("unrelated_0", "low_(0,1/3)", "mid_[1/3,2/3)", "related_[2/3,1]")
RELATED_CUTOFF = 2 / 3
TOLERANCE = 1e-6
TOP_K = 10  # tie-aware agreement with the source ranking
HEAD = 100  # retrieved head kept per query for ordering and IVF_PQ ground truth
BETA = 0.01  # allowed false-negative rate per related-pair comparison
EPSILON = 0.01  # allowed probability of any unintended match, per query over N candidates
REFERENCE_T = 0.2157  # the earlier chat's D=2048 value, evaluated as a fixed reference only

BIN = 1e-4
METRICS = {
    "abs_numerical_error": (0.0, 0.5),
    "unrelated_score": (-1.0, 1.0),
    "related_score": (-1.0, 1.0),
}


def tiers(exact: torch.Tensor) -> torch.Tensor:
    t = torch.full(exact.shape, 3, dtype=torch.int8)
    t = torch.where(exact < RELATED_CUTOFF - TOLERANCE, 2, t)
    t = torch.where(exact < 1 / 3 - TOLERANCE, 1, t)
    return torch.where(exact <= TOLERANCE, 0, t).to(torch.int8)


def chunks(n: int):
    """Chunk ranges that never cross a nested-prefix boundary."""
    edges = sorted(set(range(0, n, CHUNK)) | {p for p in PREFIXES if p < n} | {n})
    for start, stop in pairwise(edges):
        yield start, stop, next(i for i, p in enumerate(PREFIXES) if stop <= p)


class Histograms:
    """All histograms of one scoring unit in one flat tensor, filled by one bincount per chunk."""

    def __init__(self, metrics, segments: int):
        self.metrics = list(metrics)
        self.bins = {m: round((METRICS[m][1] - METRICS[m][0]) / BIN) for m in self.metrics}
        self.offset, total = {}, 0
        for m in self.metrics:
            self.offset[m] = total
            total += self.bins[m]
        self.per_segment = total
        self.counts = torch.zeros(segments * total + 1, dtype=torch.long)  # last slot discards

    def index(
        self, metric: str, values: torch.Tensor, mask: torch.Tensor, segment: int
    ) -> torch.Tensor:
        b = ((values - METRICS[metric][0]) * (1 / BIN)).clamp(0, self.bins[metric] - 1).long()
        return torch.where(
            mask, b + (segment * self.per_segment + self.offset[metric]), len(self.counts) - 1
        )

    def add(self, parts: list[torch.Tensor]):
        self.counts += torch.bincount(
            torch.cat([p.reshape(-1) for p in parts]), minlength=len(self.counts)
        )

    def segment(self, metric: str, segment: int) -> torch.Tensor:
        start = segment * self.per_segment + self.offset[metric]
        return self.counts[start : start + self.bins[metric]]


@dataclass
class Block:
    """Codebook-independent quantities for one candidate chunk."""

    base: person.Baselines
    unrelated: torch.Tensor
    related: torch.Tensor
    tier: torch.Tensor


class Unit:
    """Running state for one (D, seed) codebook."""

    def __init__(self, book, q_sum: torch.Tensor, thresholds: dict, calibration: bool):
        self.book, self.calibration = book, calibration
        self.q_sum, self.thresholds = q_sum, thresholds
        q = len(q_sum)
        if calibration:
            self.hist = Histograms(["related_score"], 1)
            return
        self.hist = Histograms(METRICS, len(PREFIXES))
        inf = torch.full((q,), torch.inf)
        self.max_unrelated, self.min_unrelated = -inf, inf.clone()
        self.counts = {
            f"{kind}_{k}": torch.zeros(q, dtype=torch.long)
            for k in thresholds
            for kind in ("false_pos", "false_pos_abs", "false_neg")
        }
        self.head = {
            "score": torch.full((q, HEAD), -torch.inf),
            "exact": torch.full((q, HEAD), -torch.inf),
            "index": torch.full((q, HEAD), -1, dtype=torch.long),
        }

    def update(self, c_sum: torch.Tensor, c_index: torch.Tensor, block: Block, segment: int):
        score = person.cosine(self.q_sum, c_sum)
        h = self.hist
        if self.calibration:
            h.add([h.index("related_score", score, block.related, 0)])
            return
        base = block.base
        h.add(
            [
                h.index("abs_numerical_error", (score - base.exact).abs(), base.valid, segment),
                h.index("unrelated_score", score, block.unrelated, segment),
                h.index("related_score", score, block.related, segment),
            ]
        )
        unrelated = torch.where(block.unrelated, score, torch.nan)
        self.max_unrelated = torch.fmax(
            self.max_unrelated, unrelated.nan_to_num(-torch.inf).amax(1)
        )
        self.min_unrelated = torch.fmin(self.min_unrelated, unrelated.nan_to_num(torch.inf).amin(1))
        related = torch.where(block.related, score, torch.inf)
        for key, t in self.thresholds.items():
            self.counts[f"false_pos_{key}"] += (unrelated >= t).sum(1)
            self.counts[f"false_pos_abs_{key}"] += (unrelated.abs() >= t).sum(1)
            self.counts[f"false_neg_{key}"] += (related < t).sum(1)

        top_score, top_pos = torch.where(base.valid, score, -torch.inf).topk(
            min(HEAD, score.shape[1]), dim=1
        )
        new = {
            "score": top_score,
            "exact": base.exact.gather(1, top_pos),
            "index": c_index[top_pos],
        }
        merged = {k: torch.cat([self.head[k], new[k]], 1) for k in self.head}
        # Deterministic order: score descending, then record index ascending.
        by_index = merged["index"].argsort(dim=1, stable=True)
        merged = {k: v.gather(1, by_index) for k, v in merged.items()}
        order = merged["score"].argsort(dim=1, descending=True, stable=True)[:, :HEAD]
        self.head = {k: v.gather(1, order) for k, v in merged.items()}


def discordant_fraction(exact: torch.Tensor) -> torch.Tensor:
    """Share of differently-scored exact pairs in the retrieved head that are out of order."""
    later_higher = exact[:, None, :] > exact[:, :, None] + TOLERANCE
    differ = (exact[:, None, :] - exact[:, :, None]).abs() > TOLERANCE
    finite = torch.isfinite(exact)
    upper = torch.ones(exact.shape[1], exact.shape[1], dtype=torch.bool).triu(1)
    ok = finite[:, None, :] & finite[:, :, None] & upper
    total = (differ & ok).sum((1, 2))
    return torch.where(total > 0, (later_higher & ok).sum((1, 2)) / total.clamp(min=1), 0.0)


@dataclass
class SourceState:
    """Codebook-independent running state (evaluation only)."""

    q: int
    n_valid: torch.Tensor = field(init=False)
    n_tier: torch.Tensor = field(init=False)
    exact_top: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.n_valid = torch.zeros(self.q, dtype=torch.long)
        self.n_tier = torch.zeros(self.q, 4, dtype=torch.long)
        self.exact_top = torch.full((self.q, TOP_K), -torch.inf)

    def update(self, block: Block):
        valid, tier = block.base.valid, block.tier.long()
        self.n_valid += valid.sum(1)
        self.n_tier += torch.stack([(valid & (tier == t)).sum(1) for t in range(4)], 1)
        k = min(TOP_K, valid.shape[1])
        top = torch.where(valid, block.base.exact, -torch.inf).topk(k, 1).values
        self.exact_top = torch.cat([self.exact_top, top], 1).topk(TOP_K, 1).values


def query_rows(
    unit_key, unit: Unit, source: SourceState, n: int, queries: Records, ranks: torch.Tensor
) -> dict:
    """Per-query state after the first n candidates, with agreement measured against S_exact."""
    kth = source.exact_top[:, TOP_K - 1]
    head = unit.head
    rows = {
        "dimension": unit_key[0],
        "seed": unit_key[1],
        "candidates": n,
        "query_rank": ranks,
        "record_index": queries.record_index,
        "valid_candidates": source.n_valid.clone(),
        **{f"pairs_{name}": source.n_tier[:, t].clone() for t, name in enumerate(TIER_NAMES)},
        "max_unrelated": unit.max_unrelated.clone(),
        "min_unrelated": unit.min_unrelated.clone(),
        "exact_best": source.exact_top[:, 0].clone(),
        "bundle_top1_exact": head["exact"][:, 0].clone(),
        "precision_at_10": (head["exact"][:, :TOP_K] >= kth[:, None] - TOLERANCE).float().mean(1),
        "head_discordant": discordant_fraction(head["exact"]),
        **{k: v.clone() for k, v in unit.counts.items()},
    }
    q = len(ranks)
    return {k: (v.tolist() if torch.is_tensor(v) else [v] * q) for k, v in rows.items()}


def run_pass(
    db,
    records: Records,
    query_offsets: list[int],
    ranks: torch.Tensor,
    units,
    *,
    candidates: int,
    thresholds: dict | None = None,
    log=print,
):
    """Stream candidates once for every codebook.

    Calibration (thresholds=None) keeps only related-pair score histograms.
    """
    calibration = thresholds is None
    books = {u: store.read_codebook(db, *u) for u in units}
    queries = records.take(query_offsets)
    state = {
        u: Unit(
            books[u],
            person.encode_sum(books[u], queries),
            {} if calibration else thresholds[u],
            calibration,
        )
        for u in units
    }
    source = None if calibration else SourceState(len(query_offsets))
    per_query, started = [], time.time()
    for start, stop, segment in chunks(candidates):
        chunk = records.take(slice(start, stop))
        base = person.baselines(queries, chunk)
        tier = tiers(base.exact)
        block = Block(base, base.valid & (tier == 0), base.valid & (tier == 3), tier)
        if not calibration:
            source.update(block)
        for u, unit in state.items():
            unit.update(person.encode_sum(books[u], chunk), chunk.record_index, block, segment)
        if not calibration and stop == PREFIXES[segment]:
            for u, unit in state.items():
                per_query.append(query_rows(u, unit, source, stop, queries, ranks))
            log(f"    N={stop:>7,}  {time.time() - started:6.0f}s")
    return state, source, per_query


def per_query_frame(parts: list[dict]) -> pl.DataFrame:
    return pl.concat([pl.DataFrame(p) for p in parts]).with_columns(
        pl.col(pl.Float64).cast(pl.Float32)
    )


def histogram_frame(state: dict) -> pl.DataFrame:
    """Nonzero histogram bins for every unit, segment and metric."""
    parts = []
    for u, unit in state.items():
        for segment, n in enumerate(PREFIXES):
            for metric in unit.hist.metrics:
                counts = unit.hist.segment(metric, segment)
                nonzero = counts.nonzero().flatten()
                if len(nonzero):
                    parts.append(
                        pl.DataFrame(
                            {
                                "dimension": u[0],
                                "seed": u[1],
                                "segment_end": n,
                                "metric": metric,
                                "bin": nonzero.tolist(),
                                "count": counts[nonzero].tolist(),
                            }
                        )
                    )
    return pl.concat(parts)


def calibrated_threshold(related_hist: torch.Tensor) -> tuple[float, int, float]:
    """Largest bin edge T with at most BETA of related calibration pairs strictly below it."""
    total = int(related_hist.sum())
    below = torch.cumsum(related_hist, 0)
    index = int(torch.searchsorted(below, torch.tensor(int(BETA * total)), right=True))
    achieved = 1 - (int(below[index - 1]) if index > 0 else 0) / total
    return round(METRICS["related_score"][0] + index * BIN, 4), total, achieved
