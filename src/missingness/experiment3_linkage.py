"""HYP-118 Experiment 3: weights from informativeness (Fellegi-Sunter record linkage).

Task: find each query person's noisy duplicate among everyone else. A query is a
complete fixture record. Its duplicate (data/missingness/duplicates.parquet) has
each value replaced by a different one with probability 0.1 and then fields missing
completely at random at the run's rate. Distractors are every other fixture record,
masked at the same rate; the query's own original record is excluded.

Equal property weights treat a region match (1 in 5 by chance) like a three-interest
match. Fellegi and Sunter (1969) weight each comparison level by log(m / u):

  m  P(level | the pair is the same person)
  u  P(level | the pair is not), estimated from random pairs (no labels needed:
     true matches are a negligible share of random pairs)

A property either side is missing contributes 0, neither for nor against.

Scorers, all over the same per-property comparisons (policy.py):

  cosine           equal weights, normalized-bundle cosine (the Experiment 2 default)
  fs_known         m from 20,000 labelled training duplicates
  fs_em            m estimated without labels by EM, u held fixed (the Splink recipe).
                   Each blocking rule fixes one property, and EM estimates m only for
                   the others, whose non-match levels stay random under the rule:
                     same region  -> m for age, job and interests
                     same job     -> m for region
                   (Blocking on overall similarity makes every blocked pair look like
                   a match, and EM collapses to "all pairs match".)
  fs_u_only        label-free and EM-free: u from data, m assumed (agree 0.9, the rest
                   of m spread like u). Like IDF, the weight comes from how common an
                   agreement is by chance.
  fs_linear        the u-only weights collapsed to agree/disagree and applied linearly
                   to the graded similarities, which a single dot product plus a
                   stored per-record bias can serve (see policy.linear_fellegi_sunter)
  cosine_weighted  cosine normalization with the u-only agreement weights log(m/u)
                   applied to the query's property terms: informativeness weights on
                   top of Experiment 2's denominator, servable by one dot product

Fellegi-Sunter scores a missing property as 0, the same "missing is neutral"
treatment as Gower, so it inherits Gower's failure in dense pools (Experiment 2).
cosine_weighted separates the two choices: who decides the weights, and what a
missing property costs.

With only four low-cardinality properties, a stranger who agrees on everything by
chance is about as common as a true duplicate, so EM is weakly identified here;
the experiment reports how far its weights land from the labelled ones.

Grid (design choices for this run): MCAR rates 0.1, 0.3, 0.5; the full pool and a
1% subsample, since Experiment 2 showed pool density changes what missingness costs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import polars as pl
import torch

import person
import policy as P
from experiment2_dial import subsample

RATES = (0.1, 0.3, 0.5)
DENSITIES = (1.0, 0.01)
K = 10
U_PAIRS_PER_QUERY = 5_000  # random pool records per calibration query for u
BLOCK_CAP = 500  # EM: at most this many blocked pool records per calibration query, at random
# Blocking rules: (name, properties the rule fixes, properties EM estimates under it).
BLOCKS = (("same_region", (2,), (0, 1, 3)), ("same_job", (1,), (2,)))
M_AGREE = 0.9  # fs_u_only's assumed agreement rate for a true match: about 1 - error rate
EM_ITERATIONS = 500
SMOOTHING = 0.5  # add-half counts so no level gets probability 0
PROPERTIES = ("age", "job", "region", "interests")
LEVEL_NAMES = (
    ("same", "one_apart", "further"),
    ("agree", "disagree"),
    ("agree", "disagree"),
    ("all", "some", "none"),
)


def level_frequencies(g: torch.Tensor, weight: torch.Tensor | None = None) -> list[torch.Tensor]:
    """Per-property level probabilities among pairs where both sides know the property."""
    g = g.reshape(-1, 4)
    w = torch.ones(len(g)) if weight is None else weight.reshape(-1)
    out = []
    for f, n in enumerate(P.LEVELS):
        known = g[:, f] >= 0
        counts = torch.bincount(g[known, f], weights=w[known], minlength=n).float() + SMOOTHING
        out.append(counts / counts.sum())
    return out


def paired_levels(a: person.Records, b: person.Records, batch: int = 1000) -> torch.Tensor:
    """Comparison vectors of row i of `a` against row i of `b`."""
    parts = []
    for start in range(0, len(a), batch):
        g = P.levels(
            P.compare(a.take(slice(start, start + batch)), b.take(slice(start, start + batch)))
        )
        parts.append(torch.diagonal(g, dim1=0, dim2=1).T)
    return torch.cat(parts)


def block_rule(name: str):
    """Pairs a blocking rule admits: both sides know the fixed properties and they agree."""
    fixed = next(f for n, f, _ in BLOCKS if n == name)

    def admits(p: P.Pairs) -> torch.Tensor:
        return torch.stack([p.s[..., f] == 1 for f in fixed]).all(0)

    return admits


def assumed_m(u: list[torch.Tensor]) -> list[torch.Tensor]:
    """m with level 0 at M_AGREE and the remaining mass spread over other levels like u."""
    out = []
    for uf in u:
        rest = uf[1:] / uf[1:].sum() * (1 - M_AGREE)
        out.append(torch.cat([torch.tensor([M_AGREE]), rest]))
    return out


def blocked_levels(queries, pool, duplicates, name: str, g: torch.Generator) -> torch.Tensor:
    """Comparison vectors of every pair a rule admits: up to BLOCK_CAP random pool records
    per query, plus the query's duplicate when the rule admits it (no labels are used)."""
    admits = block_rule(name)
    seed = int(torch.randint(2**31, (1,), generator=g))

    def random_where_admitted(p: P.Pairs) -> torch.Tensor:
        noise = torch.rand(
            p.total.shape, generator=torch.Generator().manual_seed(seed + p.total.shape[1])
        )
        return torch.where(admits(p), noise, -torch.inf)

    rows, keys = P.top_indices(queries, pool, random_where_admitted, BLOCK_CAP)
    parts = []
    for i in range(len(queries)):
        chosen = rows[i][keys[i] > -torch.inf]
        if len(chosen):
            parts.append(P.levels(P.compare(queries.take([i]), pool.take(chosen)))[0])
    dup = P.compare(queries, duplicates)
    keep = torch.diagonal(admits(dup))
    parts.append(torch.diagonal(P.levels(dup), dim1=0, dim2=1).T[keep])
    return torch.cat(parts)


def em(
    g: torch.Tensor, u: list[torch.Tensor], free: tuple, log=print
) -> tuple[list[torch.Tensor], float]:
    """Estimate m and the match share lambda by EM on unlabelled comparison vectors, u fixed.

    Only the `free` properties enter the likelihood; the rule fixed the others.
    Conditional independence across properties, as in Fellegi-Sunter and Splink.
    Starts from "a match usually agrees": level 0 at 0.8, the rest shared equally.
    """
    g = g.clone()
    for f in range(4):
        if f not in free:
            g[:, f] = -1
    m = [torch.tensor([0.8] + [0.2 / (n - 1)] * (n - 1)) for n in P.LEVELS]
    lam = 1e-3
    for _ in range(EM_ITERATIONS):
        log_m = torch.zeros(len(g))
        log_u = torch.zeros(len(g))
        for f in range(4):
            known = g[:, f] >= 0
            level = g[:, f].clamp(min=0)
            log_m += torch.where(known, m[f][level].log(), 0.0)
            log_u += torch.where(known, u[f][level].log(), 0.0)
        posterior = torch.sigmoid(log_m - log_u + torch.logit(torch.tensor(lam)))
        new_lam = float(posterior.mean())
        new_m = level_frequencies(g, posterior)
        change = max(float((a - b).abs().max()) for a, b in zip(new_m, m))
        m, lam = new_m, new_lam
        if change < 1e-7:
            break
    return m, lam


def log_weights(m: list[torch.Tensor], u: list[torch.Tensor]) -> list[torch.Tensor]:
    """Fellegi-Sunter weights log(m / u) per property level."""
    return [(a / b).log() for a, b in zip(m, u)]


def average_precision(top_score: torch.Tensor, correct: torch.Tensor) -> float:
    """AP of "the top result is the duplicate" decisions, ranked by top-1 score.

    `correct` is the expected 0-1 label (ties placed uniformly).
    """
    order = top_score.argsort(descending=True)
    c = correct[order]
    precision = c.cumsum(0) / torch.arange(1, len(c) + 1)
    return float((precision * c).sum() / c.sum().clamp(min=1e-9))


def run(out: Path, log=print):
    signatures = person.load_fixture()
    complete_all = person.complete_records(signatures)
    calibration, evaluation = person.query_panel("calibration"), person.query_panel("evaluation")
    panels = torch.cat([calibration, evaluation])
    g = torch.Generator().manual_seed(1969)
    summary, weights_rows, per_query = [], [], []
    for rate, density in itertools.product(RATES, DENSITIES):
        keep = subsample(len(complete_all), density, panels)
        pool = person.mcar_records(signatures, rate).take(keep)
        cal_queries = complete_all.take(calibration)

        # u from random pairs: calibration queries against random pool records.
        picks = torch.randint(len(pool), (U_PAIRS_PER_QUERY,), generator=g)
        u = level_frequencies(P.levels(P.compare(cal_queries, pool.take(picks))))

        # m from labelled training duplicates.
        training = person.duplicate_records(rate, "training")
        m_known = level_frequencies(
            paired_levels(complete_all.take(training.record_index), training)
        )

        # m by EM under each blocking rule, without labels. The query's original record is
        # not in the pool's competitor set, as in evaluation.
        duplicates = person.duplicate_records(rate, "calibration")
        cal_pool = pool.take(torch.nonzero(~torch.isin(pool.record_index, calibration)).flatten())
        m_em, em_info = [None] * 4, {}
        for name, _, free in BLOCKS:
            blocked = blocked_levels(cal_queries, cal_pool, duplicates, name, g)
            m_rule, lam = em(blocked, u, free, log)
            for f in free:
                m_em[f] = m_rule[f]
            em_info[name] = (lam, len(blocked))

        m_u = assumed_m(u)
        w_known, w_em, w_u = (log_weights(m, u) for m in (m_known, m_em, m_u))
        agree = torch.tensor([float((m_u[f][0] / u[f][0]).log()) for f in range(4)])
        disagree = torch.tensor([float(((1 - m_u[f][0]) / (1 - u[f][0])).log()) for f in range(4)])
        for f, prop in enumerate(PROPERTIES):
            for level, name in enumerate(LEVEL_NAMES[f]):
                weights_rows.append(
                    {
                        "rate": rate,
                        "density": density,
                        "property": prop,
                        "level": name,
                        "u": float(u[f][level]),
                        "m_known": float(m_known[f][level]),
                        "m_em": float(m_em[f][level]),
                        "m_assumed": float(m_u[f][level]),
                        "weight_known": float(w_known[f][level]),
                        "weight_em": float(w_em[f][level]),
                        "weight_u_only": float(w_u[f][level]),
                        "em_block": next(n for n, _, free in BLOCKS if f in free),
                        "em_match_share": em_info[next(n for n, _, free in BLOCKS if f in free)][0],
                        "em_pairs": em_info[next(n for n, _, free in BLOCKS if f in free)][1],
                    }
                )

        scorers = {
            "cosine": P.cosine,
            "fs_known": P.fellegi_sunter(w_known),
            "fs_em": P.fellegi_sunter(w_em),
            "fs_u_only": P.fellegi_sunter(w_u),
            "fs_linear": P.linear_fellegi_sunter(agree, disagree),
            "cosine_weighted": P.weighted_cosine(agree),
        }
        queries = complete_all.take(evaluation)
        dups = person.duplicate_records(rate, "evaluation")
        ranks = P.rank_relevant(queries, pool, dups, scorers, K, exclude=evaluation)
        for name, r in ranks.items():
            r1 = r.recall(1)
            top1 = torch.maximum(r.truth, r.top[:, 0])
            summary.append(
                {
                    "rate": rate,
                    "density": density,
                    "pool_size": len(pool),
                    "scorer": name,
                    "recall_at_1": float(r1.mean()),
                    "recall_at_10": float(r.recall(K).mean()),
                    "mrr": float(r.reciprocal_rank().mean()),
                    "average_precision_top1": average_precision(top1, r1),
                    "random_recall_at_10": K / len(pool),
                }
            )
            per_query.append(
                pl.DataFrame(
                    {
                        "rate": rate,
                        "density": density,
                        "scorer": name,
                        "query_record": evaluation.tolist(),
                        "duplicate_score": r.truth.tolist(),
                        "recall_at_1": r1.tolist(),
                        "recall_at_10": r.recall(K).tolist(),
                        "reciprocal_rank": r.reciprocal_rank().tolist(),
                    }
                )
            )
        log(
            f"rate={rate} density={density:.0%} "
            + " ".join(f"{n}: lambda={v[0]:.2e} pairs={v[1]:,}" for n, v in em_info.items())
            + " | "
            + " ".join(f"{s['scorer']}={s['recall_at_1']:.3f}" for s in summary[-6:])
        )
    pl.DataFrame(summary).write_csv(out / "experiment3_summary.csv", float_precision=5)
    pl.DataFrame(weights_rows).write_csv(out / "experiment3_weights.csv", float_precision=5)
    pl.concat(per_query).write_parquet(out / "experiment3_per_query.parquet")
    (out / "experiment3_config.json").write_text(
        json.dumps(
            {
                "rates": RATES,
                "densities": DENSITIES,
                "k": K,
                "error_rate": 0.1,
                "u_pairs_per_query": U_PAIRS_PER_QUERY,
                "em_blocks": {
                    n: {
                        "fixes": [PROPERTIES[f] for f in fixed],
                        "estimates": [PROPERTIES[f] for f in free],
                    }
                    for n, fixed, free in BLOCKS
                },
                "em_block_cap": BLOCK_CAP,
                "em_iterations_max": EM_ITERATIONS,
                "smoothing": SMOOTHING,
                "m_assumed_agree": M_AGREE,
                "levels": dict(zip(PROPERTIES, LEVEL_NAMES)),
                "queries": "complete evaluation-panel records (400); EM uses the calibration panel (200)",
                "scores": "exact comparisons (orthogonal atoms)",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=person.OUT)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.output_dir, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
