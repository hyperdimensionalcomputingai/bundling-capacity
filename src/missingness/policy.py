"""Exact similarity policies for records with missing values (HYP-118 Experiments 2-4).

With orthogonal atoms, the dot product of two property-normalized bundles is the
sum of per-property similarities over the properties both records know:

  S(q, c) = sum over p known by both of s_p(q, c)

  s_age            1 - |i - j| / 9, the intended ordinal age kernel
  s_job, s_region  [same value]
  s_interests      |shared| / sqrt(n_q * n_c), the Ochiai (set-cosine) coefficient
                   over each side's known interests

A missing-value policy is a choice of denominator for that same numerator:

  gower     S / (properties both know)            missing is neutral (Gower 1971)
  cosine    S / sqrt(p_q * p_c)                   what normalized-bundle cosine gives
  mismatch  S / p_q                               missing counts as a mismatch
  pivoted   S / (sqrt(p_q) * ((1 - s) * pivot + s * sqrt(p_c)))
            Singhal et al. (1996): s = 1 is cosine, s = 0 ranks like mismatch

p_q and p_c count each record's known properties. Scores here are exact policy
values; the MAP implementation of the same policies is measured in Experiment 4.

Search is streamed over the 920,000-record pool in chunks, with every policy
scored from one set of per-property similarities. Ties are handled in
expectation: a record tied with e others at a cut is placed uniformly among them.

Two kinds of relevance are used. The *own copy* is the query person's masked
record (or noisy duplicate). *Hidden-truth relevance* scores each pool record's
complete version, so a policy is judged by how well its ranking of the masked data
recovers the ranking it would give with nothing missing, as Singhal et al. judged
length normalization against relevance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from person import AGE_LEVELS, MAX_INTERESTS, PRESENT, Records

CHUNK = 16_384
TOLERANCE = 1e-6  # scores within this are a tie; exact policy scores are ratios of small integers
COVERAGE_BINS = 5  # 0-4 known properties


def known(r: Records) -> tuple[torch.Tensor, torch.Tensor]:
    """Which properties carry a term (n, 4), and how many interests are known (n,)."""
    n = (r.interests >= 0).sum(1)
    has_interests = (r.state[:, 3] == PRESENT) & (n > 0)
    return torch.cat([r.state[:, :3] == PRESENT, has_interests[:, None]], 1), n


@dataclass
class Pairs:
    """Per-property agreement for every (query, candidate) pair in one chunk."""

    s: torch.Tensor  # (q, c, 4) per-property similarity, 0 where either side is missing
    both: torch.Tensor  # (q, c, 4) both records know the property
    shared: torch.Tensor  # (q, c) shared known interests
    n_q: torch.Tensor  # (q, 1) query's known interests
    n_c: torch.Tensor  # (1, c) candidate's known interests
    p_q: torch.Tensor  # (q, 1) query's known properties
    p_c: torch.Tensor  # (1, c) candidate's known properties
    age_gap: torch.Tensor  # (q, c) |age band difference|

    @property
    def total(self) -> torch.Tensor:
        return self.s.sum(-1)


def compare(q: Records, c: Records) -> Pairs:
    kq, nq = known(q)
    kc, nc = known(c)
    both = kq[:, None, :] & kc[None, :, :]
    gap = (q.value[:, None, 0].int() - c.value[None, :, 0].int()).abs()
    age = 1 - gap.float() / (AGE_LEVELS - 1)
    same = (q.value[:, None, 1:] == c.value[None, :, 1:]).float()
    shared = torch.zeros(len(q), len(c), dtype=torch.int8)
    for i in range(MAX_INTERESTS):
        for j in range(MAX_INTERESTS):
            shared += (
                (q.interests[:, None, i] == c.interests[None, :, j])
                & (q.interests[:, None, i] >= 0)
            ).to(torch.int8)
    interests = shared.float() / (nq[:, None] * nc[None, :]).float().sqrt().clamp(min=1)
    s = torch.stack([age, same[..., 0], same[..., 1], interests], -1)
    return Pairs(
        torch.where(both, s, 0.0),
        both,
        shared,
        nq[:, None],
        nc[None, :],
        kq.sum(1, keepdim=True).float(),
        kc.sum(1)[None, :].float(),
        gap,
    )


# --------------------------------------------------------------------------- policies


def gower(p: Pairs) -> torch.Tensor:
    return p.total / p.both.sum(-1).clamp(min=1)


def cosine(p: Pairs) -> torch.Tensor:
    return p.total / (p.p_q * p.p_c).sqrt().clamp(min=1)


def mismatch(p: Pairs) -> torch.Tensor:
    return p.total / p.p_q.clamp(min=1)


def pivoted(slope: float, pivot: float):
    """Singhal's pivoted normalizer applied to the candidate's length sqrt(p_c)."""

    def score(p: Pairs) -> torch.Tensor:
        length = (1 - slope) * pivot + slope * p.p_c.sqrt()
        return p.total / (p.p_q.sqrt().clamp(min=1) * length.clamp(min=1e-3))

    return score


def mean_length(pool: Records) -> float:
    """Singhal's pivot: the collection's average old normalizer, here mean sqrt(p_c)."""
    k, _ = known(pool)
    p = k.sum(1).float()
    return float(p[p > 0].sqrt().mean())


# --------------------------------------------------------------------------- streamed search


def expected_recall(greater: torch.Tensor, equal: torch.Tensor, k: int) -> torch.Tensor:
    """P(relevant record in the top k) when it is tied with `equal` others, placed uniformly."""
    return ((k - greater).float() / (equal + 1).float()).clamp(0, 1)


def expected_reciprocal_rank(greater: torch.Tensor, equal: torch.Tensor) -> torch.Tensor:
    """Mean of 1/r over r = greater+1 .. greater+equal+1."""
    harmonic = lambda n: torch.digamma(n.double() + 1) + 0.5772156649015329
    return ((harmonic(greater + equal + 1) - harmonic(greater)) / (equal + 1).double()).float()


@dataclass
class Ranks:
    truth: torch.Tensor  # (q,) the relevant record's score
    greater: torch.Tensor  # (q,) pool records scoring above the relevant record
    equal: torch.Tensor  # (q,) pool records tied with it
    top: torch.Tensor  # (q, k) best pool scores, for the kth-score cut

    def recall(self, k: int) -> torch.Tensor:
        return expected_recall(self.greater, self.equal, k)

    def reciprocal_rank(self) -> torch.Tensor:
        return expected_reciprocal_rank(self.greater, self.equal)


def rank_relevant(
    queries: Records,
    pool: Records,
    relevant: Records,
    policies: dict,
    k: int = 10,
    exclude: torch.Tensor | None = None,
) -> dict[str, Ranks]:
    """Rank each query's relevant record (row i of `relevant`) against the pool.

    `exclude` holds one pool record_index per query that is not a competitor: the
    relevant record's own pool copy (Experiment 2) or the duplicated original
    (Experiment 3). Pool records with no known property are never candidates.
    """
    truth = {name: torch.diagonal(f(compare(queries, relevant))) for name, f in policies.items()}
    q = len(queries)
    out = {
        name: Ranks(
            truth[name],
            torch.zeros(q, dtype=torch.long),
            torch.zeros(q, dtype=torch.long),
            torch.full((q, k), -torch.inf),
        )
        for name in policies
    }
    for start in range(0, len(pool), CHUNK):
        chunk = pool.take(slice(start, start + CHUNK))
        pairs = compare(queries, chunk)
        valid = pairs.p_c > 0
        if exclude is not None:
            valid = valid & (chunk.record_index[None, :] != exclude[:, None])
        for name, f in policies.items():
            score = torch.where(valid, f(pairs), -torch.inf)
            t = truth[name][:, None]
            r = out[name]
            r.greater += (score > t + TOLERANCE).sum(1)
            r.equal += ((score - t).abs() <= TOLERANCE).sum(1)
            r.top = (
                torch.cat([r.top, score.topk(min(k, score.shape[1]), 1).values], 1)
                .topk(k, 1)
                .values
            )
    return out


def top_indices(
    queries: Records, pool: Records, score, k: int, exclude: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each query's k best pool rows (positions in `pool`) and their scores."""
    q = len(queries)
    best = torch.full((q, k), -torch.inf)
    where = torch.full((q, k), -1, dtype=torch.long)
    for start in range(0, len(pool), CHUNK):
        chunk = pool.take(slice(start, start + CHUNK))
        pairs = compare(queries, chunk)
        valid = pairs.p_c > 0
        if exclude is not None:
            valid = valid & (chunk.record_index[None, :] != exclude[:, None])
        sc = torch.where(valid, score(pairs), -torch.inf)
        top = sc.topk(min(k, sc.shape[1]), 1)
        merged = torch.cat([best, top.values], 1)
        positions = torch.cat([where, top.indices + start], 1)
        keep = merged.topk(k, 1).indices
        best, where = merged.gather(1, keep), positions.gather(1, keep)
    return where, best


# --------------------------------------------------------------------------- record linkage levels

LEVELS = (3, 2, 2, 3)  # comparison levels per property; -1 means either side is missing


def levels(p: Pairs) -> torch.Tensor:
    """Fellegi-Sunter comparison vector (q, c, 4), Splink-style discrete levels.

    age        0 same band, 1 one band apart, 2 further
    job        0 agree, 1 disagree
    region     0 agree, 1 disagree
    interests  0 every interest both could compare is shared (shared = min(n_q, n_c)),
               1 some shared, 2 none
    """
    age = torch.where(p.age_gap == 0, 0, torch.where(p.age_gap == 1, 1, 2))
    job = torch.where(p.s[..., 1] == 1, 0, 1)
    region = torch.where(p.s[..., 2] == 1, 0, 1)
    full = p.shared.long() == torch.minimum(p.n_q, p.n_c)
    interests = torch.where(full, 0, torch.where(p.shared > 0, 1, 2))
    g = torch.stack([age, job, region, interests], -1)
    return torch.where(p.both, g, -1)


def fellegi_sunter(weights: list[torch.Tensor]):
    """Score = sum over properties both know of log(m / u) at the observed level; missing adds 0."""

    def score(p: Pairs) -> torch.Tensor:
        g = levels(p)
        total = torch.zeros(g.shape[:-1])
        for f, w in enumerate(weights):
            total += torch.where(g[..., f] >= 0, w[g[..., f].clamp(min=0)], 0.0)
        return total

    return score


def linear_fellegi_sunter(agree: torch.Tensor, disagree: torch.Tensor):
    """A dot-product-servable approximation: sum over known-by-both of d_p + (a_p - d_p) * s_p.

    a_p and d_p are the binary agreement and disagreement weights. With a complete
    query, the d_p terms depend only on which properties the candidate knows, so they
    fold into one stored per-record bias; the rest is a query-weighted dot product of
    property-normalized bundles.
    """

    def score(p: Pairs) -> torch.Tensor:
        both = p.both.float()
        return (both * (disagree + (agree - disagree) * p.s)).sum(-1)

    return score


def weighted_cosine(weights: torch.Tensor):
    """Cosine with query-side property weights: sum of w_p * s_p over known-by-both, / sqrt(p_q p_c).

    Scaling each property term of the *query* bundle by w_p keeps every stored record
    as it is, so one index serves any weighting; the denominator is cosine's.
    """

    def score(p: Pairs) -> torch.Tensor:
        return (p.s * weights).sum(-1) / (p.p_q * p.p_c).sqrt().clamp(min=1)

    return score


def relevance(p: Pairs) -> torch.Tensor:
    """Hidden-truth relevance: equal-weight property similarity of the *complete* records.

    Scored on complete records, every property is known, so this is S / 4. It is
    the reference each policy tries to recover from masked data; MCAR masking is
    independent of it, so relevance is independent of a record's coverage.
    """
    return p.total / 4


def scan_kth(
    queries: Records, pool: Records, complete: Records, policies: dict, k: int = 10
) -> dict[str, torch.Tensor]:
    """Each query's kth-best score under every policy (masked pool) and under relevance (complete)."""
    q = len(queries)
    top = {n: torch.full((q, k), -torch.inf) for n in (*policies, "relevance")}
    for start in range(0, len(pool), CHUNK):
        masked = compare(queries, pool.take(slice(start, start + CHUNK)))
        valid = masked.p_c > 0
        scores = {n: torch.where(valid, f(masked), -torch.inf) for n, f in policies.items()}
        scores["relevance"] = relevance(
            compare(queries, complete.take(slice(start, start + CHUNK)))
        )
        for n, score in scores.items():
            top[n] = torch.cat([top[n], score.topk(k, 1).values], 1).topk(k, 1).values
    return {n: t[:, k - 1] for n, t in top.items()}


@dataclass
class TopK:
    coverage: torch.Tensor  # (q, 5) expected top-k slots held by records with 0-4 known properties
    hits: torch.Tensor  # (q,) expected top-k slots held by relevant records (relevance >= its kth)


def scan_top(
    queries: Records,
    pool: Records,
    complete: Records,
    policies: dict,
    kth: dict[str, torch.Tensor],
    k: int = 10,
) -> dict[str, TopK]:
    """What each policy's top k contains: coverage mix and relevant records, ties in expectation.

    Records above a policy's kth score are retrieved; records tied at it share the
    remaining slots. The "relevance" entry describes the relevant set itself.
    """
    q = len(queries)
    names = (*policies, "relevance")
    above = {n: torch.zeros(q, COVERAGE_BINS) for n in names}
    tied = {n: torch.zeros(q, COVERAGE_BINS) for n in names}
    hits_above = {n: torch.zeros(q) for n in names}
    hits_tied = {n: torch.zeros(q) for n in names}
    for start in range(0, len(pool), CHUNK):
        masked = compare(queries, pool.take(slice(start, start + CHUNK)))
        valid = masked.p_c > 0
        truth = relevance(compare(queries, complete.take(slice(start, start + CHUNK))))
        relevant = (truth >= kth["relevance"][:, None] - TOLERANCE).float()
        bins = torch.nn.functional.one_hot(masked.p_c[0].long(), COVERAGE_BINS).float()
        scores = {n: torch.where(valid, f(masked), -torch.inf) for n, f in policies.items()}
        scores["relevance"] = truth
        for n, score in scores.items():
            cut = kth[n][:, None]
            a = (score > cut + TOLERANCE).float()
            t = ((score - cut).abs() <= TOLERANCE).float()
            above[n] += a @ bins
            tied[n] += t @ bins
            hits_above[n] += (a * relevant).sum(1)
            hits_tied[n] += (t * relevant).sum(1)
    out = {}
    for n in names:
        free = (k - above[n].sum(1, keepdim=True)).clamp(min=0)
        share = free / tied[n].sum(1, keepdim=True).clamp(min=1)
        out[n] = TopK(above[n] + tied[n] * share, hits_above[n] + hits_tied[n] * share[:, 0])
    return out


def coverage_counts(pool: Records) -> torch.Tensor:
    """Pool records with 0-4 known properties."""
    k, _ = known(pool)
    return torch.bincount(k.sum(1), minlength=COVERAGE_BINS)
