"""PERSON records with missing values, the MAP-I encoder and the exact similarity baselines.

Signatures come from the scale study's fixture in data/scale/; the missingness
masks, pairwise pairs and noisy duplicates live in data/missingness/.

A record has four fields: an ordinal age band, a job category, a home region and
a set of three interests, so six bound facts when complete. A missing field is
handled by one of two strategies:

  omit   the field contributes no fact
  token  the field contributes ROLE * NULL[spelling]: one shared vector per null
         spelling ("null", "no_value", ""), each an independent random atom

A missing interest set is one token fact, the same weight as any other field.

`encode_normalized` is the property-normalized alternative studied here:
each known field is scaled to unit length before bundling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import torchhd

ROOT = Path(__file__).resolve().parents[2]
SCALE_DATA = ROOT / "data" / "scale"  # the 920,000 factorial signatures
DATA = ROOT / "data" / "missingness"  # the missingness mask and pairwise pairs
OUT = ROOT / "results" / "missingness"
FIELDS = ("age_band", "job_category", "home_region", "interests")
SIZES = {"age_band": 10, "job_category": 8, "home_region": 5, "interests": 25}
AGE_LEVELS = SIZES["age_band"]
MAX_INTERESTS = 3
STRATEGIES = ("omit", "token")
NULL_SPELLINGS = ("null", "no_value", "")
PRESENT, MISSING = 0, 1


# --------------------------------------------------------------------------- records


@dataclass
class Records:
    """Records as int8 tensors. Missing values are -1; `spelling` names each missing field's null token."""

    record_index: torch.Tensor  # (n,) int64
    state: torch.Tensor  # (n, 4) PRESENT or MISSING per field
    value: torch.Tensor  # (n, 3) age / job / region codes
    interests: torch.Tensor  # (n, 3) sorted distinct interests
    spelling: torch.Tensor  # (n, 4) index into NULL_SPELLINGS, used only where MISSING

    def __len__(self):
        return len(self.record_index)

    def take(self, index) -> Records:
        return Records(
            *(
                x[index]
                for x in (self.record_index, self.state, self.value, self.interests, self.spelling)
            )
        )

    def content_weight(self) -> torch.Tensor:
        """Number of content facts: age, job, region and each known interest."""
        scalars = (self.state[:, :3] == PRESENT).sum(1)
        return scalars + torch.where(self.state[:, 3] == PRESENT, (self.interests >= 0).sum(1), 0)

    def without(self, missing: torch.Tensor, spelling: torch.Tensor | None = None) -> Records:
        """A copy with the fields in `missing` (n, 4 bool) removed."""
        state = torch.where(missing, MISSING, self.state).to(torch.int8)
        value = torch.where(missing[:, :3], -1, self.value).to(torch.int8)
        interests = torch.where(missing[:, 3:], -1, self.interests).to(torch.int8)
        spelling = self.spelling if spelling is None else spelling.to(torch.int8)
        return Records(self.record_index, state, value, interests, spelling)


def load_fixture(data_dir: Path = SCALE_DATA):
    signatures = pl.read_parquet(data_dir / "signatures.parquet")
    if signatures.height != 920_000 or not signatures["record_index"].equals(
        pl.Series("record_index", range(920_000), dtype=pl.Int32)
    ):
        raise ValueError("Expected 920,000 signatures in record_index order")
    return signatures


def complete_records(signatures: pl.DataFrame) -> Records:
    cols = lambda names: signatures.select(names).to_torch().to(torch.int8)
    n = signatures.height
    return Records(
        signatures["record_index"].to_torch().long(),
        torch.zeros(n, 4, dtype=torch.int8),
        cols(list(FIELDS[:3])),
        cols(["interest_1", "interest_2", "interest_3"]),
        torch.zeros(n, 4, dtype=torch.int8),
    )


def masked_records(signatures: pl.DataFrame, data_dir: Path = DATA) -> Records:
    """Experiment 3: every field independently missing at 30%, all nulls spelled "null"."""
    mask = pl.read_parquet(data_dir / "missingness.parquet")
    if not mask["record_index"].equals(signatures["record_index"]):
        raise ValueError("Mask rows must align with signatures")
    missing = mask.select([f"{f}_missing" for f in FIELDS]).to_torch().bool()
    return complete_records(signatures).without(missing)


def from_values(record_index, value: torch.Tensor, interests: torch.Tensor) -> Records:
    """Records from value codes where -1 marks a missing value.

    A scalar field is MISSING when its value is -1. The interest property is MISSING
    only when all three slots are -1; otherwise the known slots are its values.
    """
    state = torch.cat([value < 0, (interests < 0).all(1, keepdim=True)], 1).to(torch.int8)
    n = len(value)
    return Records(
        torch.as_tensor(record_index).long(),
        state,
        value.to(torch.int8),
        interests.to(torch.int8),
        torch.zeros(n, 4, dtype=torch.int8),
    )


def mcar_records(signatures: pl.DataFrame, rate: float, data_dir: Path = DATA) -> Records:
    """The fixture with one fixed missing-completely-at-random mask at `rate`."""
    mask = pl.read_parquet(data_dir / "mcar.parquet").filter(pl.col("rate") == rate)
    if not mask["record_index"].equals(signatures["record_index"]):
        raise ValueError(f"No MCAR mask aligned with the signatures at rate {rate}")
    complete = complete_records(signatures)
    scalar = mask.select([f"{f}_missing" for f in FIELDS[:3]]).to_torch().bool()
    slots = mask.select([f"interest_{k}_missing" for k in (1, 2, 3)]).to_torch().bool()
    return from_values(
        complete.record_index,
        torch.where(scalar, -1, complete.value),
        torch.where(slots, -1, complete.interests),
    )


def duplicate_records(rate: float, purpose: str, data_dir: Path = DATA) -> Records:
    """Experiment 3: noisy duplicates. `record_index` is the source record they copy."""
    rows = pl.read_parquet(data_dir / "duplicates.parquet").filter(
        (pl.col("rate") == rate) & (pl.col("purpose") == purpose)
    )
    cols = lambda names: rows.select(names).to_torch().to(torch.int8)
    return from_values(
        rows["source_record"].to_torch(),
        cols(list(FIELDS[:3])),
        cols(["interest_1", "interest_2", "interest_3"]),
    )


def query_panel(name: str, data_dir: Path = SCALE_DATA) -> torch.Tensor:
    """Record indices of the scale study's calibration or evaluation query panel, in rank order."""
    rows = (
        pl.read_parquet(data_dir / "query_panels.parquet")
        .filter(pl.col("panel") == name)
        .sort("query_rank")
    )
    return rows["record_index"].to_torch().long()


# --------------------------------------------------------------------------- MAP-I encoder

ATOM_FAMILIES = ("role", "age_level", "job_category", "home_region", "interests", "null")


def age_levels(dimension: int, generator: torch.Generator) -> torch.Tensor:
    """Ordinal levels: flip one of nine near-equal partitions of D/2 coordinates per step."""
    start = torchhd.random(1, dimension, vsa="MAP", generator=generator, dtype=torch.float32)[0]
    chosen = torch.randperm(dimension, generator=generator)[: dimension // 2]
    levels = [start.clone()]
    for part in torch.tensor_split(chosen, AGE_LEVELS - 1):
        nxt = levels[-1].clone()
        nxt[part] *= -1
        levels.append(nxt)
    return torch.stack(levels)


def make_atoms(dimension: int, seed: int) -> dict[str, torch.Tensor]:
    """Independent bipolar roles, values and null tokens; ordinal age levels.

    Seeds: torch.Generator().manual_seed(1000 * seed + offset) per family.
    """

    def gen(offset):
        return torch.Generator().manual_seed(1000 * seed + offset)

    atoms = {
        "role": torchhd.random(4, dimension, vsa="MAP", generator=gen(1), dtype=torch.float32),
        "age_level": age_levels(dimension, gen(2)),
    }
    for i, field in enumerate(FIELDS[1:]):
        atoms[field] = torchhd.random(
            SIZES[field], dimension, vsa="MAP", generator=gen(3 + i), dtype=torch.float32
        )
    atoms["null"] = torchhd.random(
        len(NULL_SPELLINGS), dimension, vsa="MAP", generator=gen(7), dtype=torch.float32
    )
    # One tie-break vector per property for majority-sign normalization (Experiment 4).
    atoms["tiebreak"] = torchhd.random(
        4, dimension, vsa="MAP", generator=gen(8), dtype=torch.float32
    )
    return atoms


@dataclass
class Codebook:
    """Bound fact tables built from one set of atoms (normally read back from LanceDB).

    Each field table holds the bound value facts, then ROLE * NULL[spelling] for
    each spelling, then a zero row that stands for "no term".
    """

    dimension: int
    seed: int
    tables: list[torch.Tensor]
    level_similarity: torch.Tensor
    atoms: dict

    @classmethod
    def from_atoms(cls, dimension: int, seed: int, atoms: dict[str, torch.Tensor]) -> Codebook:
        roles, nulls = atoms["role"], atoms["null"]
        values = [atoms["age_level"]] + [atoms[f] for f in FIELDS[1:]]
        zero = torch.zeros(1, dimension)
        tables = [
            torch.cat([torchhd.bind(roles[f], values[f]), torchhd.bind(roles[f], nulls), zero])
            for f in range(4)
        ]
        levels = atoms["age_level"]
        return cls(dimension, seed, tables, torchhd.cosine_similarity(levels, levels), atoms)


def codebook(dimension: int, seed: int) -> Codebook:
    return Codebook.from_atoms(dimension, seed, make_atoms(dimension, seed))


def encode_sum(book: Codebook, records: Records, strategy: str) -> torch.Tensor:
    """Unnormalized additive bundles. Every coordinate is an integer in [-6, 6]."""
    total = torch.zeros(len(records), book.dimension)
    missing = records.state == MISSING
    for f, field in enumerate(FIELDS):
        size = SIZES[field]
        no_term = size + len(NULL_SPELLINGS)
        null_row = size + records.spelling[:, f].long()
        marker = (
            torch.where(missing[:, f], null_row, no_term)
            if strategy == "token"
            else torch.full_like(null_row, no_term)
        )
        if f < 3:
            total += book.tables[f][torch.where(missing[:, f], marker, records.value[:, f].long())]
            continue
        for k in range(MAX_INTERESTS):
            present = ~missing[:, f] & (records.interests[:, k] >= 0)
            slot = torch.where(
                present,
                records.interests[:, k].long(),
                marker if k == 0 else torch.full_like(marker, no_term),
            )
            total += book.tables[f][slot]
    return total


def property_terms(book: Codebook, records: Records, normalized: bool) -> torch.Tensor:
    """Each property's own term, (4, n, D); a record's bundle is their sum.

    Unnormalized, a property's term is the sum of its bound facts. Normalized, that
    sum is divided by its L2 norm, so every known property has unit length. Omit strategy.
    """
    terms = []
    for f in range(len(FIELDS)):
        others = torch.ones(len(records), len(FIELDS), dtype=torch.bool)
        others[:, f] = False
        part = encode_sum(book, records.without(others), "omit")
        terms.append(part / part.norm(dim=1, keepdim=True).clamp(min=1) if normalized else part)
    return torch.stack(terms)


def encode_sign(book: Codebook, records: Records) -> torch.Tensor:
    """Majority-sign normalization: each known property's bundle is thresholded to +-1.

    sign(ROLE * sum of values) = ROLE * sign(sum of values), because the role is bipolar.
    With an even number of values some coordinates sum to 0; half a fixed random
    tie-break vector per property decides them, and never changes an odd sum. Every
    known property then has norm sqrt(D), the bundle stays integer-valued (float16 is
    exact), and unknown properties contribute nothing (Kanerva 2009).
    """
    total = torch.zeros(len(records), book.dimension)
    terms = property_terms(book, records, normalized=False)
    for f in range(len(FIELDS)):
        part = terms[f]
        known = part.abs().sum(1, keepdim=True) > 0
        total += torch.where(known, torch.sign(part + 0.5 * book.atoms["tiebreak"][f]), 0.0)
    return total


def encode_normalized(book: Codebook, records: Records) -> torch.Tensor:
    """Property-normalized bundles: each known field is scaled to unit length before bundling.

    A field's term is ROLE * (sum of its known values) / ||sum of its known values||.
    Binding a bipolar role preserves norms, so this equals the field's bound facts
    from `encode_sum`, divided by their norm. Each observed field then contributes
    the same norm however many values it lists; a field with no known value
    contributes nothing. Omit strategy only.
    Coordinates are no longer integers, so float32 dot products are not exact ties.
    """
    return property_terms(book, records, normalized=True).sum(0)  # unknown properties stay zero


def cosine(a_sum: torch.Tensor, b_sum: torch.Tensor, paired: bool = False) -> torch.Tensor:
    """Cosine from integer-valued sums. Dot products are exact in float32 (|a.b| <= 36 D),
    so records with identical vectors tie exactly. `paired` compares row i with row i."""
    a_norm = a_sum.square().sum(1).sqrt().clamp(min=1)
    b_norm = b_sum.square().sum(1).sqrt().clamp(min=1)
    if paired:
        return (a_sum * b_sum).sum(1) / (a_norm * b_norm)
    return (a_sum @ b_sum.T) / a_norm[:, None] / b_norm[None, :]


# --------------------------------------------------------------------------- exact baselines


@dataclass
class Baselines:
    content: torch.Tensor  # S_content: similarity supported by what both records know
    encoder: torch.Tensor  # S_encoder: exact cosine of the encoded terms, if atoms were orthogonal
    valid: torch.Tensor  # both records have content and the candidate is not the query itself


def baselines(a: Records, b: Records, strategy: str, paired: bool = False) -> Baselines:
    """S_content and S_encoder for every (a, b) pair, or row-wise when `paired`.

    Content numerator: k_age + [same job] + [same region] + |shared interests|,
    counted only when both sides know the field; denominator sqrt(w_a * w_b),
    the geometric mean of their content facts. With tokens, each field both
    records spell the same way adds 1 to the encoder numerator, and every token
    counts as a term. Ratios of small integers, so float32 is ample.
    """

    def pair(x, y):
        return (x, y) if paired else (x[:, None], y[None, :])

    sa, sb = pair(a.state, b.state)
    both = (sa == PRESENT) & (sb == PRESENT)
    va, vb = pair(a.value, b.value)
    ia, ib = pair(a.interests, b.interests)
    age = torch.where(
        both[..., 0],
        1 - (va[..., 0].int() - vb[..., 0].int()).abs().float() / (AGE_LEVELS - 1),
        0.0,
    )
    same = ((va[..., 1:] == vb[..., 1:]) & both[..., 1:3]).sum(-1)
    shared = torch.zeros(age.shape, dtype=torch.int8)
    for i in range(MAX_INTERESTS):
        for j in range(MAX_INTERESTS):
            shared += ((ia[..., i] == ib[..., j]) & (ia[..., i] >= 0)).to(torch.int8)
    content_num = age + same + torch.where(both[..., 3], shared, 0)
    wa, wb = pair(a.content_weight(), b.content_weight())
    content = content_num / (wa * wb).float().sqrt().clamp(min=1)
    if strategy == "token":
        pa, pb = pair(a.spelling, b.spelling)
        tokens = ((sa == MISSING) & (sb == MISSING) & (pa == pb)).sum(-1)
        ta, tb = pair(
            a.content_weight() + (a.state == MISSING).sum(1),
            b.content_weight() + (b.state == MISSING).sum(1),
        )
        encoder = (content_num + tokens) / (ta * tb).float().sqrt().clamp(min=1)
    else:
        encoder = content
    ra, rb = pair(a.record_index, b.record_index)
    valid = (wa > 0) & (wb > 0) & (ra != rb)
    return Baselines(content, encoder, valid)


def normalized_baseline(a: Records, b: Records) -> torch.Tensor:
    """Exact cosine of `encode_normalized` bundles, row i against row i, if atoms were orthogonal.

    Each known field is a unit vector, so a record's squared norm is its number of
    known fields. A field both rows know adds k_age for age, [same value] for job
    and region, and |shared| / sqrt(n_a * n_b) for interests, where n counts each
    side's known interests. Omit strategy only.
    """

    def known(r):
        """Which properties carry a unit term, and how many interests are known."""
        n = (r.interests >= 0).sum(1)
        has_interests = (r.state[:, 3] == PRESENT) & (n > 0)
        return torch.cat([r.state[:, :3] == PRESENT, has_interests[:, None]], 1), n

    ka, na = known(a)
    kb, nb = known(b)
    both = ka & kb
    age = 1 - (a.value[:, 0].int() - b.value[:, 0].int()).abs().float() / (AGE_LEVELS - 1)
    same = (a.value[:, 1:] == b.value[:, 1:]).float()
    shared = sum(
        ((a.interests[:, i] == b.interests[:, j]) & (a.interests[:, i] >= 0)).float()
        for i in range(MAX_INTERESTS)
        for j in range(MAX_INTERESTS)
    )
    interests = shared / (na * nb).float().sqrt().clamp(min=1)
    terms = torch.stack([age, same[:, 0], same[:, 1], interests], 1)
    dot = torch.where(both, terms, 0.0).sum(1)
    return dot / (ka.sum(1) * kb.sum(1)).float().sqrt().clamp(min=1)
