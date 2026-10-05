"""PERSON records, the MAP-I encoder and the exact similarity baselines.

A record has four fields: an ordinal age band, a job category, a home region and
a set of three interests, so six bound facts when complete. A missing field is
handled by one of two strategies:

  omit   the field contributes no fact
  token  the field contributes ROLE * NULL[spelling]: one shared vector per null
         spelling ("null", "no_value", ""), each an independent random atom

A missing interest set is one token fact, the same weight as any other field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import torchhd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "scale-missingness"
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


def load_fixture(data_dir: Path = DATA):
    signatures = pl.read_parquet(data_dir / "signatures.parquet")
    if signatures.height != 920_000 or not signatures["record_index"].equals(
        pl.Series("record_index", range(920_000), dtype=pl.Int32)
    ):
        raise ValueError("Expected 920,000 signatures in record_index order")
    return signatures


def fixture_hashes(data_dir: Path = DATA) -> dict[str, str]:
    names = (
        "signatures.parquet",
        "missingness.parquet",
        "query_panels.parquet",
        "pairwise_pairs.parquet",
    )
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in names}


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
