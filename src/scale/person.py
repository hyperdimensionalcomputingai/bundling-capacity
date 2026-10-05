"""PERSON records, the MAP-I encoder and the exact similarity baseline.

A record has four fields: an ordinal age band, a job category, a home region and
a set of three interests, so six bound facts. Every fixture record is complete;
missing values are studied separately in src/missingness (HYP-118).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import torchhd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "scale"
FIELDS = ("age_band", "job_category", "home_region", "interests")
SIZES = {"age_band": 10, "job_category": 8, "home_region": 5, "interests": 25}
AGE_LEVELS = SIZES["age_band"]
MAX_INTERESTS = 3
FACTS = 3 + MAX_INTERESTS  # age, job, region and three interests


# --------------------------------------------------------------------------- records


@dataclass
class Records:
    """Complete records as int8 tensors."""

    record_index: torch.Tensor  # (n,) int64
    value: torch.Tensor  # (n, 3) age / job / region codes
    interests: torch.Tensor  # (n, 3) sorted distinct interests

    def __len__(self):
        return len(self.record_index)

    def take(self, index) -> Records:
        return Records(*(x[index] for x in (self.record_index, self.value, self.interests)))


def load_fixture(data_dir: Path = DATA):
    signatures = pl.read_parquet(data_dir / "signatures.parquet")
    if signatures.height != 920_000 or not signatures["record_index"].equals(
        pl.Series("record_index", range(920_000), dtype=pl.Int32)
    ):
        raise ValueError("Expected 920,000 signatures in record_index order")
    return signatures


def fixture_hashes(data_dir: Path = DATA) -> dict[str, str]:
    names = ("signatures.parquet", "query_panels.parquet")
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in names}


def complete_records(signatures: pl.DataFrame) -> Records:
    cols = lambda names: signatures.select(names).to_torch().to(torch.int8)
    return Records(
        signatures["record_index"].to_torch().long(),
        cols(list(FIELDS[:3])),
        cols(["interest_1", "interest_2", "interest_3"]),
    )


# --------------------------------------------------------------------------- MAP-I encoder

ATOM_FAMILIES = ("role", "age_level", "job_category", "home_region", "interests")


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
    """Independent bipolar roles and values; ordinal age levels.

    Seeds: torch.Generator().manual_seed(1000 * seed + offset) per family. Offset 7
    (null tokens) belongs to src/missingness; leaving it unused here keeps every
    other family identical to the atoms that study regenerates.
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
    return atoms


@dataclass
class Codebook:
    """Bound fact tables built from one set of atoms (normally read back from LanceDB).

    tables[f][v] is ROLE[f] * VALUE[f][v], the bound fact for value v of field f.
    """

    dimension: int
    seed: int
    tables: list[torch.Tensor]
    level_similarity: torch.Tensor
    atoms: dict

    @classmethod
    def from_atoms(cls, dimension: int, seed: int, atoms: dict[str, torch.Tensor]) -> Codebook:
        roles = atoms["role"]
        values = [atoms["age_level"]] + [atoms[f] for f in FIELDS[1:]]
        tables = [torchhd.bind(roles[f], values[f]) for f in range(4)]
        levels = atoms["age_level"]
        return cls(dimension, seed, tables, torchhd.cosine_similarity(levels, levels), atoms)


def codebook(dimension: int, seed: int) -> Codebook:
    return Codebook.from_atoms(dimension, seed, make_atoms(dimension, seed))


def encode_sum(book: Codebook, records: Records) -> torch.Tensor:
    """Unnormalized additive bundles of the six facts. Every coordinate is an integer in [-6, 6]."""
    total = torch.zeros(len(records), book.dimension)
    for f in range(3):
        total += book.tables[f][records.value[:, f].long()]
    for k in range(MAX_INTERESTS):
        total += book.tables[3][records.interests[:, k].long()]
    return total


def cosine(a_sum: torch.Tensor, b_sum: torch.Tensor, paired: bool = False) -> torch.Tensor:
    """Cosine from integer-valued sums. Dot products are exact in float32 (|a.b| <= 36 D),
    so records with identical vectors tie exactly. `paired` compares row i with row i."""
    a_norm = a_sum.square().sum(1).sqrt().clamp(min=1)
    b_norm = b_sum.square().sum(1).sqrt().clamp(min=1)
    if paired:
        return (a_sum * b_sum).sum(1) / (a_norm * b_norm)
    return (a_sum @ b_sum.T) / a_norm[:, None] / b_norm[None, :]


# --------------------------------------------------------------------------- exact baseline


@dataclass
class Baselines:
    exact: torch.Tensor  # S_exact: the cosine the encoder intends, if atoms were orthogonal
    valid: torch.Tensor  # the candidate is not the query itself


def baselines(a: Records, b: Records, paired: bool = False) -> Baselines:
    """S_exact for every (a, b) pair, or row-wise when `paired`.

    Numerator: k_age + [same job] + [same region] + |shared interests|, with the
    intended linear age kernel k_age = 1 - |i - j| / 9. Both records have six unit
    facts, so the denominator is 6. Ratios of small integers, so float32 is ample.
    """

    def pair(x, y):
        return (x, y) if paired else (x[:, None], y[None, :])

    va, vb = pair(a.value, b.value)
    ia, ib = pair(a.interests, b.interests)
    age = 1 - (va[..., 0].int() - vb[..., 0].int()).abs().float() / (AGE_LEVELS - 1)
    same = (va[..., 1:] == vb[..., 1:]).sum(-1)
    shared = torch.zeros(age.shape, dtype=torch.int8)
    for i in range(MAX_INTERESTS):
        for j in range(MAX_INTERESTS):
            shared += (ia[..., i] == ib[..., j]).to(torch.int8)
    ra, rb = pair(a.record_index, b.record_index)
    return Baselines((age + same + shared) / FACTS, ra != rb)
