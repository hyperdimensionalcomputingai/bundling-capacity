"""Build the controlled factorial PERSON fixture for the scale study.

Every combination of 10 age bands, 8 job categories, 5 home regions and three
distinct interests from 25 appears exactly once: 920,000 integer signatures.
Nothing here is sampled from people, and no names or free text are generated.

Run from the repository root:

    uv run --project src/scale python data/scale/generate.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import polars as pl
import torch

HERE = Path(__file__).resolve().parent
SEED = "hyp83-person-v2"
AGE_BANDS = ("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+")
JOB_CATEGORIES = (
    "education",
    "engineering",
    "finance",
    "healthcare",
    "hospitality",
    "public_service",
    "retail",
    "skilled_trades",
)
HOME_REGIONS = ("north", "south", "east", "west", "central")
INTERESTS = (
    "baking",
    "birdwatching",
    "board_games",
    "chess",
    "climbing",
    "cycling",
    "film",
    "gardening",
    "hiking",
    "history",
    "jazz",
    "knitting",
    "languages",
    "painting",
    "photography",
    "poetry",
    "running",
    "sailing",
    "science_fiction",
    "swimming",
    "tennis",
    "theatre",
    "volunteering",
    "woodworking",
    "yoga",
)
TRIPLES = torch.tensor(list(itertools.combinations(range(len(INTERESTS)), 3)), dtype=torch.int8)
CELLS = len(AGE_BANDS) * len(JOB_CATEGORIES) * len(HOME_REGIONS)
RECORDS = CELLS * len(TRIPLES)

# Query panels. Endpoint ages are deliberately over-represented: among complete
# records only an age-1 or age-10 query has candidates with zero content similarity.
PANELS = {
    "calibration": {"endpoint_each": 40, "interior_each": 15},
    "evaluation": {"endpoint_each": 80, "interior_each": 30},
}


def rng(*parts: object) -> torch.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in (SEED, *parts)).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1)


def balanced_cell_order() -> torch.Tensor:
    """Order the 400 (age, job, region) cells so every prefix has even marginals.

    Ten blocks of 40 cells. Within block b, position k uses age k mod 10,
    job (k - b // 5) mod 8 and region (k - b % 5) mod 5. Each field therefore
    cycles through its values without gaps, so any run of positions is balanced
    to within one. A block contains exactly the cells with age - region = b mod 5
    and age - job = b // 5 mod 2, so the ten blocks cover all 400 cells once.
    """
    order = [(k % 10, (k - b // 5) % 8, (k - b % 5) % 5) for b in range(10) for k in range(40)]
    if len(set(order)) != CELLS:
        raise AssertionError("Cell order must visit every (age, job, region) cell once")
    return torch.tensor(order, dtype=torch.int8)


def signatures() -> pl.DataFrame:
    order = balanced_cell_order()
    # Each cell visits its 2,300 interest triples in its own fixed shuffled order.
    triple_order = torch.stack(
        [torch.randperm(len(TRIPLES), generator=rng("triples", *cell.tolist())) for cell in order]
    )
    index = torch.arange(RECORDS)
    slot, round_ = index % CELLS, index // CELLS
    cell = order[slot]
    triples = TRIPLES[triple_order[slot, round_]]
    return pl.DataFrame(
        {
            "record_index": index.to(torch.int32).numpy(),
            "age_band": cell[:, 0].numpy(),
            "job_category": cell[:, 1].numpy(),
            "home_region": cell[:, 2].numpy(),
            "interest_1": triples[:, 0].numpy(),
            "interest_2": triples[:, 1].numpy(),
            "interest_3": triples[:, 2].numpy(),
        }
    )


def query_panels(frame: pl.DataFrame) -> pl.DataFrame:
    lookup = {
        sig: i
        for i, sig in enumerate(
            frame.select(
                "age_band", "job_category", "home_region", "interest_1", "interest_2", "interest_3"
            ).rows()
        )
    }
    used, rows = set(), []
    for panel, design in PANELS.items():
        ages = (
            [0] * design["endpoint_each"]
            + [9] * design["endpoint_each"]
            + [a for a in range(1, 9) for _ in range(design["interior_each"])]
        )
        size = len(ages)
        if size % len(JOB_CATEGORIES) or size % len(HOME_REGIONS):
            raise ValueError("Panel size must balance jobs and regions")
        g = rng("panel", panel)
        ages = torch.tensor(ages)[torch.randperm(size, generator=g)]
        jobs = torch.arange(len(JOB_CATEGORIES)).repeat(size // len(JOB_CATEGORIES))[
            torch.randperm(size, generator=g)
        ]
        regions = torch.arange(len(HOME_REGIONS)).repeat(size // len(HOME_REGIONS))[
            torch.randperm(size, generator=g)
        ]
        for rank in range(size):
            while True:
                triple = TRIPLES[int(torch.randint(len(TRIPLES), (1,), generator=g))].tolist()
                record = lookup[(int(ages[rank]), int(jobs[rank]), int(regions[rank]), *triple)]
                if record not in used:
                    break
            used.add(record)
            rows.append({"panel": panel, "query_rank": rank, "record_index": record})
    return pl.DataFrame(
        rows, schema={"panel": pl.String, "query_rank": pl.Int16, "record_index": pl.Int32}
    )


def check(frame: pl.DataFrame):
    if frame.height != 920_000 or RECORDS != 920_000 or len(INTERESTS) != 25:
        raise AssertionError("Expected 10 x 8 x 5 x C(25, 3) = 920,000 signatures")
    keys = ["age_band", "job_category", "home_region", "interest_1", "interest_2", "interest_3"]
    if frame.select(keys).n_unique() != RECORDS:
        raise AssertionError("Signatures must be unique")
    if not (
        (frame["interest_1"] < frame["interest_2"]) & (frame["interest_2"] < frame["interest_3"])
    ).all():
        raise AssertionError("Interest triples must be three distinct sorted values")
    for n in (100, 1_000, 10_000, 100_000, 920_000):
        prefix = frame.head(n)
        for field, size in (("age_band", 10), ("job_category", 8), ("home_region", 5)):
            counts = prefix[field].value_counts()["count"]
            if counts.len() != size or counts.max() - counts.min() > 1:
                raise AssertionError(f"Prefix {n:,} is unbalanced for {field}")


def main():
    frame = signatures()
    check(frame)
    outputs = {
        "signatures.parquet": frame,
        "query_panels.parquet": query_panels(frame),
    }
    for name, table in outputs.items():
        table.write_parquet(HERE / name, statistics=False)
    (HERE / "domains.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "age_band": AGE_BANDS,
                "job_category": JOB_CATEGORIES,
                "home_region": HOME_REGIONS,
                "interests": INTERESTS,
                "panels": PANELS,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Wrote {frame.height:,} signatures and the query panels to {HERE}"
    )


if __name__ == "__main__":
    main()
