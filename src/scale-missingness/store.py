"""LanceDB storage (results/scale-missingness/experiment.lancedb).

  symbols_d{D}      every atom for every seed: roles, age levels, job, region and
                    interest values, and the three null tokens (a few MB in total)
  records_d2048_s11 the unnormalized complete-record bundle for all 920,000
                    records at the one (D, seed) Experiment 4 indexes (~3.8 GB)

Vectors are fixed-size lists of float16, cast to float32 on read. Atoms are
bipolar and a bundle is a sum of six bipolar facts, so every stored coordinate
is a small integer and float16 holds it exactly. All other bundles are built
on the fly, in float32, from the stored atoms.
"""

from __future__ import annotations

import lancedb
import person
import pyarrow as pa
import pyarrow.compute as pc
import torch
from person import Codebook, Records


def symbol_schema(dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("seed", pa.int32()),
            pa.field("family", pa.string()),
            pa.field("position", pa.int16()),
            pa.field("vector", pa.list_(pa.float16(), dimension)),
        ]
    )


def record_schema(dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("record_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float16(), dimension)),
        ]
    )


def to_f16_column(vectors: torch.Tensor, dimension: int) -> pa.FixedSizeListArray:
    """The float16 write boundary. Refuses values float16 cannot hold exactly."""
    if vectors.dtype != torch.float32 or vectors.shape[1] != dimension:
        raise ValueError("Expected float32 vectors of the table's dimension")
    if not bool((vectors == vectors.round()).all()) or float(vectors.abs().max()) > 2048:
        raise ValueError("Stored hypervectors must be small integers so float16 is exact")
    values = pa.array(vectors.to(torch.float16).contiguous().reshape(-1).numpy(), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(values, dimension)


def from_f16_column(column, dimension: int) -> torch.Tensor:
    """The float32 read boundary."""
    flat = column.combine_chunks().values if isinstance(column, pa.ChunkedArray) else column.values
    return (
        torch.from_numpy(flat.to_numpy(zero_copy_only=False).copy())
        .reshape(-1, dimension)
        .to(torch.float32)
    )


def symbols_table(dimension: int) -> str:
    return f"symbols_d{dimension}"


def records_table(dimension: int, seed: int) -> str:
    return f"records_d{dimension}_s{seed}"


def write_symbols(db, dimension: int, seeds) -> None:
    rows = {"seed": [], "family": [], "position": []}
    vectors = []
    for seed in seeds:
        atoms = person.make_atoms(dimension, seed)
        for family in person.ATOM_FAMILIES:
            for position, vector in enumerate(atoms[family]):
                rows["seed"].append(seed)
                rows["family"].append(family)
                rows["position"].append(position)
                vectors.append(vector)
    schema = symbol_schema(dimension)
    table = pa.Table.from_arrays(
        [
            pa.array(rows["seed"], pa.int32()),
            pa.array(rows["family"]),
            pa.array(rows["position"], pa.int16()),
            to_f16_column(torch.stack(vectors), dimension),
        ],
        schema=schema,
    )
    stored = db.create_table(symbols_table(dimension), table, schema=schema, mode="overwrite")
    if stored.schema != schema:
        raise AssertionError("LanceDB changed the symbol schema")


def read_codebook(db, dimension: int, seed: int) -> Codebook:
    table = db.open_table(symbols_table(dimension)).to_arrow()  # a few hundred rows
    table = table.filter(pc.equal(table["seed"], seed))
    if table.num_rows == 0:
        raise ValueError(f"No stored atoms for D={dimension}, seed={seed}")
    vectors = from_f16_column(table["vector"], dimension)
    families, positions = table["family"].to_pylist(), table["position"].to_pylist()
    atoms = {
        f: vectors[
            [
                i
                for _, i in sorted(
                    (p, i) for i, (g, p) in enumerate(zip(families, positions)) if g == f
                )
            ]
        ]
        for f in person.ATOM_FAMILIES
    }
    return Codebook.from_atoms(dimension, seed, atoms)


def write_records(db, name: str, book: Codebook, records: Records, chunk: int = 8192):
    """Encode and write complete-record bundles in bounded batches, in record_index order."""
    dimension = book.dimension
    schema = record_schema(dimension)

    def batches():
        for start in range(0, len(records), chunk):
            block = records.take(slice(start, start + chunk))
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(block.record_index.numpy(), pa.int32()),
                    to_f16_column(person.encode_sum(book, block, "omit"), dimension),
                ],
                schema=schema,
            )

    table = db.create_table(
        name, pa.RecordBatchReader.from_batches(schema, batches()), mode="overwrite"
    )
    if table.schema != schema:
        raise AssertionError("LanceDB changed the record schema")
    return table


def read_rows(table, dimension: int, offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows by position, in the order requested, as float32."""
    arrow = table.take_offsets(offsets).to_arrow()
    index = torch.from_numpy(arrow["record_index"].to_numpy().astype("int64"))
    vectors = from_f16_column(arrow["vector"], dimension)
    order = torch.argsort(index)
    index, vectors = index[order], vectors[order]
    if not torch.equal(index, torch.as_tensor(sorted(offsets), dtype=torch.int64)):
        raise AssertionError("Stored rows must be in record_index order")
    rank = torch.argsort(torch.argsort(torch.as_tensor(offsets)))
    return index[rank], vectors[rank]


def connect(path) -> lancedb.DBConnection:
    return lancedb.connect(str(path))
