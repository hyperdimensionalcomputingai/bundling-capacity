"""Measure answer cleanup and profile geometry in an ordered MAP bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lancedb
import polars as pl
import pyarrow as pa
import torch
import torchhd


ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = (512, 2048, 4096, 8192)
WIDTHS = (5, 10, 20, 30, 40, 50)
SEEDS = (11, 23, 37, 53, 71)
SCORING_KEY = "IPIP Big-Five Factor Markers, 50-item sample, accessed 2026-09-23"

ITEMS = pa.schema([
    pa.field("item_id", pa.string()), pa.field("position", pa.int16()),
    pa.field("factor_id", pa.int8()), pa.field("factor_name", pa.string()),
    pa.field("key_direction", pa.string()), pa.field("statement", pa.string()),
])
RESPONSES = pa.schema([
    pa.field("respondent_id", pa.string()), pa.field("item_id", pa.string()),
    pa.field("raw_response", pa.int8()), pa.field("scored_response", pa.int8()),
])


def vector_schema(dimension: int, symbols: bool) -> pa.Schema:
    fields = [pa.field("seed", pa.int32())]
    if symbols:
        fields += [pa.field("kind", pa.string()), pa.field("namespace", pa.string()),
                   pa.field("symbol_id", pa.string())]
    else:
        fields += [pa.field("respondent_id", pa.string()), pa.field("bundle_width", pa.int8())]
    return pa.schema(fields + [pa.field("vector", pa.list_(pa.float16(), dimension))])


def write_vectors(db, name: str, dimension: int, metadata: list[dict], vectors: torch.Tensor, *, symbols: bool):
    schema = vector_schema(dimension, symbols)
    if vectors.shape != (len(metadata), dimension) or vectors.dtype != torch.float32:
        raise ValueError(f"Wrong shape or dtype for {name}")
    # Storage alone is float16. Bipolar keys, levels and sums <=50 are exact in it.
    values = pa.array(vectors.to(torch.float16).contiguous().reshape(-1).numpy(), type=pa.float16())
    columns = [pa.array([row[field.name] for row in metadata], type=field.type)
               for field in list(schema)[:-1]]
    columns.append(pa.FixedSizeListArray.from_arrays(values, dimension))
    table = pa.Table.from_arrays(columns, schema=schema)
    stored = db.create_table(name, table, schema=schema, mode="overwrite")
    if stored.schema != schema:
        raise AssertionError(f"LanceDB changed {name}'s schema")
    return stored


def read_vectors(table, dimension: int, *, symbols: bool):
    arrow = table.to_arrow()
    if arrow.schema != vector_schema(dimension, symbols):
        raise AssertionError("Expected fixed-size float16 hypervectors")
    metadata = arrow.drop(["vector"]).to_pylist()
    flat = arrow["vector"].combine_chunks().values.to_numpy(zero_copy_only=False).copy()
    return metadata, torch.from_numpy(flat).reshape(-1, dimension).to(torch.float32)


def source(data_dir: Path):
    items = pl.read_csv(data_dir / "items.csv").to_arrow().cast(ITEMS)
    profiles = pl.read_csv(data_dir / "profiles.csv")
    raw = pl.read_csv(data_dir / "responses.csv")
    item_ids = items["item_id"].to_pylist()
    respondent_ids = profiles["profile_id"].to_list()
    keying = items["key_direction"].to_pylist()
    if item_ids != [f"Q{i:02d}" for i in range(1, 51)]:
        raise ValueError("The published Q01–Q50 item order is required")
    if len(respondent_ids) != 31 or len(set(respondent_ids)) != 31:
        raise ValueError("Expected 31 fixed synthetic profiles")
    if keying[18] != "+" or keying[48] != "-":
        raise ValueError("Q19/Q49 scoring key is inconsistent")
    if set(keying) != {"+", "-"}:
        raise ValueError("Unknown key direction")
    factors = items["factor_id"].to_pylist()
    if any(factors.count(factor) != 10 for factor in range(1, 6)):
        raise ValueError("Expected ten items from each IPIP factor")
    if items["factor_name"][3].as_py() != "Emotional Stability":
        raise ValueError("Factor IV must point toward Emotional Stability")
    lookup = {(r["profile_id"], r["item_id"]): r["raw_answer"] for r in raw.iter_rows(named=True)}
    if len(lookup) != 1550 or raw.height != 1550 or set(lookup) != {
        (p, i) for p in respondent_ids for i in item_ids
    }:
        raise ValueError("Expected exactly one response at every profile/item position")
    raw_matrix = torch.tensor([[lookup[p, i] for i in item_ids] for p in respondent_ids], dtype=torch.int8)
    if not bool(((raw_matrix >= 1) & (raw_matrix <= 5)).all()):
        raise ValueError("Responses must be 1–5")
    reversed_items = torch.tensor([direction == "-" for direction in keying], dtype=torch.bool)
    scored = torch.where(reversed_items[None, :], 6 - raw_matrix, raw_matrix)
    if not torch.equal(torch.where(reversed_items[None, :], 6 - scored, scored), raw_matrix):
        raise AssertionError("Raw/scored inversion failed")
    # Q19=5 and Q49=1 are both scored 5, but their item IDs remain separate.
    example = torch.tensor([5, 1], dtype=torch.int8)
    if tuple(torch.where(reversed_items[[18, 48]], 6 - example, example).tolist()) != (5, 5):
        raise AssertionError("Reverse-scoring example failed")
    rows = [{"respondent_id": p, "item_id": item_ids[i],
             "raw_response": int(raw_matrix[j, i]), "scored_response": int(scored[j, i])}
            for j, p in enumerate(respondent_ids) for i in range(50)]
    return items, respondent_ids, raw_matrix, scored, reversed_items, pa.Table.from_pylist(rows, schema=RESPONSES)


def encoder(dimension: int, seed: int):
    keys = torchhd.random(50, dimension, vsa="MAP", dtype=torch.float32,
                          generator=torch.Generator().manual_seed(100 * seed + 5))
    levels = torchhd.level(5, dimension, vsa="MAP", randomness=0.0, dtype=torch.float32,
                           generator=torch.Generator().manual_seed(100 * seed + 4))
    similarities = torchhd.cosine_similarity(levels, levels)
    adjacent = torch.stack([similarities[i, i + 1] for i in range(4)])
    distant = torch.stack([similarities[i, i + 3] for i in range(2)])
    if not bool(adjacent.min() > distant.max()):
        raise AssertionError("The five TorchHD levels lack ordered geometry")
    return keys, levels, similarities


def bundles_at_widths(keys: torch.Tensor, levels: torch.Tensor, scored: torch.Tensor,
                      widths: tuple[int, ...]):
    current = None
    outputs = {}
    for i in range(max(widths)):
        fact = torchhd.bind(keys[i].unsqueeze(0), levels[scored[:, i].long() - 1])
        current = fact if current is None else torchhd.bundle(current, fact)
        if i + 1 in widths:
            outputs[i + 1] = current.clone().to(torch.float32)
    return outputs


def rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties, used only for descriptive Spearman correlation."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = rankdata(a), rankdata(b)
    x, y = x - x.mean(), y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x @ y / denom).item()) if denom > 0 else float("nan")


def geometry(vectors: torch.Tensor, scored: torch.Tensor, level_similarity: torch.Tensor,
             respondent_ids: list[str], dimension: int, seed: int, width: int):
    normalized = torch.nn.functional.normalize(vectors, dim=1)
    vector_cosine = normalized @ normalized.T
    pairs = []
    for a in range(len(respondent_ids)):
        for b in range(a + 1, len(respondent_ids)):
            distance = (scored[a, :width].float() - scored[b, :width].float()).abs().mean()
            oracle = level_similarity[scored[a, :width].long() - 1,
                                      scored[b, :width].long() - 1].mean()
            pairs.append({"dimension": dimension, "seed": seed, "bundle_width": width,
                          "respondent_a": respondent_ids[a], "respondent_b": respondent_ids[b],
                          "bundle_cosine": float(vector_cosine[a, b]),
                          "mean_absolute_response_difference": float(distance),
                          "oracle_codebook_cosine": float(oracle)})
    frame = pl.DataFrame(pairs)
    rank_rows = []
    for reference, sign in (("response_difference", -1), ("oracle_codebook", 1)):
        col = "mean_absolute_response_difference" if sign == -1 else "oracle_codebook_cosine"
        ref = torch.tensor(frame[col].to_list(), dtype=torch.float32) * sign
        observed = torch.tensor(frame["bundle_cosine"].to_list(), dtype=torch.float32)
        matches = 0
        overlap = 0.0
        for a in range(len(respondent_ids)):
            candidates = [b for b in range(len(respondent_ids)) if b != a]
            truth = []
            found = []
            for b in candidates:
                lo, hi = min(a, b), max(a, b)
                idx = lo * (2 * len(respondent_ids) - lo - 1) // 2 + hi - lo - 1
                truth.append(float(ref[idx]))
                found.append(float(vector_cosine[a, b]))
            top_true = sorted(range(len(candidates)), key=lambda i: (-truth[i], candidates[i]))[:3]
            top_found = sorted(range(len(candidates)), key=lambda i: (-found[i], candidates[i]))[:3]
            matches += top_true[0] == top_found[0]
            overlap += len(set(top_true) & set(top_found)) / 3
        rank_rows.append({"dimension": dimension, "seed": seed, "bundle_width": width,
                          "reference": reference, "spearman": spearman(ref, observed),
                          "nearest_neighbor_agreement": matches / len(respondent_ids),
                          "top3_neighbor_overlap": overlap / len(respondent_ids)})
    return pairs, rank_rows


def evaluate(stored_profiles, stored_symbols, dimension: int, seeds: tuple[int, ...],
             widths: tuple[int, ...], scored: torch.Tensor, reversed_items: torch.Tensor,
             respondent_ids: list[str], item_ids: list[str], db_table):
    profile_meta, profile_vectors = read_vectors(stored_profiles, dimension, symbols=False)
    symbol_meta, symbol_vectors = read_vectors(stored_symbols, dimension, symbols=True)
    symbol_lookup = {(m["seed"], m["kind"], m["symbol_id"]): symbol_vectors[i]
                     for i, m in enumerate(symbol_meta)}
    profile_lookup = {(m["seed"], m["bundle_width"], m["respondent_id"]): profile_vectors[i]
                      for i, m in enumerate(profile_meta)}
    results, pair_results, geometry_results = [], [], []
    for seed in seeds:
        keys = torch.stack([symbol_lookup[seed, "question", item_id] for item_id in item_ids])
        levels = torch.stack([symbol_lookup[seed, "answer", str(level)] for level in range(1, 6)])
        level_similarity = torchhd.cosine_similarity(levels, levels)
        for width in widths:
            profiles = torch.stack([profile_lookup[seed, width, p] for p in respondent_ids])
            for p, respondent_id in enumerate(respondent_ids):
                # Score one respondent at a time to bound memory at every dimension.
                unbound = torchhd.bind(profiles[p].unsqueeze(0), keys[:width])
                scores = torchhd.cosine_similarity(unbound, levels)
                order = torch.argsort(scores, dim=1, descending=True, stable=True)
                predictions = order[:, 0] + 1
                best = scores.gather(1, order[:, :1]).squeeze(1)
                runner_up = scores.gather(1, order[:, 1:2]).squeeze(1)
                if p == 0:
                    # Exact flat LanceDB search checks the same five-row filtered
                    # cleanup. The full sweep scores those five stored rows in
                    # TorchHD to avoid 96,100 query-planning round trips.
                    for position in sorted(set((0, width // 2, width - 1))):
                        found = (db_table.search(unbound[position].tolist(), vector_column_name="vector")
                                 .where(f"kind = 'answer' AND namespace = 'ordered_1_to_5' AND seed = {seed}", prefilter=True)
                                 .distance_type("cosine").limit(5).to_list())
                        expected = [int(x) + 1 for x in order[position].tolist()]
                        if len(found) != 5 or [int(row["symbol_id"]) for row in found] != expected:
                            raise AssertionError(f"LanceDB flat cleanup disagrees with TorchHD: "
                                                 f"{[row['symbol_id'] for row in found]} vs {expected}")
                for i in range(width):
                    predicted_scored = int(predictions[i])
                    actual_scored = int(scored[p, i])
                    raw_prediction = 6 - predicted_scored if bool(reversed_items[i]) else predicted_scored
                    actual_raw = 6 - actual_scored if bool(reversed_items[i]) else actual_scored
                    results.append({"dimension": dimension, "seed": seed, "bundle_width": width,
                                    "respondent_id": respondent_id, "item_id": item_ids[i],
                                    "scored_response": actual_scored, "predicted_scored": predicted_scored,
                                    "raw_response": actual_raw, "predicted_raw": raw_prediction,
                                    "correct": predicted_scored == actual_scored,
                                    "absolute_ordinal_error": abs(predicted_scored - actual_scored),
                                    "winner_margin": float(best[i] - runner_up[i])})
            pairs, ranks = geometry(profiles, scored, level_similarity, respondent_ids, dimension, seed, width)
            pair_results.extend(pairs)
            geometry_results.extend(ranks)
    return results, pair_results, geometry_results


def run(data_dir: Path, output_dir: Path, dimensions: tuple[int, ...] = DIMENSIONS,
        seeds: tuple[int, ...] = SEEDS, widths: tuple[int, ...] = WIDTHS):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    items, respondent_ids, raw, scored, reversed_items, responses = source(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(output_dir / "experiment.lancedb"))
    db.create_table("items", items, schema=ITEMS, mode="overwrite")
    db.create_table("responses", responses, schema=RESPONSES, mode="overwrite")
    all_items, all_pairs, all_geometry, diagnostics = [], [], [], []
    for dimension in dimensions:
        symbol_meta, symbol_vectors, profile_meta, profile_vectors = [], [], [], []
        for seed in seeds:
            keys, levels, level_similarity = encoder(dimension, seed)
            key_cosines = torchhd.cosine_similarity(keys, keys)
            off_diagonal = key_cosines[torch.triu_indices(50, 50, offset=1).unbind()].abs()
            diagnostics.append({"dimension": dimension, "seed": seed,
                                "mean_absolute_question_cosine": float(off_diagonal.mean()),
                                "max_absolute_question_cosine": float(off_diagonal.max()),
                                **{f"level_{a+1}_{b+1}_cosine": float(level_similarity[a, b])
                                   for a in range(5) for b in range(a + 1, 5)}})
            for i, item_id in enumerate(items["item_id"].to_pylist()):
                symbol_meta.append({"seed": seed, "kind": "question", "namespace": "ipip_item",
                                    "symbol_id": item_id})
                symbol_vectors.append(keys[i])
            for i in range(5):
                symbol_meta.append({"seed": seed, "kind": "answer", "namespace": "ordered_1_to_5",
                                    "symbol_id": str(i + 1)})
                symbol_vectors.append(levels[i])
            for width, bundles in bundles_at_widths(keys, levels, scored, widths).items():
                for p, respondent_id in enumerate(respondent_ids):
                    profile_meta.append({"seed": seed, "respondent_id": respondent_id,
                                         "bundle_width": width})
                    profile_vectors.append(bundles[p])
        stored_symbols = write_vectors(db, f"symbols_d{dimension}", dimension, symbol_meta,
                                       torch.stack(symbol_vectors), symbols=True)
        stored_profiles = write_vectors(db, f"profiles_d{dimension}", dimension, profile_meta,
                                        torch.stack(profile_vectors), symbols=False)
        items_out, pairs_out, geometry_out = evaluate(
            stored_profiles, stored_symbols, dimension, seeds, widths, scored, reversed_items,
            respondent_ids, items["item_id"].to_pylist(), stored_symbols)
        all_items.extend(items_out)
        all_pairs.extend(pairs_out)
        all_geometry.extend(geometry_out)
        print(f"D={dimension:,}: {len(items_out):,} item readouts, {len(pairs_out):,} profile pairs", flush=True)
    item_frame = pl.DataFrame(all_items)
    item_frame.write_parquet(output_dir / "item_readouts.parquet")
    pl.DataFrame(all_pairs).write_parquet(output_dir / "profile_pairs.parquet")
    pl.DataFrame(all_geometry).write_csv(output_dir / "geometry_by_seed.csv")
    pl.DataFrame(diagnostics).write_csv(output_dir / "encoder_diagnostics.csv")
    (item_frame.group_by("dimension", "seed", "bundle_width")
     .agg(pl.col("correct").mean().alias("exact_accuracy"),
          pl.col("absolute_ordinal_error").mean().alias("mean_absolute_error"),
          (pl.col("absolute_ordinal_error") <= 1).mean().alias("within_one"),
          pl.col("winner_margin").mean().alias("mean_winner_margin"),
          pl.len().alias("queries"))
     .sort("dimension", "seed", "bundle_width").write_csv(output_dir / "recovery_by_seed.csv"))
    (output_dir / "manifest.json").write_text(json.dumps({
        "design": "independent MAP item keys, one shared TorchHD ordered five-level codebook; unthresholded sum",
        "dimensions": dimensions, "widths": widths, "seeds": seeds,
        "source_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                          for name in ("items.csv", "profiles.csv", "responses.csv")},
        "scoring_key": SCORING_KEY, "question_seed_formula": "100 * seed + 5",
        "level_seed_formula": "100 * seed + 4", "torchhd_version": torchhd.__version__,
        "lancedb_version": lancedb.__version__, "polars_version": pl.__version__,
        "pyarrow_version": pa.__version__, "torch_version": torch.__version__,
        "storage_dtype": "float16", "compute_dtype": "float32",
        "cleanup": "Stored answer vectors, exact five-way TorchHD cosine; three filtered flat LanceDB searches per seed/dimension/width agree in ordering",
        "respondents": len(respondent_ids), "synthetic_profiles": True,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "qa-encoding")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "qa-encoding")
    parser.add_argument("--dimensions", nargs="+", type=int, choices=DIMENSIONS, default=DIMENSIONS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--widths", nargs="+", type=int, choices=WIDTHS, default=WIDTHS)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, tuple(args.dimensions), tuple(args.seeds), tuple(args.widths))


if __name__ == "__main__":
    main()
