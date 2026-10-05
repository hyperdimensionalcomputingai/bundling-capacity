"""End-to-end checks on a small store: write float16, read, cast to float32, compute."""

import pytest
import torch

import experiment as ex
import person
import store

SMALL = 3_000


@pytest.fixture(scope="module")
def small_store(tmp_path_factory):
    complete = person.complete_records(person.load_fixture()).take(slice(0, SMALL))
    db = store.connect(tmp_path_factory.mktemp("store") / "test.lancedb")
    store.write_symbols(db, 512, (11,))
    store.write_records(
        db, store.records_table(512, 11), store.read_codebook(db, 512, 11), complete, chunk=1_000
    )
    return db, complete


def test_float16_round_trip_is_exact(small_store):
    db, complete = small_store
    stored, fresh = store.read_codebook(db, 512, 11), person.codebook(512, 11)
    for family in person.ATOM_FAMILIES:
        assert torch.equal(stored.atoms[family], fresh.atoms[family])
    index, vectors = store.read_rows(
        db.open_table(store.records_table(512, 11)), 512, [2_999, 5, 1_000]
    )
    assert index.tolist() == [2_999, 5, 1_000] and vectors.dtype == torch.float32
    assert torch.equal(vectors, person.encode_sum(fresh, complete.take([2_999, 5, 1_000])))


def test_write_boundary_rejects_values_float16_cannot_hold():
    with pytest.raises(ValueError):
        store.to_f16_column(torch.full((1, 4), 0.1), 4)


def test_chunks_respect_prefix_boundaries(monkeypatch):
    monkeypatch.setattr(ex, "CHUNK", 300)
    monkeypatch.setattr(ex, "PREFIXES", (100, 1_000, 3_000))
    spans = list(ex.chunks(3_000))
    assert spans[0] == (0, 100, 0)
    assert all(stop - start <= 300 for start, stop, _ in spans)
    assert {stop for _, stop, _ in spans} >= {100, 1_000, 3_000}


def test_streamed_pass_matches_brute_force(small_store, monkeypatch):
    monkeypatch.setattr(ex, "CHUNK", 256)
    monkeypatch.setattr(ex, "PREFIXES", (100, 1_000, SMALL))
    db, cand = small_store
    offsets, ranks, unit = list(range(0, SMALL, 150)), torch.arange(20), (512, 11)
    thresholds = {unit: {"calibrated": 0.3, "reference": 0.2157}}
    state, _, rows = ex.run_pass(
        db,
        cand,
        offsets,
        ranks,
        [unit],
        candidates=SMALL,
        thresholds=thresholds,
        log=lambda m: None,
    )
    book = person.codebook(512, 11)
    final = next(r for r in rows if r["candidates"][0] == SMALL)
    q = cand.take(offsets)
    score = person.cosine(person.encode_sum(book, q), person.encode_sum(book, cand))
    base = person.baselines(q, cand)
    tier = ex.tiers(base.exact)
    unrelated, related = base.valid & (tier == 0), base.valid & (tier == 3)
    assert torch.allclose(
        torch.tensor(final["max_unrelated"]), torch.where(unrelated, score, -torch.inf).amax(1)
    )
    assert final["false_pos_calibrated"] == (unrelated & (score >= 0.3)).sum(1).tolist()
    assert final["false_neg_calibrated"] == (related & (score < 0.3)).sum(1).tolist()
    assert final["valid_candidates"] == base.valid.sum(1).tolist()
    masked = torch.where(base.valid, score, -torch.inf)
    for i in range(len(offsets)):
        order = sorted(range(SMALL), key=lambda j: (-float(masked[i, j]), j))[: ex.HEAD]
        assert state[unit].head["index"][i].tolist() == order
    counted = sum(int(state[unit].hist.segment("abs_numerical_error", s).sum()) for s in range(3))
    assert counted == int(base.valid.sum())


def test_threshold_keeps_at_most_beta_of_related_pairs_below():
    scores = 0.6 + 0.05 * torch.randn(100_000, generator=torch.Generator().manual_seed(0))
    h = ex.Histograms(["related_score"], 1)
    h.add([h.index("related_score", scores, torch.ones_like(scores, dtype=torch.bool), 0)])
    threshold, total, achieved = ex.calibrated_threshold(h.segment("related_score", 0))
    assert total == 100_000
    assert (
        (scores < threshold).float().mean()
        <= ex.BETA
        < (scores < threshold + ex.BIN).float().mean()
    )
    assert achieved >= 1 - ex.BETA
