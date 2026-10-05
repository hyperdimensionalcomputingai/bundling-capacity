import pytest
import torch

import person
from person import MISSING, PRESENT, Codebook, Records


@pytest.fixture(scope="module")
def signatures():
    return person.load_fixture()


def test_fixture_is_the_full_factorial_space(signatures):
    keys = ["age_band", "job_category", "home_region", "interest_1", "interest_2", "interest_3"]
    assert signatures.select(keys).n_unique() == 920_000


def test_mask_rate_and_records(signatures):
    masked = person.masked_records(signatures)
    missing = masked.state == MISSING
    assert all(abs(missing[:, f].float().mean().item() - 0.3) < 0.002 for f in range(4))
    assert torch.equal(masked.value == -1, missing[:, :3])


# ----------------------------------------------------------------- encoder


def test_age_levels_follow_the_linear_ordinal_kernel():
    for dimension in (512, 8192):
        book = person.codebook(dimension, 11)
        i, j = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
        # Nine near-equal partitions of D/2 coordinates: cumulative rounding stays under 2.5 flips.
        assert (book.level_similarity - (1 - (i - j).abs() / 9)).abs().max() < 5 / dimension
        assert abs(book.level_similarity[0, 9]) < 1e-6


def walsh(dimension):
    h = torch.ones(1, 1)
    while h.shape[0] < dimension:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    return h


def orthogonal_book(dimension=1024):
    """Walsh atoms: w_a * w_b = w_(a xor b), so every bound fact below is a distinct Walsh row.

    Age levels are Walsh rows too, so the age kernel here is the identity, not ordinal.
    """
    w = walsh(dimension)
    atoms = {
        "role": w[[0, 256, 512, 768]],
        "age_level": w[1:11],
        "job_category": w[1:9],
        "home_region": w[1:6],
        "interests": w[1:26],
        "null": w[[100, 101, 102]],
    }
    return Codebook.from_atoms(dimension, 0, atoms)


def random_records(n, seed, missing_rate=0.0, spellings=1):
    g = torch.Generator().manual_seed(seed)
    value = torch.stack([torch.randint(0, s, (n,), generator=g) for s in (10, 8, 5)], 1).to(
        torch.int8
    )
    interests = torch.stack(
        [torch.randperm(25, generator=g)[:3].sort().values for _ in range(n)]
    ).to(torch.int8)
    base = Records(
        torch.arange(n),
        torch.zeros(n, 4, dtype=torch.int8),
        value,
        interests,
        torch.zeros(n, 4, dtype=torch.int8),
    )
    return base.without(
        torch.rand(n, 4, generator=g) < missing_rate,
        torch.randint(0, spellings, (n, 4), generator=g),
    )


@pytest.mark.parametrize("strategy", person.STRATEGIES)
def test_orthogonal_atoms_make_map_cosine_equal_the_encoder_similarity(strategy):
    book = orthogonal_book()
    records = random_records(300, 1, 0.3, spellings=3)
    # With identity age levels, S_encoder uses exact age matches; build records with ages 0 or 9 only.
    records.value[:, 0] = torch.where(records.value[:, 0] >= 0, (records.value[:, 0] % 2) * 9, -1)
    sums = person.encode_sum(book, records, strategy)
    base = person.baselines(records, records, strategy)
    assert (person.cosine(sums, sums) - base.encoder)[base.valid].abs().max() < 1e-5


def test_omit_encoder_similarity_is_the_content_similarity():
    records = random_records(400, 2, 0.4)
    base = person.baselines(records, records, "omit")
    assert torch.equal(base.encoder, base.content)


def test_shared_token_adds_one_per_shared_missing_field():
    q = Records(
        torch.tensor([0]),
        torch.tensor([[MISSING, PRESENT, PRESENT, PRESENT]], dtype=torch.int8),
        torch.tensor([[-1, 2, 1]], dtype=torch.int8),
        torch.tensor([[0, 1, 2]], dtype=torch.int8),
        torch.zeros(1, 4, dtype=torch.int8),
    )
    c = Records(
        torch.tensor([1]),
        q.state.clone(),
        torch.tensor([[-1, 2, 3]], dtype=torch.int8),
        torch.tensor([[5, 6, 7]], dtype=torch.int8),
        torch.zeros(1, 4, dtype=torch.int8),
    )
    assert person.baselines(q, c, "omit").content.item() == pytest.approx(
        1 / 5
    )  # shared job over 5 facts each
    assert person.baselines(q, c, "token").encoder.item() == pytest.approx(
        2 / 6
    )  # job + shared token over 6
    c.spelling[0, 0] = 1  # "null" vs "no_value"
    assert person.baselines(q, c, "token").encoder.item() == pytest.approx(1 / 6)


def test_strategies_agree_without_missing_values_and_a_missing_interest_set_is_one_token():
    book = person.codebook(2048, 23)
    complete = random_records(50, 3)
    assert torch.equal(
        person.encode_sum(book, complete, "omit"), person.encode_sum(book, complete, "token")
    )
    no_interests = complete.without(torch.tensor([[False, False, False, True]]).expand(50, 4))
    expected = (
        sum(book.tables[f][complete.value[:, f].long()] for f in range(3)) + book.tables[3][25]
    )
    assert torch.equal(person.encode_sum(book, no_interests, "token"), expected)


def test_bundle_coordinates_are_small_integers():
    sums = person.encode_sum(person.codebook(1024, 53), random_records(200, 8, 0.2), "token")
    assert torch.equal(sums, sums.round()) and sums.abs().max() <= 6


def test_paired_baselines_match_the_diagonal():
    a, b = random_records(40, 9, 0.3, 3), random_records(40, 10, 0.3, 3)
    full = person.baselines(a, b, "token")
    paired = person.baselines(a, b, "token", paired=True)
    assert torch.allclose(paired.encoder, full.encoder.diagonal())
    assert torch.allclose(paired.content, full.content.diagonal())


# ----------------------------------------------------------------- property normalization (Experiment 5)


def test_orthogonal_atoms_make_normalized_cosine_equal_its_baseline():
    book = orthogonal_book()
    a, b = random_records(300, 11, 0.3), random_records(300, 12, 0.3)
    # Some rows lose individual interests, not only the whole set.
    a.interests[::3, 2] = -1
    b.interests[::4, 1:] = -1
    for r in (a, b):  # identity age levels: keep ages at 0 or 9, as above
        r.value[:, 0] = torch.where(r.value[:, 0] >= 0, (r.value[:, 0] % 2) * 9, -1)
    measured = person.cosine(
        person.encode_normalized(book, a), person.encode_normalized(book, b), paired=True
    )
    assert (measured - person.normalized_baseline(a, b)).abs().max() < 1e-5


def test_every_known_property_has_unit_norm():
    book = person.codebook(2048, 23)
    records = random_records(50, 13)
    records.interests[:25, 1:] = -1  # one interest or three: same property norm
    norms = person.encode_normalized(book, records).norm(dim=1)
    assert torch.allclose(norms, torch.full((50,), 2.0), atol=0.15)  # sqrt(4 unit properties)


def test_normalization_experiment_reproduces_the_plan_predictions():
    import normalization

    ideal = normalization.expected(normalization.build_records()[0])
    # A_0..A_3, then B
    plan = {
        "without_normalization": [0.707, 0.816, 0.913, 1.0, 0.833],
        "with_normalization": [0.866, 0.894, 0.954, 1.0, 0.750],
    }
    for encoding, values in plan.items():
        assert torch.allclose(ideal[encoding], torch.tensor(values), atol=6e-4)
