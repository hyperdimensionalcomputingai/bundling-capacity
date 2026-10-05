import pytest
import torch

import person
from person import Codebook, Records


@pytest.fixture(scope="module")
def signatures():
    return person.load_fixture()


def test_fixture_is_the_full_factorial_space(signatures):
    keys = ["age_band", "job_category", "home_region", "interest_1", "interest_2", "interest_3"]
    assert signatures.select(keys).n_unique() == 920_000


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
    }
    return Codebook.from_atoms(dimension, 0, atoms)


def random_records(n, seed):
    g = torch.Generator().manual_seed(seed)
    value = torch.stack([torch.randint(0, s, (n,), generator=g) for s in (10, 8, 5)], 1).to(
        torch.int8
    )
    interests = torch.stack(
        [torch.randperm(25, generator=g)[:3].sort().values for _ in range(n)]
    ).to(torch.int8)
    return Records(torch.arange(n), value, interests)


def test_orthogonal_atoms_make_map_cosine_equal_the_exact_similarity():
    book = orthogonal_book()
    records = random_records(300, 1)
    # With identity age levels, S_exact needs exact age matches; use ages 0 or 9 only.
    records.value[:, 0] = (records.value[:, 0] % 2) * 9
    sums = person.encode_sum(book, records)
    base = person.baselines(records, records)
    assert (person.cosine(sums, sums) - base.exact)[base.valid].abs().max() < 1e-5


def test_bundle_coordinates_are_small_integers():
    sums = person.encode_sum(person.codebook(1024, 53), random_records(200, 8))
    assert torch.equal(sums, sums.round()) and sums.abs().max() <= 6


def test_paired_baselines_match_the_diagonal():
    a, b = random_records(40, 9), random_records(40, 10)
    full = person.baselines(a, b)
    paired = person.baselines(a, b, paired=True)
    assert torch.allclose(paired.exact, full.exact.diagonal())
