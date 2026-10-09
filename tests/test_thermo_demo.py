import numpy as np
import thermo_demo as td


def test_frames_have_matched_mean_within_animal():
    v = td.make_animal(np.random.default_rng(0), "B")
    means = v.mean(axis=(1, 2))
    assert np.allclose(means, means[0], atol=1e-9)   # every frame shifted to the animal baseline


def test_group_b_is_more_heterogeneous_than_group_a():
    rng = np.random.default_rng(1)
    sd_a = np.mean([td.make_animal(rng, "A")[0].std() for _ in range(5)])
    sd_b = np.mean([td.make_animal(rng, "B")[0].std() for _ in range(5)])
    assert sd_b > sd_a


def test_permutation_p_floor_and_symmetry():
    a, b = np.arange(6.0), np.arange(6.0) + 100
    assert abs(td.exact_permutation_p(a, b) - 2 / 924) < 1e-12   # smallest attainable p for 6 vs 6
    assert td.exact_permutation_p(a, b) == td.exact_permutation_p(b, a)


def test_identical_groups_give_p_one():
    a = np.array([1.0, 2, 3, 4, 5, 6])
    assert td.exact_permutation_p(a, a.copy()) == 1.0


def test_animal_level_controls_false_positives_frame_level_does_not():
    sim = td.simulate_false_positives(nsim=300, seed=3)
    assert sim["fp_rate_animal_level"] < 0.10
    assert sim["fp_rate_frame_level"] > 0.30
