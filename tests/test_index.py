"""Contract tests for the SSIM index.

Every test calls the public stubs in ``ssim.index``.  Numerical comparisons use
an absolute tolerance of 1e-9 (``rel=0`` is intentional).
"""

import math

import pytest

from ssim.index import (
    box_filter,
    downsample,
    local_stats,
    mean_ssim,
    ms_ssim,
    ssim_batch,
    ssim_map,
    validate,
)

C1 = 0.01 ** 2
C2 = 0.03 ** 2

SAMPLE_A = [
    [0.0, 0.1, 0.2, 0.3, 0.4],
    [0.2, 0.5, 0.8, 0.4, 0.1],
    [0.9, 0.7, 0.3, 0.6, 0.2],
    [0.1, 0.4, 0.8, 1.0, 0.5],
    [0.0, 0.5, 0.9, 0.3, 0.7],
]

SAMPLE_B = [
    [0.0, 0.2, 0.1, 0.4, 0.3],
    [0.3, 0.4, 0.9, 0.2, 0.2],
    [0.8, 0.6, 0.4, 0.5, 0.3],
    [0.2, 0.3, 0.7, 0.9, 0.6],
    [0.1, 0.4, 0.8, 0.5, 0.6],
]

SAMPLE_C = [
    [0.05, 0.25, 0.45, 0.65, 0.85],
    [0.15, 0.35, 0.55, 0.75, 0.95],
    [0.25, 0.45, 0.65, 0.85, 0.05],
    [0.35, 0.55, 0.75, 0.95, 0.15],
    [0.45, 0.65, 0.85, 0.05, 0.25],
]

SAMPLE_D = [
    [0.85, 0.65, 0.45, 0.25, 0.05],
    [0.75, 0.55, 0.35, 0.15, 0.95],
    [0.65, 0.45, 0.25, 0.05, 0.85],
    [0.55, 0.35, 0.15, 0.95, 0.75],
    [0.45, 0.25, 0.05, 0.85, 0.65],
]


def _flat(matrix):
    return [value for row in matrix for value in row]


def _assert_matrix_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        assert actual_row == pytest.approx(expected_row, rel=0.0, abs=1e-9)


def _scaled(matrix, factor):
    return [[value * factor for value in row] for row in matrix]


def test_box_filter_edge_normalization():
    """Defect 1: a border window must divide by its valid count, not 9."""
    image = [[1.0, 2.0], [3.0, 4.0]]
    expected = [[2.5, 2.5], [2.5, 2.5]]
    _assert_matrix_close(box_filter(image, 1), expected)


def test_local_stats_uses_population_moments():
    """Defect 2: variance and covariance must use the same population N."""
    a = [[0.0, 1.0], [1.0, 1.0]]
    b = [[0.0, 1.0], [1.0, 0.0]]
    stats = local_stats(a, b, 1)

    assert set(stats) == {"mu_a", "mu_b", "var_a", "var_b", "cov"}
    assert _flat(stats["mu_a"]) == pytest.approx([0.75] * 4, rel=0.0, abs=1e-9)
    assert _flat(stats["mu_b"]) == pytest.approx([0.5] * 4, rel=0.0, abs=1e-9)
    assert _flat(stats["var_a"]) == pytest.approx([0.1875] * 4, rel=0.0, abs=1e-9)
    assert _flat(stats["var_b"]) == pytest.approx([0.25] * 4, rel=0.0, abs=1e-9)
    assert _flat(stats["cov"]) == pytest.approx([0.125] * 4, rel=0.0, abs=1e-9)


def test_ssim_map_covariance_uses_same_normalization():
    """Defect 3: the covariance term must carry the same 1/N as variances."""
    a = [[0.0, 1.0], [1.0, 1.0]]
    b = [[0.0, 1.0], [1.0, 0.0]]

    mu_a, mu_b = 0.75, 0.5
    var_a, var_b, cov = 0.1875, 0.25, 0.125
    expected = (
        (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2)
    ) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    )

    _assert_matrix_close(ssim_map(a, b, 1, C1, C2), [[expected] * 2] * 2)


def test_ssim_map_scales_stabilizers_by_input_range():
    """Defect 4: normalized c1/c2 must be multiplied by L**2."""
    raw_a = [
        [0.0, 64.0, 255.0],
        [128.0, 255.0, 0.0],
        [255.0, 0.0, 128.0],
    ]
    raw_b = [
        [0.0, 32.0, 255.0],
        [255.0, 128.0, 0.0],
        [0.0, 64.0, 255.0],
    ]
    norm_a = _scaled(raw_a, 1.0 / 255.0)
    norm_b = _scaled(raw_b, 1.0 / 255.0)

    raw_map = ssim_map(raw_a, raw_b, 1, C1, C2)
    norm_map = ssim_map(norm_a, norm_b, 1, C1, C2)
    _assert_matrix_close(raw_map, norm_map)


def test_mean_ssim_weights_by_valid_window_area():
    """Defect 5: mean_ssim must weight each map pixel by its valid area."""
    ssim_values = ssim_map(SAMPLE_A, SAMPLE_B, 1, C1, C2)
    counts = [
        [4.0, 6.0, 6.0, 6.0, 4.0],
        [6.0, 9.0, 9.0, 9.0, 6.0],
        [6.0, 9.0, 9.0, 9.0, 6.0],
        [6.0, 9.0, 9.0, 9.0, 6.0],
        [4.0, 6.0, 6.0, 6.0, 4.0],
    ]

    numerator = 0.0
    denominator = 0.0
    for row in range(len(ssim_values)):
        for col in range(len(ssim_values[row])):
            weight = counts[row][col]
            numerator += ssim_values[row][col] * weight
            denominator += weight

    expected = numerator / denominator
    unweighted = sum(_flat(ssim_values)) / len(_flat(ssim_values))
    assert abs(expected - unweighted) > 1e-6
    assert mean_ssim(SAMPLE_A, SAMPLE_B) == pytest.approx(
        expected, rel=0.0, abs=1e-9
    )


def test_ms_ssim_normalizes_level_weights():
    """Defect 6: [1, 1] and [0.5, 0.5] are the same weighting."""
    a = [
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
    b = [
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ]

    unnormalized = ms_ssim(a, b, [1.0, 1.0])
    normalized = ms_ssim(a, b, [0.5, 0.5])
    assert unnormalized == pytest.approx(normalized, rel=0.0, abs=1e-9)


def test_downsample_preserves_odd_tail():
    """Defect 7: ceil-sized output must keep the odd last row and column."""
    image = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    expected = [[3.0, 4.5], [7.5, 9.0]]
    _assert_matrix_close(downsample(image), expected)


def test_downsample_averages_instead_of_subsampling():
    """Defect 8: a 2x2 block mean must not alias to one sample."""
    _assert_matrix_close(downsample([[0.0, 2.0], [0.0, 2.0]]), [[1.0]])
    _assert_matrix_close(downsample([[0.0, 1.0], [1.0, 0.0]]), [[0.5]])


def test_validate_detects_size_mismatch():
    """Defect 9: mismatched shapes, NaN and a bad radius are all invalid."""
    valid = [[0.0] * 3 for _ in range(3)]
    mismatched = [[0.0] * 4 for _ in range(3)]
    with_nan = [[0.0, float("nan")], [0.0, 0.0]]
    finite = [[0.0, 0.0], [0.0, 0.0]]
    assert validate(valid, mismatched, 1)
    assert validate(with_nan, finite, 1)
    assert validate(valid, valid, -1)
    assert validate(valid, valid, 1) == []


def test_ssim_batch_does_not_leak_stats_between_pairs():
    """Defect 10: batch scores must not depend on pair order."""
    config = {"radius": 1}
    pair_one = (SAMPLE_A, SAMPLE_B)
    pair_two = (SAMPLE_C, SAMPLE_D)

    forward = ssim_batch([pair_one, pair_two], config)["scores"]
    reverse = ssim_batch([pair_two, pair_one], config)["scores"]

    assert len(forward) == 2
    assert len(reverse) == 2
    assert forward[0] == pytest.approx(reverse[1], rel=0.0, abs=1e-9)
    assert forward[1] == pytest.approx(reverse[0], rel=0.0, abs=1e-9)


def test_identical_images_score_one():
    """Identical images give exactly one at one and several scales."""
    assert mean_ssim(SAMPLE_A, SAMPLE_A) == pytest.approx(
        1.0, rel=0.0, abs=1e-9
    )
    assert ms_ssim(SAMPLE_A, SAMPLE_A, [0.5, 0.5]) == pytest.approx(
        1.0, rel=0.0, abs=1e-9
    )
    _assert_matrix_close(
        ssim_map(SAMPLE_A, SAMPLE_A, 1, C1, C2),
        [[1.0] * len(SAMPLE_A[0]) for _ in SAMPLE_A],
    )


def test_constant_images_are_finite():
    """Constant images have defined, finite scores and non-negative variance."""
    constant_a = [[0.2] * 4 for _ in range(4)]
    constant_b = [[0.8] * 4 for _ in range(4)]

    score = mean_ssim(constant_a, constant_b)
    assert math.isfinite(score)
    assert -1.0 <= score <= 1.0

    multi = ms_ssim(constant_a, constant_b, [0.5, 0.5])
    assert math.isfinite(multi)
    assert -1.0 <= multi <= 1.0

    stats = local_stats(constant_a, constant_a, 1)
    assert _flat(stats["var_a"]) == pytest.approx([0.0] * 16, rel=0.0, abs=1e-9)
    assert _flat(stats["var_b"]) == pytest.approx([0.0] * 16, rel=0.0, abs=1e-9)
    assert _flat(stats["cov"]) == pytest.approx([0.0] * 16, rel=0.0, abs=1e-9)


def test_8bit_and_normalized_inputs_agree():
    """The 8-bit and [0, 1] encodings of a picture must score the same."""
    raw_a = [
        [0.0, 64.0, 255.0],
        [128.0, 255.0, 0.0],
        [255.0, 0.0, 128.0],
    ]
    raw_b = [
        [0.0, 32.0, 255.0],
        [255.0, 128.0, 0.0],
        [0.0, 64.0, 255.0],
    ]
    norm_a = _scaled(raw_a, 1.0 / 255.0)
    norm_b = _scaled(raw_b, 1.0 / 255.0)

    raw_score = mean_ssim(raw_a, raw_b)
    norm_score = mean_ssim(norm_a, norm_b)
    assert raw_score == pytest.approx(norm_score, rel=0.0, abs=1e-9)
    assert math.isfinite(raw_score)
    assert -1.0 <= raw_score <= 1.0


def test_ms_ssim_scale_count_is_consistent():
    """Normalized weights make zero-weight extra scales harmless."""
    single = ms_ssim(SAMPLE_A, SAMPLE_B, [1.0])
    padded = ms_ssim(SAMPLE_A, SAMPLE_B, [1.0, 0.0, 0.0])
    baseline = mean_ssim(SAMPLE_A, SAMPLE_B)

    assert single == pytest.approx(baseline, rel=0.0, abs=1e-9)
    assert padded == pytest.approx(single, rel=0.0, abs=1e-9)


def test_ssim_batch_matches_single_pair_runs():
    """Each batch entry equals the same pair scored in its own batch."""
    config = {"radius": 1}
    pair_one = (SAMPLE_A, SAMPLE_B)
    pair_two = (SAMPLE_C, SAMPLE_D)

    together = ssim_batch([pair_one, pair_two], config)["scores"]
    alone_one = ssim_batch([pair_one], config)["scores"][0]
    alone_two = ssim_batch([pair_two], config)["scores"][0]

    assert together[0] == pytest.approx(alone_one, rel=0.0, abs=1e-9)
    assert together[1] == pytest.approx(alone_two, rel=0.0, abs=1e-9)
