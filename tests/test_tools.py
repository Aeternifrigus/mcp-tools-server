import numpy as np
import pytest

from app.tools import analyze_survival, compare_samples

# ── tools.py ──

def test_survival_requires_matching_lengths():
    with pytest.raises(ValueError, match="must match"):
        analyze_survival([1, 2, 3], [1, 0])


def test_survival_single_group_no_logrank():
    result = analyze_survival([10, 20, 30], [1, 1, 0])
    assert "logrank_test" not in result
    assert "all" in result["groups"]


def test_survival_detects_a_real_hazard_difference():
    rng = np.random.default_rng(1)
    fast = rng.exponential(10, 100).tolist()
    slow = rng.exponential(40, 100).tolist()
    result = analyze_survival(
        fast + slow, [1] * 200, ["fast"] * 100 + ["slow"] * 100
    )
    assert result["logrank_test"]["significant_at_05"] is True
    assert result["logrank_test"]["p_value"] < 0.01


def test_survival_stays_quiet_on_identical_groups():
    rng = np.random.default_rng(1)
    a = rng.exponential(15, 100).tolist()
    b = rng.exponential(15, 100).tolist()
    result = analyze_survival(a + b, [1] * 200, ["a"] * 100 + ["b"] * 100)
    assert result["logrank_test"]["significant_at_05"] is False


def test_compare_samples_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        compare_samples([1, 2, 3], [1, 2])


def test_compare_samples_requires_minimum_observations():
    with pytest.raises(ValueError, match="at least 2"):
        compare_samples([1], [2])


def test_compare_samples_recovers_a_known_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(100, 5, 150).tolist()
    b = (np.array(a) - 20 + rng.normal(0, 1, 150)).tolist()
    result = compare_samples(a, b)
    assert result["mean_difference"] == pytest.approx(20, abs=1.5)
    assert result["significant_at_05"] is True
    assert result["difference_bootstrap_ci"][0] > 0


def test_compare_samples_finds_no_difference_for_identical_input():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 100).tolist()
    result = compare_samples(a, list(a))
    assert result["mean_difference"] == pytest.approx(0.0, abs=1e-9)
    assert result["significant_at_05"] is False
