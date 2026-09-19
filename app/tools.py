"""Statistical functions behind the MCP tools.

No MCP imports here, so they can be tested on their own.
"""
from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats


def _bootstrap_ci(data: np.ndarray, statistic_fn=np.mean, n_bootstrap: int = 1000,
                   confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    boot = np.array([statistic_fn(data[rng.integers(0, n, size=n)]) for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _km_median(durations: np.ndarray, events: np.ndarray) -> float:
    """Kaplan-Meier median: the first time the survival estimate reaches 0.5.

    Returns inf when the curve never gets that low (median not reached),
    matching lifelines' median_survival_time_.
    """
    times = np.unique(durations[events == 1])
    if times.size == 0:
        return float("inf")
    at_risk = (durations[None, :] >= times[:, None]).sum(axis=1)
    deaths = ((durations[None, :] == times[:, None]) & (events[None, :] == 1)).sum(axis=1)
    survival = np.cumprod(1.0 - deaths / at_risk)
    reached = np.nonzero(survival <= 0.5)[0]
    return float(times[reached[0]]) if reached.size else float("inf")


def _bootstrap_median_survival_ci(
    durations: np.ndarray, events: np.ndarray, n_bootstrap: int = 1000,
    confidence: float = 0.95, seed: int = 0,
) -> list[float] | None:
    """Percentile CI on the Kaplan-Meier median, refitting the curve on each
    resample so censored observations are handled the same way as in the
    point estimate.

    Returns None when the median is not reached in too many resamples for an
    interval to mean anything.
    """
    rng = np.random.default_rng(seed)
    n = len(durations)
    medians = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        medians[i] = _km_median(durations[idx], events[idx])
    finite = medians[np.isfinite(medians)]
    if finite.size < 0.9 * n_bootstrap:
        return None
    alpha = 1 - confidence
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def analyze_survival(
    durations: list[float],
    events: list[int],
    group_labels: list[str] | None = None,
) -> dict:
    """
    Fits a Kaplan-Meier curve (per group if labels are given) and, when
    exactly two groups are present, runs a log-rank test between them.
    """
    if len(durations) != len(events):
        raise ValueError(f"durations ({len(durations)}) and events ({len(events)}) must match")
    if group_labels is not None and len(group_labels) != len(durations):
        raise ValueError("group_labels must match durations in length")

    durations_arr = np.asarray(durations, dtype=float)
    events_arr = np.asarray(events, dtype=int)
    if group_labels is not None:
        groups = np.asarray(group_labels)
    else:
        groups = np.full(len(durations), "all")

    result: dict = {"groups": {}}
    for g in np.unique(groups):
        mask = groups == g
        kmf = KaplanMeierFitter()
        kmf.fit(durations_arr[mask], event_observed=events_arr[mask])
        median = kmf.median_survival_time_
        if mask.sum() > 1 and np.isfinite(median):
            median_ci = _bootstrap_median_survival_ci(
                durations_arr[mask], events_arr[mask], seed=0
            )
        else:
            median_ci = None
        result["groups"][str(g)] = {
            "n": int(mask.sum()),
            "n_events": int(events_arr[mask].sum()),
            "median_survival": float(median) if np.isfinite(median) else None,
            # None when the median is not reached or cannot be bootstrapped.
            "median_bootstrap_ci": median_ci,
        }

    unique_groups = list(np.unique(groups))
    if len(unique_groups) == 2:
        m1, m2 = groups == unique_groups[0], groups == unique_groups[1]
        lr = logrank_test(
            durations_arr[m1], durations_arr[m2],
            event_observed_A=events_arr[m1], event_observed_B=events_arr[m2],
        )
        result["logrank_test"] = {
            "groups_compared": [str(unique_groups[0]), str(unique_groups[1])],
            "p_value": float(lr.p_value),
            "significant_at_05": bool(lr.p_value < 0.05),
        }

    return result


def compare_samples(sample_a: list[float], sample_b: list[float]) -> dict:
    """
    Paired t-test and Wilcoxon signed-rank test, plus a bootstrap CI on the mean
    difference. Both tests are returned because they can disagree when the data
    isn't normal.
    """
    a, b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"paired comparison requires equal-length samples, got {len(a)} and {len(b)}"
        )
    if len(a) < 2:
        raise ValueError("need at least 2 paired observations")

    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # wilcoxon raises when every difference is zero
        w_stat, w_p = 0.0, 1.0

    ci_lo, ci_hi = _bootstrap_ci(diff, seed=0)

    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference": float(np.mean(diff)),
        "difference_bootstrap_ci": [ci_lo, ci_hi],
        "paired_t_test": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon_test": {"statistic": float(w_stat), "p_value": float(w_p)},
        "significant_at_05": bool(t_p < 0.05),
    }


def detect_drift(reference: list[float], current: list[float]) -> dict:
    """
    KS test plus Population Stability Index. KS is weak in the tails and PSI
    depends on binning, so both are reported.
    """
    ref, cur = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if len(ref) < 2 or len(cur) < 2:
        raise ValueError("need at least 2 observations in each sample")

    ks_stat, ks_p = stats.ks_2samp(ref, cur)

    n_bins = 10
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_prop = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_prop = np.clip(cur_counts / len(cur), 1e-6, None)
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))

    return {
        "ks_test": {"statistic": float(ks_stat), "p_value": float(ks_p)},
        "psi": psi,
        "psi_interpretation": (
            "no significant shift" if psi < 0.1 else
            "moderate shift, monitor" if psi < 0.25 else
            "significant shift, investigate"
        ),
        "drift_detected": bool(ks_p < 0.05 or psi >= 0.25),
    }
