"""Statistical functions behind the MCP tools.

No MCP imports here, so they can be tested on their own.
"""
from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def _bootstrap_ci(data: np.ndarray, statistic_fn=np.mean, n_bootstrap: int = 1000,
                   confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    boot = np.array([statistic_fn(data[rng.integers(0, n, size=n)]) for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


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
        if mask.sum() > 1:
            ci_lo, ci_hi = _bootstrap_ci(durations_arr[mask], seed=0)
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        result["groups"][str(g)] = {
            "n": int(mask.sum()),
            "n_events": int(events_arr[mask].sum()),
            "median_survival": float(median) if np.isfinite(median) else None,
            "median_bootstrap_ci": [ci_lo, ci_hi],
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
