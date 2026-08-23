"""
Evaluation for the MCP tools: quality, latency and cost.

Quality uses inputs with a known answer. Latency is timed through
server.call_tool, so protocol overhead is included. Cost is an estimate from
latency and an assumed compute rate, since the tools run locally.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field

from app.server import server

# Assumed rate for the cost estimate, not a real price.
ILLUSTRATIVE_COMPUTE_RATE_USD_PER_VCPU_HOUR = 0.05


@dataclass
class QualityCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class LatencyStats:
    tool_name: str
    n_calls: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass
class EvalReport:
    quality_checks: list[QualityCheck] = field(default_factory=list)
    latency: list[LatencyStats] = field(default_factory=list)

    @property
    def quality_pass_rate(self) -> float:
        if not self.quality_checks:
            return 0.0
        return sum(c.passed for c in self.quality_checks) / len(self.quality_checks)

    def estimated_cost_usd(self, calls_per_day: int) -> dict[str, float]:
        """Monthly compute cost per tool from mean latency and a daily call volume."""
        out = {}
        for lat in self.latency:
            vcpu_hours_per_month = (lat.mean_ms / 1000 / 3600) * calls_per_day * 30
            out[lat.tool_name] = round(
                vcpu_hours_per_month * ILLUSTRATIVE_COMPUTE_RATE_USD_PER_VCPU_HOUR, 4
            )
        return out


# ── quality: inputs with a known answer ──

async def run_quality_checks() -> list[QualityCheck]:
    checks: list[QualityCheck] = []

    # survival_analysis: different hazards should be significant, identical ones not
    import numpy as np
    rng = np.random.default_rng(1)

    fast = rng.exponential(10, 80).tolist()
    slow = rng.exponential(30, 80).tolist()
    r = await server.call_tool("survival_analysis", {
        "durations": fast + slow,
        "events": [1] * 160,
        "group_labels": ["fast"] * 80 + ["slow"] * 80,
    })
    sig = r.structured_content["logrank_test"]["significant_at_05"]
    checks.append(QualityCheck(
        "survival_analysis detects a real hazard difference", sig,
        f"p={r.structured_content['logrank_test']['p_value']:.2e}",
    ))

    same_a = rng.exponential(15, 80).tolist()
    same_b = rng.exponential(15, 80).tolist()
    r = await server.call_tool("survival_analysis", {
        "durations": same_a + same_b,
        "events": [1] * 160,
        "group_labels": ["a"] * 80 + ["b"] * 80,
    })
    not_sig = not r.structured_content["logrank_test"]["significant_at_05"]
    checks.append(QualityCheck(
        "survival_analysis stays quiet on identical-hazard groups", not_sig,
        f"p={r.structured_content['logrank_test']['p_value']:.2f}",
    ))

    # compare_two_samples: a known injected 10-unit difference must be recovered
    a = rng.normal(50, 5, 100).tolist()
    b = (np.array(a) - 10 + rng.normal(0, 1, 100)).tolist()
    r = await server.call_tool("compare_two_samples", {"sample_a": a, "sample_b": b})
    recovered = abs(r.structured_content["mean_difference"] - 10) < 1.5
    checks.append(QualityCheck(
        "compare_two_samples recovers a known 10-unit difference", recovered,
        f"measured={r.structured_content['mean_difference']:.2f}",
    ))

    # check_distribution_drift: identical samples should show no drift;
    # a genuine shift should be flagged
    stable = rng.normal(0, 1, 200).tolist()
    r = await server.call_tool("check_distribution_drift", {
        "reference": stable, "current": rng.normal(0, 1, 200).tolist(),
    })
    checks.append(QualityCheck(
        "check_distribution_drift stays quiet on stable data",
        not r.structured_content["drift_detected"],
        f"psi={r.structured_content['psi']:.3f}",
    ))

    r = await server.call_tool("check_distribution_drift", {
        "reference": stable, "current": rng.normal(3, 1, 200).tolist(),
    })
    checks.append(QualityCheck(
        "check_distribution_drift catches a genuine shift",
        r.structured_content["drift_detected"],
        f"psi={r.structured_content['psi']:.3f}",
    ))

    return checks


# ── latency: timed through server.call_tool ──

async def measure_latency(tool_name: str, arguments: dict, n_calls: int = 50) -> LatencyStats:
    timings_ms = []
    for _ in range(n_calls):
        start = time.perf_counter()
        await server.call_tool(tool_name, arguments)
        timings_ms.append((time.perf_counter() - start) * 1000)

    timings_sorted = sorted(timings_ms)
    return LatencyStats(
        tool_name=tool_name,
        n_calls=n_calls,
        mean_ms=statistics.mean(timings_ms),
        p50_ms=timings_sorted[int(0.50 * n_calls)],
        p95_ms=timings_sorted[min(int(0.95 * n_calls), n_calls - 1)],
        p99_ms=timings_sorted[min(int(0.99 * n_calls), n_calls - 1)],
        max_ms=max(timings_ms),
    )


async def run_full_evaluation(calls_per_day: int = 1000) -> EvalReport:
    report = EvalReport()
    report.quality_checks = await run_quality_checks()

    import numpy as np
    rng = np.random.default_rng(2)

    report.latency.append(await measure_latency(
        "survival_analysis",
        {"durations": rng.exponential(20, 100).tolist(), "events": [1] * 100,
         "group_labels": ["a"] * 50 + ["b"] * 50},
    ))
    report.latency.append(await measure_latency(
        "compare_two_samples",
        {"sample_a": rng.normal(0, 1, 100).tolist(), "sample_b": rng.normal(1, 1, 100).tolist()},
    ))
    report.latency.append(await measure_latency(
        "check_distribution_drift",
        {"reference": rng.normal(0, 1, 200).tolist(), "current": rng.normal(0, 1, 200).tolist()},
    ))

    return report


def print_report(report: EvalReport, calls_per_day: int = 1000) -> None:
    print("=" * 62)
    print(f"QUALITY  ({report.quality_pass_rate:.0%} passed)")
    print("=" * 62)
    for c in report.quality_checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] {c.name}  ({c.detail})")

    print()
    print("=" * 62)
    print("LATENCY  (timed through server.call_tool)")
    print("=" * 62)
    for lat in report.latency:
        print(f"  {lat.tool_name:26s} mean={lat.mean_ms:6.2f}ms  "
              f"p50={lat.p50_ms:6.2f}ms  p95={lat.p95_ms:6.2f}ms  p99={lat.p99_ms:6.2f}ms")

    print()
    print("=" * 62)
    print(f"COST  (estimate at {calls_per_day}/day, "
          f"${ILLUSTRATIVE_COMPUTE_RATE_USD_PER_VCPU_HOUR}/vCPU-hour assumed)")
    print("=" * 62)
    for tool_name, cost in report.estimated_cost_usd(calls_per_day).items():
        print(f"  {tool_name:26s} ~${cost:.4f}/month in compute time")


if __name__ == "__main__":
    result = asyncio.run(run_full_evaluation())
    print_report(result)
