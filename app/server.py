"""
MCP server exposing the functions in tools.py.

Run:      python -m app.server
Inspect:  npx @modelcontextprotocol/inspector python -m app.server
"""
from mcp.server.mcpserver import MCPServer

from app.tools import analyze_survival, compare_samples, detect_drift

server = MCPServer(
    name="stats-tools",
    description="Statistical analysis tools: survival analysis, paired "
                "significance testing, and distribution drift detection.",
)


@server.tool()
def survival_analysis(
    durations: list[float],
    events: list[int],
    group_labels: list[str] | None = None,
) -> dict:
    """
    Fit a Kaplan-Meier survival curve and, if exactly two groups are given
    (e.g. two carriers, two treatment arms), run a log-rank significance
    test between them.

    Args:
        durations: observed time for each record (e.g. days until delivery).
        events: 1 if the event was observed, 0 if the record is censored
            (e.g. still in transit / still ongoing as of the data cutoff).
        group_labels: optional group name per record, for comparing exactly
            two groups via log-rank test.
    """
    return analyze_survival(durations, events, group_labels)


@server.tool()
def compare_two_samples(sample_a: list[float], sample_b: list[float]) -> dict:
    """
    Compare two paired samples (e.g. a new model's per-case scores vs a
    baseline's, on the same cases) with a paired t-test and a Wilcoxon
    signed-rank test, plus a bootstrap confidence interval on the mean
    difference.

    Args:
        sample_a: first sample's values, one per paired observation.
        sample_b: second sample's values, same length and same pairing order.
    """
    return compare_samples(sample_a, sample_b)


@server.tool()
def check_distribution_drift(reference: list[float], current: list[float]) -> dict:
    """
    Check whether a current sample has drifted from a reference sample,
    using both a Kolmogorov-Smirnov test and the Population Stability Index.
    Useful for monitoring whether a model's input feature distribution has
    shifted since training.

    Args:
        reference: baseline values (e.g. a feature's values at training time).
        current: recent values to check for drift against the reference.
    """
    return detect_drift(reference, current)


if __name__ == "__main__":
    server.run()
