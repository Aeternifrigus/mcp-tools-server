# mcp-stats-tools

An MCP server with three statistical tools that an agent can call:

- `survival_analysis`: Kaplan-Meier curves, plus a log-rank test when exactly two
  groups are given
- `compare_two_samples`: paired t-test, Wilcoxon signed-rank test and a bootstrap
  CI on the mean difference
- `check_distribution_drift`: Kolmogorov-Smirnov test and Population Stability Index
  between a reference and a current sample

No API keys or paid services. The tools are plain statistics, not model calls.

## Running

```bash
pip install -r requirements.txt
make test        # 18 tests
make evaluate    # quality, latency and cost report
make run         # server over stdio
```

To poke at it with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python -m app.server
```

## Layout

```
app/tools.py      the statistics, no MCP imports
app/server.py     MCP tool definitions
app/evaluate.py   evaluation report
tests/            unit tests for tools.py and in-process tests through the server
```

The logic in `tools.py` raises `ValueError` on bad input. `server.py` converts that
to `ToolError`, because the SDK treats any other exception as an unexpected crash
and the client never sees the message.

Tool functions return `dict[str, Any]` rather than a bare `dict`. With a bare
`dict` the SDK can't build an output schema and results only come back as text,
not `structured_content`.

## Evaluation

`python -m app.evaluate` prints three sections.

**Quality.** Inputs with a known answer: two groups with different hazards should
give a significant log-rank test and two identical ones should not, a 10-unit
shift should be recovered by `compare_two_samples`, and drift should be flagged for
a shifted sample but not a stable one.

**Latency.** 50 calls per tool through `server.call_tool`, so argument validation
and serialisation are included. On my machine drift takes about 1 ms, the paired
comparison about 9 ms and survival analysis about 40 ms (it bootstraps).

**Cost.** The tools run locally, so there's no bill to read from. The report
estimates monthly compute cost from mean latency, a daily call volume and an
assumed per-vCPU-hour rate set at the top of `evaluate.py`.

## Limitations

- Only stdio transport is used. HTTP isn't exercised.
- The tests call the server in-process. Nothing here connects a real agent to it.
- The cost figure is an estimate, not a measured bill.
