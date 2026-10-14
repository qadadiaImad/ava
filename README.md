# ava — Totem-implied repo: control and bad-curve detection

Repo parameters for independent price verification are implied from Totem: forwards from consensus call/put
prices, dividends from the separate Totem dividend service. Because the two services have different contributor
panels, the implied repo curve sometimes shows irregular shapes or jumps that the market does not justify.

This repository holds the methodology note and a Python prototype:

- **Report:** [`docs/totem_repo_control.pdf`](docs/totem_repo_control.pdf). It covers the mechanisms, the
  div-agnostic control identity, the structural detector, the benchmark results and the production inputs.
- **Package:** `repo_control`, which needs only numpy and scipy.

## What the prototype does

| Step | Module |
|---|---|
| Forwards from consensus call/put prices (put-call parity zero crossing, implied discount factor as a diagnostic) | `parity` |
| Allocation of Totem dividend buckets to ex-dates; exact implied-repo bootstrap | `buckets`, `curves` |
| Noise model of the Totem − FO carry difference from Totem dispersions and contributor counts (nested panels, panel gap, drop-off jumps) | `noise`, `pipeline` |
| Structural detector: matched-filter templates (level, step, spike, season) after removing a smooth repo difference, calibrated false-alarm rate, attribution in natural units, repaired curve, FO-mismark test | `detect` |
| Control lines: dividend, repo, forward (Totem forwards with FO dividends) and non-forward lines; split materiality from sensitivities | `control` |
| Minimal pricer for the bucketed sensitivities used in the examples | `pricer` |
| Synthetic contributor-level Totem market, scenarios, matched-FPR benchmark against naive and B-spline + GMM baselines | `synthetic`, `benchmark`, `baselines` |

## Running it

```bash
pip install -e .[benchmark,report,dev]
pytest                                                   # 22 tests
python scripts/run_benchmark.py                          # ~5 min, writes results/benchmark.json.gz
PYTHONPATH=. python scripts/build_report.py              # writes docs/totem_repo_control.pdf
```

On real data, build a `TotemSnapshot` and an `FOSnapshot` (see `repo_control/pipeline.py`). Then call
`imply_totem_repo` and run `StructuralDetector.detect` on `implied.delta_carry` and `implied.cov`. Calibrate the
detector's variance scale on clean history with `null_statistics` and `calibrate_to_fpr`.
