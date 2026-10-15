# ava — Totem-implied repo: control and bad-curve detection

**English** | [Français](README.fr.md)

Repo parameters for independent price verification are implied from Totem: forwards from consensus call/put prices,
dividends from the separate Totem dividend service. Because the two services have different contributor panels, the
implied repo curve sometimes shows irregular shapes or jumps that the market does not justify.

This repository holds the methodology note and a Python prototype:

- **Methodology note:** [`docs/totem_repo_note_fr.pdf`](docs/totem_repo_note_fr.pdf) (French) and
  [`docs/totem_repo_note_en.pdf`](docs/totem_repo_note_en.pdf) (English). The note states its three goals on the first
  page (understand the bad curves, control the repo position independently of the dividend contributors, detect bad
  curves), follows one running example throughout, and gives the proofs. Sources are in `docs/latex/`.
- **Package:** `repo_control`, which needs only numpy and scipy.

## What the prototype does

| Step | Module |
|---|---|
| Forwards from consensus call/put prices (put-call parity zero crossing, implied discount factor as a diagnostic) | `parity` |
| Allocation of Totem dividend periods to ex-dates; exact implied-repo bootstrap | `buckets`, `curves` |
| Noise model of the Totem − FO carry difference from Totem dispersions and contributor counts (nested panels, panel gap, drop-off jumps) | `noise`, `pipeline` |
| Structural detector: matched-filter tests of typical shapes (level, step, spike, season) after removing a smooth repo difference, calibrated false-alarm rate, explanation in natural units, repaired curve, FO-mismark test | `detect` |
| Control lines: dividend, repo, forward (Totem forwards with FO dividends) and non-forward lines; materiality of the split from sensitivities | `control` |
| Minimal pricer for the sensitivities used in the examples | `pricer` |
| Synthetic contributor-level Totem market, scenarios, matched-FPR benchmark against naive and B-spline + Gaussian mixture baselines | `synthetic`, `benchmark`, `baselines` |

## Running it

```bash
pip install -e .[benchmark,report,dev]
pytest                                                   # 22 tests
python scripts/run_benchmark.py                          # ~5 min, writes results/benchmark.json.gz
PYTHONPATH=. python scripts/build_latex.py               # figures, tables and both notes (needs TeX Live)
```

On real data, build a `TotemSnapshot` and an `FOSnapshot` (see `repo_control/pipeline.py`). Then call
`imply_totem_repo` and run `StructuralDetector.detect` on `implied.delta_carry` and `implied.cov`. Calibrate the
detector's variance scale on clean history with `null_statistics` and `calibrate_to_fpr`.
