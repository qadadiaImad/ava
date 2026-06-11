# 🧮 Vega Netting Studio — EBA Prudent Valuation (AVA MPU)

**Volatility-sensitivity netting under the EBA variance test**, as formalised in the
technical note *"Netting des sensibilités de volatilité sous contrainte de test de
variance — formalisation dans le cadre EBA Prudent Valuation (AVA / MPU)"*
(Delegated Regulation (EU) 2016/101, art. 9(5) & art. 89).

A production-grade IPV framework + a colorful Streamlit studio: optimal netting of the
vega surface under the tracking-error variance test, scenario scoring of any custom
netting scheme, spectral diagnostics, adverse correlation stress, and the full
art. 9(5) audit trail.

---

## What it does

| Theory (note section) | Implementation |
|---|---|
| Uncertainty model ΔΠ = ν′Δσ, Σ = DρD (§2) | `ebanetting/model.py` — `UncertaintyModel` |
| AVA extremes: add-up (2) / full diversification (3) | `ava_brut`, `ava_full` |
| Kronecker separability ρ = ρ_mat ⊗ ρ_strike (§3.1, eq. 4) | `MarketDataBundle.covariance()` |
| Netting as aggregation operator P, W, Σ̃ = WΣW′ (§4, def. 1) | `ebanetting/netting.py` — `NettingScheme` |
| Variance test TE² ≤ (1−α)·Var(ΔΠ), R² ≥ α (§5, eq. 7) | `NettingScheme.evaluate()` |
| Conservatism floor AVA(P) ≥ κ√(ν′Σν) (eq. 8) | `SchemeEvaluation.passes_floor` |
| Two-bucket closed form (§6.2, eqs. 10–11) | `two_bucket()` — unit-tested against the engine |
| Spectral (PCA) lower bound, K*(α) (§6.3, Lemma 1) | `ebanetting/spectral.py` |
| Greedy agglomerative algorithm under budget (§6.4) | `ebanetting/optimizer.py` — `greedy_netting()` |
| Rectangular pavings of the tenor × strike grid (§3.2) | rectangle-union merge candidates |
| Lagrangian efficient frontier (Remark 2) | `lagrangian_frontier()` |
| Adverse correlation stress ρ → max(ρ−δ,−1) (step 4, §7) | `stress_bundle()`, `robust_netting()` |
| Hierarchical netting / smile decomposition level–RR–fly (§3.2) | `scenario.smile_decomposition()` |
| Art. 9(5) evidence pack (§7) | `ebanetting/reporting.py` |

## Quick start

```bash
cd eba-vega-netting
pip install -r requirements.txt
streamlit run app.py
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

## The app

- **🏛️ Theory** — the complete framework with every equation of the note, from the
  regulatory frame to the algorithm and the IPV implementation pitfalls.
- **📊 Market Data** — vega surface, consensus-dispersion surface, tenor/strike
  correlation heatmaps, AVA envelope KPIs, and the data contract.
- **🧠 Optimal Netting** — the greedy optimiser: retained rectangular partition on the
  grid, AVA waterfall (add-up → netted vs the diversification floor), budget-consumption
  path, Lagrangian efficient frontier, robust mode with correlation-stress regimes.
- **🎯 Scenario Lab** — score **any** netting scheme: edit the (M × K) label matrix in
  place, upload one, or start from presets / the optimal solution. Returns the
  **variance score R²** (gauge vs α), the EBA verdict (test 7 + floor 8), the AVA
  impact and per-set statistics. Includes the two-bucket closed-form sandbox.
- **🔬 Spectral & Smile** — scree/loadings of Σ weighted by the vega profile, K*(α),
  leading eigenmodes on the grid, and the per-tenor level / risk-reversal / butterfly
  decomposition with per-tranche tests.
- **📋 Audit & Export** — the art. 9(5) evidence pack: retained partition, realised R²,
  frontier, stress table, greedy trace, SHA-256 input fingerprint — one JSON download.

## Plugging your data (the open pipe)

The app is **bank-agnostic** by design. The data pipe is deliberately left open behind
two abstract interfaces in [`ebanetting/datasource.py`](ebanetting/datasource.py):

```python
class DataSource(ABC):
    def load(self) -> MarketDataBundle: ...

class ScenarioSource(ABC):
    def load(self) -> np.ndarray:  # (M, K) int label matrix
        ...
```

Implement `DataSource.load()` against your golden sources (risk-system vegas,
Totem/Markit consensus dispersions, broker ranges) and everything — optimiser, scenario
scoring, audit pack — works unchanged. The file-based reference implementation is the
**JSON contract** (`JSONBundleSource`), one document per as-of / underlying / valuation
exposure:

```jsonc
{
  "meta":        { "asof": "2026-06-11", "underlying": "...", "desk": "...",
                   "vega_unit": "EUR per vol point", "uncertainty_unit": "vol points" },
  "tenors":      ["1M", "3M", ...],          // M tenor labels
  "tenor_years": [0.0833, 0.25, ...],        // M year fractions
  "strikes":     [0.80, 0.90, ...],          // K moneyness levels K/F (recommended, §3.3)
  "vega":        [[...], ...],               // M x K signed vegas  ∂V/∂σ(T, K)
  "s":           [[...], ...],               // M x K uncertainty std-devs, > 0
  "corr_mat":    [[...], ...],               // M x M  (Kronecker factor, eq. 4)
  "corr_strike": [[...], ...],               // K x K  (Kronecker factor, eq. 4)
  "corr_full":   null                        // (MK x MK) — overrides the Kronecker pair
}
```

Conventions: row-major vectorisation (bucket *i = m·K + k*); strikes in **moneyness
K/F** as recommended by §3.3 (estimating Σ on an absolute-strike grid inflates
strike-axis correlations and over-justifies netting — a non-conservative bias);
correlations estimated on **variations** of consensus marks with Ledoit–Wolf shrinkage.
A ready sample lives at [`data/sample_bundle.json`](data/sample_bundle.json).

Scenarios use the same philosophy: an `(M, K)` integer label matrix
(`{"labels": [[...]]}` in JSON), cells sharing a label forming one netting set —
producible by any in-house system via `ScenarioSource`.

## Library usage (no UI)

```python
from ebanetting import (
    JSONBundleSource, UncertaintyModel, KAPPA_90,
    greedy_netting, score_scenario, build_audit_pack,
)

bundle = JSONBundleSource("data/sample_bundle.json").load()
model = UncertaintyModel(bundle=bundle, kappa=KAPPA_90)

result = greedy_netting(model, alpha=0.90)          # optimal netting
print(result.scheme.n_sets, result.evaluation.r2, result.evaluation.ava)

report = score_scenario(bundle, my_labels, alpha=0.90, kappa=KAPPA_90)
print(report.variance_score, report.verdict)         # score any custom scheme

pack = build_audit_pack(bundle, result.evaluation, kappa=KAPPA_90,
                        history=result.history)      # art. 9(5) evidence
```

## Project layout

```
eba-vega-netting/
├── app.py                  # Streamlit studio
├── ebanetting/             # quant library (UI-independent, fully tested)
│   ├── datasource.py       #   data contract + abstract sources (plug your pipe here)
│   ├── model.py            #   ΔΠ, Var, AVA extremes
│   ├── netting.py          #   partition operator, variance test, floor, two-bucket
│   ├── optimizer.py        #   greedy under budget, frontier, correlation stress
│   ├── spectral.py         #   PCA bound, K*(α)
│   ├── scenario.py         #   scenario scoring, presets, smile decomposition
│   └── reporting.py        #   art. 9(5) audit pack
├── ui/charts.py            # Plotly figure builders
├── data/sample_bundle.json # JSON contract example
└── tests/test_core.py      # closed-forms vs engine, budget/floor invariants
```

## Caveats (faithful to the note, §7)

- First-order model: for books with material volga/vanna (cliquets, barriers), validate
  that second-order terms do not reorder the sets — otherwise net on full scenarios.
- The correlation stress must always be **adverse to netting** (the test protects
  against under-estimation).
- Demo data is synthetic; nothing here is investment or regulatory advice.
