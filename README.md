# 🧮 Vega Netting Studio — EBA Prudent Valuation (AVA MPU)

**Volatility-sensitivity netting under the EBA variance test**, as formalised in the
technical note *"Netting des sensibilités vega sous le test de variance — cadre EBA
Valorisation Prudente"* (Delegated Regulation (EU) 2016/101, art. 9(5) & art. 89).

A production-grade IPV framework + a colorful Streamlit studio: passage of the granular
vega onto the consensus test nodes (with restriction matrices when the shocks live on a
finer grid), optimal netting under the tracking-error variance test, the decoupled
two-run architecture (portfolio-free dendrogram + per-book AVA), stability across shock
families, scenario scoring of any custom netting scheme, spectral diagnostics, adverse
correlation stress, and the full art. 9(5) audit trail.

> **No tensor products.** Everything is computed with ordinary matrix products and sum
> reductions on the (tenor, strike) grid: the passage is the sandwich Ñ = AᵀT·N·AK, and
> under the decoupled-correlation hypothesis every variance is the matrix form of
> Property 1, `Var⟨V,Δσ⟩ = Σ( (V∘s) ∘ (ρ_mat @ (V∘s) @ ρ_strike) )`. The (MK × MK)
> Kronecker covariance is never assembled.

---

## What it does

| Theory (note section) | Implementation |
|---|---|
| Uncertainty model ΔΠ = ⟨N, Δσ⟩, Property 1 (§2) | `ebanetting/model.py` — `UncertaintyModel` |
| AVA extremes: add-up (2) / full diversification (3) | `ava_brut`, `ava_full` |
| Quadratic forms in matrix sandwich form, no Kronecker (§2 + annex) | `MarketDataBundle.variance_of/covariance_of` |
| Passage to the test nodes Ñ = AᵀT·N·AK, Th. 1 (§3) | `ebanetting/passage.py` — `project_bundle()` |
| Passage conventions quadrant / equal / interp (§3.2, Def. 1) | `passage_matrix()` |
| Passage tracking error vs surface weights (eq. 5) | `PassageResult.te2_by_convention` |
| Netting as aggregation operator on the nodes (§5, Def. 4) | `ebanetting/netting.py` — `NettingScheme` |
| Variance test TE² ≤ (1−α)·Var(ΔΠ), R² ≥ α (§4, eq. 6, Def. 3) | `NettingScheme.evaluate()` |
| Conservatism floor AVA(P) ≥ κ√Var(ΔΠ) (eq. 7) | `SchemeEvaluation.passes_floor` |
| Two-node closed form (§5.2, Théorème 2, eq. 8) | `two_bucket()` — unit-tested against the engine |
| Spectral diagnostic: factors of Σ, risk map, Théorème 3 floor (§6.1) | `ebanetting/spectral.py` — `spectral_diagnostic()` |
| Spectral cleaning of a noisy Σ̂ + subspace stability (§6.1.4, U3/U4) | `spectral_clean()`, `subspace_stability()` |
| Smile deformation model & Théorème 4 go/no-go (§6.2, Déf. 5) | `fit_deformation_model()`, `tranche_collapse_variance()` |
| Collapse refinement: level netted, RR/FLY carved out (§6.2, S2) | `tranche_refinement()` |
| Greedy agglomerative algorithm under budget (§7.3) | `ebanetting/optimizer.py` — `greedy_netting()` |
| Rectangular contiguous pavings of the node grid (§7.3) | rectangle-union merge candidates |
| Adverse correlation stress ρ → max(ρ−δ,−1) (step 4, §7.3) | `stress_bundle()`, `robust_netting()` |
| Restriction without loss R_T Δσ_fin R_Kᵀ, off-grid residual (§3.5, Prop. 5, Rem. 4) | `restriction_matrix()`, `offgrid_residual()` |
| Shock families & stability test — same partition, other Σ (§7.4) | `stability_report()`, `stable_cut()` |
| Base-risk distance d_ij and portfolio-free dendrogram (§7.5, Run 1) | `ebanetting/clustering.py` — `build_dendrogram()` |
| Portfolio-free TE bound, TE ≤ ε·AVA_brut/κ (§7.5, Prop. 8) | `DecoupledResult.te_bound` |
| Run 2: AVA on the cut + per-book TE²/Var check (§7.5) | `decoupled_netting()`, `evaluate_cut()` |
| Decoupling hypothesis ρ2D = ρ_mat·ρ_strike, entrywise (annex §8) | per-axis `corr_mat` / `corr_strike` inputs |
| Art. 9(5) evidence pack (§9) | `ebanetting/reporting.py` |
| Lagrangian efficient frontier (complementary) | `lagrangian_frontier()` |
| **Two-layer companion note** — smile model, Th. 1 disagreement d_ij | `ebanetting/twolayer.py` — `SmileModel`, `model_distance()` |
| Generated correlation ρ(x) = s₀/sₓ (Prop. 2) | `generated_correlation()` |
| Tranche dendrogram, pivot-linkage, most-liquid pivots (A3) | `tranche_dendrogram()`, `cut_tranche()` |
| Exact group residual, general pivot + idio floor (Th. 2) | `group_residual_variance()` vs `majorant_residual_sd()` |
| Barycenter pivot x_p = RR₀/m kills RR exactly (sec. 6.3) | `barycenter_pivot()` |
| Decisions N1–N4: collapse / extraction / scission (sec. 6.4) | `evaluate_book()` — never all-or-nothing |
| **Engine sheet (volet 2/2)** — flip-flop separable estimation | `ebanetting/engine.py` — `flip_flop()`, `fit_engine_model()` |
| 2-D sandwich X = Y·B·Zᵀ + E, exposure G = Yᵀ N Z, Σ_B, idio map | `EngineModel` (orthonormal bases, leverage-corrected idio) |
| T1–T6 battery + automatic decisions (majorant mode, torsion factor, idio exclusions) | `EngineModel.tests` |
| Exact TE² = tr(Aᵀ ΣT A ΣK) + ΣA²σε² on ONE global gap pattern | `EngineModel.variance_of()`, `evaluate_book_engine()` (E1–E6) |
| GA = Yᵀ A Z extraction diagnostic, AVA, floor | `EngineRun` |
| Structure: stress, closed-form distances, clustering, portfolio-free flag | `engine_dendrogram()`, `cut_engine()` |
| Stability: survival frequency, ARI, principal angles | `survival_frequencies()`, `adjusted_rand_index()`, `principal_angle_cosines()` |
| Select: frozen cut sweep + minimum-benefit guard across families | `select_cut()`, `min_benefit_guard()` |
| Golden material: synthetic generator, invariances | `simulate_panel()` + test suite |

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
- **🔁 Passage** — step 1 of the note: project the granular vega onto the consensus
  pillars (sandwich Ñ = AᵀT·N·AK), check vega conservation (Prop. 3), and compare the
  tracking error of the quadrant / equal / interp conventions (Théorème 1).
- **🧠 Optimal Netting** — the joint greedy optimiser: retained rectangular partition on
  the grid, AVA waterfall (add-up → netted vs the diversification floor),
  budget-consumption path, Lagrangian efficient frontier, robust mode with
  correlation-stress regimes.
- **🧬 Two-Layer** — the companion methodology end to end on a tranche: the layer-1
  deformation model (editable σ_S, σ_C, σ_ε), the disagreement matrix and the generated
  correlation curve, the dendrogram cut, then the layer-2 book decisions with the exact
  TE² vs the majorant — including the case where the exact signed computation
  authorises what layer 1 alone would have refused.
- **🌳 Structure & Stability** — the decoupled architecture of §7.5: the portfolio-free
  dendrogram on the base risk d_ij (Run 1), the ε cut slider with the Prop. 8 bound and
  the per-book TE²/Var check (Run 2), and the §7.4 stability protocol replaying the
  variance test of the same partition under alternative covariances (stress regimes or
  an uploaded Σ_daily / Σ_bidask bundle) — unstable fusions are undone.
- **🎯 Scenario Lab** — score **any** netting scheme: edit the (M × K) label matrix in
  place, upload one, or start from presets / the optimal solution. Returns the
  **variance score R²** (gauge vs α), the EBA verdict (test 6 + floor 7), the AVA
  impact and per-set statistics. Includes the two-node closed-form sandbox (Th. 2).
- **🔬 Spectral & Smile** — §6: the factors of Σ at the nodes (Prop. 6–7), the book's
  risk map c_ℓ²λ_ℓ, the Théorème 3 floor K*(α), the inertia ratio τ_L, leading
  eigendirections on the grid, and the per-tranche net level / risk-reversal /
  butterfly decomposition (Théorème 4) with per-tranche tests.
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
  "strikes":     [0.80, 0.90, ...],          // K moneyness levels K/F (recommended, §9)
  "vega":        [[...], ...],               // M x K signed vegas  ∂V/∂σ(T, K)
  "s":           [[...], ...],               // M x K uncertainty std-devs, > 0
  "corr_mat":    [[...], ...],               // M x M  (decoupled tenor correlation, annex)
  "corr_strike": [[...], ...],               // K x K  (decoupled strike correlation, annex)
  "corr_full":   null                        // (MK x MK) — overrides the decoupled pair
}
```

Conventions: row-major vectorisation (bucket *i = m·K + k*, only relevant for
`corr_full`); strikes in **moneyness K/F** as recommended by §9 (estimating the
uncertainty model on an absolute-strike grid inflates
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
│   ├── datasource.py       #   data contract + matrix-form quadratic engine (no kron)
│   ├── model.py            #   ΔΠ, Var, AVA extremes
│   ├── passage.py          #   step 1: passage (sandwich, Th. 1) + restriction (Prop. 5)
│   ├── clustering.py       #   sec. 7.5 dendrogram / Prop. 8 / Run 2 + sec. 7.4 stability
│   ├── twolayer.py         #   companion note: smile model, exact layer-2, N1-N4
│   ├── engine.py           #   engine sheet: flip-flop, sandwich, T1-T6, E1-E6, select
│   ├── netting.py          #   partition operator, variance test, floor, two-node
│   ├── optimizer.py        #   greedy under budget, frontier, correlation stress
│   ├── spectral.py         #   sec. 6.1: factors of Σ, Th. 3 floor, cleaning, stability
│   ├── scenario.py         #   scenario scoring, presets, smile decomposition
│   └── reporting.py        #   art. 9(5) audit pack
├── ui/charts.py            # Plotly figure builders
├── data/sample_bundle.json # JSON contract example
└── tests/test_core.py      # closed-forms vs engine, budget/floor invariants
```

## Caveats (faithful to the note, §9)

- First-order model: for books with material volga/vanna (cliquets, barriers), validate
  that second-order terms do not reorder the sets — otherwise net on full scenarios.
- The correlation stress must always be **adverse to netting** (the test protects
  against under-estimation).
- Demo data is synthetic; nothing here is investment or regulatory advice.
