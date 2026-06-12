"""Vega Netting Studio — EBA Prudent Valuation (AVA MPU) variance test.

Streamlit front-end for the ``ebanetting`` library. Implements the full
theory of the technical note "Netting des sensibilités vega sous le test
de variance" (Delegated Regulation (EU) 2016/101): passage to the test
nodes, variance test, conservatism floor and greedy optimal netting —
all in ordinary matrix algebra (no tensor / Kronecker products).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from ebanetting import (
    KAPPA_90,
    SmileModel,
    MarketDataBundle,
    NettingScheme,
    SyntheticDataSource,
    UncertaintyModel,
    audit_json,
    build_audit_pack,
    build_dendrogram,
    decoupled_netting,
    greedy_netting,
    preset_labels,
    project_bundle,
    robust_netting,
    cut_tranche,
    evaluate_book,
    generated_correlation,
    model_distance,
    score_scenario,
    smile_decomposition,
    tranche_dendrogram,
    spectral_diagnostic,
    stable_cut,
    stress_bundle,
    two_bucket,
)
from ui import charts

# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Vega Netting Studio — EBA AVA MPU",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .hero {
        background: linear-gradient(110deg, #FF5C8A 0%, #B388EB 35%, #5BC0EB 70%, #3DDC97 100%);
        border-radius: 18px; padding: 26px 34px; margin-bottom: 14px;
        box-shadow: 0 8px 30px rgba(91,192,235,0.25);
      }
      .hero h1 { color: #0B1026; margin: 0; font-size: 1.9rem; letter-spacing: -0.5px; }
      .hero p  { color: #15203f; margin: 6px 0 0 0; font-size: 1.0rem; font-weight: 500; }
      .badge {
        display: inline-block; padding: 6px 16px; border-radius: 999px;
        font-weight: 700; font-size: 0.95rem; letter-spacing: 0.5px;
      }
      .badge-pass { background: rgba(61,220,151,0.18); color: #3DDC97; border: 1.5px solid #3DDC97; }
      .badge-fail { background: rgba(255,92,138,0.18); color: #FF5C8A; border: 1.5px solid #FF5C8A; }
      div[data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(91,192,235,0.10), rgba(179,136,235,0.10));
        border: 1px solid rgba(120,140,220,0.25);
        border-radius: 14px; padding: 14px 16px;
      }
      div[data-testid="stMetricLabel"] { color: #9fb0e8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧮 Vega Netting Studio</h1>
      <p>Volatility-sensitivity netting under the EBA variance test — AVA Market Price Uncertainty,
      Delegated Regulation (EU) 2016/101, art. 9(5) &amp; art. 89 · IPV production framework</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached computations
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_greedy(bundle_json: str, alpha: float, kappa: float, weighting: str):
    bundle = MarketDataBundle.from_json(bundle_json)
    model = UncertaintyModel(bundle=bundle, kappa=kappa)
    return greedy_netting(model, alpha=alpha, weighting=weighting)


@st.cache_data(show_spinner=False)
def cached_robust(bundle_json: str, alpha: float, kappa: float, weighting: str, deltas: tuple):
    bundle = MarketDataBundle.from_json(bundle_json)
    return robust_netting(bundle, alpha=alpha, kappa=kappa, deltas=list(deltas), weighting=weighting)


@st.cache_resource(show_spinner=False)
def cached_dendrogram(bundle_json: str):
    """Run 1 (sec. 7.5) — depends only on s and rho, never on the book."""
    return build_dendrogram(MarketDataBundle.from_json(bundle_json))


def eur(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"€{v / 1e6:,.2f}M"
    if a >= 1e3:
        return f"€{v / 1e3:,.1f}k"
    return f"€{v:,.0f}"


def verdict_badge(ok: bool, text_ok: str = "PASS", text_ko: str = "FAIL") -> str:
    cls = "badge-pass" if ok else "badge-fail"
    return f'<span class="badge {cls}">{text_ok if ok else text_ko}</span>'


# --------------------------------------------------------------------------- #
# Sidebar — data source (abstract, bank-pluggable) and parameters
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Inputs")
    st.subheader("Market data source")
    source_kind = st.radio(
        "Source",
        ["Synthetic demo book", "Upload JSON bundle"],
        help="Banks plug their own data by implementing `ebanetting.DataSource` "
        "or producing the JSON contract (see Market Data tab).",
    )
    if source_kind == "Synthetic demo book":
        book = st.selectbox(
            "Demo book",
            ["mixed_desk", "smile_book", "calendar_book"],
            format_func={
                "mixed_desk": "Mixed desk (calendar + RR + short flies)",
                "smile_book": "Smile book (long ATM, short wings)",
                "calendar_book": "Calendar book (short vs long tenors)",
            }.get,
        )
        seed = st.number_input("Seed", min_value=0, max_value=10_000, value=7, step=1)
        bundle = SyntheticDataSource(book=book, seed=int(seed)).load()
    else:
        uploaded = st.file_uploader("MarketDataBundle JSON", type=["json"])
        if uploaded is None:
            st.info("Waiting for a bundle — falling back to the demo book.")
            bundle = SyntheticDataSource().load()
        else:
            bundle = MarketDataBundle.from_json(uploaded.read().decode("utf-8"))

    issues = bundle.validate()
    if issues:
        st.error("Bundle validation failed:\n\n- " + "\n- ".join(issues))
        st.stop()

    st.subheader("Step 1 — passage to test nodes (sec. 3)")
    granular_bundle = bundle
    passage_res = None
    use_passage = st.toggle(
        "Project granular vega onto consensus pillars",
        value=False,
        help="Transport the system-grid vega onto the pillars where the "
        "consensus uncertainty (s, ρ) is actually observable — the sandwich "
        "Ñ = Aᵀ_T N A_K of Def. 2. Théorème 1: with interp weights aligned "
        "on the surface construction, the passage destroys no information.",
    )
    if use_passage:
        default_t = sorted(set(list(range(0, bundle.M, 2)) + [bundle.M - 1]))
        default_k = sorted(set(list(range(0, bundle.K, 2)) + [bundle.K - 1]))
        pillar_t = st.multiselect(
            "Pillar tenors", options=list(range(bundle.M)), default=default_t,
            format_func=lambda i: bundle.tenors[i],
        )
        pillar_k = st.multiselect(
            "Pillar strikes", options=list(range(bundle.K)), default=default_k,
            format_func=lambda i: str(bundle.strikes[i]),
        )
        convention = st.selectbox(
            "Passage convention (sec. 3.2)",
            ["interp", "equal", "quadrant"],
            format_func={
                "interp": "Interp-weighted (= surface weights, TE = 0 — Th. 1)",
                "equal": "Equal-weighted (½/½ on bracketing pillars)",
                "quadrant": "Quadrant (nearest pillar)",
            }.get,
        )
        if pillar_t and pillar_k:
            passage_res = project_bundle(bundle, pillar_t, pillar_k, convention)
            bundle = passage_res.node_bundle
        else:
            st.warning("Select at least one pillar per axis — passage skipped.")

    st.subheader("Regulatory parameters")
    alpha = st.slider(
        "α — variance-test threshold (R² ≥ α)", 0.80, 0.99, 0.90, 0.01,
        help="Usual range 0.90–0.95, consistent with the 90% RTS confidence level.",
    )
    kappa = st.number_input(
        "κ — uncertainty quantile multiplier", value=round(KAPPA_90, 4),
        min_value=0.5, max_value=3.0, step=0.01,
        help="z₀.₉₀ ≈ 1.2816 under the Gaussian hypothesis (art. 89), or an "
        "empirically calibrated factor.",
    )
    weighting = st.selectbox(
        "Representative shock Δσ̃ₖ",
        ["pivot", "vega", "equal"],
        format_func={
            "pivot": "Pivot bucket (max |ν·s| of the set)",
            "vega": "|vega|-weighted average",
            "equal": "Equal-weighted average",
        }.get,
    )
    grid_word = "test nodes" if passage_res is not None else "buckets"
    st.caption(
        f"Grid: **{bundle.M} tenors × {bundle.K} strikes = {bundle.n} {grid_word}** · "
        f"{bundle.meta.get('underlying', '—')} · as-of {bundle.meta.get('asof', '—')}"
    )

model = UncertaintyModel(bundle=bundle, kappa=float(kappa))
bundle_json = bundle.to_json(sort_keys=True)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
(tab_theory, tab_data, tab_passage, tab_optimal, tab_structure, tab_twolayer,
 tab_scenario, tab_spectral, tab_audit) = st.tabs(
    ["🏛️ Theory", "📊 Market Data", "🔁 Passage", "🧠 Optimal Netting",
     "🌳 Structure & Stability", "🧬 Two-Layer", "🎯 Scenario Lab",
     "🔬 Spectral & Smile", "📋 Audit & Export"]
)

# =========================================================================== #
# THEORY
# =========================================================================== #
with tab_theory:
    st.markdown("## The complete framework, from regulation to algorithm")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(
            """
### 1 · Regulatory frame
Delegated Regulation (EU) **2016/101** (EBA RTS on prudent valuation) requires, under the
*core approach*, **AVA** computed per *valuation exposure* at a **90% confidence level**
(art. 89). Article **9(5)** and the related EBA Q&A allow the **Market Price Uncertainty**
AVA to be computed on a **netted exposure** instead of the gross sum — *provided the
institution demonstrates that positions genuinely offset against the parameter
uncertainty*. For vega the question becomes: **at which granularity of the volatility
surface (tenor × strike) may signed sensitivities be summed before applying the
uncertainty spread?**

### 2 · Uncertainty model
The surface is discretised into $n = M \\times K$ buckets. With $\\nu_i = \\partial V / \\partial \\sigma_i$
the bucket vegas and $\\Delta\\sigma_i$ the centred mid-uncertainty
($\\mathrm{sd}=s_i$, correlation $\\rho$, $\\Sigma = D\\rho D$), the first-order valuation
P&L uncertainty is
            """
        )
        st.latex(r"\Delta\Pi=\nu^{\top}\Delta\sigma,\qquad \operatorname{Var}(\Delta\Pi)=\nu^{\top}\Sigma\,\nu \tag{1}")
        st.markdown("**The two AVA extremes** (no netting vs full diversification):")
        st.latex(r"\mathrm{AVA}_{\text{brut}}=\kappa\sum_{i=1}^{n}\lvert\nu_i\rvert s_i \tag{2}")
        st.latex(r"\mathrm{AVA}_{\text{full}}=\kappa\sqrt{\nu^{\top}\Sigma\,\nu}\;\le\;\mathrm{AVA}_{\text{brut}} \tag{3}")
        st.markdown(
            """
The gap (2)−(3) is the **maximal theoretical netting benefit**. The regulator does not
grant (3) by default: the dependence structure must be demonstrated robust, hence the
variance test on **partial, interpretable aggregations**.

### 3 · Step 1: the passage to the test nodes (sec. 3, Théorème 1)
Uncertainty is observable only at the consensus pillars. The granular vega is
transported there with two **passage matrices** (non-negative rows summing to 1,
Def. 1) and the 2-D **sandwich** — an ordinary matrix product on each side:
            """
        )
        st.latex(r"\tilde N = A_T^{\top}\, N\, A_K \in \mathbb{R}^{q_T\times q_K} \quad\text{(Def. 2)}")
        st.markdown(
            """
**Théorème 1 (coherence passage–surface)**: when the passage weights equal the
**interpolation weights of the pricing-surface construction** ($A=B$), the passage
destroys no information at first order — $\\mathrm{TE}_{\\text{passage}}=0$. Any other
convention (quadrant, equal-weighted) creates a quantifiable error (eq. 5) that
consumes the variance-test budget before any netting. Consequence (Remarque 3): the
passage matrices are **not an optimisation variable** — they are pinned by the surface
construction; only the downstream partition is optimised.

**Sec. 3.5 — when the shock grid differs from the sensi grid.** Three regimes, one
recipe: shocks *coarser* than the sensitivities (Totem pillars) → contract the vega
(sandwich, Théorème 1); shocks *finer* (daily variations of the full system surface,
e.g. 350×100 vs 10×10 parametrisation pillars) → **restriction without loss**
(Propriété 5): 0/1 selection matrices read the fine surface at the pillars,
$\\Delta\\sigma^{\\text{pil}} = R_T\\,\\Delta\\sigma^{\\text{fin}}\\,R_K^{\\top}$, exact because
the interpolation reproduces its nodes ($R\\,B = I$); identical grids → identity. The
out-of-grid component $\\Delta\\sigma^{\\text{fin}} - B_T(R_T\\Delta\\sigma^{\\text{fin}}R_K^\\top)B_K^\\top$
(Remarque 4) is measurable daily: its size gauges the adequacy of the pillar set.
Everything downstream operates on the **pivot grid**, indifferent to the shock family
that fed it.

### 3b · 2-D uncertainty: the decoupling hypothesis (annex)
With per-axis correlations, the 2-D correlation decouples **entrywise**,
$\\rho_{(a,b),(a',b')}=\\rho^{\\text{mat}}_{aa'}\\,\\rho^{\\text{strike}}_{bb'}$ — only
$\\binom{q_T}{2}+\\binom{q_K}{2}$ parameters, each interpretable and stressable. **No
Kronecker / tensor product is ever assembled**: every variance is evaluated in matrix
form with usual matrix products and a final sum reduction (Property 1),
            """
        )
        st.latex(
            r"\operatorname{Var}\langle V,\Delta\sigma\rangle"
            r"=\big\langle V\!\circ\! s,\ \rho^{\text{mat}}\,(V\!\circ\! s)\,\rho^{\text{strike}}\big\rangle"
        )
        st.markdown(
            """
Admissible partitions are restricted to **contiguous rectangles**
$\\mathcal N_r = T_r\\times S_r$ on the node grid (sec. 7.3) — interpretable and
documentable under art. 9(5). **Beware the absolute-strike grid** (sec. 8): fixed
strikes drift in moneyness with spot, inflating strike-axis correlations and
**over-justifying netting** — a non-conservative bias. Estimate $s,\\rho$ in moneyness
$x=K/F(T)$ (or delta) coordinates and remap vegas via the interpolation Jacobian; if
stuck in absolute strike, stress $\\rho^{\\text{strike}}$ downward.
            """
        )
    with c2:
        st.markdown(
            """
### 4 · Netting as an aggregation operator (Def. 4)
A **netting scheme** is a partition $\\mathcal P=\\{\\mathcal G_1,\\dots,\\mathcal G_S\\}$
of the test nodes. Set $r$ has netted exposure $m_r=\\sum_{(a,b)\\in\\mathcal G_r}\\tilde N_{ab}$
and a representative shock $\\Delta\\tilde\\sigma_r=\\langle W_r,\\Delta\\sigma\\rangle$
with $W_r$ a row-stochastic weight grid. The proxy P&L and the retained (conservative,
add-up across sets) AVA are
            """
        )
        st.latex(r"\widehat{\Delta\Pi}_{\mathcal P}=\sum_r m_r\,\Delta\tilde\sigma_r,\qquad \mathrm{AVA}(\mathcal P)=\kappa\sum_{r}\lvert m_r\rvert\,\tilde s_r \quad\text{(Def. 4)}")
        st.markdown("### 5 · The variance test (Def. 3)")
        st.latex(
            r"\mathrm{TE}^2(\mathcal P)=\operatorname{Var}\!\big(\Delta\Pi-\widehat{\Delta\Pi}_{\mathcal P}\big)"
            r"=\operatorname{Var}\Big\langle \tilde N-\textstyle\sum_r m_r W_r,\ \Delta\sigma\Big\rangle\;\le\;(1-\alpha)\operatorname{Var}(\Delta\Pi) \tag{6}"
        )
        st.markdown(
            "equivalently **R² ≥ α** (α ∈ [0.90, 0.95]) — the prudent-valuation analogue of "
            "FRTB P&L-attribution tests. (6) controls **fidelity**; the **conservatism floor** controls level:"
        )
        st.latex(r"\mathrm{AVA}(\mathcal P)\;\ge\;\kappa\sqrt{\operatorname{Var}(\Delta\Pi)} \tag{7}")
        st.markdown("### 6 · Optimal netting")
        st.latex(
            r"\mathcal P^{\star}=\arg\min_{\mathcal P}\;\kappa\sum_{r}\Big|\sum_{(a,b)\in\mathcal G_r}\tilde N_{ab}\Big|\,\tilde s_r"
            r"\quad\text{s.c. } \mathrm{TE}^2(\mathcal P)\le(1-\alpha)\operatorname{Var}(\Delta\Pi) \text{ and } (7) \tag{10}"
        )
        st.markdown(
            "Combinatorial (Bell numbers), NP-hard → greedy controlled construction. "
            "**Two-node closed form** (Théorème 2 — net $j$ onto pivot $i$):"
        )
        st.latex(r"\mathrm{TE}^2=\nu_j^2\big(s_i^2+s_j^2-2\rho s_i s_j\big) \quad\text{(Th. 2 i)}")
        st.latex(
            r"\text{admissible}\iff \rho\;\ge\;\frac{s_i^2+s_j^2}{2 s_i s_j}"
            r"-\frac{(1-\alpha)\operatorname{Var}(\Delta\Pi)}{2\,\nu_j^2\,s_i s_j} \tag{8}"
        )
        st.markdown(
            """
*Economic reading*: netting requires **high correlation and homogeneous uncertainties**
(adjacent smile points net; a 1M ATM against a 10Y wing does not), and the larger the
netted vega relative to total risk, the more demanding the threshold — **you cannot hide
a large short behind an approximate proxy**. Same-sign merges are admissible but
pointless: optimal netting targets opposite-sign, strongly correlated sets.
            """
        )

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 6.1 · The spectral diagnostic — Théorème 3")
        st.latex(
            r"\Sigma=\sum_\ell \lambda_\ell u_\ell u_\ell^{\top},\qquad"
            r"\operatorname{Var}(\Delta\Pi)=\sum_\ell c_\ell^2\lambda_\ell,\quad c_\ell=\langle\nu,u_\ell\rangle"
        )
        st.markdown(
            """
The factors $\\xi_\\ell=u_\\ell^{\\top}\\Delta\\sigma$ are decorrelated with variances
$\\lambda_\\ell$ (Prop. 6); the map $\\{c_\\ell^2\\lambda_\\ell\\}$ (Prop. 7) is the **risk
map** of the book. **Théorème 3 (spectral floor)**: for any scheme whose representative
shocks live in $\\mathrm{Vect}(u_1,\\dots,u_L)$ — the realistic description of coarse
schemes with smooth weights — $\\mathrm{TE}^2\\ge\\sum_{\\ell>L}c_\\ell^2\\lambda_\\ell$: the
tail variance of the book is incompressible. (The hypothesis is necessary: with $w=\\nu$
a single set replicates $\\Delta\\Pi$ exactly — Remarque 5.) **Four usages** (6.1.4):
U1 the inertia ratio $\\tau_L$ — compressibility of the uncertainty; U2 the risk map —
*why* a test fails and where to refine; U3 **spectral cleaning** of a noisy
$\\hat\\Sigma$ — flatten the eigenvalue tail to its mean, with the conservatism guard;
U4 stability of the dominant subspaces across estimation windows. **The limit**: a
factor is *not* a netting set — dense signed weights are no valuation exposure; the
spectrum measures, the partition nets.

### 6.2 · Smile decomposition — Théorème 4
Under the **two-mode deformation model** (Déf. 5, eq. 9),
$\\Delta\\sigma_{a,j}=\\Delta\\sigma_{a,\\text{ATM}}+(x_j{-}x_0)\\Delta S_a+(x_j{-}x_0)^2\\Delta C_a+\\varepsilon_{a,j}$
— empirical, testable by regressing daily strike moves on ATM/slope/curvature — the
residual of collapsing tranche $a$ onto its ATM pivot is **exactly**
$R_a=\\mathrm{RR}_a\\,\\Delta S_a+\\mathrm{FLY}_a\\,\\Delta C_a+\\sum_j\\nu_j\\varepsilon_{a,j}$
(Théorème 4): the **net level $m_a$ nets perfectly whatever its size**, and the
go/no-go is closed-form,
$\\operatorname{Var}(R_a)=\\mathrm{RR}_a^2\\operatorname{Var}\\Delta S+\\mathrm{FLY}_a^2\\operatorname{Var}\\Delta C+2\\,\\mathrm{RR}_a\\mathrm{FLY}_a\\operatorname{Cov}+\\sigma_\\varepsilon^2\\sum_j\\nu_j^2$
— not "insufficient correlation" but "the book carries this much net RR on this
tranche" (S1). When the collapse fails, the refinement keeps the netted level and
carves $\\mathrm{RR}_a$, $\\mathrm{FLY}_a$ out in **add-up** with their own consensus
uncertainties (S2): only the genuinely unjustified netting is given up. RR/FLY are the
book's coordinates in the standard smile-strategy basis (Remarque 6).
            """
        )
    with c4:
        st.markdown("### 7.3 · Greedy agglomerative algorithm under variance budget")
        st.markdown(
            """
With budget $B=(1-\\alpha)\\operatorname{Var}(\\Delta\\Pi)$:

1. **Init**: singletons, $\\mathrm{TE}^2=0$.
2. **Iterate**: for every fusible pair of adjacent rectangular sets compute the AVA
   reduction $g$ and the residual-variance cost $c$; merge the pair maximising $g/c$
   (zero-cost merges first) while $\\mathrm{TE}^2+c\\le B$.
3. **Stop** when no merge fits the budget; verify the floor (7), otherwise unwind the
   least efficient fusion.
4. **Stability**: replay under rolling estimation windows and **adverse correlation
   stress** $\\rho\\to\\max(\\rho-\\delta,-1)$, $\\delta\\sim0.1\\!-\\!0.2$; retain only fusions
   robust across regimes — this is what makes the scheme defendable in model review.

### 7.4 · Shock families and stability
One formalism, several covariances: $\\Sigma^{\\text{totem}}$ (AVA level + official
test), $\\Sigma^{\\text{bid-ask}}$ (unwind-cost level), $\\Sigma^{\\text{daily}}$ (daily
surface moves restricted to the pillars, Prop. 5 — **not** a prudential level, but the
densest $\\rho$ estimator and the **stability test**: replay the variance test of the
*same* partition under $\\Sigma^{\\text{daily}}$ windows and stresses; any fusion that
fails a regime is unstable and undone). Regulatory asymmetry: refusing a valid fusion
is allowed, keeping an invalidated one is not.

### 7.5 · Decoupled architecture — structure ⟂ level
**Run 1 (structure, portfolio-free)**: hierarchical clustering of the nodes on the
**base risk** $d_{ij}=\\sqrt{s_i^2+s_j^2-2\\rho_{ij}s_is_j}$ (the cost of Th. 2 (i) is
$\\nu_j^2 d_{ij}^2$) → a **dendrogram**; a scheme is a cut at height $\\varepsilon$.
**Propriété 8**: $d_{j,\\text{pivot}}\\le\\varepsilon s_j$ per node implies, for *any*
book, $\\mathrm{TE}\\le\\varepsilon\\sum_j|\\nu_j|s_j=\\varepsilon\\,\\mathrm{AVA}_{\\text{brut}}/\\kappa$.
**Run 2 (level)**: evaluate $\\mathrm{AVA}=\\kappa\\sum_r|m_r|\\tilde s_r$ on the cut with
the day's Totem $s$ — the only remaining freedom is the scalar cut height. The
mandatory per-book check is the single closed-form ratio $\\mathrm{TE}^2/\\mathrm{Var}(\\Delta\\Pi)$
(lower the cut if it fails). A partition that moves with the book is a validation red
flag; here the structure is slow and justified, the level is fast and per-book.

### 8 · IPV implementation points
- **$s_i$**: Totem inter-contributor dispersion (sd or interquantile ranges rescaled to
  90%), else broker ranges / liquidity proxies. Few contributors ⇒ **inflate** $s_i$,
  don't smooth it.
- **$\\rho$**: estimate on **variations** of consensus marks, never levels
  (cointegration inflates level correlations and over-justifies netting). When the
  Totem history is short, the **daily variations of the system surface restricted to
  the pillars (Prop. 5)** give a denser estimator — retain the **less netting-favourable**
  of the two, and cross with the stability test (sec. 7.4). Shrinkage recommended in
  all cases.
- **Regulatory asymmetry**: the test protects against *under*-estimation — in doubt,
  the correlation stress must be **adverse to netting**.
- **Documentation**: keep {retained partition, realised R², frontier, stress results}
  per computation date — the art. 9(5) evidence (see Audit tab).
- **Non-linearity**: (1) is first-order; for material volga/vanna books (cliquets,
  barriers), validate that second-order terms do not reorder the sets — else net on
  full scenarios rather than sensitivities.
            """
        )

# =========================================================================== #
# MARKET DATA
# =========================================================================== #
with tab_data:
    meta = bundle.meta
    st.markdown(
        f"**{meta.get('underlying', 'Portfolio')}** · {meta.get('desk', '')} · "
        f"as-of **{meta.get('asof', '—')}** · vega in *{meta.get('vega_unit', 'EUR/vol pt')}*, "
        f"uncertainty in *{meta.get('uncertainty_unit', 'vol pts')}*"
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Var(ΔΠ) — Property 1", eur(np.sqrt(model.var_total)) + "²" if False else f"{model.var_total:,.0f} €²")
    k2.metric("AVA add-up — eq. (2)", eur(model.ava_brut))
    k3.metric("AVA full diversification — eq. (3)", eur(model.ava_full))
    k4.metric(
        "Max netting benefit", eur(model.ava_brut - model.ava_full),
        delta=f"-{(1 - model.ava_full / model.ava_brut):.0%} of add-up" if model.ava_brut > 0 else None,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.heatmap(bundle.vega, bundle.strikes, bundle.tenors,
                           "Vega surface N (signed, EUR / vol pt)", diverging=True,
                           colorbar_title="ν"),
            width="stretch",
        )
        if bundle.corr_mat is not None:
            st.plotly_chart(
                charts.heatmap(bundle.corr_mat, bundle.tenors, bundle.tenors,
                               "ρ_mat — tenor correlation of Δσ", colorscale="Sunset",
                               colorbar_title="ρ"),
                width="stretch",
            )
    with c2:
        st.plotly_chart(
            charts.heatmap(bundle.s, bundle.strikes, bundle.tenors,
                           "Uncertainty s (consensus dispersion, vol pts)",
                           colorscale="Plasma", colorbar_title="s"),
            width="stretch",
        )
        if bundle.corr_strike is not None:
            st.plotly_chart(
                charts.heatmap(bundle.corr_strike, bundle.strikes, bundle.strikes,
                               "ρ_strike — moneyness correlation of Δσ", colorscale="Sunset",
                               colorbar_title="ρ"),
                width="stretch",
            )
        elif bundle.corr_full is not None:
            st.plotly_chart(
                charts.heatmap(bundle.corr_full, list(range(bundle.n)), list(range(bundle.n)),
                               "ρ — full bucket correlation", colorscale="Sunset"),
                width="stretch",
            )

    with st.expander("📦 Data contract — plug your own pipe here", expanded=False):
        st.markdown(
            """
The app is **bank-agnostic**. Two integration points, both in `ebanetting/datasource.py`:

1. **`DataSource`** (abstract): implement `load() -> MarketDataBundle` against your golden
   sources — risk-system vegas, Totem/Markit dispersions, broker ranges. The reference
   implementation `JSONBundleSource` reads the JSON contract below.
2. **`ScenarioSource`** (abstract): implement `load() -> (M,K) int label matrix` to feed
   candidate netting schemes (e.g. the desk's current convention) into the Scenario Lab.

**JSON contract** (one document per as-of / underlying / valuation exposure) — strikes in
moneyness K/F as recommended in sec. 8; correlations either decoupled
(decoupled `corr_mat` × `corr_strike`, annex of the note — combined entrywise,
never as a Kronecker matrix) or full (`corr_full`, takes precedence):
            """
        )
        st.code(
            json.dumps(
                {
                    "meta": {"asof": "YYYY-MM-DD", "underlying": "...", "desk": "...",
                             "vega_unit": "EUR per vol point", "uncertainty_unit": "vol points"},
                    "tenors": ["1M", "3M", "..."],
                    "tenor_years": [0.0833, 0.25],
                    "strikes": [0.8, 1.0, 1.2],
                    "vega": [["M x K matrix"]],
                    "s": [["M x K matrix, > 0"]],
                    "corr_mat": [["M x M"]],
                    "corr_strike": [["K x K"]],
                    "corr_full": None,
                },
                indent=2,
            ),
            language="json",
        )
        st.download_button(
            "⬇️ Download current bundle as JSON template",
            data=bundle.to_json(indent=2),
            file_name="market_data_bundle.json",
            mime="application/json",
        )

# =========================================================================== #
# PASSAGE
# =========================================================================== #
with tab_passage:
    st.markdown(
        "### Step 1 — passage from the system grid to the test nodes (sec. 3)\n"
        "Consensus uncertainty lives at a few pillars; the granular vega is "
        "transported there with the **sandwich of passage matrices** "
        "Ñ = Aᵀ_T·N·A_K (Def. 2) — two ordinary matrix products, one per axis. "
        "**Théorème 1**: aligning the passage weights on the interpolation "
        "weights of the surface construction makes the passage lossless "
        "(TE_passage = 0); quadrant / equal-weighted conventions create a "
        "quantifiable tracking error (eq. 5) charged against the variance budget."
    )
    if passage_res is None:
        st.info(
            "Enable **“Project granular vega onto consensus pillars”** in the "
            "sidebar to activate the passage. The rest of the app then runs on "
            "the projected node grid, as the note prescribes."
        )
    else:
        pr = passage_res
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            "Nodes", f"{bundle.M} × {bundle.K} = {bundle.n}",
            delta=f"from {granular_bundle.M} × {granular_bundle.K} = {granular_bundle.n} buckets",
        )
        p2.metric(
            "Vega conservation — Prop. 3",
            "OK" if pr.conserves_vega else "BROKEN",
            delta=f"Σ N = {pr.vega_total_granular:,.0f} → Σ Ñ = {pr.vega_total_nodes:,.0f}",
            delta_color="off",
        )
        budget_passage = (1.0 - float(alpha)) * pr.var_reference
        p3.metric(
            "TE²_passage — eq. (5)",
            f"{pr.te2_passage:,.0f} €²",
            delta=(
                f"{pr.te2_passage / budget_passage:.1%} of budget"
                if budget_passage > 0 else "—"
            ),
            delta_color="off",
        )
        p4.metric("Convention", pr.convention)
        if pr.convention != "interp":
            st.warning(
                "Non-interp passage: the tracking error above consumes the "
                "variance-test budget **before any netting** (Théorème 1 / "
                "Remarque 1). Document it and deduct it (sec. 8)."
            )

        cpa, cpb = st.columns(2)
        with cpa:
            st.plotly_chart(
                charts.heatmap(
                    granular_bundle.vega, granular_bundle.strikes, granular_bundle.tenors,
                    "Granular vega N (system grid)", diverging=True, colorbar_title="ν",
                ),
                width="stretch",
            )
        with cpb:
            st.plotly_chart(
                charts.heatmap(
                    bundle.vega, bundle.strikes, bundle.tenors,
                    "Projected vega Ñ = Aᵀ_T N A_K (test nodes)", diverging=True,
                    colorbar_title="ν̃",
                ),
                width="stretch",
            )

        st.markdown("#### Tracking error of the three conventions (vs interp weights — Théorème 1)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "convention": conv,
                        "TE²_passage (€²)": round(te2, 0),
                        "share of budget": (
                            f"{te2 / budget_passage:.1%}" if budget_passage > 0 else "—"
                        ),
                        "lossless (Th. 1)": "✓" if te2 <= 1e-9 * max(pr.var_reference, 1.0) else "✗",
                    }
                    for conv, te2 in pr.te2_by_convention.items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander("🔎 Passage matrices A_T and A_K (Def. 1 — rows sum to 1)"):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**A_T (tenor axis)**")
                st.dataframe(
                    pd.DataFrame(pr.a_t, index=granular_bundle.tenors, columns=bundle.tenors),
                    width="stretch",
                )
            with cb:
                st.markdown("**A_K (strike axis)**")
                st.dataframe(
                    pd.DataFrame(
                        pr.a_k,
                        index=[str(s) for s in granular_bundle.strikes],
                        columns=[str(s) for s in bundle.strikes],
                    ),
                    width="stretch",
                )

# =========================================================================== #
# OPTIMAL NETTING
# =========================================================================== #
with tab_optimal:
    st.markdown("### Greedy agglomerative optimisation under variance budget (sec. 7.3)")
    opt_col1, opt_col2 = st.columns([1, 1])
    with opt_col1:
        robust_mode = st.toggle(
            "🛡️ Robust mode — retain only fusions surviving adverse correlation stress (step 4)",
            value=False,
        )
    with opt_col2:
        deltas = st.multiselect(
            "Stress magnitudes δ (ρ → max(ρ−δ, −1))",
            [0.05, 0.10, 0.15, 0.20], default=[0.10, 0.20],
            disabled=not robust_mode,
        )

    with st.spinner("Running greedy agglomeration…"):
        result = cached_greedy(bundle_json, float(alpha), float(kappa), weighting)
    if robust_mode and deltas:
        with st.spinner("Replaying under correlation stress…"):
            rb = cached_robust(bundle_json, float(alpha), float(kappa), weighting, tuple(sorted(deltas)))
        ev = rb["robust_evaluation"]
        scheme = rb["robust_scheme"]
        history = result.history
    else:
        rb = None
        ev = result.evaluation
        scheme = result.scheme
        history = result.history

    st.session_state["optimal_labels"] = scheme.labels.tolist()
    st.session_state["optimal_eval"] = ev
    st.session_state["optimal_history"] = history
    st.session_state["robust_pack"] = rb

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Netting sets", f"{scheme.n_sets}", delta=f"from {bundle.n} buckets")
    m2.metric("Variance score R²", f"{ev.r2:.2%}", delta=f"{(ev.r2 - alpha) * 100:+.2f} pts vs α")
    m3.metric("AVA netted — Def. 4", eur(ev.ava), delta=f"-{ev.ava_saving_pct:.0%} vs add-up")
    m4.metric("Budget used", f"{ev.te2 / ev.budget:.0%}" if ev.budget > 0 else "—",
              delta=f"TE² = {ev.te2:,.0f} €²")
    m5.markdown(
        f"**Variance test (6)** {verdict_badge(ev.passes_variance)}<br><br>"
        f"**Floor (7)** {verdict_badge(ev.passes_floor)}",
        unsafe_allow_html=True,
    )
    if result.rolled_back:
        st.warning(f"Floor (7) initially violated — {result.rolled_back} merge(s) rolled back (step 3).")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.plotly_chart(
            charts.partition_figure(
                scheme.labels, bundle.vega, bundle.tenors, bundle.strikes,
                title="Retained netting sets (rectangular paving, sec. 7.3)",
                set_stats=ev.set_stats,
            ),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            charts.ava_waterfall(model.ava_brut, ev.ava, model.ava_full),
            width="stretch",
        )

    st.plotly_chart(
        charts.merge_history_figure(history, model.budget(alpha), model.ava_brut),
        width="stretch",
    )
    if rb is not None:
        st.markdown("#### 🛡️ Stress regimes — robustness of the fusions")
        rows = []
        for d, run in rb["runs"].items():
            e = run.evaluation
            rows.append(
                {
                    "regime": "base" if d == 0.0 else f"ρ − {d:.2f}",
                    "sets": run.scheme.n_sets,
                    "R²": round(e.r2, 4),
                    "AVA": round(e.ava, 0),
                    "TE²/B": round(e.te2 / e.budget, 3) if e.budget > 0 else None,
                    "variance test": "PASS" if e.passes_variance else "FAIL",
                    "floor": "PASS" if e.passes_floor else "FAIL",
                }
            )
        rows.append(
            {
                "regime": "robust (common refinement)",
                "sets": scheme.n_sets,
                "R²": round(ev.r2, 4),
                "AVA": round(ev.ava, 0),
                "TE²/B": round(ev.te2 / ev.budget, 3) if ev.budget > 0 else None,
                "variance test": "PASS" if ev.passes_variance else "FAIL",
                "floor": "PASS" if ev.passes_floor else "FAIL",
            }
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("🔎 Per-set detail"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "set": s.set_id,
                        "buckets": s.size,
                        "net vega m_k": round(s.net_vega, 0),
                        "gross Σ|ν|": round(s.gross_vega, 0),
                        "offset ratio": f"{s.offset_ratio:.0%}",
                        "s̃_k": round(s.s_tilde, 3),
                        "AVA add-up": round(s.ava_addup, 0),
                        "AVA netted": round(s.ava_netted, 0),
                    }
                    for s in ev.set_stats
                ]
            ),
            width="stretch",
            hide_index=True,
        )

# =========================================================================== #
# STRUCTURE & STABILITY (sec. 7.4 / 7.5)
# =========================================================================== #
with tab_structure:
    st.markdown(
        "### Decoupled architecture — the structure by the variance, the level by the AVA (sec. 7.5)\n"
        "**Run 1** clusters the test nodes hierarchically on the **base risk** "
        "d_ij = √(s_i² + s_j² − 2ρ_ij s_i s_j) — Théorème 2 (i): fusing j onto pivot i "
        "costs TE² = ν_j²·d_ij². The dendrogram depends only on Σ, **never on the book**: "
        "the structure is stable from portfolio to portfolio. A netting scheme is a "
        "**cut** of the tree at height ε; Propriété 8 guarantees, for *any* book, "
        "TE ≤ ε·Σ|ν_j|s_j = ε·AVA_brut/κ. **Run 2** evaluates the AVA on the cut with "
        "the day's Totem s — no re-optimisation, plus the mandatory per-book check "
        "TE²/Var(ΔΠ) (the cut is lowered if it fails)."
    )
    with st.spinner("Run 1 — building the portfolio-free dendrogram…"):
        dendro = cached_dendrogram(bundle_json)
    if dendro.merges:
        heights = [m.height for m in dendro.merges]
        h_max = max(heights)
        h_default = float(np.median(heights))
    else:
        h_max, h_default = 1.0, 0.5
    eps_cut = st.slider(
        "ε — cut height (Prop. 8: per-node base risk ≤ ε · own uncertainty)",
        0.0, float(np.ceil(h_max * 1.05 * 100) / 100), h_default, 0.01,
    )
    with st.spinner("Run 2 — evaluating the day's book on the cut…"):
        dres = decoupled_netting(bundle, alpha=float(alpha), kappa=float(kappa),
                                 epsilon=float(eps_cut), dendrogram=dendro)
    ev_d = dres.evaluation

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Netting sets", f"{ev_d.scheme.n_sets}",
              delta=f"{dres.n_merges} merges (of {len(dendro.merges)})")
    d2.metric("ε realised", f"{dres.epsilon_realised:.3f}",
              delta=f"{dres.lowered} merge(s) undone by the book check" if dres.lowered else "cut as requested",
              delta_color="off")
    d3.metric("Prop. 8 bound on TE", f"{dres.te_bound:,.0f}",
              delta=f"actual TE = {np.sqrt(max(ev_d.te2, 0)):,.0f}", delta_color="off")
    d4.metric("Variance score R²", f"{ev_d.r2:.2%}",
              delta=f"{(ev_d.r2 - alpha) * 100:+.2f} pts vs α")
    d5.metric("AVA (Run 2) — Def. 4", eur(ev_d.ava), delta=f"-{ev_d.ava_saving_pct:.0%} vs add-up")

    cd1, cd2 = st.columns([1.1, 1])
    with cd1:
        st.plotly_chart(
            charts.dendrogram_figure(dendro.merges, float(eps_cut), dres.n_merges),
            width="stretch",
        )
    with cd2:
        st.plotly_chart(
            charts.partition_figure(
                dres.labels, bundle.vega, bundle.tenors, bundle.strikes,
                title="Cut of the dendrogram — structure sets (Run 1)",
                set_stats=ev_d.set_stats,
            ),
            width="stretch",
        )
    st.caption(
        "Compare with the joint greedy (Optimal Netting tab): the decoupled scheme may "
        "leave some AVA on the table, but it does not move with the book — re-optimising "
        "the partition every date with the day's vegas is itself a red flag in validation "
        "(sec. 7.5)."
    )

    st.divider()
    st.markdown(
        "#### Stability across shock families (sec. 7.4)\n"
        "Same partition, same transport, same formulas — only Σ changes. The variance "
        "test of the retained cut is replayed under alternative covariances; any fusion "
        "failing a regime is unstable and undone (one may refuse a valid fusion, never "
        "keep an invalidated one)."
    )
    stab_deltas = st.multiselect(
        "Adverse-correlation regimes standing in for Σ_daily windows (ρ → max(ρ−δ, −1))",
        [0.05, 0.10, 0.15, 0.20, 0.30], default=[0.10, 0.20],
    )
    alt_upload = st.file_uploader(
        "…and/or upload an alternative-covariance bundle (e.g. Σ_daily restricted to "
        "the pillars via Prop. 5, or Σ_bidask) — same grid and vega, different s/ρ",
        type=["json"], key="alt_cov",
    )
    alternatives = {f"ρ − {d:.2f}": stress_bundle(bundle, d) for d in stab_deltas}
    if alt_upload is not None:
        alt_b = MarketDataBundle.from_json(alt_upload.read().decode("utf-8"))
        if (alt_b.M, alt_b.K) == (bundle.M, bundle.K):
            alternatives["uploaded Σ"] = alt_b
        else:
            st.error("Alternative bundle grid does not match — ignored.")
    if alternatives:
        with st.spinner("Replaying the variance test in every regime…"):
            stab = stable_cut(bundle, alternatives, alpha=float(alpha),
                              kappa=float(kappa), epsilon=float(eps_cut),
                              dendrogram=dendro)
        if stab["undone"]:
            st.warning(
                f"{stab['undone']} fusion(s) unstable under at least one alternative "
                f"covariance — undone. Retained: {stab['n_merges']} merges, "
                f"{stab['evaluation'].scheme.n_sets} sets."
            )
        else:
            st.success("Every fusion of the cut survives all regimes — the structure is stable.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "regime": r["regime"],
                        "R²": round(r["r2"], 4),
                        "TE²/B": round(r["te2"] / r["budget"], 3) if r["budget"] > 0 else None,
                        "variance test": "PASS" if r["passes_variance"] else "FAIL",
                    }
                    for r in stab["report"]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.session_state["stability_pack"] = {
            "epsilon": float(eps_cut),
            "undone": stab["undone"],
            "n_merges": stab["n_merges"],
            "regimes": stab["report"],
            "labels": stab["labels"].tolist(),
        }

# =========================================================================== #
# TWO-LAYER (companion methodology note)
# =========================================================================== #
with tab_twolayer:
    st.markdown(
        "### The two-layer methodology — underlying (structure) × book (evaluation)\n"
        "**Layer 1, per underlying (no book):** the smile of a tranche moves through "
        "three modes — level, slope ΔS, curvature ΔC — plus an idiosyncratic noise "
        "(Déf. 1). The model generates the disagreement between any two points "
        "(Théorème 1) and the correlation curve ρ(x) = s₀/sₓ (Prop. 2); the dendrogram "
        "of contiguous strike groups is the **map of possible fusions**. "
        "**Layer 2, per book:** the exact residual of netting a group on its pivot is "
        "carried by the *signed* aggregates RR_p and FLY_p (Théorème 2) — the net level "
        "always nets. The barycenter pivot x_p = RR₀/m kills RR exactly (sec. 6.3), and "
        "the decisions are never all-or-nothing: collapse / extraction / scission "
        "(N2–N4). The exact computation can **authorise what the layer-1 majorant "
        "Σ|ν_j|d_jp wrongly refused** — it stops double-counting what cancels."
    )

    tl1, tl2 = st.columns([1, 2.2])
    with tl1:
        tranche_idx = st.selectbox(
            "Tranche (one maturity line of the vega matrix)",
            list(range(bundle.M)), format_func=lambda i: bundle.tenors[i],
        )
        st.markdown("**Layer-1 model** (A1 — from daily regressions; editable):")
        s0_tl = st.number_input("s₀ — level shock sd", value=1.0, min_value=0.0, step=0.1)
        ss_tl = st.number_input("σ_S — slope (skew) shock sd", value=2.0, min_value=0.0, step=0.1)
        sc_tl = st.number_input("σ_C — curvature shock sd", value=10.0, min_value=0.0, step=0.5)
        se_tl = st.number_input("σ_ε — idiosyncratic sd", value=0.1, min_value=0.0, step=0.01)
        smodel = SmileModel(s0=float(s0_tl), sigma_s=float(ss_tl),
                            sigma_c=float(sc_tl), sigma_eps=float(se_tl))
        x_tl = np.asarray(bundle.strikes, dtype=float) - 1.0
        nu_tl = bundle.vega[int(tranche_idx)]
        s_tl = smodel.point_uncertainty(x_tl)
        dist_tl = model_distance(x_tl, smodel)
        merges_tl = tranche_dendrogram(x_tl, dist_tl, s_tl)
        h_def = float(np.median([m.height for m in merges_tl])) if merges_tl else 0.5
        h_top = max((m.height for m in merges_tl), default=1.0)
        eps_tl = st.slider("cut height (disagreement d)", 0.0,
                           float(np.ceil(h_top * 105) / 100), h_def, 0.01)
    with tl2:
        cda, cdb = st.columns(2)
        with cda:
            st.plotly_chart(
                charts.heatmap(
                    dist_tl, bundle.strikes, bundle.strikes,
                    "Disagreement matrix D = (d_ij) — Théorème 1, no book",
                    colorscale="Plasma", colorbar_title="d", height=360,
                ),
                width="stretch",
            )
        with cdb:
            x_dense = np.linspace(x_tl.min(), x_tl.max(), 200)
            st.plotly_chart(
                charts.correlation_curve_figure(
                    x_dense, generated_correlation(x_dense, smodel),
                    x_tl, generated_correlation(x_tl, smodel), height=360,
                ),
                width="stretch",
            )
        st.plotly_chart(
            charts.dendrogram_figure(
                merges_tl, float(eps_tl),
                sum(1 for m in merges_tl if m.height <= eps_tl), height=330,
            ),
            width="stretch",
        )

    groups_tl = cut_tranche(merges_tl, bundle.K, float(eps_tl))
    res_tl = evaluate_book(groups_tl, nu_tl, x_tl, smodel,
                           alpha=float(alpha), kappa=float(kappa))
    st.markdown(f"#### Layer 2 — book decisions on tranche **{bundle.tenors[int(tranche_idx)]}** (N1–N4)")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Groups (layer-1 cut)", f"{len(groups_tl)}",
              delta=f"from {bundle.K} strikes")
    t2.metric("TE² exact vs majorant²",
              f"{res_tl.te2:,.0f}",
              delta=f"majorant {res_tl.te2_majorant:,.0f}", delta_color="off")
    t3.metric("Budget (1−α)·Var(ΔΠ)", f"{res_tl.budget:,.0f}",
              delta="PASS" if res_tl.passes_variance else "FAIL", delta_color="off")
    t4.metric("AVA (tranche)", eur(res_tl.ava), delta=f"-{res_tl.ava_saving_pct:.0%} vs add-up")
    t5.markdown(
        f"**Variance test** {verdict_badge(res_tl.passes_variance)}<br><br>"
        f"**Floor** {verdict_badge(res_tl.passes_floor)}",
        unsafe_allow_html=True,
    )
    if res_tl.te2_majorant > res_tl.budget >= res_tl.te2:
        st.success(
            "The layer-1 majorant alone would have **refused** this netting "
            f"({res_tl.te2_majorant:,.0f} > {res_tl.budget:,.0f}); the exact signed "
            "computation authorises it — the compensations between groups are real "
            "(sec. 5: the correlation measures the disagreement, the book decides "
            "whether it hurts)."
        )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "group (strikes)": ", ".join(str(bundle.strikes[j]) for j in d.members),
                    "pivot": bundle.strikes[d.pivot_index],
                    "m (net level)": round(d.m, 1),
                    "RR_p": round(d.rr, 3),
                    "FLY_p": round(d.fly, 3),
                    "Var(R) exact (Th. 2)": round(d.var_exact, 2),
                    "majorant Σ|ν|d": round(d.majorant_sd, 2),
                    "decision": d.decision,
                    "AVA": round(d.ava, 1),
                }
                for d in res_tl.decisions
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Pivots are the quoted nodes nearest the vega barycenter RR₀/m — the choice "
        "that annihilates the slope term (sec. 6.3). Extraction keeps the net level "
        "netted and provisions RR/FLY in add-up with their own uncertainties; only the "
        "genuinely unjustified netting is given up."
    )

# =========================================================================== #
# SCENARIO LAB
# =========================================================================== #
with tab_scenario:
    st.markdown(
        "### Score any netting scheme you want to test\n"
        "A scenario is an **(M × K) integer label matrix** over the tenor × strike grid — "
        "cells sharing a label form one netting set. Edit the grid, upload a matrix, or "
        "plug a `ScenarioSource`; the engine returns the **variance score R²**, the EBA "
        "verdict and the AVA impact."
    )

    preset_names = {
        "Per tenor (collapse the smile)": "per_tenor",
        "Per strike column": "per_strike",
        "Quadrants (short/long × low/high)": "quadrants",
        "Single global set": "global",
        "Singletons (no netting)": "singletons",
        "Optimal (from the optimiser)": "optimal",
    }
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        chosen_preset = st.selectbox("Start from", list(preset_names.keys()), index=0)
    with cc2:
        scen_upload = st.file_uploader(
            "…or upload a scenario JSON  {\"labels\": [[…]]}", type=["json"], key="scen_up"
        )

    if scen_upload is not None:
        labels_init = np.asarray(json.loads(scen_upload.read().decode("utf-8"))["labels"], dtype=int)
    elif preset_names[chosen_preset] == "optimal" and "optimal_labels" in st.session_state:
        labels_init = np.asarray(st.session_state["optimal_labels"], dtype=int)
    elif preset_names[chosen_preset] == "optimal":
        st.info("Run the optimiser first — falling back to per-tenor.")
        labels_init = preset_labels("per_tenor", bundle.M, bundle.K)
    else:
        labels_init = preset_labels(preset_names[chosen_preset], bundle.M, bundle.K)

    if labels_init.shape != (bundle.M, bundle.K):
        st.error(f"Scenario shape {labels_init.shape} does not match the grid ({bundle.M}, {bundle.K}).")
        st.stop()

    st.markdown("**Edit the set labels** (integers — same label = same netting set):")
    df_labels = pd.DataFrame(labels_init, index=bundle.tenors, columns=[str(s) for s in bundle.strikes])
    edited = st.data_editor(df_labels, width="stretch", key=f"editor_{chosen_preset}_{scen_upload is not None}")
    try:
        labels = edited.to_numpy(dtype=int)
    except (TypeError, ValueError):
        st.error("Labels must be integers.")
        st.stop()

    report = score_scenario(bundle, labels, alpha=float(alpha), kappa=float(kappa), weighting=weighting)
    ev_s = report.evaluation

    g1, g2 = st.columns([1, 1.4])
    with g1:
        st.plotly_chart(charts.r2_gauge(report.variance_score, alpha), width="stretch")
        ok = ev_s.admissible
        st.markdown(
            f"<div style='text-align:center'>{verdict_badge(ok, '✅ ' + report.verdict, '❌ ' + report.verdict)}</div>",
            unsafe_allow_html=True,
        )
        st.metric("AVA of the scenario — Def. 4", eur(ev_s.ava),
                  delta=f"-{ev_s.ava_saving_pct:.0%} vs add-up")
        st.metric("TE² vs budget", f"{ev_s.te2:,.0f} €²",
                  delta=f"budget {ev_s.budget:,.0f} €²", delta_color="off")
        st.markdown(
            f"**Variance test (6)** {verdict_badge(ev_s.passes_variance)} &nbsp;&nbsp; "
            f"**Floor (7)** {verdict_badge(ev_s.passes_floor)}",
            unsafe_allow_html=True,
        )
    with g2:
        st.plotly_chart(
            charts.partition_figure(
                NettingScheme.from_labels(labels).labels, bundle.vega, bundle.tenors,
                bundle.strikes, title="Scenario netting sets", set_stats=ev_s.set_stats,
            ),
            width="stretch",
        )

    st.download_button(
        "⬇️ Export scenario labels (JSON)",
        data=json.dumps({"labels": labels.tolist()}, indent=2),
        file_name="netting_scenario.json",
        mime="application/json",
    )
    st.session_state["scenario_eval"] = ev_s
    st.session_state["scenario_labels"] = labels.tolist()

    with st.expander("🧪 Two-node sandbox — the closed-form intuition (Théorème 2, eq. 8)"):
        s1c, s2c, s3c = st.columns(3)
        with s1c:
            nu_i_tb = st.number_input("ν_i (pivot vega)", value=1000.0, step=50.0)
            nu_j_tb = st.number_input("ν_j (netted vega)", value=-300.0, step=50.0)
        with s2c:
            s_i_tb = st.number_input("s_i", value=0.50, min_value=0.01, step=0.05)
            s_j_tb = st.number_input("s_j", value=0.55, min_value=0.01, step=0.05)
        with s3c:
            rho_tb = st.slider("ρ", -1.0, 1.0, 0.90, 0.01)
            var_choice = st.radio("Var(ΔΠ) in the RHS of (6)", ["pair variance", "portfolio variance"],
                                  horizontal=False)
        if var_choice == "portfolio variance":
            var_tb = model.var_total
        else:
            var_tb = (
                nu_i_tb ** 2 * s_i_tb ** 2 + nu_j_tb ** 2 * s_j_tb ** 2
                + 2 * nu_i_tb * nu_j_tb * rho_tb * s_i_tb * s_j_tb
            )
        cf = two_bucket(nu_i_tb, nu_j_tb, s_i_tb, s_j_tb, rho_tb, var_tb, float(alpha), float(kappa))
        r1, r2_, r3, r4 = st.columns(4)
        r1.metric("TE² — Th. 2 (i)", f"{cf['te2']:,.0f}")
        r2_.metric("ρ_min — eq. (8)", f"{cf['rho_min']:.3f}" if cf["rho_min"] > -1 else "always")
        r3.metric("AVA gain", eur(cf["ava_gain"]))
        r4.markdown("**Admissible**<br>" + verdict_badge(cf["admissible"], "YES", "NO"), unsafe_allow_html=True)
        st.plotly_chart(
            charts.two_bucket_figure(s_i_tb, s_j_tb, nu_i_tb, var_tb, float(alpha), nu_j_tb, rho_tb),
            width="stretch",
        )

# =========================================================================== #
# SPECTRAL & SMILE
# =========================================================================== #
with tab_spectral:
    st.markdown(
        "### The spectral diagnostic (sec. 6.1) — the spectrum measures, the partition nets\n"
        "Σ at the nodes is diagonalised into decorrelated factors ξ_ℓ = u_ℓᵀΔσ "
        "(Prop. 6); the book's variance decomposes exactly as Σ c_ℓ²λ_ℓ with "
        "c_ℓ = ⟨ν, u_ℓ⟩ (Prop. 7). Théorème 3: schemes whose representative shocks "
        "live in the span of the first L factors cannot beat the tail floor."
    )
    diag = spectral_diagnostic(model, alpha=float(alpha))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("K*(α) — spectral floor on #sets (Théorème 3)", f"{diag.k_star}")
    s2.metric("Top-3 modes explain", f"{diag.explained_ratio(3):.1%}", delta="of the book's variance (Prop. 7)")
    s3.metric("Inertia τ₃ — U1", f"{diag.inertia[3]:.1%}", delta="of tr(Σ) = Σs² (compressibility)")
    s4.metric("λ₁ / λ₂", f"{diag.eigenvalues[0] / max(diag.eigenvalues[1], 1e-12):.1f}×")
    st.plotly_chart(charts.spectral_figure(diag), width="stretch")

    st.markdown("#### Leading eigendirections u_ℓ of Σ on the grid — the *non-nettable directions* live in the tail")
    mode_cols = st.columns(3)
    names = ["mode 1 (level)", "mode 2 (term/skew)", "mode 3 (smile)"]
    for l, col in enumerate(mode_cols):
        with col:
            col.plotly_chart(
                charts.heatmap(
                    diag.mode(l, bundle.M, bundle.K), bundle.strikes, bundle.tenors,
                    names[l], diverging=True, height=330,
                ),
                width="stretch",
            )

    st.divider()
    rows = smile_decomposition(bundle)
    st.plotly_chart(charts.smile_decomposition_figure(rows, float(alpha)), width="stretch")
    st.caption(
        "Tranches whose R² falls below α carry material net risk-reversal / butterfly: "
        "their smile-shape exposure must stay in add-up (stage 1 of the hierarchical "
        "netting, sec. 6.2 / Théorème 4); level vegas of passing tranches proceed to stage-2 "
        "term-structure netting under ρ_mat."
    )

# =========================================================================== #
# AUDIT & EXPORT
# =========================================================================== #
with tab_audit:
    st.markdown(
        "### Art. 9(5) evidence pack\n"
        "Everything the validator and the supervisor will ask for: the retained "
        "partition, realised R², stress results and the full greedy trace — "
        "fingerprinted against the input bundle, per computation date."
    )
    which = st.radio(
        "Build the pack for", ["Optimal scheme", "Current scenario"], horizontal=True
    )
    if which == "Optimal scheme" and "optimal_eval" in st.session_state:
        ev_a = st.session_state["optimal_eval"]
        history_a = st.session_state.get("optimal_history")
    elif which == "Current scenario" and "scenario_eval" in st.session_state:
        ev_a = st.session_state["scenario_eval"]
        history_a = None
    else:
        st.info("Visit the corresponding tab first so the evaluation exists.")
        ev_a = None

    if ev_a is not None:
        rb_pack = st.session_state.get("robust_pack")
        stress_results = None
        if rb_pack is not None:
            stress_results = {
                ("base" if d == 0.0 else f"delta_{d}"): {
                    "n_sets": r.scheme.n_sets,
                    "r2": r.evaluation.r2,
                    "ava": r.evaluation.ava,
                    "passes_variance": r.evaluation.passes_variance,
                    "passes_floor": r.evaluation.passes_floor,
                }
                for d, r in rb_pack["runs"].items()
            }
        pack = build_audit_pack(
            bundle, ev_a, kappa=float(kappa),
            stress_results=stress_results,
            history=history_a,
        )
        if st.session_state.get("stability_pack") is not None:
            pack["stability_6_4"] = st.session_state["stability_pack"]
        text = audit_json(pack)
        a1, a2 = st.columns([1, 1])
        with a1:
            st.download_button(
                "⬇️ Download audit pack (JSON)", data=text,
                file_name=f"ava_netting_audit_{bundle.meta.get('asof', 'latest')}.json",
                mime="application/json", type="primary",
            )
            st.download_button(
                "⬇️ Download input bundle (JSON)", data=bundle.to_json(indent=2),
                file_name="market_data_bundle.json", mime="application/json",
            )
        with a2:
            st.metric("Input fingerprint (SHA-256)", pack["input"]["fingerprint_sha256"])
        st.json(pack, expanded=2)

st.markdown(
    "<div style='text-align:center; color:#5b6694; padding-top: 24px;'>"
    "Vega Netting Studio · EBA Prudent Valuation AVA MPU · for IPV / model-validation use — "
    "synthetic demo data, not investment advice</div>",
    unsafe_allow_html=True,
)
