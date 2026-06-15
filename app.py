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
from pathlib import Path

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
      <p>Netting des sensibilités de volatilité sous le test de variance EBA — AVA Market Price Uncertainty,
      Règlement délégué (UE) 2016/101, art. 9(5) &amp; art. 89 · cadre de production IPV</p>
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
    st.header("⚙️ Entrées")
    st.subheader("Source de données de marché")
    source_kind = st.radio(
        "Source",
        ["Book de démo synthétique", "Charger un bundle JSON"],
        help="Les banques branchent leurs propres données en implémentant "
        "`ebanetting.DataSource` ou en produisant le contrat JSON "
        "(voir l'onglet Données de marché).",
    )
    if source_kind == "Book de démo synthétique":
        book = st.selectbox(
            "Book de démo",
            ["mixed_desk", "smile_book", "calendar_book"],
            format_func={
                "mixed_desk": "Desk mixte (calendaires + RR + flies short)",
                "smile_book": "Book de smile (long ATM, short sur les ailes)",
                "calendar_book": "Book calendaire (maturités courtes vs longues)",
            }.get,
        )
        seed = st.number_input("Graine (seed)", min_value=0, max_value=10_000, value=7, step=1)
        bundle = SyntheticDataSource(book=book, seed=int(seed)).load()
    else:
        uploaded = st.file_uploader("MarketDataBundle JSON", type=["json"])
        if uploaded is None:
            st.info("En attente d'un bundle — repli sur le book de démo.")
            bundle = SyntheticDataSource().load()
        else:
            bundle = MarketDataBundle.from_json(uploaded.read().decode("utf-8"))

    issues = bundle.validate()
    if issues:
        st.error("Échec de la validation du bundle :\n\n- " + "\n- ".join(issues))
        st.stop()

    st.subheader("Étape 1 — passage aux nœuds de test (sec. 3)")
    granular_bundle = bundle
    passage_res = None
    use_passage = st.toggle(
        "Projeter le vega granulaire sur les piliers du consensus",
        value=False,
        help="Transporte le vega de la grille système vers les piliers où "
        "l'incertitude consensus (s, ρ) est réellement observable — le sandwich "
        "Ñ = Aᵀ_T N A_K de la Déf. 2. Théorème 1 : avec des poids de passage "
        "alignés sur la construction de la surface, le passage ne détruit "
        "aucune information.",
    )
    if use_passage:
        default_t = sorted(set(list(range(0, bundle.M, 2)) + [bundle.M - 1]))
        default_k = sorted(set(list(range(0, bundle.K, 2)) + [bundle.K - 1]))
        pillar_t = st.multiselect(
            "Piliers de maturité", options=list(range(bundle.M)), default=default_t,
            format_func=lambda i: bundle.tenors[i],
        )
        pillar_k = st.multiselect(
            "Piliers de strike", options=list(range(bundle.K)), default=default_k,
            format_func=lambda i: str(bundle.strikes[i]),
        )
        convention = st.selectbox(
            "Convention de passage (sec. 3.2)",
            ["interp", "equal", "quadrant"],
            format_func={
                "interp": "Poids d'interpolation (= poids de la surface, TE = 0 — Th. 1)",
                "equal": "Équipondérée (½/½ sur les piliers encadrants)",
                "quadrant": "Quadrant (pilier le plus proche)",
            }.get,
        )
        if pillar_t and pillar_k:
            passage_res = project_bundle(bundle, pillar_t, pillar_k, convention)
            bundle = passage_res.node_bundle
        else:
            st.warning("Sélectionnez au moins un pilier par axe — passage ignoré.")

    st.subheader("Paramètres réglementaires")
    alpha = st.slider(
        "α — seuil du test de variance (R² ≥ α)", 0.80, 0.99, 0.90, 0.01,
        help="Plage usuelle 0,90–0,95, cohérente avec le niveau de confiance de 90 % des RTS.",
    )
    kappa = st.number_input(
        "κ — multiplicateur du quantile d'incertitude", value=round(KAPPA_90, 4),
        min_value=0.5, max_value=3.0, step=0.01,
        help="z₀.₉₀ ≈ 1,2816 sous l'hypothèse gaussienne (art. 89), ou un "
        "facteur calibré empiriquement.",
    )
    weighting = st.selectbox(
        "Choc représentatif Δσ̃ₖ",
        ["pivot", "vega", "equal"],
        format_func={
            "pivot": "Bucket pivot (max |ν·s| de l'ensemble)",
            "vega": "Moyenne pondérée par |vega|",
            "equal": "Moyenne équipondérée",
        }.get,
    )
    grid_word = "nœuds de test" if passage_res is not None else "buckets"
    st.caption(
        f"Grille : **{bundle.M} maturités × {bundle.K} strikes = {bundle.n} {grid_word}** · "
        f"{bundle.meta.get('underlying', '—')} · au {bundle.meta.get('asof', '—')}"
    )

model = UncertaintyModel(bundle=bundle, kappa=float(kappa))
bundle_json = bundle.to_json(sort_keys=True)

# --------------------------------------------------------------------------- #
# Per-tab documentation (rendered in the Documentation tab)
# --------------------------------------------------------------------------- #
DOC_DIR = Path(__file__).parent / "docs"
DOC_PAGES = {
    "🏛️ Théorie": "readme_theorie.md",
    "📊 Données de marché": "readme_donnees_marche.md",
    "🔁 Passage": "readme_passage.md",
    "🧠 Netting optimal": "readme_netting_optimal.md",
    "🌳 Structure & Stabilité": "readme_structure_stabilite.md",
    "🧬 Deux couches": "readme_deux_couches.md",
    "🎯 Labo scénarios": "readme_labo_scenarios.md",
    "🔬 Spectral & Smile": "readme_spectral_smile.md",
    "📋 Audit & Export": "readme_audit_export.md",
}


@st.cache_data(show_spinner=False)
def load_doc(filename: str) -> str:
    return (DOC_DIR / filename).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
(tab_theory, tab_data, tab_passage, tab_optimal, tab_structure, tab_twolayer,
 tab_scenario, tab_spectral, tab_audit, tab_docs) = st.tabs(
    ["🏛️ Théorie", "📊 Données de marché", "🔁 Passage", "🧠 Netting optimal",
     "🌳 Structure & Stabilité", "🧬 Deux couches", "🎯 Labo scénarios",
     "🔬 Spectral & Smile", "📋 Audit & Export", "📖 Documentation"]
)

# =========================================================================== #
# THEORY
# =========================================================================== #
with tab_theory:
    st.markdown("## Le cadre complet, de la réglementation à l'algorithme")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(
            """
### 1 · Cadre réglementaire
Le Règlement délégué (UE) **2016/101** (RTS EBA sur la valorisation prudente) impose, sous
l'*approche principale* (core approach), des **AVA** calculées par *exposition de
valorisation* à un **niveau de confiance de 90 %** (art. 89). L'article **9(5)** et la
Q&A EBA associée autorisent le calcul de l'AVA **Market Price Uncertainty** sur une
**exposition nettée** plutôt que sur la somme brute — *à condition que l'établissement
démontre que les positions se compensent réellement face à l'incertitude du paramètre*.
Pour le vega, la question devient : **à quelle granularité de la surface de volatilité
(maturité × strike) peut-on sommer les sensibilités signées avant d'appliquer l'écart
d'incertitude ?**

### 2 · Modèle d'incertitude
La surface est discrétisée en $n = M \\times K$ buckets. Avec $\\nu_i = \\partial V / \\partial \\sigma_i$
les vegas par bucket et $\\Delta\\sigma_i$ l'incertitude mid centrée
($\\mathrm{sd}=s_i$, corrélation $\\rho$, $\\Sigma = D\\rho D$), l'incertitude de P&L de
valorisation au premier ordre est
            """
        )
        st.latex(r"\Delta\Pi=\nu^{\top}\Delta\sigma,\qquad \operatorname{Var}(\Delta\Pi)=\nu^{\top}\Sigma\,\nu \tag{1}")
        st.markdown("**Les deux extrêmes de l'AVA** (aucun netting vs diversification complète) :")
        st.latex(r"\mathrm{AVA}_{\text{brut}}=\kappa\sum_{i=1}^{n}\lvert\nu_i\rvert s_i \tag{2}")
        st.latex(r"\mathrm{AVA}_{\text{full}}=\kappa\sqrt{\nu^{\top}\Sigma\,\nu}\;\le\;\mathrm{AVA}_{\text{brut}} \tag{3}")
        st.markdown(
            """
L'écart (2)−(3) est le **bénéfice de netting théorique maximal**. Le régulateur
n'accorde pas (3) par défaut : la structure de dépendance doit être démontrée robuste,
d'où le test de variance sur des **agrégations partielles et interprétables**.

### 3 · Étape 1 : le passage aux nœuds de test (sec. 3, Théorème 1)
L'incertitude n'est observable qu'aux piliers du consensus. Le vega granulaire y est
transporté par deux **matrices de passage** (lignes positives sommant à 1,
Déf. 1) et le **sandwich** 2-D — un produit matriciel ordinaire de chaque côté :
            """
        )
        st.latex(r"\tilde N = A_T^{\top}\, N\, A_K \in \mathbb{R}^{q_T\times q_K} \quad\text{(Def. 2)}")
        st.markdown(
            """
**Théorème 1 (cohérence passage–surface)** : lorsque les poids de passage égalent les
**poids d'interpolation de la construction de la surface de pricing** ($A=B$), le passage
ne détruit aucune information au premier ordre — $\\mathrm{TE}_{\\text{passage}}=0$. Toute
autre convention (quadrant, équipondérée) crée une erreur quantifiable (éq. 5) qui
consomme le budget du test de variance avant tout netting. Conséquence (Remarque 3) : les
matrices de passage ne sont **pas une variable d'optimisation** — elles sont fixées par la
construction de la surface ; seule la partition en aval est optimisée.

**Sec. 3.5 — quand la grille de chocs diffère de la grille de sensis.** Trois régimes, une
recette : chocs *plus grossiers* que les sensibilités (piliers Totem) → contracter le vega
(sandwich, Théorème 1) ; chocs *plus fins* (variations quotidiennes de la surface système
complète, p. ex. 350×100 contre des piliers de paramétrisation 10×10) → **restriction sans
perte** (Propriété 5) : des matrices de sélection 0/1 lisent la surface fine aux piliers,
$\\Delta\\sigma^{\\text{pil}} = R_T\\,\\Delta\\sigma^{\\text{fin}}\\,R_K^{\\top}$, exact parce que
l'interpolation reproduit ses nœuds ($R\\,B = I$) ; grilles identiques → identité. La
composante hors grille $\\Delta\\sigma^{\\text{fin}} - B_T(R_T\\Delta\\sigma^{\\text{fin}}R_K^\\top)B_K^\\top$
(Remarque 4) est mesurable quotidiennement : sa taille jauge l'adéquation du jeu de
piliers. Tout l'aval opère sur la **grille pivot**, indifférent à la famille de chocs qui
l'a alimentée.

### 3b · Incertitude 2-D : l'hypothèse de découplage (annexe)
Avec des corrélations par axe, la corrélation 2-D se découple **terme à terme**,
$\\rho_{(a,b),(a',b')}=\\rho^{\\text{mat}}_{aa'}\\,\\rho^{\\text{strike}}_{bb'}$ — seulement
$\\binom{q_T}{2}+\\binom{q_K}{2}$ paramètres, chacun interprétable et stressable. **Aucun
produit de Kronecker / tensoriel n'est jamais assemblé** : chaque variance s'évalue sous
forme matricielle avec des produits matriciels usuels et une réduction par somme finale
(Propriété 1),
            """
        )
        st.latex(
            r"\operatorname{Var}\langle V,\Delta\sigma\rangle"
            r"=\big\langle V\!\circ\! s,\ \rho^{\text{mat}}\,(V\!\circ\! s)\,\rho^{\text{strike}}\big\rangle"
        )
        st.markdown(
            """
Les partitions admissibles sont restreintes aux **rectangles contigus**
$\\mathcal N_r = T_r\\times S_r$ sur la grille de nœuds (sec. 7.3) — interprétables et
documentables au titre de l'art. 9(5). **Attention à la grille en strike absolu**
(sec. 8) : des strikes fixes dérivent en moneyness avec le spot, gonflant les
corrélations de l'axe strike et **sur-justifiant le netting** — un biais non
conservateur. Estimez $s,\\rho$ en coordonnées de moneyness $x=K/F(T)$ (ou de delta) et
re-mappez les vegas via le jacobien d'interpolation ; à défaut, stressez
$\\rho^{\\text{strike}}$ à la baisse.
            """
        )
    with c2:
        st.markdown(
            """
### 4 · Le netting comme opérateur d'agrégation (Déf. 4)
Un **schéma de netting** est une partition $\\mathcal P=\\{\\mathcal G_1,\\dots,\\mathcal G_S\\}$
des nœuds de test. L'ensemble $r$ porte l'exposition nettée $m_r=\\sum_{(a,b)\\in\\mathcal G_r}\\tilde N_{ab}$
et un choc représentatif $\\Delta\\tilde\\sigma_r=\\langle W_r,\\Delta\\sigma\\rangle$
avec $W_r$ une grille de poids stochastique en ligne. Le P&L proxy et l'AVA retenue
(conservatrice, add-up entre ensembles) sont
            """
        )
        st.latex(r"\widehat{\Delta\Pi}_{\mathcal P}=\sum_r m_r\,\Delta\tilde\sigma_r,\qquad \mathrm{AVA}(\mathcal P)=\kappa\sum_{r}\lvert m_r\rvert\,\tilde s_r \quad\text{(Def. 4)}")
        st.markdown("### 5 · Le test de variance (Déf. 3)")
        st.latex(
            r"\mathrm{TE}^2(\mathcal P)=\operatorname{Var}\!\big(\Delta\Pi-\widehat{\Delta\Pi}_{\mathcal P}\big)"
            r"=\operatorname{Var}\Big\langle \tilde N-\textstyle\sum_r m_r W_r,\ \Delta\sigma\Big\rangle\;\le\;(1-\alpha)\operatorname{Var}(\Delta\Pi) \tag{6}"
        )
        st.markdown(
            "de façon équivalente **R² ≥ α** (α ∈ [0,90 ; 0,95]) — l'analogue, en valorisation "
            "prudente, des tests d'attribution de P&L FRTB. (6) contrôle la **fidélité** ; le "
            "**plancher de conservatisme** contrôle le niveau :"
        )
        st.latex(r"\mathrm{AVA}(\mathcal P)\;\ge\;\kappa\sqrt{\operatorname{Var}(\Delta\Pi)} \tag{7}")
        st.markdown("### 6 · Netting optimal")
        st.latex(
            r"\mathcal P^{\star}=\arg\min_{\mathcal P}\;\kappa\sum_{r}\Big|\sum_{(a,b)\in\mathcal G_r}\tilde N_{ab}\Big|\,\tilde s_r"
            r"\quad\text{s.c. } \mathrm{TE}^2(\mathcal P)\le(1-\alpha)\operatorname{Var}(\Delta\Pi) \text{ and } (7) \tag{10}"
        )
        st.markdown(
            "Combinatoire (nombres de Bell), NP-difficile → construction gloutonne contrôlée. "
            "**Forme fermée à deux nœuds** (Théorème 2 — netter $j$ sur le pivot $i$) :"
        )
        st.latex(r"\mathrm{TE}^2=\nu_j^2\big(s_i^2+s_j^2-2\rho s_i s_j\big) \quad\text{(Th. 2 i)}")
        st.latex(
            r"\text{admissible}\iff \rho\;\ge\;\frac{s_i^2+s_j^2}{2 s_i s_j}"
            r"-\frac{(1-\alpha)\operatorname{Var}(\Delta\Pi)}{2\,\nu_j^2\,s_i s_j} \tag{8}"
        )
        st.markdown(
            """
*Lecture économique* : le netting exige **corrélation élevée et incertitudes homogènes**
(des points de smile adjacents se nettent ; un ATM 1M contre une aile 10Y, non), et plus
le vega netté est grand relativement au risque total, plus le seuil est exigeant — **on
ne cache pas un gros short derrière un proxy approximatif**. Les fusions de même signe
sont admissibles mais inutiles : le netting optimal vise des ensembles de signes opposés
fortement corrélés.
            """
        )

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 6.1 · Le diagnostic spectral — Théorème 3")
        st.latex(
            r"\Sigma=\sum_\ell \lambda_\ell u_\ell u_\ell^{\top},\qquad"
            r"\operatorname{Var}(\Delta\Pi)=\sum_\ell c_\ell^2\lambda_\ell,\quad c_\ell=\langle\nu,u_\ell\rangle"
        )
        st.markdown(
            """
Les facteurs $\\xi_\\ell=u_\\ell^{\\top}\\Delta\\sigma$ sont décorrélés, de variances
$\\lambda_\\ell$ (Prop. 6) ; la carte $\\{c_\\ell^2\\lambda_\\ell\\}$ (Prop. 7) est la **carte
de risque** du book. **Théorème 3 (plancher spectral)** : pour tout schéma dont les chocs
représentatifs vivent dans $\\mathrm{Vect}(u_1,\\dots,u_L)$ — la description réaliste des
schémas grossiers à poids lisses — $\\mathrm{TE}^2\\ge\\sum_{\\ell>L}c_\\ell^2\\lambda_\\ell$ : la
variance de queue du book est incompressible. (L'hypothèse est nécessaire : avec $w=\\nu$
un seul ensemble réplique $\\Delta\\Pi$ exactement — Remarque 5.) **Quatre usages** (6.1.4) :
U1 le ratio d'inertie $\\tau_L$ — compressibilité de l'incertitude ; U2 la carte de risque —
*pourquoi* un test échoue et où raffiner ; U3 le **nettoyage spectral** d'un
$\\hat\\Sigma$ bruité — aplatir la queue des valeurs propres à sa moyenne, avec le garde-fou
de conservatisme ; U4 la stabilité des sous-espaces dominants entre fenêtres d'estimation.
**La limite** : un facteur n'est *pas* un ensemble de netting — des poids signés denses ne
sont pas une exposition de valorisation ; le spectre mesure, la partition nette.

### 6.2 · Décomposition du smile — Théorème 4
Sous le **modèle de déformation à deux modes** (Déf. 5, éq. 9),
$\\Delta\\sigma_{a,j}=\\Delta\\sigma_{a,\\text{ATM}}+(x_j{-}x_0)\\Delta S_a+(x_j{-}x_0)^2\\Delta C_a+\\varepsilon_{a,j}$
— empirique, testable en régressant les mouvements quotidiens par strike sur
ATM/pente/courbure — le résidu de l'écrasement de la tranche $a$ sur son pivot ATM est
**exactement**
$R_a=\\mathrm{RR}_a\\,\\Delta S_a+\\mathrm{FLY}_a\\,\\Delta C_a+\\sum_j\\nu_j\\varepsilon_{a,j}$
(Théorème 4) : le **niveau net $m_a$ se nette parfaitement quelle que soit sa taille**, et
le go/no-go est en forme fermée,
$\\operatorname{Var}(R_a)=\\mathrm{RR}_a^2\\operatorname{Var}\\Delta S+\\mathrm{FLY}_a^2\\operatorname{Var}\\Delta C+2\\,\\mathrm{RR}_a\\mathrm{FLY}_a\\operatorname{Cov}+\\sigma_\\varepsilon^2\\sum_j\\nu_j^2$
— non pas « corrélation insuffisante » mais « le book porte tant de RR net sur cette
tranche » (S1). Quand l'écrasement échoue, le raffinement conserve le niveau netté et
extrait $\\mathrm{RR}_a$, $\\mathrm{FLY}_a$ en **add-up** avec leurs propres incertitudes
consensus (S2) : seul le netting réellement injustifié est abandonné. RR/FLY sont les
coordonnées du book dans la base standard des stratégies de smile (Remarque 6).
            """
        )
    with c4:
        st.markdown("### 7.3 · Algorithme glouton agglomératif sous budget de variance")
        st.markdown(
            """
Avec le budget $B=(1-\\alpha)\\operatorname{Var}(\\Delta\\Pi)$ :

1. **Init** : singletons, $\\mathrm{TE}^2=0$.
2. **Itérer** : pour chaque paire fusible d'ensembles rectangulaires adjacents, calculer
   la réduction d'AVA $g$ et le coût en variance résiduelle $c$ ; fusionner la paire
   maximisant $g/c$ (fusions à coût nul d'abord) tant que $\\mathrm{TE}^2+c\\le B$.
3. **Arrêt** quand plus aucune fusion ne tient dans le budget ; vérifier le plancher (7),
   sinon défaire la fusion la moins efficace.
4. **Stabilité** : rejouer sous fenêtres d'estimation glissantes et **stress de
   corrélation adverse** $\\rho\\to\\max(\\rho-\\delta,-1)$, $\\delta\\sim0{,}1\\!-\\!0{,}2$ ; ne
   retenir que les fusions robustes à travers les régimes — c'est ce qui rend le schéma
   défendable en revue de modèle.

### 7.4 · Familles de chocs et stabilité
Un formalisme, plusieurs covariances : $\\Sigma^{\\text{totem}}$ (niveau d'AVA + test
officiel), $\\Sigma^{\\text{bid-ask}}$ (niveau de coût de débouclage), $\\Sigma^{\\text{daily}}$
(mouvements quotidiens de la surface restreints aux piliers, Prop. 5 — **pas** un niveau
prudentiel, mais l'estimateur de $\\rho$ le plus dense et le **test de stabilité** :
rejouer le test de variance de la *même* partition sous fenêtres et stress de
$\\Sigma^{\\text{daily}}$ ; toute fusion qui échoue dans un régime est instable et défaite).
Asymétrie réglementaire : refuser une fusion valide est permis, conserver une fusion
invalidée ne l'est pas.

### 7.5 · Architecture découplée — structure ⟂ niveau
**Run 1 (structure, sans portefeuille)** : clustering hiérarchique des nœuds sur le
**risque de base** $d_{ij}=\\sqrt{s_i^2+s_j^2-2\\rho_{ij}s_is_j}$ (le coût du Th. 2 (i) est
$\\nu_j^2 d_{ij}^2$) → un **dendrogramme** ; un schéma est une coupe à la hauteur $\\varepsilon$.
**Propriété 8** : $d_{j,\\text{pivot}}\\le\\varepsilon s_j$ par nœud implique, pour *tout*
book, $\\mathrm{TE}\\le\\varepsilon\\sum_j|\\nu_j|s_j=\\varepsilon\\,\\mathrm{AVA}_{\\text{brut}}/\\kappa$.
**Run 2 (niveau)** : évaluer $\\mathrm{AVA}=\\kappa\\sum_r|m_r|\\tilde s_r$ sur la coupe avec
le $s$ Totem du jour — la seule liberté restante est la hauteur de coupe scalaire. Le
contrôle obligatoire par book est l'unique ratio en forme fermée $\\mathrm{TE}^2/\\mathrm{Var}(\\Delta\\Pi)$
(abaisser la coupe s'il échoue). Une partition qui bouge avec le book est un signal
d'alerte en validation ; ici la structure est lente et justifiée, le niveau rapide et
par book.

### 8 · Points de mise en œuvre IPV
- **$s_i$** : dispersion inter-contributeurs Totem (écart-type ou intervalles
  interquantiles remis à l'échelle 90 %), sinon fourchettes brokers / proxys de
  liquidité. Peu de contributeurs ⇒ **gonfler** $s_i$, pas le lisser.
- **$\\rho$** : à estimer sur les **variations** des marks de consensus, jamais les
  niveaux (la cointégration gonfle les corrélations de niveaux et sur-justifie le
  netting). Quand l'historique Totem est court, les **variations quotidiennes de la
  surface système restreintes aux piliers (Prop. 5)** donnent un estimateur plus dense —
  retenir le **moins favorable au netting** des deux, et croiser avec le test de
  stabilité (sec. 7.4). Shrinkage recommandé dans tous les cas.
- **Asymétrie réglementaire** : le test protège contre la *sous*-estimation — dans le
  doute, le stress de corrélation doit être **adverse au netting**.
- **Documentation** : conserver {partition retenue, R² réalisé, frontière, résultats de
  stress} par date de calcul — la preuve art. 9(5) (voir l'onglet Audit).
- **Non-linéarité** : (1) est au premier ordre ; pour les books à volga/vanna matériels
  (cliquets, barrières), valider que les termes de second ordre ne réordonnent pas les
  ensembles — sinon netter sur des scénarios complets plutôt que sur des sensibilités.
            """
        )

# =========================================================================== #
# MARKET DATA
# =========================================================================== #
with tab_data:
    meta = bundle.meta
    st.markdown(
        f"**{meta.get('underlying', 'Portefeuille')}** · {meta.get('desk', '')} · "
        f"au **{meta.get('asof', '—')}** · vega en *{meta.get('vega_unit', 'EUR/pt de vol')}*, "
        f"incertitude en *{meta.get('uncertainty_unit', 'pts de vol')}*"
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Var(ΔΠ) — Propriété 1", eur(np.sqrt(model.var_total)) + "²" if False else f"{model.var_total:,.0f} €²")
    k2.metric("AVA add-up — éq. (2)", eur(model.ava_brut))
    k3.metric("AVA diversification complète — éq. (3)", eur(model.ava_full))
    k4.metric(
        "Bénéfice de netting max", eur(model.ava_brut - model.ava_full),
        delta=f"-{(1 - model.ava_full / model.ava_brut):.0%} de l'add-up" if model.ava_brut > 0 else None,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.heatmap(bundle.vega, bundle.strikes, bundle.tenors,
                           "Surface de vega N (signée, EUR / pt de vol)", diverging=True,
                           colorbar_title="ν"),
            width="stretch",
        )
        if bundle.corr_mat is not None:
            st.plotly_chart(
                charts.heatmap(bundle.corr_mat, bundle.tenors, bundle.tenors,
                               "ρ_mat — corrélation des Δσ entre maturités", colorscale="Sunset",
                               colorbar_title="ρ"),
                width="stretch",
            )
    with c2:
        st.plotly_chart(
            charts.heatmap(bundle.s, bundle.strikes, bundle.tenors,
                           "Incertitude s (dispersion du consensus, pts de vol)",
                           colorscale="Plasma", colorbar_title="s"),
            width="stretch",
        )
        if bundle.corr_strike is not None:
            st.plotly_chart(
                charts.heatmap(bundle.corr_strike, bundle.strikes, bundle.strikes,
                               "ρ_strike — corrélation des Δσ entre moneyness", colorscale="Sunset",
                               colorbar_title="ρ"),
                width="stretch",
            )
        elif bundle.corr_full is not None:
            st.plotly_chart(
                charts.heatmap(bundle.corr_full, list(range(bundle.n)), list(range(bundle.n)),
                               "ρ — corrélation complète des buckets", colorscale="Sunset"),
                width="stretch",
            )

    with st.expander("📦 Contrat de données — branchez votre propre pipe ici", expanded=False):
        st.markdown(
            """
L'application est **agnostique de la banque**. Deux points d'intégration, tous deux dans
`ebanetting/datasource.py` :

1. **`DataSource`** (abstraite) : implémentez `load() -> MarketDataBundle` contre vos
   sources golden — vegas du système de risque, dispersions Totem/Markit, fourchettes
   brokers. L'implémentation de référence `JSONBundleSource` lit le contrat JSON ci-dessous.
2. **`ScenarioSource`** (abstraite) : implémentez `load() -> matrice de labels entiers (M,K)`
   pour injecter des schémas de netting candidats (p. ex. la convention actuelle du desk)
   dans le Labo scénarios.

**Contrat JSON** (un document par date / sous-jacent / exposition de valorisation) — strikes
en moneyness K/F comme recommandé en sec. 8 ; corrélations soit découplées
(`corr_mat` × `corr_strike`, annexe de la note — combinées terme à terme,
jamais en matrice de Kronecker), soit complètes (`corr_full`, prioritaire) :
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
            "⬇️ Télécharger le bundle courant comme gabarit JSON",
            data=bundle.to_json(indent=2),
            file_name="market_data_bundle.json",
            mime="application/json",
        )

# =========================================================================== #
# PASSAGE
# =========================================================================== #
with tab_passage:
    st.markdown(
        "### Étape 1 — passage de la grille système aux nœuds de test (sec. 3)\n"
        "L'incertitude consensus ne vit qu'à quelques piliers ; le vega granulaire "
        "y est transporté par le **sandwich des matrices de passage** "
        "Ñ = Aᵀ_T·N·A_K (Déf. 2) — deux produits matriciels ordinaires, un par axe. "
        "**Théorème 1** : aligner les poids de passage sur les poids d'interpolation "
        "de la construction de la surface rend le passage sans perte "
        "(TE_passage = 0) ; les conventions quadrant / équipondérée créent une "
        "tracking error quantifiable (éq. 5) imputée au budget de variance."
    )
    if passage_res is None:
        st.info(
            "Activez **« Projeter le vega granulaire sur les piliers du consensus »** "
            "dans la barre latérale pour déclencher le passage. Le reste de "
            "l'application tourne alors sur la grille de nœuds projetée, comme le "
            "prescrit la note."
        )
    else:
        pr = passage_res
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            "Nœuds", f"{bundle.M} × {bundle.K} = {bundle.n}",
            delta=f"depuis {granular_bundle.M} × {granular_bundle.K} = {granular_bundle.n} buckets",
        )
        p2.metric(
            "Conservation du vega — Prop. 3",
            "OK" if pr.conserves_vega else "ROMPUE",
            delta=f"Σ N = {pr.vega_total_granular:,.0f} → Σ Ñ = {pr.vega_total_nodes:,.0f}",
            delta_color="off",
        )
        budget_passage = (1.0 - float(alpha)) * pr.var_reference
        p3.metric(
            "TE²_passage — éq. (5)",
            f"{pr.te2_passage:,.0f} €²",
            delta=(
                f"{pr.te2_passage / budget_passage:.1%} du budget"
                if budget_passage > 0 else "—"
            ),
            delta_color="off",
        )
        p4.metric("Convention", pr.convention)
        if pr.convention != "interp":
            st.warning(
                "Passage non-interp : la tracking error ci-dessus consomme le "
                "budget du test de variance **avant tout netting** (Théorème 1 / "
                "Remarque 1). Documentez-la et déduisez-la (sec. 8)."
            )

        cpa, cpb = st.columns(2)
        with cpa:
            st.plotly_chart(
                charts.heatmap(
                    granular_bundle.vega, granular_bundle.strikes, granular_bundle.tenors,
                    "Vega granulaire N (grille système)", diverging=True, colorbar_title="ν",
                ),
                width="stretch",
            )
        with cpb:
            st.plotly_chart(
                charts.heatmap(
                    bundle.vega, bundle.strikes, bundle.tenors,
                    "Vega projeté Ñ = Aᵀ_T N A_K (nœuds de test)", diverging=True,
                    colorbar_title="ν̃",
                ),
                width="stretch",
            )

        st.markdown("#### Tracking error des trois conventions (vs poids d'interpolation — Théorème 1)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "convention": conv,
                        "TE²_passage (€²)": round(te2, 0),
                        "part du budget": (
                            f"{te2 / budget_passage:.1%}" if budget_passage > 0 else "—"
                        ),
                        "sans perte (Th. 1)": "✓" if te2 <= 1e-9 * max(pr.var_reference, 1.0) else "✗",
                    }
                    for conv, te2 in pr.te2_by_convention.items()
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander("🔎 Matrices de passage A_T et A_K (Déf. 1 — lignes sommant à 1)"):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**A_T (axe des maturités)**")
                st.dataframe(
                    pd.DataFrame(pr.a_t, index=granular_bundle.tenors, columns=bundle.tenors),
                    width="stretch",
                )
            with cb:
                st.markdown("**A_K (axe des strikes)**")
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
    st.markdown("### Optimisation gloutonne agglomérative sous budget de variance (sec. 7.3)")
    opt_col1, opt_col2 = st.columns([1, 1])
    with opt_col1:
        robust_mode = st.toggle(
            "🛡️ Mode robuste — ne retenir que les fusions survivant au stress de corrélation adverse (étape 4)",
            value=False,
        )
    with opt_col2:
        deltas = st.multiselect(
            "Amplitudes de stress δ (ρ → max(ρ−δ, −1))",
            [0.05, 0.10, 0.15, 0.20], default=[0.10, 0.20],
            disabled=not robust_mode,
        )

    with st.spinner("Agglomération gloutonne en cours…"):
        result = cached_greedy(bundle_json, float(alpha), float(kappa), weighting)
    if robust_mode and deltas:
        with st.spinner("Rejeu sous stress de corrélation…"):
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
    m1.metric("Ensembles de netting", f"{scheme.n_sets}", delta=f"depuis {bundle.n} buckets")
    m2.metric("Score de variance R²", f"{ev.r2:.2%}", delta=f"{(ev.r2 - alpha) * 100:+.2f} pts vs α")
    m3.metric("AVA nettée — Déf. 4", eur(ev.ava), delta=f"-{ev.ava_saving_pct:.0%} vs add-up")
    m4.metric("Budget consommé", f"{ev.te2 / ev.budget:.0%}" if ev.budget > 0 else "—",
              delta=f"TE² = {ev.te2:,.0f} €²")
    m5.markdown(
        f"**Test de variance (6)** {verdict_badge(ev.passes_variance)}<br><br>"
        f"**Plancher (7)** {verdict_badge(ev.passes_floor)}",
        unsafe_allow_html=True,
    )
    if result.rolled_back:
        st.warning(f"Plancher (7) initialement violé — {result.rolled_back} fusion(s) défaite(s) (étape 3).")

    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.plotly_chart(
            charts.partition_figure(
                scheme.labels, bundle.vega, bundle.tenors, bundle.strikes,
                title="Ensembles de netting retenus (pavage rectangulaire, sec. 7.3)",
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
        st.markdown("#### 🛡️ Régimes de stress — robustesse des fusions")
        rows = []
        for d, run in rb["runs"].items():
            e = run.evaluation
            rows.append(
                {
                    "régime": "base" if d == 0.0 else f"ρ − {d:.2f}",
                    "ensembles": run.scheme.n_sets,
                    "R²": round(e.r2, 4),
                    "AVA": round(e.ava, 0),
                    "TE²/B": round(e.te2 / e.budget, 3) if e.budget > 0 else None,
                    "test de variance": "PASS" if e.passes_variance else "FAIL",
                    "plancher": "PASS" if e.passes_floor else "FAIL",
                }
            )
        rows.append(
            {
                "régime": "robuste (raffinement commun)",
                "ensembles": scheme.n_sets,
                "R²": round(ev.r2, 4),
                "AVA": round(ev.ava, 0),
                "TE²/B": round(ev.te2 / ev.budget, 3) if ev.budget > 0 else None,
                "test de variance": "PASS" if ev.passes_variance else "FAIL",
                "plancher": "PASS" if ev.passes_floor else "FAIL",
            }
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("🔎 Détail par ensemble"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ensemble": s.set_id,
                        "buckets": s.size,
                        "vega net m_k": round(s.net_vega, 0),
                        "brut Σ|ν|": round(s.gross_vega, 0),
                        "taux de compensation": f"{s.offset_ratio:.0%}",
                        "s̃_k": round(s.s_tilde, 3),
                        "AVA add-up": round(s.ava_addup, 0),
                        "AVA nettée": round(s.ava_netted, 0),
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
        "### Architecture découplée — la structure par la variance, le niveau par l'AVA (sec. 7.5)\n"
        "**Run 1** regroupe hiérarchiquement les nœuds de test sur le **risque de base** "
        "d_ij = √(s_i² + s_j² − 2ρ_ij s_i s_j) — Théorème 2 (i) : fusionner j sur le pivot i "
        "coûte TE² = ν_j²·d_ij². Le dendrogramme ne dépend que de Σ, **jamais du book** : "
        "la structure est stable de portefeuille en portefeuille. Un schéma de netting est "
        "une **coupe** de l'arbre à la hauteur ε ; la Propriété 8 garantit, pour *tout* book, "
        "TE ≤ ε·Σ|ν_j|s_j = ε·AVA_brut/κ. **Run 2** évalue l'AVA sur la coupe avec "
        "le s Totem du jour — aucune ré-optimisation, plus le contrôle obligatoire par book "
        "TE²/Var(ΔΠ) (la coupe est abaissée s'il échoue)."
    )
    with st.spinner("Run 1 — construction du dendrogramme sans portefeuille…"):
        dendro = cached_dendrogram(bundle_json)
    if dendro.merges:
        heights = [m.height for m in dendro.merges]
        h_max = max(heights)
        h_default = float(np.median(heights))
    else:
        h_max, h_default = 1.0, 0.5
    eps_cut = st.slider(
        "ε — hauteur de coupe (Prop. 8 : risque de base par nœud ≤ ε · incertitude propre)",
        0.0, float(np.ceil(h_max * 1.05 * 100) / 100), h_default, 0.01,
    )
    with st.spinner("Run 2 — évaluation du book du jour sur la coupe…"):
        dres = decoupled_netting(bundle, alpha=float(alpha), kappa=float(kappa),
                                 epsilon=float(eps_cut), dendrogram=dendro)
    ev_d = dres.evaluation

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Ensembles de netting", f"{ev_d.scheme.n_sets}",
              delta=f"{dres.n_merges} fusions (sur {len(dendro.merges)})")
    d2.metric("ε réalisé", f"{dres.epsilon_realised:.3f}",
              delta=f"{dres.lowered} fusion(s) défaite(s) par le contrôle book" if dres.lowered else "coupe telle que demandée",
              delta_color="off")
    d3.metric("Borne Prop. 8 sur TE", f"{dres.te_bound:,.0f}",
              delta=f"TE réalisé = {np.sqrt(max(ev_d.te2, 0)):,.0f}", delta_color="off")
    d4.metric("Score de variance R²", f"{ev_d.r2:.2%}",
              delta=f"{(ev_d.r2 - alpha) * 100:+.2f} pts vs α")
    d5.metric("AVA (Run 2) — Déf. 4", eur(ev_d.ava), delta=f"-{ev_d.ava_saving_pct:.0%} vs add-up")

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
                title="Coupe du dendrogramme — ensembles de structure (Run 1)",
                set_stats=ev_d.set_stats,
            ),
            width="stretch",
        )
    st.caption(
        "Comparez avec le glouton joint (onglet Netting optimal) : le schéma découplé "
        "peut laisser un peu d'AVA sur la table, mais il ne bouge pas avec le book — "
        "ré-optimiser la partition à chaque date avec les vegas du jour est en soi un "
        "signal d'alerte en validation (sec. 7.5)."
    )

    st.divider()
    st.markdown(
        "#### Stabilité entre familles de chocs (sec. 7.4)\n"
        "Même partition, même transport, mêmes formules — seul Σ change. Le test de "
        "variance de la coupe retenue est rejoué sous des covariances alternatives ; "
        "toute fusion qui échoue dans un régime est instable et défaite (on peut refuser "
        "une fusion valide, jamais conserver une fusion invalidée)."
    )
    stab_deltas = st.multiselect(
        "Régimes de corrélation adverse tenant lieu de fenêtres Σ_daily (ρ → max(ρ−δ, −1))",
        [0.05, 0.10, 0.15, 0.20, 0.30], default=[0.10, 0.20],
    )
    alt_upload = st.file_uploader(
        "…et/ou charger un bundle de covariance alternative (p. ex. Σ_daily restreinte "
        "aux piliers via la Prop. 5, ou Σ_bidask) — même grille et même vega, s/ρ différents",
        type=["json"], key="alt_cov",
    )
    alternatives = {f"ρ − {d:.2f}": stress_bundle(bundle, d) for d in stab_deltas}
    if alt_upload is not None:
        alt_b = MarketDataBundle.from_json(alt_upload.read().decode("utf-8"))
        if (alt_b.M, alt_b.K) == (bundle.M, bundle.K):
            alternatives["Σ chargée"] = alt_b
        else:
            st.error("La grille du bundle alternatif ne correspond pas — ignoré.")
    if alternatives:
        with st.spinner("Rejeu du test de variance dans chaque régime…"):
            stab = stable_cut(bundle, alternatives, alpha=float(alpha),
                              kappa=float(kappa), epsilon=float(eps_cut),
                              dendrogram=dendro)
        if stab["undone"]:
            st.warning(
                f"{stab['undone']} fusion(s) instable(s) sous au moins une covariance "
                f"alternative — défaites. Retenu : {stab['n_merges']} fusions, "
                f"{stab['evaluation'].scheme.n_sets} ensembles."
            )
        else:
            st.success("Chaque fusion de la coupe survit à tous les régimes — la structure est stable.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "régime": r["regime"],
                        "R²": round(r["r2"], 4),
                        "TE²/B": round(r["te2"] / r["budget"], 3) if r["budget"] > 0 else None,
                        "test de variance": "PASS" if r["passes_variance"] else "FAIL",
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
        "### La méthodologie à deux couches — sous-jacent (structure) × book (évaluation)\n"
        "**Couche 1, par sous-jacent (sans book) :** le smile d'une tranche bouge selon "
        "trois modes — niveau, pente ΔS, courbure ΔC — plus un bruit idiosyncratique "
        "(Déf. 1). Le modèle engendre le désaccord entre deux points quelconques "
        "(Théorème 1) et la courbe de corrélation ρ(x) = s₀/sₓ (Prop. 2) ; le dendrogramme "
        "des groupes de strikes contigus est la **carte des fusions possibles**. "
        "**Couche 2, par book :** le résidu exact du netting d'un groupe sur son pivot est "
        "porté par les agrégats *signés* RR_p et FLY_p (Théorème 2) — le niveau net se "
        "nette toujours. Le pivot barycentre x_p = RR₀/m annule RR exactement (sec. 6.3), et "
        "les décisions ne sont jamais tout-ou-rien : écrasement / extraction / scission "
        "(N2–N4). Le calcul exact peut **autoriser ce que le majorant de couche 1 "
        "Σ|ν_j|d_jp refusait à tort** — il cesse de compter deux fois ce qui s'annule."
    )

    tl1, tl2 = st.columns([1, 2.2])
    with tl1:
        tranche_idx = st.selectbox(
            "Tranche (une ligne de maturité de la matrice de vega)",
            list(range(bundle.M)), format_func=lambda i: bundle.tenors[i],
        )
        st.markdown("**Modèle de couche 1** (A1 — issu des régressions quotidiennes ; éditable) :")
        s0_tl = st.number_input("s₀ — écart-type du choc de niveau", value=1.0, min_value=0.0, step=0.1)
        ss_tl = st.number_input("σ_S — écart-type du choc de pente (skew)", value=2.0, min_value=0.0, step=0.1)
        sc_tl = st.number_input("σ_C — écart-type du choc de courbure", value=10.0, min_value=0.0, step=0.5)
        se_tl = st.number_input("σ_ε — écart-type idiosyncratique", value=0.1, min_value=0.0, step=0.01)
        smodel = SmileModel(s0=float(s0_tl), sigma_s=float(ss_tl),
                            sigma_c=float(sc_tl), sigma_eps=float(se_tl))
        x_tl = np.asarray(bundle.strikes, dtype=float) - 1.0
        nu_tl = bundle.vega[int(tranche_idx)]
        s_tl = smodel.point_uncertainty(x_tl)
        dist_tl = model_distance(x_tl, smodel)
        merges_tl = tranche_dendrogram(x_tl, dist_tl, s_tl)
        h_def = float(np.median([m.height for m in merges_tl])) if merges_tl else 0.5
        h_top = max((m.height for m in merges_tl), default=1.0)
        eps_tl = st.slider("hauteur de coupe (désaccord d)", 0.0,
                           float(np.ceil(h_top * 105) / 100), h_def, 0.01)
    with tl2:
        cda, cdb = st.columns(2)
        with cda:
            st.plotly_chart(
                charts.heatmap(
                    dist_tl, bundle.strikes, bundle.strikes,
                    "Matrice de désaccord D = (d_ij) — Théorème 1, sans book",
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
    st.markdown(f"#### Couche 2 — décisions book sur la tranche **{bundle.tenors[int(tranche_idx)]}** (N1–N4)")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Groupes (coupe couche 1)", f"{len(groups_tl)}",
              delta=f"depuis {bundle.K} strikes")
    t2.metric("TE² exact vs majorant²",
              f"{res_tl.te2:,.0f}",
              delta=f"majorant {res_tl.te2_majorant:,.0f}", delta_color="off")
    t3.metric("Budget (1−α)·Var(ΔΠ)", f"{res_tl.budget:,.0f}",
              delta="PASS" if res_tl.passes_variance else "FAIL", delta_color="off")
    t4.metric("AVA (tranche)", eur(res_tl.ava), delta=f"-{res_tl.ava_saving_pct:.0%} vs add-up")
    t5.markdown(
        f"**Test de variance** {verdict_badge(res_tl.passes_variance)}<br><br>"
        f"**Plancher** {verdict_badge(res_tl.passes_floor)}",
        unsafe_allow_html=True,
    )
    if res_tl.te2_majorant > res_tl.budget >= res_tl.te2:
        st.success(
            "Le majorant de couche 1 aurait à lui seul **refusé** ce netting "
            f"({res_tl.te2_majorant:,.0f} > {res_tl.budget:,.0f}) ; le calcul signé "
            "exact l'autorise — les compensations entre groupes sont réelles "
            "(sec. 5 : la corrélation mesure le désaccord, le book décide "
            "s'il fait mal)."
        )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "groupe (strikes)": ", ".join(str(bundle.strikes[j]) for j in d.members),
                    "pivot": bundle.strikes[d.pivot_index],
                    "m (niveau net)": round(d.m, 1),
                    "RR_p": round(d.rr, 3),
                    "FLY_p": round(d.fly, 3),
                    "Var(R) exacte (Th. 2)": round(d.var_exact, 2),
                    "majorant Σ|ν|d": round(d.majorant_sd, 2),
                    "décision": d.decision,
                    "AVA": round(d.ava, 1),
                }
                for d in res_tl.decisions
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Les pivots sont les nœuds cotés les plus proches du barycentre de vega RR₀/m — "
        "le choix qui annihile le terme de pente (sec. 6.3). L'extraction garde le niveau "
        "net netté et provisionne RR/FLY en add-up avec leurs propres incertitudes ; seul "
        "le netting réellement injustifié est abandonné."
    )

# =========================================================================== #
# SCENARIO LAB
# =========================================================================== #
with tab_scenario:
    st.markdown(
        "### Scorez n'importe quel schéma de netting que vous voulez tester\n"
        "Un scénario est une **matrice de labels entiers (M × K)** sur la grille "
        "maturité × strike — les cellules partageant un label forment un ensemble de "
        "netting. Éditez la grille, chargez une matrice ou branchez une `ScenarioSource` ; "
        "le moteur renvoie le **score de variance R²**, le verdict EBA et l'impact AVA."
    )

    preset_names = {
        "Par maturité (écraser le smile)": "per_tenor",
        "Par colonne de strike": "per_strike",
        "Quadrants (court/long × bas/haut)": "quadrants",
        "Ensemble global unique": "global",
        "Singletons (aucun netting)": "singletons",
        "Optimal (issu de l'optimiseur)": "optimal",
    }
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        chosen_preset = st.selectbox("Partir de", list(preset_names.keys()), index=0)
    with cc2:
        scen_upload = st.file_uploader(
            "…ou charger un scénario JSON  {\"labels\": [[…]]}", type=["json"], key="scen_up"
        )

    if scen_upload is not None:
        labels_init = np.asarray(json.loads(scen_upload.read().decode("utf-8"))["labels"], dtype=int)
    elif preset_names[chosen_preset] == "optimal" and "optimal_labels" in st.session_state:
        labels_init = np.asarray(st.session_state["optimal_labels"], dtype=int)
    elif preset_names[chosen_preset] == "optimal":
        st.info("Lancez d'abord l'optimiseur — repli sur le netting par maturité.")
        labels_init = preset_labels("per_tenor", bundle.M, bundle.K)
    else:
        labels_init = preset_labels(preset_names[chosen_preset], bundle.M, bundle.K)

    if labels_init.shape != (bundle.M, bundle.K):
        st.error(f"La forme du scénario {labels_init.shape} ne correspond pas à la grille ({bundle.M}, {bundle.K}).")
        st.stop()

    st.markdown("**Éditez les labels d'ensemble** (entiers — même label = même ensemble de netting) :")
    df_labels = pd.DataFrame(labels_init, index=bundle.tenors, columns=[str(s) for s in bundle.strikes])
    edited = st.data_editor(df_labels, width="stretch", key=f"editor_{chosen_preset}_{scen_upload is not None}")
    try:
        labels = edited.to_numpy(dtype=int)
    except (TypeError, ValueError):
        st.error("Les labels doivent être des entiers.")
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
        st.metric("AVA du scénario — Déf. 4", eur(ev_s.ava),
                  delta=f"-{ev_s.ava_saving_pct:.0%} vs add-up")
        st.metric("TE² vs budget", f"{ev_s.te2:,.0f} €²",
                  delta=f"budget {ev_s.budget:,.0f} €²", delta_color="off")
        st.markdown(
            f"**Test de variance (6)** {verdict_badge(ev_s.passes_variance)} &nbsp;&nbsp; "
            f"**Plancher (7)** {verdict_badge(ev_s.passes_floor)}",
            unsafe_allow_html=True,
        )
    with g2:
        st.plotly_chart(
            charts.partition_figure(
                NettingScheme.from_labels(labels).labels, bundle.vega, bundle.tenors,
                bundle.strikes, title="Ensembles de netting du scénario", set_stats=ev_s.set_stats,
            ),
            width="stretch",
        )

    st.download_button(
        "⬇️ Exporter les labels du scénario (JSON)",
        data=json.dumps({"labels": labels.tolist()}, indent=2),
        file_name="netting_scenario.json",
        mime="application/json",
    )
    st.session_state["scenario_eval"] = ev_s
    st.session_state["scenario_labels"] = labels.tolist()

    with st.expander("🧪 Bac à sable à deux nœuds — l'intuition en forme fermée (Théorème 2, éq. 8)"):
        s1c, s2c, s3c = st.columns(3)
        with s1c:
            nu_i_tb = st.number_input("ν_i (vega pivot)", value=1000.0, step=50.0)
            nu_j_tb = st.number_input("ν_j (vega netté)", value=-300.0, step=50.0)
        with s2c:
            s_i_tb = st.number_input("s_i", value=0.50, min_value=0.01, step=0.05)
            s_j_tb = st.number_input("s_j", value=0.55, min_value=0.01, step=0.05)
        with s3c:
            rho_tb = st.slider("ρ", -1.0, 1.0, 0.90, 0.01)
            var_choice = st.radio("Var(ΔΠ) au membre de droite de (6)", ["variance de la paire", "variance du portefeuille"],
                                  horizontal=False)
        if var_choice == "variance du portefeuille":
            var_tb = model.var_total
        else:
            var_tb = (
                nu_i_tb ** 2 * s_i_tb ** 2 + nu_j_tb ** 2 * s_j_tb ** 2
                + 2 * nu_i_tb * nu_j_tb * rho_tb * s_i_tb * s_j_tb
            )
        cf = two_bucket(nu_i_tb, nu_j_tb, s_i_tb, s_j_tb, rho_tb, var_tb, float(alpha), float(kappa))
        r1, r2_, r3, r4 = st.columns(4)
        r1.metric("TE² — Th. 2 (i)", f"{cf['te2']:,.0f}")
        r2_.metric("ρ_min — éq. (8)", f"{cf['rho_min']:.3f}" if cf["rho_min"] > -1 else "toujours")
        r3.metric("Gain d'AVA", eur(cf["ava_gain"]))
        r4.markdown("**Admissible**<br>" + verdict_badge(cf["admissible"], "OUI", "NON"), unsafe_allow_html=True)
        st.plotly_chart(
            charts.two_bucket_figure(s_i_tb, s_j_tb, nu_i_tb, var_tb, float(alpha), nu_j_tb, rho_tb),
            width="stretch",
        )

# =========================================================================== #
# SPECTRAL & SMILE
# =========================================================================== #
with tab_spectral:
    st.markdown(
        "### Le diagnostic spectral (sec. 6.1) — le spectre mesure, la partition nette\n"
        "Σ aux nœuds est diagonalisée en facteurs décorrélés ξ_ℓ = u_ℓᵀΔσ "
        "(Prop. 6) ; la variance du book se décompose exactement en Σ c_ℓ²λ_ℓ avec "
        "c_ℓ = ⟨ν, u_ℓ⟩ (Prop. 7). Théorème 3 : les schémas dont les chocs représentatifs "
        "vivent dans le sous-espace des L premiers facteurs ne peuvent pas battre le "
        "plancher de queue."
    )
    diag = spectral_diagnostic(model, alpha=float(alpha))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("K*(α) — plancher spectral sur le nb d'ensembles (Théorème 3)", f"{diag.k_star}")
    s2.metric("Les 3 premiers modes expliquent", f"{diag.explained_ratio(3):.1%}", delta="de la variance du book (Prop. 7)")
    s3.metric("Inertie τ₃ — U1", f"{diag.inertia[3]:.1%}", delta="de tr(Σ) = Σs² (compressibilité)")
    s4.metric("λ₁ / λ₂", f"{diag.eigenvalues[0] / max(diag.eigenvalues[1], 1e-12):.1f}×")
    st.plotly_chart(charts.spectral_figure(diag), width="stretch")

    st.markdown("#### Directions propres dominantes u_ℓ de Σ sur la grille — les *directions non nettables* vivent dans la queue")
    mode_cols = st.columns(3)
    names = ["mode 1 (niveau)", "mode 2 (terme/skew)", "mode 3 (smile)"]
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
        "Les tranches dont le R² tombe sous α portent un risk-reversal / butterfly net "
        "matériel : leur exposition de forme de smile doit rester en add-up (étape 1 du "
        "netting hiérarchique, sec. 6.2 / Théorème 4) ; les vegas de niveau des tranches "
        "qui passent poursuivent vers le netting de structure par terme de l'étape 2 "
        "sous ρ_mat."
    )

# =========================================================================== #
# AUDIT & EXPORT
# =========================================================================== #
with tab_audit:
    st.markdown(
        "### Dossier de preuve art. 9(5)\n"
        "Tout ce que le validateur et le superviseur demanderont : la partition "
        "retenue, le R² réalisé, les résultats de stress et la trace gloutonne "
        "complète — avec empreinte du bundle d'entrée, par date de calcul."
    )
    which = st.radio(
        "Construire le dossier pour", ["Schéma optimal", "Scénario courant"], horizontal=True
    )
    if which == "Schéma optimal" and "optimal_eval" in st.session_state:
        ev_a = st.session_state["optimal_eval"]
        history_a = st.session_state.get("optimal_history")
    elif which == "Scénario courant" and "scenario_eval" in st.session_state:
        ev_a = st.session_state["scenario_eval"]
        history_a = None
    else:
        st.info("Visitez d'abord l'onglet correspondant pour que l'évaluation existe.")
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
                "⬇️ Télécharger le dossier d'audit (JSON)", data=text,
                file_name=f"ava_netting_audit_{bundle.meta.get('asof', 'latest')}.json",
                mime="application/json", type="primary",
            )
            st.download_button(
                "⬇️ Télécharger le bundle d'entrée (JSON)", data=bundle.to_json(indent=2),
                file_name="market_data_bundle.json", mime="application/json",
            )
        with a2:
            st.metric("Empreinte de l'entrée (SHA-256)", pack["input"]["fingerprint_sha256"])
        st.json(pack, expanded=2)

# =========================================================================== #
# DOCUMENTATION
# =========================================================================== #
with tab_docs:
    st.markdown("## 📖 Documentation des onglets")
    st.caption(
        "Une fiche par onglet : ce qui tourne à l'intérieur (UI → moteur → flux de "
        "données) et la raison réglementaire et mathématique derrière chaque calcul. "
        "Les fiches sources sont les fichiers `docs/readme_*.md` du dépôt."
    )
    dc1, dc2 = st.columns([2, 1])
    with dc1:
        doc_choice = st.selectbox("Onglet à documenter", list(DOC_PAGES), index=0)
    try:
        doc_text = load_doc(DOC_PAGES[doc_choice])
    except OSError as exc:
        st.error(f"Fiche introuvable : {DOC_PAGES[doc_choice]} ({exc}).")
    else:
        with dc2:
            st.download_button(
                "⬇️ Télécharger cette fiche (Markdown)",
                data=doc_text,
                file_name=DOC_PAGES[doc_choice],
                mime="text/markdown",
            )
        st.divider()
        st.markdown(doc_text)

st.markdown(
    "<div style='text-align:center; color:#5b6694; padding-top: 24px;'>"
    "Vega Netting Studio · Valorisation prudente EBA, AVA MPU · usage IPV / validation de modèle — "
    "données de démonstration synthétiques, pas un conseil en investissement</div>",
    unsafe_allow_html=True,
)
