"""ebanetting — vega netting under the EBA Prudent Valuation variance test.

Implements the technical note "Netting des sensibilités vega sous le test
de variance" (AVA MPU, Delegated Regulation (EU) 2016/101): the passage
of granular vega onto the test nodes (sandwich of passage matrices,
Theoreme 1), netting as an aggregation operator, the tracking-error
variance test, the conservatism floor, the two-node closed form
(Theoreme 2) and the greedy agglomerative optimiser with adverse
correlation stress.

Everything is computed with ordinary matrix products and sum reductions
on the (tenor, strike) grid — no Kronecker / tensor product is ever
assembled, even under the decoupled-correlation hypothesis (annex).
"""

from .datasource import (
    DataSource,
    JSONBundleSource,
    JSONScenarioSource,
    MarketDataBundle,
    MatrixScenarioSource,
    ScenarioSource,
    SyntheticDataSource,
    nearest_correlation,
)
from .model import KAPPA_90, UncertaintyModel
from .netting import NettingScheme, SchemeEvaluation, SetStat, two_bucket
from .optimizer import (
    GreedyResult,
    MergeStep,
    greedy_netting,
    lagrangian_frontier,
    robust_netting,
    stress_bundle,
)
from .passage import (
    PassageResult,
    offgrid_residual,
    passage_matrix,
    project,
    project_bundle,
    restrict_shocks,
    restriction_matrix,
)
from .clustering import (
    DecoupledResult,
    Dendrogram,
    base_risk_distance,
    build_dendrogram,
    cut_pivots,
    decoupled_netting,
    evaluate_cut,
    stability_report,
    stable_cut,
)
from .reporting import audit_json, build_audit_pack
from .scenario import ScenarioReport, preset_labels, score_scenario, smile_decomposition
from .spectral import SpectralDiagnostic, spectral_diagnostic

__version__ = "2.1.0"

__all__ = [
    "PassageResult",
    "passage_matrix",
    "project",
    "project_bundle",
    "restriction_matrix",
    "restrict_shocks",
    "offgrid_residual",
    "DecoupledResult",
    "Dendrogram",
    "base_risk_distance",
    "build_dendrogram",
    "cut_pivots",
    "decoupled_netting",
    "evaluate_cut",
    "stability_report",
    "stable_cut",
    "DataSource",
    "JSONBundleSource",
    "JSONScenarioSource",
    "MarketDataBundle",
    "MatrixScenarioSource",
    "ScenarioSource",
    "SyntheticDataSource",
    "nearest_correlation",
    "KAPPA_90",
    "UncertaintyModel",
    "NettingScheme",
    "SchemeEvaluation",
    "SetStat",
    "two_bucket",
    "GreedyResult",
    "MergeStep",
    "greedy_netting",
    "lagrangian_frontier",
    "robust_netting",
    "stress_bundle",
    "audit_json",
    "build_audit_pack",
    "ScenarioReport",
    "preset_labels",
    "score_scenario",
    "smile_decomposition",
    "SpectralDiagnostic",
    "spectral_diagnostic",
]
