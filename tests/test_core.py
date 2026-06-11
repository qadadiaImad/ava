import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebanetting import (
    KAPPA_90,
    MarketDataBundle,
    NettingScheme,
    SyntheticDataSource,
    UncertaintyModel,
    greedy_netting,
    nearest_correlation,
    preset_labels,
    robust_netting,
    score_scenario,
    spectral_diagnostic,
    stress_bundle,
    two_bucket,
)


@pytest.fixture
def bundle() -> MarketDataBundle:
    return SyntheticDataSource(book="mixed_desk", seed=3).load()


@pytest.fixture
def model(bundle) -> UncertaintyModel:
    return UncertaintyModel(bundle=bundle, kappa=KAPPA_90)


def test_bundle_valid(bundle):
    assert bundle.validate() == []


def test_kronecker_covariance_entries(bundle):
    """Sigma[(m,k),(m',k')] = s_mk s_m'k' rho_mat[m,m'] rho_strike[k,k']."""
    sigma = bundle.covariance()
    K = bundle.K
    for (m, k, mp, kp) in [(0, 0, 3, 5), (2, 4, 2, 4), (7, 1, 1, 7)]:
        expected = (
            bundle.s[m, k]
            * bundle.s[mp, kp]
            * bundle.corr_mat[m, mp]
            * bundle.corr_strike[k, kp]
        )
        assert sigma[m * K + k, mp * K + kp] == pytest.approx(expected)


def test_extremes(model):
    """Singletons: TE^2 = 0, AVA = add-up (eq. 2); and AVA_full <= AVA_brut."""
    scheme = NettingScheme.singletons(model.bundle.M, model.bundle.K)
    ev = scheme.evaluate(model, alpha=0.9)
    assert ev.te2 == pytest.approx(0.0, abs=1e-9 * model.var_total)
    assert ev.ava == pytest.approx(model.ava_brut, rel=1e-12)
    assert model.ava_full <= model.ava_brut


def test_two_bucket_closed_form_matches_engine():
    """Eq. (10) against the generic TE^2 engine on an isolated pair."""
    nu_i, nu_j, s_i, s_j, rho = 1200.0, -800.0, 0.5, 0.62, 0.9
    bundle = MarketDataBundle(
        tenors=["1Y"],
        tenor_years=[1.0],
        strikes=[0.95, 1.05],
        vega=np.array([[nu_i, nu_j]]),
        s=np.array([[s_i, s_j]]),
        corr_mat=np.array([[1.0]]),
        corr_strike=np.array([[1.0, rho], [rho, 1.0]]),
    )
    model = UncertaintyModel(bundle=bundle)
    # pivot weighting picks i (|nu_i| s_i = 600 > |nu_j| s_j = 496)
    scheme = NettingScheme.from_labels(np.array([[0, 0]]), weighting="pivot")
    ev = scheme.evaluate(model, alpha=0.9)
    cf = two_bucket(nu_i, nu_j, s_i, s_j, rho, model.var_total, 0.9, KAPPA_90)
    assert ev.te2 == pytest.approx(cf["te2"], rel=1e-10)
    assert ev.ava == pytest.approx(KAPPA_90 * abs(nu_i + nu_j) * s_i, rel=1e-10)
    assert cf["admissible"] == ev.passes_variance


def test_two_bucket_threshold_is_sharp():
    """rho just above rho_min passes, just below fails (eq. 11)."""
    # homogeneous uncertainties and a netted position small relative to
    # total risk, otherwise eq. (11) has no solution below rho = 1
    nu_i, nu_j, s_i, s_j, alpha = 1000.0, -300.0, 0.5, 0.5, 0.9

    def make(rho):
        b = MarketDataBundle(
            tenors=["1Y"],
            tenor_years=[1.0],
            strikes=[0.95, 1.05],
            vega=np.array([[nu_i, nu_j]]),
            s=np.array([[s_i, s_j]]),
            corr_mat=np.array([[1.0]]),
            corr_strike=np.array([[1.0, rho], [rho, 1.0]]),
        )
        return UncertaintyModel(bundle=b)

    # locate the threshold rho* where TE^2(rho) = (1 - alpha) Var(rho) by
    # bisection (var_total itself depends on rho, so eq. 11 is implicit)
    def excess(rho):
        cf = two_bucket(nu_i, nu_j, s_i, s_j, rho, make(rho).var_total, alpha, KAPPA_90)
        return cf["te2"] - cf["budget"]

    lo, hi = -0.99, 0.99
    assert excess(lo) > 0 > excess(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    rho = 0.5 * (lo + hi)
    scheme = NettingScheme.from_labels(np.array([[0, 0]]), weighting="pivot")
    above = scheme.evaluate(make(min(rho + 0.02, 0.999)), alpha)
    below = scheme.evaluate(make(rho - 0.02), alpha)
    assert above.passes_variance
    assert not below.passes_variance


def test_r2_definition(model):
    labels = preset_labels("per_tenor", model.bundle.M, model.bundle.K)
    ev = NettingScheme.from_labels(labels).evaluate(model, alpha=0.9)
    assert ev.r2 == pytest.approx(1.0 - ev.te2 / ev.var_total)
    assert ev.te2 >= 0.0


def test_greedy_respects_budget_and_floor(model):
    alpha = 0.9
    res = greedy_netting(model, alpha=alpha)
    ev = res.evaluation
    assert ev.te2 <= model.budget(alpha) + 1e-9
    assert ev.passes_variance and ev.passes_floor
    # AVA must improve monotonically along the merge history
    avas = [h.ava for h in res.history]
    assert all(a >= b - 1e-9 for a, b in zip(avas, avas[1:]))
    assert ev.ava <= model.ava_brut + 1e-9
    # netted AVA never below the diversification floor
    assert ev.ava >= model.ava_full - 1e-9


def test_greedy_beats_brut(model):
    res = greedy_netting(model, alpha=0.9)
    assert res.evaluation.ava < model.ava_brut


def test_spectral_bound_consistency(model):
    diag = spectral_diagnostic(model, alpha=0.9)
    # K = 0 residual is the whole vega-weighted variance = Var(DeltaPi)
    assert diag.residual_curve[0] == pytest.approx(model.var_total, rel=1e-8)
    assert diag.residual_curve[-1] == pytest.approx(0.0, abs=1e-6 * model.var_total)
    assert np.all(np.diff(diag.residual_curve) <= 1e-9)
    assert 1 <= diag.k_star <= model.bundle.n
    # the greedy solution can never use fewer sets than the spectral bound
    res = greedy_netting(model, alpha=0.9)
    assert res.scheme.n_sets >= diag.k_star


def test_stress_is_adverse(bundle):
    """Reducing intra-set correlation must not shrink the tracking error."""
    model = UncertaintyModel(bundle=bundle)
    stressed = UncertaintyModel(bundle=stress_bundle(bundle, 0.15))
    labels = preset_labels("per_tenor", bundle.M, bundle.K)
    scheme = NettingScheme.from_labels(labels)
    assert scheme.evaluate(stressed, 0.9).te2 >= scheme.evaluate(model, 0.9).te2 * 0.99


def test_stressed_corr_is_valid(bundle):
    stressed = stress_bundle(bundle, 0.2)
    assert stressed.validate() == []


def test_robust_partition_is_refinement(bundle):
    out = robust_netting(bundle, alpha=0.9, kappa=KAPPA_90, deltas=[0.1, 0.2])
    base = out["runs"][0.0].scheme.labels
    robust = out["robust_scheme"].labels
    # refinement: two cells together in the robust partition were together
    # in the base partition
    flat_b, flat_r = base.flatten(), robust.flatten()
    for k in np.unique(flat_r):
        idx = np.flatnonzero(flat_r == k)
        assert len(np.unique(flat_b[idx])) == 1
    assert out["robust_evaluation"].passes_variance


def test_scenario_scoring_roundtrip(bundle):
    labels = preset_labels("quadrants", bundle.M, bundle.K)
    report = score_scenario(bundle, labels, alpha=0.9, kappa=KAPPA_90)
    assert 0.0 <= report.variance_score <= 1.0 or report.variance_score < 0
    assert report.verdict in {
        "ADMISSIBLE",
        "REJECTED — variance test (R² < α)",
        "REJECTED — conservatism floor (8)",
        "REJECTED — variance test and floor",
    }


def test_json_roundtrip(bundle):
    clone = MarketDataBundle.from_json(bundle.to_json())
    assert clone.validate() == []
    np.testing.assert_allclose(clone.covariance(), bundle.covariance())
    m1 = UncertaintyModel(bundle=bundle)
    m2 = UncertaintyModel(bundle=clone)
    assert m1.ava_brut == pytest.approx(m2.ava_brut)


def test_nearest_correlation():
    rho = np.array([[1.0, 0.95, -0.9], [0.95, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = nearest_correlation(rho)
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-10
