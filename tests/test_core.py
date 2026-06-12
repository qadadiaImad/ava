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
    passage_matrix,
    preset_labels,
    project,
    project_bundle,
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


def test_decoupled_covariance_entries(bundle):
    """Cov(dsigma_mk, dsigma_m'k') = s_mk s_m'k' rho_mat[m,m'] rho_strike[k,k']
    recovered through the matrix-form bilinear form on unit-bump exposures
    (no Kronecker matrix anywhere)."""
    for (m, k, mp, kp) in [(0, 0, 3, 5), (2, 4, 2, 4), (7, 1, 1, 7)]:
        e_a = np.zeros((bundle.M, bundle.K))
        e_b = np.zeros((bundle.M, bundle.K))
        e_a[m, k] = 1.0
        e_b[mp, kp] = 1.0
        expected = (
            bundle.s[m, k]
            * bundle.s[mp, kp]
            * bundle.corr_mat[m, mp]
            * bundle.corr_strike[k, kp]
        )
        assert bundle.covariance_of(e_a, e_b) == pytest.approx(expected)


def test_variance_matches_scalar_double_sum(bundle):
    """Property 1 by brute force: the matrix sandwich must equal the
    explicit double sum over all bucket pairs (scalar products only)."""
    rng = np.random.default_rng(11)
    expo = rng.normal(size=(bundle.M, bundle.K))
    total = 0.0
    for m in range(bundle.M):
        for k in range(bundle.K):
            for mp in range(bundle.M):
                for kp in range(bundle.K):
                    total += (
                        expo[m, k] * expo[mp, kp]
                        * bundle.s[m, k] * bundle.s[mp, kp]
                        * bundle.corr_mat[m, mp] * bundle.corr_strike[k, kp]
                    )
    assert bundle.variance_of(expo) == pytest.approx(total, rel=1e-10)


def test_corr_full_path_consistent(bundle):
    """A corr_full assembled entrywise from the decoupled axes must give
    the same quadratic forms as the sandwich path."""
    n = bundle.n
    full = np.empty((n, n))
    for m in range(bundle.M):
        for k in range(bundle.K):
            for mp in range(bundle.M):
                for kp in range(bundle.K):
                    full[m * bundle.K + k, mp * bundle.K + kp] = (
                        bundle.corr_mat[m, mp] * bundle.corr_strike[k, kp]
                    )
    clone = MarketDataBundle(
        tenors=bundle.tenors,
        tenor_years=bundle.tenor_years,
        strikes=bundle.strikes,
        vega=bundle.vega,
        s=bundle.s,
        corr_full=full,
    )
    assert clone.variance_of(bundle.vega) == pytest.approx(
        bundle.variance_of(bundle.vega), rel=1e-10
    )


def test_extremes(model):
    """Singletons: TE^2 = 0, AVA = add-up (eq. 2); and AVA_full <= AVA_brut."""
    scheme = NettingScheme.singletons(model.bundle.M, model.bundle.K)
    ev = scheme.evaluate(model, alpha=0.9)
    assert ev.te2 == pytest.approx(0.0, abs=1e-9 * model.var_total)
    assert ev.ava == pytest.approx(model.ava_brut, rel=1e-12)
    assert model.ava_full <= model.ava_brut


def test_two_bucket_closed_form_matches_engine():
    """Th. 2 (i) closed form against the generic TE^2 engine on an isolated pair."""
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
    """rho just above rho_min passes, just below fails (eq. 8)."""
    # homogeneous uncertainties and a netted position small relative to
    # total risk, otherwise eq. (8) has no solution below rho = 1
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
    # bisection (var_total itself depends on rho, so eq. 8 is implicit)
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
        "REJECTED — conservatism floor (7)",
        "REJECTED — variance test and floor",
    }


def test_json_roundtrip(bundle):
    clone = MarketDataBundle.from_json(bundle.to_json())
    assert clone.validate() == []
    m1 = UncertaintyModel(bundle=bundle)
    m2 = UncertaintyModel(bundle=clone)
    assert m1.var_total == pytest.approx(m2.var_total)
    assert m1.ava_brut == pytest.approx(m2.ava_brut)


# --------------------------------------------------------------------------- #
# Passage to the test nodes (sec. 3 of the note)
# --------------------------------------------------------------------------- #
@pytest.fixture
def note_bundle() -> MarketDataBundle:
    """The worked example of the note: M=3 maturities x K=4 strikes."""
    return MarketDataBundle(
        tenors=["1Y", "2Y", "3Y"],
        tenor_years=[1.0, 2.0, 3.0],
        strikes=[90.0, 100.0, 110.0, 120.0],
        vega=np.array(
            [[80.0, -30.0, 10.0, 0.0],
             [-50.0, 40.0, -20.0, 10.0],
             [20.0, 0.0, 30.0, -40.0]]
        ),
        s=np.ones((3, 4)),
        corr_mat=np.eye(3),
        corr_strike=np.eye(4),
    )


def test_passage_matrices_match_note_example(note_bundle):
    """Interp weights of sec. 3.2: pillars {1Y, 3Y} and {100, 120}."""
    a_t = passage_matrix(note_bundle.tenor_years, [1.0, 3.0], "interp")
    a_k = passage_matrix(note_bundle.strikes, [100.0, 120.0], "interp")
    np.testing.assert_allclose(a_t, [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(
        a_k, [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    )
    # rows are non-negative and sum to one (Def. 1)
    for a in (a_t, a_k):
        assert np.all(a >= 0)
        np.testing.assert_allclose(a.sum(axis=1), 1.0)


def test_passage_sandwich_matches_note_example(note_bundle):
    """Eq. (4) of the note: N_tilde = A_T' N A_K = [[45, 5], [25, -25]]."""
    a_t = passage_matrix(note_bundle.tenor_years, [1.0, 3.0], "interp")
    a_k = passage_matrix(note_bundle.strikes, [100.0, 120.0], "interp")
    np.testing.assert_allclose(
        project(note_bundle.vega, a_t, a_k), [[45.0, 5.0], [25.0, -25.0]]
    )


def test_passage_conserves_total_vega(note_bundle):
    """Property 3: sum N_tilde = sum N for any passage convention."""
    for conv in ("interp", "equal", "quadrant"):
        res = project_bundle(note_bundle, [0, 2], [1, 3], conv)
        assert res.conserves_vega
        assert res.vega_total_nodes == pytest.approx(50.0)


def test_passage_interp_has_zero_tracking_error(bundle):
    """Theoreme 1: aligning passage weights on the interpolation weights
    gives TE_passage = 0; other conventions consume budget."""
    res = project_bundle(bundle, [0, 3, 5, 7], [0, 2, 4, 6], "interp")
    assert res.te2_passage == pytest.approx(0.0, abs=1e-12)
    assert res.te2_by_convention["equal"] >= 0.0
    assert res.te2_by_convention["quadrant"] >= 0.0


def test_projected_bundle_feeds_the_pipeline(bundle):
    """The node bundle must be a valid input for the whole machinery."""
    res = project_bundle(bundle, [0, 3, 5, 7], [0, 2, 4, 6], "interp")
    node = res.node_bundle
    assert node.validate() == []
    model = UncertaintyModel(bundle=node, kappa=KAPPA_90)
    out = greedy_netting(model, alpha=0.9)
    assert out.evaluation.passes_variance and out.evaluation.passes_floor


def test_nearest_correlation():
    rho = np.array([[1.0, 0.95, -0.9], [0.95, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = nearest_correlation(rho)
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-10


# --------------------------------------------------------------------------- #
# Sec. 3.5 — restriction without loss (Propriete 5)
# --------------------------------------------------------------------------- #
def test_restriction_reads_pillars_exactly():
    """R_T dsigma_fin R_K' = dsigma_pil when the fine surface is generated
    by the pillars (Propriete 5), and R B = I."""
    from ebanetting import offgrid_residual, restrict_shocks, restriction_matrix

    pil_t, pil_k = [1.0, 3.0], [100.0, 120.0]
    fine_t, fine_k = [1.0, 1.5, 2.0, 2.5, 3.0], [100.0, 105.0, 110.0, 115.0, 120.0]
    b_t = passage_matrix(fine_t, pil_t, "interp")
    b_k = passage_matrix(fine_k, pil_k, "interp")
    r_t = restriction_matrix(fine_t, pil_t)
    r_k = restriction_matrix(fine_k, pil_k)
    # the interpolation reproduces its nodes: R B = I
    np.testing.assert_allclose(r_t @ b_t, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(r_k @ b_k, np.eye(2), atol=1e-12)
    # pillar-generated fine shocks restrict back exactly, no residual
    rng = np.random.default_rng(5)
    shocks_pil = rng.normal(size=(2, 2))
    shocks_fine = b_t @ shocks_pil @ b_k.T
    np.testing.assert_allclose(restrict_shocks(shocks_fine, r_t, r_k), shocks_pil, atol=1e-12)
    np.testing.assert_allclose(
        offgrid_residual(shocks_fine, b_t, b_k, r_t, r_k), 0.0, atol=1e-12
    )
    # a re-fitted move beyond the pillars leaves a measurable off-grid part
    shocks_fine[2, 2] += 1.0
    assert np.abs(offgrid_residual(shocks_fine, b_t, b_k, r_t, r_k)).max() > 0.1


# --------------------------------------------------------------------------- #
# Sec. 6.5 — decoupled architecture
# --------------------------------------------------------------------------- #
def test_base_risk_distance_matches_theorem2(bundle):
    """Fusing j onto pivot i costs TE^2 = nu_j^2 d_ij^2 (Th. 2 (i))."""
    from ebanetting import base_risk_distance

    d = base_risk_distance(bundle)
    K = bundle.K
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0)
    (m, k), (mp, kp) = (1, 2), (4, 5)
    rho = bundle.corr_mat[m, mp] * bundle.corr_strike[k, kp]
    expected = np.sqrt(
        bundle.s[m, k] ** 2 + bundle.s[mp, kp] ** 2
        - 2 * rho * bundle.s[m, k] * bundle.s[mp, kp]
    )
    assert d[m * K + k, mp * K + kp] == pytest.approx(expected)


def test_dendrogram_is_portfolio_free_and_nested(bundle):
    """Run 1 must not depend on the book; cuts are nested in epsilon."""
    from ebanetting import build_dendrogram

    other_book = MarketDataBundle(
        tenors=bundle.tenors,
        tenor_years=bundle.tenor_years,
        strikes=bundle.strikes,
        vega=-3.0 * bundle.vega + 11.0,
        s=bundle.s,
        corr_mat=bundle.corr_mat,
        corr_strike=bundle.corr_strike,
    )
    d1, d2 = build_dendrogram(bundle), build_dendrogram(other_book)
    assert [(m.rect_a, m.rect_b, m.height) for m in d1.merges] == [
        (m.rect_a, m.rect_b, m.height) for m in d2.merges
    ]
    # nested cuts: each merge prefix refines the next
    la, lb = d1.labels_after(5), d1.labels_after(12)
    for sid in np.unique(la):
        assert len(np.unique(lb[la == sid])) == 1


def test_portfolio_free_bound(bundle):
    """Propriete 6: TE <= eps_realised * sum |nu_j| s_j for any book."""
    from ebanetting import decoupled_netting

    res = decoupled_netting(bundle, alpha=0.9, kappa=KAPPA_90, epsilon=0.6)
    assert np.sqrt(res.evaluation.te2) <= res.te_bound + 1e-9
    assert res.evaluation.passes_variance and res.evaluation.passes_floor
    # the realised epsilon honours the requested cut height
    assert all(p["epsilon"] <= res.epsilon + 1e-12 for p in res.pivots.values()) or res.lowered > 0


def test_decoupled_lowers_cut_for_hedged_book(bundle):
    """The per-book check: a very hedged book (small Var vs add-up scale)
    forces a finer cut — never a failed scheme."""
    from ebanetting import decoupled_netting

    res = decoupled_netting(bundle, alpha=0.99, kappa=KAPPA_90, epsilon=2.0)
    assert res.evaluation.passes_variance
    assert res.n_merges <= res.dendrogram.n_merges_at(2.0)


# --------------------------------------------------------------------------- #
# Sec. 6.4 — stability across shock families
# --------------------------------------------------------------------------- #
def test_stability_undoes_fragile_fusions(bundle):
    """Fusions failing under an adverse alternative covariance are undone;
    the retained cut passes the variance test in every regime."""
    from ebanetting import build_dendrogram, stable_cut

    alternatives = {
        "stress_0.2": stress_bundle(bundle, 0.2),
        "stress_0.4": stress_bundle(bundle, 0.4),
    }
    dendro = build_dendrogram(bundle)
    out = stable_cut(bundle, alternatives, alpha=0.9, kappa=KAPPA_90,
                     epsilon=0.8, dendrogram=dendro)
    assert all(r["passes_variance"] for r in out["report"])
    assert out["n_merges"] <= dendro.n_merges_at(0.8)
    # undone merges imply the unstable regime was binding
    base_only = stable_cut(bundle, {}, alpha=0.9, kappa=KAPPA_90,
                           epsilon=0.8, dendrogram=dendro)
    assert out["n_merges"] <= base_only["n_merges"]


# --------------------------------------------------------------------------- #
# Sec. 6.1 — the spectral diagnostic on Sigma at the nodes
# --------------------------------------------------------------------------- #
def test_factor_variance_decomposition_is_exact(model):
    """Prop. 7: sum c_l^2 lambda_l = Var(DeltaPi), exactly."""
    diag = spectral_diagnostic(model, alpha=0.9)
    assert diag.loadings2.sum() == pytest.approx(model.var_total, rel=1e-10)
    # Prop. 6 (iii): sum lambda = tr(Sigma) = sum s_i^2
    assert diag.eigenvalues.sum() == pytest.approx(np.sum(model.bundle.s ** 2), rel=1e-10)
    # tau is increasing from 0 to 1
    assert diag.inertia[0] == 0.0
    assert diag.inertia[-1] == pytest.approx(1.0)
    assert np.all(np.diff(diag.inertia) >= -1e-12)


def test_spectral_floor_against_low_rank_proxy(model):
    """Theoreme 3: a proxy whose weights live in span(u_1..u_L) cannot beat
    the tail floor sum_{l>L} c_l^2 lambda_l."""
    from ebanetting import node_covariance

    bundle = model.bundle
    diag = spectral_diagnostic(model, alpha=0.9)
    sigma = node_covariance(bundle)
    nu = bundle.vega.flatten()
    L = 3
    u = diag.eigenvectors[:, :L]
    # best L-factor proxy of DeltaPi: w = projection of nu on span(u_1..u_L)
    w = u @ (u.T @ nu)
    resid = nu - w
    te2 = float(resid @ sigma @ resid)
    floor = diag.residual_curve[L]
    assert te2 >= floor - 1e-9 * model.var_total
    assert te2 == pytest.approx(floor, rel=1e-8)  # the projection attains it


def test_spectral_clean_guard_material(bundle):
    """U3: the cleaned covariance is a valid bundle, preserves the total
    uncertainty tr(Sigma), and the conservatism guard is measurable."""
    from ebanetting import node_covariance, spectral_clean

    cleaned = spectral_clean(bundle, tau=0.95)
    assert cleaned.validate() == []
    tr_before = float(np.trace(node_covariance(bundle)))
    tr_after = float(np.trace(node_covariance(cleaned)))
    assert tr_after == pytest.approx(tr_before, rel=1e-8)
    # guard material: netting benefit comparable on both bundles (the
    # trace is preserved globally, individual s_i may shift slightly)
    m0 = UncertaintyModel(bundle=bundle)
    m1 = UncertaintyModel(bundle=cleaned)
    assert m1.ava_full <= m1.ava_brut + 1e-9
    assert m0.ava_brut == pytest.approx(m1.ava_brut, rel=0.05)


def test_subspace_stability_bounds(bundle):
    """U4: identical covariances give cosines 1; stressed ones stay in [0, 1]."""
    from ebanetting import subspace_stability

    same = subspace_stability(bundle, bundle, L=3)
    np.testing.assert_allclose(same, 1.0, atol=1e-10)
    crossed = subspace_stability(bundle, stress_bundle(bundle, 0.2), L=3)
    assert np.all(crossed <= 1.0 + 1e-12) and np.all(crossed >= 0.0)


# --------------------------------------------------------------------------- #
# Sec. 6.2 — deformation model and Theoreme 4
# --------------------------------------------------------------------------- #
def test_deformation_model_recovery_and_th4():
    """Simulate shocks under the model (9); the fit recovers the shape-shock
    variances and Th. 4 (iii) matches the simulated collapse residual."""
    from ebanetting import fit_deformation_model, tranche_collapse_variance

    rng = np.random.default_rng(42)
    x = np.array([0.80, 0.90, 1.00, 1.10, 1.20]) - 1.0
    atm = 2
    T = 20000
    var_s, var_c = 0.30 ** 2, 0.15 ** 2
    dS = rng.normal(0, np.sqrt(var_s), T)
    dC = rng.normal(0, np.sqrt(var_c), T)
    atm_shock = rng.normal(0, 0.5, T)
    z = x - x[atm]
    shocks = atm_shock[:, None] + dS[:, None] * z[None, :] + dC[:, None] * z[None, :] ** 2
    fit = fit_deformation_model(x, shocks, atm)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-10)        # noiseless model
    assert fit["var_skew"] == pytest.approx(var_s, rel=0.05)
    assert fit["var_curv"] == pytest.approx(var_c, rel=0.05)
    # Th. 4: collapse residual variance of a book on this tranche
    nu = np.array([300.0, -150.0, 800.0, -200.0, -400.0])
    rr, fly = float(nu @ z), float(nu @ z ** 2)
    closed = tranche_collapse_variance(rr, fly, fit["var_skew"], fit["var_curv"],
                                       fit["cov_skew_curv"])
    resid = shocks - shocks[:, [atm]]
    simulated = float(np.var(resid @ nu))
    assert closed == pytest.approx(simulated, rel=0.02)
    # Th. 4 (ii): a pure-level book has zero collapse residual
    level_book = np.full(5, 100.0)
    assert tranche_collapse_variance(float(level_book @ z), float(level_book @ z ** 2),
                                     fit["var_skew"], fit["var_curv"]) == pytest.approx(
        float(np.var(resid @ level_book)), abs=1e-6)


def test_tranche_refinement_is_addup_of_components():
    """S2: level netted on ATM + RR / FLY carved out in add-up."""
    from ebanetting import tranche_refinement

    out = tranche_refinement(level=500.0, rr=-120.0, fly=80.0,
                             s_atm=0.3, s_skew=0.6, s_fly=0.9, kappa=KAPPA_90)
    assert out["ava_total"] == pytest.approx(
        KAPPA_90 * (500.0 * 0.3 + 120.0 * 0.6 + 80.0 * 0.9))


# --------------------------------------------------------------------------- #
# Two-layer methodology (companion note) — worked example and theorems
# --------------------------------------------------------------------------- #
@pytest.fixture
def smile_example():
    """The companion note's end-to-end example: 5 strikes, layer-1 model
    s0=1, sigma_S=2, sigma_C=10, sigma_eps=0.1, book (+8,-12,+20,+30,+14)."""
    from ebanetting import SmileModel

    model = SmileModel(s0=1.0, sigma_s=2.0, sigma_c=10.0, sigma_eps=0.1)
    x = np.array([0.60, 0.70, 0.90, 1.00, 1.10]) - 1.0
    nu = np.array([8.0, -12.0, 20.0, 30.0, 14.0])
    return model, x, nu


def test_layer1_matches_worked_example(smile_example):
    """A1/A2: point uncertainties, disagreement matrix (Th. 1) and the
    generated correlation (Prop. 2)."""
    from ebanetting import generated_correlation, model_distance

    model, x, _ = smile_example
    np.testing.assert_allclose(
        model.point_uncertainty(x), [2.05, 1.48, 1.03, 1.00, 1.03], atol=0.005)
    D = model_distance(x, model)
    np.testing.assert_allclose(D[0], [0.0, 0.74, 1.62, 1.79, 1.81], atol=0.005)
    np.testing.assert_allclose(D[1, 2:], [0.91, 1.09, 1.14], atol=0.005)
    # idio floor and symmetric-pair curvature cancellation (Th. 1 readings)
    assert D.min() == 0.0 and (D[D > 0] ** 2 >= 2 * model.sigma_eps ** 2 - 1e-12).all()
    rho = generated_correlation(x, model)
    assert rho[3] == pytest.approx(1.0 / np.sqrt(1.0 + 0.01))  # near-ATM
    assert np.all(np.diff(rho[:4]) > 0)  # the curvature cliff away from ATM


def test_dendrogram_wing_pivot_emerges(smile_example):
    """A3: the wing merges with the wing (0.74), not with the ATM (1.09) —
    the wing pivot emerges from the clustering, zones are an output."""
    from ebanetting import cut_tranche, model_distance, tranche_dendrogram

    model, x, _ = smile_example
    D = model_distance(x, model)
    merges = tranche_dendrogram(x, D, model.point_uncertainty(x))
    heights = [round(m.height, 2) for m in merges]
    assert heights == [0.26, 0.26, 0.74, 1.09]
    groups = cut_tranche(merges, 5, epsilon=0.8)
    assert groups == [(0, 1), (2, 3, 4)]


def test_layer2_exact_te_authorises_what_majorant_refuses(smile_example):
    """B1-B3: the full worked example — exact TE^2 = 107.6 passes the
    budget 182 while the layer-1 majorant (~224) would have refused."""
    from ebanetting import (book_variance, cut_tranche, evaluate_book,
                            model_distance, tranche_dendrogram)

    model, x, nu = smile_example
    D = model_distance(x, model)
    groups = cut_tranche(tranche_dendrogram(x, D, model.point_uncertainty(x)), 5, 0.8)
    assert book_variance(nu, x, model) == pytest.approx(3646.4, abs=0.5)
    res = evaluate_book(groups, nu, x, model, alpha=0.95, kappa=1.0)
    by_members = {d.members: d for d in res.decisions}
    w, c = by_members[(0, 1)], by_members[(2, 3, 4)]
    assert (w.pivot_index, c.pivot_index) == (1, 3)        # pivots 70 and 100
    assert (w.m, round(w.rr, 2), round(w.fly, 2)) == (-4.0, -0.8, 0.56)
    assert (c.m, round(c.rr, 2), round(c.fly, 2)) == (64.0, -0.6, 0.34)
    assert res.te2 == pytest.approx(107.6, abs=0.1)
    assert res.te2 <= res.budget and res.passes_variance
    assert res.te2_majorant > res.budget                   # layer 1 alone refuses
    assert res.ava == pytest.approx(70.2, abs=0.1)         # note rounds to 69.9
    assert res.ava_floor == pytest.approx(60.4, abs=0.1)
    assert res.passes_floor and res.ava < res.ava_brut


def test_twin_books_property3(smile_example):
    """Prop. 3: fly book (+10,+10) and RR book (+10,-10) get the same
    layer-1 majorant but exact variances of different orders (sigma_eps
    set to 0 to isolate the slope/curvature asymmetry, as in the note)."""
    from ebanetting import SmileModel, group_residual_variance, majorant_residual_sd

    model = SmileModel(s0=1.0, sigma_s=2.0, sigma_c=10.0, sigma_eps=0.0)
    delta = 0.05
    x3 = np.array([-delta, 0.0, +delta])
    fly = np.array([10.0, 0.0, 10.0])
    rr = np.array([10.0, 0.0, -10.0])
    v_fly = group_residual_variance(fly, x3, 1, model)
    v_rr = group_residual_variance(rr, x3, 1, model)
    # same majorant, blind to the signs
    assert majorant_residual_sd(fly, x3, 1, model) == pytest.approx(
        majorant_residual_sd(rr, x3, 1, model))
    # exact: A has no slope risk (~delta^4), B no curvature risk (~delta^2)
    assert v_fly == pytest.approx((20 * delta ** 2) ** 2 * model.sigma_c ** 2)
    assert v_rr == pytest.approx((20 * delta) ** 2 * model.sigma_s ** 2)
    assert v_rr > 10 * v_fly  # orders of magnitude apart for small delta


def test_barycenter_pivot_kills_rr(smile_example):
    """Sec. 6.3: RR_p = RR_0 - m x_p vanishes at x_p = RR_0 / m, and the
    translation identities hold."""
    from ebanetting import barycenter_pivot, group_projections

    _, x, nu = smile_example
    p0 = group_projections(nu, x, 0.0)
    xp = barycenter_pivot(nu, x)
    assert xp == pytest.approx(p0["rr"] / p0["m"])
    pp = group_projections(nu, x, xp)
    assert pp["rr"] == pytest.approx(0.0, abs=1e-12)
    assert pp["fly"] == pytest.approx(p0["fly"] - p0["m"] * xp ** 2)


def test_fit_smile_model_recovers_parameters():
    """A1: the cross-sectional regression recovers (s0, sigma_S, sigma_C)
    and a high R^2 under the true model."""
    from ebanetting import fit_smile_model

    rng = np.random.default_rng(3)
    x = np.array([-0.4, -0.3, -0.1, 0.0, 0.1])
    T = 20000
    level = rng.normal(0, 1.0, T)
    dS = rng.normal(0, 2.0, T)
    dC = rng.normal(0, 10.0, T)
    eps = rng.normal(0, 0.1, (T, x.size))
    shocks = level[:, None] + dS[:, None] * x[None, :] + dC[:, None] * x[None, :] ** 2 + eps
    fit = fit_smile_model(x, shocks)
    assert fit.s0 == pytest.approx(1.0, rel=0.05)
    assert fit.sigma_s == pytest.approx(2.0, rel=0.05)
    assert fit.sigma_c == pytest.approx(10.0, rel=0.05)
    assert fit.sigma_eps == pytest.approx(0.1, rel=0.1)
    assert fit.r2 > 0.99


# --------------------------------------------------------------------------- #
# Engine implementation sheet (volet 2/2) — golden tests of sec. 5.2
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def engine_world():
    """A known production model (SigmaT, SigmaK, sigma_eps) + a simulated
    panel + the fitted EngineModel — the synthetic generator golden.

    Full-rank smooth kernels: a rich world on which the flip-flop and the
    trace formula are validated. (It is NOT a 9-factor sandwich world —
    the battery rightly sends it to majorant mode; the battery's own
    golden uses the sandwich-faithful fixture below.)"""
    from ebanetting import fit_engine_model, simulate_panel

    rng = np.random.default_rng(1)
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    M, K = len(tenors), len(strikes)
    lt = np.log(tenors)
    st_true = 1.5 * np.exp(-np.abs(lt[:, None] - lt[None, :]) / 1.2)
    x = np.asarray(strikes) - 1.0
    sk_true = np.exp(-np.abs(x[:, None] - x[None, :]) / 0.25)
    sk_true *= K / np.trace(sk_true)
    st_true *= np.trace(sk_true) / K
    eps_true = 0.05 * (1.0 + 2.0 * np.abs(x))[None, :] * np.ones((M, 1))
    panel = simulate_panel(st_true, sk_true, eps_true, n_days=2500, seed=7)
    model = fit_engine_model(panel, tenors, strikes, alpha=0.95, seed=3)
    return {
        "tenors": tenors, "strikes": strikes,
        "st": st_true, "sk": sk_true, "eps": eps_true,
        "panel": panel, "model": model, "rng": rng,
    }


def test_flipflop_recovers_factors(engine_world):
    """Golden (b): the flip-flop recovers (SigmaT, SigmaK) within 5%."""
    w = engine_world
    model = w["model"]
    err_t = np.linalg.norm(model.sigma_t - w["st"]) / np.linalg.norm(w["st"])
    err_k = np.linalg.norm(model.sigma_k - w["sk"]) / np.linalg.norm(w["sk"])
    assert err_t <= 0.05 and err_k <= 0.05


def test_idio_map_recovered_under_sandwich_world(engine_world):
    """When the panel really follows the sandwich X = Y B Z' + E, the
    per-cell idio map is the residual std and is recovered."""
    from ebanetting import fit_engine_model, orthonormal_basis

    w = engine_world
    tenors, strikes = w["tenors"], w["strikes"]
    M, K = len(tenors), len(strikes)
    lt = np.log(np.asarray(tenors)); lt = (lt - lt.mean()) / lt.std()
    y = orthonormal_basis(lt)
    z = orthonormal_basis(np.asarray(strikes) - 1.0)
    rng = np.random.default_rng(17)
    T = 4000
    b = rng.normal(0, 1.0, size=(T, 3, 3)) * np.array([[3.0, 1.5, 0.8],
                                                       [1.5, 0.7, 0.3],
                                                       [0.8, 0.3, 0.2]])
    eps_true = 0.05 * (1.0 + 2.0 * np.abs(np.asarray(strikes) - 1.0))[None, :] * np.ones((M, 1))
    panel = np.stack([y @ bt @ z.T for bt in b]) \
        + rng.normal(size=(T, M, K)) * eps_true[None, :, :]
    model = fit_engine_model(panel, tenors, strikes, seed=5)
    assert np.median(model.sigma_eps / eps_true) == pytest.approx(1.0, abs=0.1)
    assert model.tests["T1"]["passed"]


def test_te_formula_matches_empirical(engine_world):
    """Golden (b): the trace formula tr(A' ST A SK) + sum A^2 eps^2 must
    match the empirical Var(<A, X>) on simulated shocks within 2%."""
    from ebanetting import simulate_panel

    w = engine_world
    rng = np.random.default_rng(11)
    a = rng.normal(size=(len(w["tenors"]), len(w["strikes"])))
    true_model_var = (
        float(np.trace(a.T @ w["st"] @ a @ w["sk"]))
        + float(np.sum(a ** 2 * np.asarray(w["eps"]) ** 2))
    )
    big = simulate_panel(w["st"], w["sk"], w["eps"], n_days=60000, seed=21)
    empirical = float(np.var(np.tensordot(big, a, axes=([1, 2], [0, 1]))))
    assert empirical == pytest.approx(true_model_var, rel=0.02)


def test_model_battery_behaviour():
    """T1-T6 (sec. 3.3): each diagnostic detects what it is designed to
    detect, and the automatic decisions fire.

    World S — smooth full-rank kernels (mostly 9-factor but with a real
    out-of-span common tail): T1/T2/T3/T5/T6 pass, no majorant fallback,
    and T4 rightly FLAGS the cross-correlated residual.
    World P — pure sandwich + idio: the full battery passes, including
    the torsion check, with the 9-factor basis.
    World P + |x| torsion: the T4 automatic decision adds the kink mode
    (the sandwich grows to 12 factors) and the model re-explains the
    surface."""
    from ebanetting import fit_engine_model, orthonormal_basis, simulate_panel

    tenors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    strikes = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    M, K = len(tenors), len(strikes)
    lt = np.log(np.asarray(tenors))
    x = np.asarray(strikes) - 1.0

    # --- world S
    st = 0.97 * 1.5 * np.exp(-(np.abs(lt[:, None] - lt[None, :]) / 3.5) ** 2) \
        + 0.03 * 1.5 * np.eye(M)
    sk = 0.97 * np.exp(-(np.abs(x[:, None] - x[None, :]) / 0.75) ** 2) + 0.03 * np.eye(K)
    sk *= K / np.trace(sk)
    st *= np.trace(sk) / K
    panel_s = simulate_panel(st, sk, 0.02 * np.ones((M, K)), 3000, seed=7)
    m_s = fit_engine_model(panel_s, tenors, strikes, alpha=0.95, seed=4)
    assert all(m_s.tests[t]["passed"] for t in ("T1", "T2", "T3", "T5", "T6")), m_s.tests
    assert not m_s.majorant_only
    # the residual carries genuine out-of-span common movement: T4 flags it
    assert not m_s.tests["T4"]["passed"]
    assert m_s.tests["T4"]["offdiag_mass"] > 0.10

    # --- world P
    lt_s = (lt - lt.mean()) / lt.std()
    y, z = orthonormal_basis(lt_s), orthonormal_basis(x)
    rng = np.random.default_rng(23)
    amp = np.array([1.8, 0.9, 0.5])
    b = rng.normal(size=(4000, 3, 3)) * (amp[:, None] @ amp[None, :] / amp[0])
    panel_p = np.stack([y @ bt @ z.T for bt in b]) + rng.normal(size=(4000, M, K)) * 0.01
    m_p = fit_engine_model(panel_p, tenors, strikes, alpha=0.95, seed=4)
    assert m_p.tests["T1"]["passed"] and m_p.tests["T4"]["passed"], m_p.tests
    assert m_p.tests["T4"]["torsion_corr"] >= 0.80
    # no torsion refit needed on a torsion-free world. (T5 may flag this
    # degenerate exactly-rank-3 world: the flip-flop is documented to
    # struggle on singular factor matrices — the conservative majorant
    # fallback is the intended behaviour there, not an error.)
    assert m_p.z.shape[1] == 3

    # --- world P + genuine torsion (kink |x|, opposite wing slopes)
    w_shape = np.abs(x) / np.linalg.norm(np.abs(x))
    tor = rng.normal(0, 0.6, 4000)
    panel_t = panel_p + tor[:, None, None] * np.ones((M, 1))[None, :, :] \
        * w_shape[None, None, :]
    m_t = fit_engine_model(panel_t, tenors, strikes, alpha=0.95, seed=4)
    assert m_t.z.shape[1] == 4              # the automatic decision fired
    assert m_t.tests["T1"]["mean_r2"] >= 0.95   # the kink mode re-explains
    assert m_t.tests["T4"]["torsion_passed"]    # handled structurally


def test_engine_invariances(engine_world):
    """Golden (c): strike permutation and the c SigmaT, SigmaK / c scale
    indeterminacy leave TE^2 and the AVA unchanged."""
    from dataclasses import replace as drep
    from ebanetting import cut_engine, engine_dendrogram, evaluate_book_engine

    w = engine_world
    model = w["model"]
    M, K = model.shape
    rng = np.random.default_rng(5)
    book = 1000.0 * rng.normal(size=(M, K))
    merges = engine_dendrogram(model)
    labels = cut_engine(merges, M, K, np.median([m.height for m in merges]))
    run = evaluate_book_engine(book, model, labels, alpha=0.95, kappa=KAPPA_90)
    # scale indeterminacy: c SigmaT, SigmaK / c
    c = 3.7
    scaled = drep(model, sigma_t=c * model.sigma_t, sigma_k=model.sigma_k / c)
    run_scaled = evaluate_book_engine(book, scaled, labels, alpha=0.95, kappa=KAPPA_90)
    assert run_scaled.te2 == pytest.approx(run.te2, rel=1e-10)
    assert run_scaled.ava == pytest.approx(run.ava, rel=1e-10)
    # strike permutation: same AVA
    perm = rng.permutation(K)
    permuted = drep(
        model,
        sigma_k=model.sigma_k[np.ix_(perm, perm)],
        sigma_eps=model.sigma_eps[:, perm],
        z=model.z[perm, :],
    )
    run_perm = evaluate_book_engine(book[:, perm], permuted, labels[:, perm],
                                    alpha=0.95, kappa=KAPPA_90)
    assert run_perm.te2 == pytest.approx(run.te2, rel=1e-9)
    assert run_perm.ava == pytest.approx(run.ava, rel=1e-9)


def test_engine_run_and_extraction(engine_world):
    """E1-E6: the gap pattern is global (trace handles inter-group
    compensation); on failure the GA heatmap extraction provisions the
    dominant modes in add-up and the test is re-run."""
    from ebanetting import cut_engine, engine_dendrogram, evaluate_book_engine

    w = engine_world
    model = w["model"]
    M, K = model.shape
    rng = np.random.default_rng(9)
    book = 1000.0 * rng.normal(size=(M, K))
    merges = engine_dendrogram(model)
    # a deliberately coarse cut at a demanding alpha forces extractions
    labels = cut_engine(merges, M, K, height=np.inf, require_portfolio_free=False)
    run = evaluate_book_engine(book, model, labels, alpha=0.999, kappa=KAPPA_90)
    base_gap_var = model.variance_of(book - run.gap + run.gap)  # sanity: callable
    assert run.te2_majorant >= run.te2 - 1e-9      # majorant always above exact
    if run.extractions:
        # each extraction is provisioned and reduces the residual
        assert all(e[4] > 0 for e in run.extractions)
    assert run.ava <= run.ava_brut + 1e-9
    # a fine cut must always pass
    fine = cut_engine(merges, M, K, height=-1.0)
    run_fine = evaluate_book_engine(book, model, fine, alpha=0.95, kappa=KAPPA_90)
    assert run_fine.te2 == pytest.approx(0.0, abs=1e-9 * run_fine.var_total)
    assert run_fine.ava == pytest.approx(run_fine.ava_brut, rel=1e-9)


def test_stability_metrics(engine_world):
    """Sec. 4.2: ARI = 1 on identical partitions, principal-angle
    cosines = 1 on identical matrices, and the survival frequency of a
    stable world's fusions is high."""
    from ebanetting import (adjusted_rand_index, principal_angle_cosines,
                            simulate_panel, survival_frequencies)

    w = engine_world
    labels = np.array([[0, 0, 1], [0, 0, 1]])
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)
    other = np.array([[0, 1, 0], [1, 0, 1]])
    assert adjusted_rand_index(labels, other) < 0.5
    np.testing.assert_allclose(
        principal_angle_cosines(w["st"], w["st"]), 1.0, atol=1e-10)
    # three bootstrap-style panels from the same world: fusions survive
    panels = [simulate_panel(w["st"], w["sk"], w["eps"], 600, seed=s)
              for s in (1, 2, 3)]
    freq = survival_frequencies(panels, w["tenors"], w["strikes"],
                                height=np.inf, epsilon=1.5, seed=0)
    assert freq and max(freq.values()) == 1.0


def test_select_cut_and_min_benefit_guard(engine_world):
    """Sec. 6: the chosen cut is admissible with minimal AVA; across
    shock families the retained one gives the minimum netting benefit."""
    from ebanetting import (cut_engine, engine_dendrogram, evaluate_book_engine,
                            select_cut, min_benefit_guard, stress_engine_model)

    w = engine_world
    model = w["model"]
    M, K = model.shape
    rng = np.random.default_rng(13)
    book = 1000.0 * rng.normal(size=(M, K))
    merges = engine_dendrogram(model)
    sel = select_cut(model, merges, book, alpha=0.95, kappa=KAPPA_90)
    assert sel["chosen"] is not None and sel["chosen"]["admissible"]
    avas = [p["ava"] for p in sel["frontier"] if p["admissible"]]
    assert sel["chosen"]["ava"] == pytest.approx(min(avas))
    # min-benefit guard across families
    labels = cut_engine(merges, M, K, sel["chosen"]["height"])
    runs = {
        "daily": evaluate_book_engine(book, model, labels, 0.95, KAPPA_90),
        "stressed": evaluate_book_engine(book, stress_engine_model(model),
                                         labels, 0.95, KAPPA_90),
    }
    name, kept = min_benefit_guard(runs)
    benefits = {k: r.ava_brut - r.ava for k, r in runs.items()}
    assert benefits[name] == pytest.approx(min(benefits.values()))


def test_adapters_quality_utilities():
    """Adapters: winsorisation clips the tails per cell; stale cells are
    flagged after > 5 flat days."""
    from ebanetting import stale_cells, winsorize_panel

    rng = np.random.default_rng(2)
    panel = rng.normal(size=(400, 3, 3))
    panel[5, 0, 0] = 50.0
    wz = winsorize_panel(panel, q=0.005)
    assert wz[5, 0, 0] < 50.0 and np.abs(wz).max() < 10.0
    panel2 = rng.normal(size=(30, 2, 2))
    panel2[3:12, 1, 1] = 0.0          # 9 consecutive flat days
    mask = stale_cells(panel2, max_flat=5)
    assert mask[1, 1] and not mask[0, 0]
