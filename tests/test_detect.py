import numpy as np
import pytest

from repo_control.curves import Dividends
from repo_control.detect import DetectionContext, StructuralDetector, nuisance_design

T = np.array([1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60]) / 12.0
DIVS = Dividends([0.55, 1.55, 2.55, 3.55], [3.0, 3.1, 3.2, 3.3])
CTX = DetectionContext(100.0, np.interp(T, [0, 0.55, 1.55, 2.55, 3.55, 5], [0, 3, 6.1, 9.3, 12.6, 12.6]), DIVS, np.full(T.size, 12.0), np.full(T.size, 8.0))


def _cov():
    sd = 1e-4 * (0.3 + 3 * T**1.5)
    corr = np.exp(-np.abs(np.log(T)[:, None] - np.log(T)[None, :]))
    return corr * np.outer(sd, sd)


def test_false_positive_rate_is_calibrated_under_the_null():
    rng = np.random.default_rng(3)
    cov = _cov()
    L = np.linalg.cholesky(cov)
    det = StructuralDetector(alpha=0.05, n_sim=2000)
    smooth = nuisance_design(T) @ np.array([5e-4, -2e-4])
    flags = [det.detect(T, smooth + L @ rng.standard_normal(T.size), cov, CTX).flagged for _ in range(600)]
    assert np.mean(flags) == pytest.approx(0.05, abs=0.025)


def test_null_model_absorbs_smooth_repo_difference():
    cov = _cov()
    det = StructuralDetector(alpha=0.01)
    d = det.detect(T, nuisance_design(T) @ np.array([30e-4, 5e-4]), cov, CTX)
    assert not d.flagged
    assert d.repo_level == pytest.approx(30e-4)
    assert d.fo_mismark


@pytest.mark.parametrize(
    "vector, kind, pillar",
    [
        (np.ones(T.size) * 8e-4, "level", -1),
        ((np.arange(T.size) == 4) * 25e-4, "spike", 4),
        ((np.arange(T.size) >= 7) * 60e-4, "step", 7),
    ],
)
def test_injected_template_is_recovered(vector, kind, pillar):
    rng = np.random.default_rng(4)
    cov = _cov()
    noise = np.linalg.cholesky(cov) @ rng.standard_normal(T.size) * 0.3
    d = StructuralDetector(alpha=0.01).detect(T, vector + noise, cov, CTX)
    assert d.flagged
    top = d.explanations[0]
    assert (top.kind, top.pillar) == (kind, pillar)


def test_variance_scale_calibration_hits_target():
    rng = np.random.default_rng(5)
    cov = _cov()
    L = np.linalg.cholesky(cov)
    det = StructuralDetector(alpha=0.02, n_sim=2000)
    stats = [det.null_statistics(T, 1.7 * L @ rng.standard_normal(T.size), cov, CTX) for _ in range(400)]
    kappa = det.calibrate_to_fpr(stats)
    assert 1.7**2 * 0.6 < kappa < 1.7**2 * 1.6
