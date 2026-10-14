"""Baseline detectors the structural approach is benchmarked against.

* ``naive_level_score``: largest |Totem - FO| repo gap over the pillars (bp).
* ``naive_roughness_score``: largest deviation of the Totem implied repo from
  the linear interpolation of its two neighbours (a shape check in repo space).
* ``SplineGMM``: the starter idea -- cubic B-spline coefficients of the curve
  (knots in ln T) clustered with a Gaussian mixture per category; the anomaly
  score is the negative log-likelihood.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from sklearn.mixture import GaussianMixture


def naive_level_score(delta_repo) -> float:
    return float(np.max(np.abs(np.asarray(delta_repo, dtype=float))))


def naive_roughness_score(pillars, repo) -> float:
    T = np.asarray(pillars, dtype=float)
    b = np.asarray(repo, dtype=float)
    w = (T[1:-1] - T[:-2]) / (T[2:] - T[:-2])
    interp = b[:-2] + w * (b[2:] - b[:-2])
    return float(np.max(np.abs(b[1:-1] - interp)))


class SplineGMM:
    def __init__(self, pillars, n_basis: int = 6, max_components: int = 4, seed: int = 0) -> None:
        x = np.log(np.asarray(pillars, dtype=float))
        degree = 3
        n_inner = n_basis - degree - 1
        inner = np.linspace(x[0], x[-1], n_inner + 2)[1:-1]
        knots = np.concatenate([[x[0]] * (degree + 1), inner, [x[-1]] * (degree + 1)])
        design = BSpline.design_matrix(x, knots, degree).toarray()
        self._pinv = np.linalg.pinv(design)
        self.max_components = max_components
        self.seed = seed
        self.models: dict[str, GaussianMixture] = {}

    def features(self, curve) -> np.ndarray:
        """Least-squares B-spline coefficients of a curve sampled at the pillars."""
        return self._pinv @ np.asarray(curve, dtype=float)

    def fit(self, category: str, feature_rows) -> GaussianMixture:
        X = np.asarray(feature_rows, dtype=float)
        best, best_bic = None, np.inf
        for n in range(1, self.max_components + 1):
            gm = GaussianMixture(n, covariance_type="full", reg_covar=1e-6, random_state=self.seed, n_init=3)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best, best_bic = gm, bic
        self.models[category] = best
        return best

    def score(self, category: str, features) -> float:
        return float(-self.models[category].score_samples(np.atleast_2d(features))[0])
