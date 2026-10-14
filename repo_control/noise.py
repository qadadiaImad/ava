"""Uncertainty of the Totem-implied carry from consensus statistics.

The implied cumulative repo ``B_k = b_k T_k`` at Totem pillar ``k`` mixes two
independent consensus means:

* the option panel (forward), size ``n_A,k``, contributor dispersion ``sd_F,k``;
* the dividend panel (annual buckets), size ``n_B,y``, dispersion ``sd_D,y``.

To first order ``B_tot - B_true = -e_F - e_D`` with ``e_F = d ln F`` and
``e_D = dD / (S - D)``.  Panels are assumed *nested* (contributors quoting a
long tenor also quote the shorter ones) and a contributor's error is persistent
across tenors with correlation ``rho``, so the covariance of two consensus
means is ``rho_kl sd_k sd_l / max(n_k, n_l)``.  This reproduces the jump
variance ``sd^2 (1/m - 1/n)`` when a panel shrinks from ``n`` to ``m``.
"""
from __future__ import annotations

import numpy as np


def exp_correlation(x, length: float) -> np.ndarray:
    """``exp(-|x_i - x_j| / length)`` correlation matrix."""
    x = np.asarray(x, dtype=float)
    return np.exp(-np.abs(x[:, None] - x[None, :]) / length)


def cumulative_covariance(sd, factor_weight: float = 0.5) -> np.ndarray:
    """Covariance of a cumulative error (carry, cumulative dividends) across tenors.

    A contributor's forward error is a persistent factor (level of its views)
    plus increments that arrive with each dividend.  With ``s`` the
    non-decreasing dispersion, the covariance is
    ``w s_k s_l + (1 - w) min(s_k, s_l)^2``; for pure increments the
    correlation between ``k < l`` is ``s_k / s_l``.
    """
    s = np.maximum.accumulate(np.maximum(np.asarray(sd, dtype=float), 0.0))
    return factor_weight * np.outer(s, s) + (1 - factor_weight) * np.minimum.outer(s, s) ** 2


def forward_panel_covariance(pillars, lnF_sd, n, factor_weight: float = 0.5, repo_sd: float | None = None) -> np.ndarray:
    """Covariance of the option-panel consensus error in ``ln F`` across pillars.

    Contributor forwards differ through their repo views (persistent: a level
    error grows like ``sd_b T``) and their dividend views (cumulative: factor
    plus increments at ex-dates, see :func:`cumulative_covariance`).  ``sd_b`` is read off the shortest pillars,
    where dividend uncertainty is negligible; the rest of the dispersion is
    attributed to dividends.
    """
    T = np.asarray(pillars, dtype=float)
    s = np.asarray(lnF_sd, dtype=float)
    if repo_sd is None:
        # median rather than min: the min of three noisy estimates is biased low
        repo_sd = float(np.median(s[: min(3, s.size)] / T[: min(3, s.size)]))
    repo_part = repo_sd * T
    div_sd = np.sqrt(np.maximum(s**2 - repo_part**2, 0.0))
    cov_div = cumulative_covariance(div_sd, factor_weight)
    cov = np.outer(repo_part, repo_part) + cov_div
    n = np.asarray(n, dtype=float)
    return cov / np.maximum(n[:, None], n[None, :])


def consensus_covariance(sd, n, corr) -> np.ndarray:
    """Covariance of consensus means for nested panels."""
    sd = np.asarray(sd, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.asarray(corr) * np.outer(sd, sd) / np.maximum(n[:, None], n[None, :])


def carry_covariance(
    pillars,
    forwards,
    fwd_sd,
    fwd_n,
    div_sd,
    div_n,
    alloc,
    spot: float,
    carry_div,
    bucket_mid,
    fwd_factor_weight: float = 0.5,
    div_corr_length: float = 2.0,
    quote_var=None,
    jitter: float = 1e-14,
) -> np.ndarray:
    """Null covariance of ``Delta B_k`` (Totem minus FO cumulative repo).

    The option-panel term uses :func:`forward_panel_covariance`; the dividend
    term propagates the bucket covariance through the allocation matrix.

    Parameters
    ----------
    alloc:
        ``(K, Y)`` matrix ``dD_k / dLevel_y``: carry-discounted share of bucket
        ``y`` that goes ex before pillar ``k`` under the controller's allocation.
    carry_div:
        Carry-discounted cumulative dividends ``D_k`` at the pillars.
    quote_var:
        Optional extra diagonal variance of ``ln F`` from the parity fit.
    """
    T = np.asarray(pillars, dtype=float)
    lnF_sd = np.asarray(fwd_sd, dtype=float) / np.asarray(forwards, dtype=float)
    cov_f = forward_panel_covariance(T, lnF_sd, fwd_n, fwd_factor_weight)
    if quote_var is not None:
        cov_f = cov_f + np.diag(np.asarray(quote_var, dtype=float))
    cov_u = consensus_covariance(div_sd, div_n, exp_correlation(bucket_mid, div_corr_length))
    J = np.asarray(alloc, dtype=float) / (spot - np.asarray(carry_div, dtype=float))[:, None]
    cov_d = J @ cov_u @ J.T
    cov = cov_f + cov_d
    return cov + jitter * np.eye(T.size)


def robust_dispersion(sd, cap: float = 1.5):
    """Cap isolated dispersion spikes at ``cap`` times the larger neighbour.

    A spike in contributor dispersion at one expiry (typically a split view on
    an ex-date) is a symptom, not noise: left in the covariance it would hide
    the shift it causes in the consensus mean.  Returns the capped profile and
    the ratio of each pillar's dispersion to its larger neighbour.
    """
    sd = np.asarray(sd, dtype=float)
    nb = np.empty_like(sd)
    nb[0], nb[-1] = sd[1], sd[-2]
    nb[1:-1] = np.maximum(sd[:-2], sd[2:])
    ratio = sd / np.maximum(nb, 1e-300)
    return np.minimum(sd, cap * nb), ratio


def panel_gap_sd(T, spot: float, carry_div, div_sd, n_opt, n_div):
    """Repo noise from the panel gap for disjoint panels.

    ``sd(b_implied - mean_A(b)) = sd_D * sqrt(1/n_A + 1/n_B) / (T (S - D))``
    """
    T = np.asarray(T, dtype=float)
    return np.asarray(div_sd) * np.sqrt(1.0 / np.asarray(n_opt) + 1.0 / np.asarray(n_div)) / (
        T * (spot - np.asarray(carry_div))
    )


def nested_jump_sd(sd, n_short, n_long):
    """Standard deviation of the consensus jump when a nested panel shrinks from ``n_short`` to ``n_long``."""
    return np.asarray(sd) * np.sqrt(1.0 / np.asarray(n_long) - 1.0 / np.asarray(n_short))
