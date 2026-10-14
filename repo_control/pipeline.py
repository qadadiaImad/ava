"""Controller pipeline: Totem consensus + FO marks -> implied repo, carry difference, noise.

This is the part that runs on real data.  Inputs per underlying and date:

* :class:`TotemSnapshot` -- per expiry the consensus call/put prices on a
  strike grid, the contributor count and forward dispersion; per dividend
  bucket the consensus level, dispersion and contributor count;
* :class:`FOSnapshot` -- spot, rate curve, FO repo marks at the Totem
  pillars and the FO dividend forecast (ex-dates and cash amounts).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .buckets import BucketSchedule, allocate, allocation_matrix, bucket_levels
from .curves import Dividends, ZeroCurve, bootstrap_repo, carry_dividends, forward
from .detect import DetectionContext
from .noise import carry_covariance, consensus_covariance, exp_correlation, robust_dispersion
from .parity import ParityFit, forward_from_parity


@dataclass(frozen=True)
class TotemSnapshot:
    pillars: np.ndarray
    ref_spot: float
    strikes: np.ndarray
    call: np.ndarray
    put: np.ndarray
    n_opt: np.ndarray
    fwd_sd: np.ndarray
    schedule: BucketSchedule
    div_level: np.ndarray
    div_sd: np.ndarray
    n_div: np.ndarray


@dataclass(frozen=True)
class FOSnapshot:
    spot: float
    rates: ZeroCurve
    repo: ZeroCurve
    dividends: Dividends


@dataclass
class ImpliedRepo:
    pillars: np.ndarray
    spot: float
    forwards: np.ndarray
    fo_forwards: np.ndarray
    parity: list
    totem_dividends: Dividends
    repo_totem: ZeroCurve
    repo_fwd_only: ZeroCurve
    repo_fo: np.ndarray
    carry_div: np.ndarray
    alloc: np.ndarray
    cov: np.ndarray
    level_cov: np.ndarray
    fo_levels: np.ndarray
    n_opt: np.ndarray
    n_div_at_pillar: np.ndarray
    profile: Dividends
    dispersion_ratio: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def delta_carry(self) -> np.ndarray:
        return (self.repo_totem.zeros - self.repo_fo) * self.pillars

    @property
    def delta_repo(self) -> np.ndarray:
        return self.repo_totem.zeros - self.repo_fo

    @property
    def carry_sd(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def context(self) -> DetectionContext:
        return DetectionContext(
            self.spot, self.carry_div, self.profile, self.n_opt, self.n_div_at_pillar, self.dispersion_ratio
        )


def imply_totem_repo(
    totem: TotemSnapshot,
    fo: FOSnapshot,
    allocation_profile: Dividends | None = None,
    fwd_factor_weight: float = 0.5,
    div_corr_length: float = 2.0,
) -> ImpliedRepo:
    T = np.asarray(totem.pillars, dtype=float)
    S = fo.spot
    fits: list[ParityFit] = [forward_from_parity(totem.strikes[k], totem.call[k], totem.put[k]) for k in range(T.size)]
    F = np.array([f.forward for f in fits])
    # express Totem forwards on the FO spot (first order, growth from the FO curves)
    growth = np.exp(fo.rates.cum(T) - fo.repo.cum(T))
    F = F + growth * (S - totem.ref_spot)
    profile = fo.dividends if allocation_profile is None else allocation_profile
    divs_tot = allocate(totem.schedule, totem.div_level, profile)
    repo_tot = bootstrap_repo(S, fo.rates, divs_tot, T, F)
    repo_fwd = bootstrap_repo(S, fo.rates, fo.dividends, T, F)
    repo_fo = fo.repo.zero(T)
    carry_div = carry_dividends(S, fo.rates, repo_tot, divs_tot, T)
    alloc = allocation_matrix(totem.schedule, profile, T, fo.rates, fo.repo)
    quote_var = np.array([(f.forward_se / f.forward) ** 2 for f in fits])
    fwd_sd, dispersion_ratio = robust_dispersion(totem.fwd_sd)
    cov = carry_covariance(
        T,
        F,
        fwd_sd,
        totem.n_opt,
        totem.div_sd,
        totem.n_div,
        alloc,
        S,
        carry_div,
        totem.schedule.mid,
        fwd_factor_weight=fwd_factor_weight,
        div_corr_length=div_corr_length,
        quote_var=quote_var,
    )
    level_cov = consensus_covariance(totem.div_sd, totem.n_div, exp_correlation(totem.schedule.mid, div_corr_length))
    bucket_of_pillar = np.clip(totem.schedule.index(T), 0, totem.schedule.size - 1)
    return ImpliedRepo(
        pillars=T,
        spot=S,
        forwards=F,
        fo_forwards=forward(S, fo.rates, fo.repo, fo.dividends, T),
        parity=fits,
        totem_dividends=divs_tot,
        repo_totem=repo_tot,
        repo_fwd_only=repo_fwd,
        repo_fo=repo_fo,
        carry_div=carry_div,
        alloc=alloc,
        cov=cov,
        level_cov=level_cov,
        fo_levels=bucket_levels(fo.dividends, totem.schedule),
        n_opt=np.asarray(totem.n_opt, dtype=float),
        n_div_at_pillar=np.asarray(totem.n_div, dtype=float)[bucket_of_pillar],
        profile=profile,
        dispersion_ratio=dispersion_ratio,
    )
