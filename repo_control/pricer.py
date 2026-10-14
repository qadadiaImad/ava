"""A deliberately small pricer, used to produce the bucketed sensitivities a
real FO risk system would report (repo by pillar, dividends by bucket).

The control layer only consumes those sensitivities; the pricer exists so the
prototype can check first-order control lines against full revaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr

from .buckets import BucketSchedule, allocate
from .curves import Dividends, ZeroCurve, forward


@dataclass(frozen=True)
class Market:
    spot: float
    rates: ZeroCurve
    repo: ZeroCurve
    schedule: BucketSchedule
    div_levels: np.ndarray
    profile: Dividends

    @property
    def dividends(self) -> Dividends:
        return allocate(self.schedule, self.div_levels, self.profile)

    def forward(self, T) -> np.ndarray:
        return forward(self.spot, self.rates, self.repo, self.dividends, T)

    def with_repo(self, repo: ZeroCurve) -> "Market":
        return replace(self, repo=repo)

    def with_levels(self, levels) -> "Market":
        return replace(self, div_levels=np.asarray(levels, dtype=float))


@dataclass(frozen=True)
class EuropeanOption:
    is_call: bool
    strike: float
    expiry: float
    vol: float
    notional: float = 1.0

    def price(self, m: Market) -> float:
        F = float(m.forward(self.expiry)[0])
        df = float(m.rates.df(self.expiry))
        s = self.vol * np.sqrt(self.expiry)
        d1 = (np.log(F / self.strike) + 0.5 * s * s) / s
        d2 = d1 - s
        if self.is_call:
            v = df * (F * ndtr(d1) - self.strike * ndtr(d2))
        else:
            v = df * (self.strike * ndtr(-d2) - F * ndtr(-d1))
        return self.notional * float(v)


@dataclass(frozen=True)
class EquityForward:
    expiry: float
    strike: float
    notional: float = 1.0

    def price(self, m: Market) -> float:
        F = float(m.forward(self.expiry)[0])
        return self.notional * float(m.rates.df(self.expiry)) * (F - self.strike)


@dataclass(frozen=True)
class DividendSwap:
    """Receive the realised dividends of one bucket against a fixed strike, paid at ``pay_time``."""

    bucket: int
    strike: float
    pay_time: float
    notional: float = 1.0

    def price(self, m: Market) -> float:
        return self.notional * float(m.rates.df(self.pay_time)) * (m.div_levels[self.bucket] - self.strike)


@dataclass(frozen=True)
class StockLoanFee:
    """Lend stock until ``expiry`` and receive the borrow fee: a repo-only exposure."""

    expiry: float
    notional: float = 1.0

    def price(self, m: Market) -> float:
        return self.notional * m.spot * float(m.repo.cum(self.expiry)) * float(m.rates.df(self.expiry))


def book_value(book, m: Market) -> float:
    return float(sum(inst.price(m) for inst in book))


def sensitivities(book, m: Market, h_rate: float = 1e-6, h_div: float = 1e-6):
    """Central-difference sensitivities: per unit zero repo at each repo pillar, per unit dividend level per bucket."""
    sens_repo = np.zeros(m.repo.times.size)
    for k in range(sens_repo.size):
        up = book_value(book, m.with_repo(m.repo.bumped(k, h_rate)))
        dn = book_value(book, m.with_repo(m.repo.bumped(k, -h_rate)))
        sens_repo[k] = (up - dn) / (2 * h_rate)
    sens_div = np.zeros(m.schedule.size)
    for y in range(sens_div.size):
        lv_up = m.div_levels.copy()
        lv_up[y] += h_div
        lv_dn = m.div_levels.copy()
        lv_dn[y] -= h_div
        sens_div[y] = (book_value(book, m.with_levels(lv_up)) - book_value(book, m.with_levels(lv_dn))) / (2 * h_div)
    return sens_repo, sens_div
