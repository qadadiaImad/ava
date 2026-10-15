"""Worked control examples and benchmark summaries used by the methodology notes."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from repo_control.benchmark import BenchmarkResult
from repo_control.buckets import BucketSchedule
from repo_control.control import control_lines, implied_repo_jacobian, split_materiality
from repo_control.curves import Dividends, ZeroCurve, bootstrap_repo
from repo_control.detect import StructuralDetector
from repo_control.noise import consensus_covariance, exp_correlation, nested_jump_sd
from repo_control.pricer import (
    DividendSwap,
    EquityForward,
    EuropeanOption,
    Market,
    StockLoanFee,
    book_value,
    sensitivities,
)
from repo_control.synthetic import EXPECTED_KIND, PILLARS


# ------------------------------------------------- control identity examples
def cancellation_table():
    """1Y ATM call, Totem forward 20bp below FO, three dividend panels (single PV dividend)."""
    S, r, T, K, vol = 100.0, 0.03, 1.0, 100.0, 0.25
    D_fo, q_fo = 3.5, 0.006

    def fwd(D, q):
        return (S - D) * np.exp((r - q) * T)

    def call(D, q):
        F = fwd(D, q)
        s = vol * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * s * s) / s
        return np.exp(-r * T) * (F * ndtr(d1) - K * ndtr(d1 - s))

    F_tot = fwd(D_fo, q_fo) * (1 - 0.002)
    h = 1e-6
    s_div = (call(D_fo + h, q_fo) - call(D_fo - h, q_fo)) / (2 * h)
    s_repo = (call(D_fo, q_fo + h) - call(D_fo, q_fo - h)) / (2 * h)
    rows = []
    for D_tot in (3.5, 3.0, 4.2):
        q_tot = r - np.log(F_tot / (S - D_tot)) / T
        div_line = s_div * (D_tot - D_fo)
        repo_line = s_repo * (q_tot - q_fo)
        full = call(D_tot, q_tot) - call(D_fo, q_fo)
        rows.append((D_tot, q_tot * 1e4, div_line, repo_line, div_line + repo_line, full))
    return rows


def demo_market():
    """Eleven repo pillars, six dividend periods and the three books A, B and C of the notes."""
    T = PILLARS
    sch = BucketSchedule(np.array([0, 0.2164, 1.2164, 2.219, 3.219, 4.219, 5.22]))
    prof = Dividends(np.array([0.55, 1.55, 2.55, 3.55, 4.55]), np.ones(5))
    levels = np.array([0.0, 3.5, 3.7, 3.9, 4.1, 4.3])
    m = Market(100.0, ZeroCurve.flat(0.03), ZeroCurve(T, 0.006 + 0 * T), sch, levels, prof)
    options = [EuropeanOption(True, 100, 1.0, 0.25, 10), EuropeanOption(False, 95, 2.0, 0.25, -5),
               EquityForward(3.0, 95, 2)]
    books = {
        "A: options and forwards": options,
        "B: A + short 2028 dividend swap": options + [DividendSwap(2, 3.7, 2.3, -8)],
        "C: A + 1Y stock-loan fee": options + [StockLoanFee(1.0, 3)],
    }
    return m, books


def book_lines(bumps=(-0.4, 0.0, 0.5)):
    """Control lines per book when the Totem dividend panel moves 2027-2028 by ``bump`` (cash), and materiality."""
    m, books = demo_market()
    T = PILLARS
    F_tot = m.forward(T) * (1 - 0.002)
    repo_fwd = bootstrap_repo(m.spot, m.rates, m.dividends, T, F_tot)
    J = implied_repo_jacobian(m.spot, m.rates, m.schedule, m.div_levels, m.profile, T, F_tot)
    out = []
    level_cov = consensus_covariance(0.05 * np.maximum(m.div_levels, 1e-9), np.full(m.div_levels.size, 10.0),
                                     exp_correlation(m.schedule.mid, 2.0))
    for name, book in books.items():
        sr, sd = sensitivities(book, m)
        mat = split_materiality(sd, sr, J, level_cov)
        for bump in bumps:
            lv = m.div_levels.copy()
            lv[1:3] += bump
            mt = m.with_levels(lv)
            rt = bootstrap_repo(m.spot, m.rates, mt.dividends, T, F_tot)
            cl = control_lines(sr, sd, rt.zeros - m.repo.zeros, repo_fwd.zeros - m.repo.zeros, lv - m.div_levels, J)
            full = book_value(book, mt.with_repo(rt)) - book_value(book, m)
            out.append((name, bump, cl.div_line, cl.repo_line, cl.total, cl.forward_line, cl.non_forward_div_line,
                        full))
        out.append((name, "materiality", mat.div_line_sd, mat.split_sd, mat.cancellation))
    return out


def jump_table():
    """Standard deviation of the consensus jump, in units of contributor dispersion, when a nested panel shrinks."""
    return [(n, m, float(nested_jump_sd(1.0, n, m)), 1 / np.sqrt(m)) for n, m in [(16, 12), (12, 7), (15, 7), (10, 4)]]


# ------------------------------------------------------------- benchmark
def calibrated_detector(res: BenchmarkResult, category: str) -> StructuralDetector:
    return StructuralDetector(alpha=res.fpr_target, variance_scale=res.variance_scale[category],
                              level_scale=res.level_scale.get(category, 1.0))


def attribution_table(res: BenchmarkResult):
    rows = []
    for s in list(EXPECTED_KIND) + ["fo_repo_mismark", "div_gross_net"]:
        recs = [r for r in res.records if r["scenario"] == s]
        flagged = [r for r in recs if r["flags"]["structural"]]
        rows.append(
            {
                "scenario": s,
                "detected": len(flagged) / len(recs),
                "attr_cond": float(np.mean([r["attribution_ok"] for r in flagged]))
                if flagged and s in EXPECTED_KIND else float("nan"),
                "fo_flag": float(np.mean([r["structural"]["fo_mismark"] for r in recs])),
            }
        )
    return rows
