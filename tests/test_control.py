import numpy as np
import pytest

from repo_control.buckets import BucketSchedule
from repo_control.control import control_lines, implied_repo_jacobian, split_materiality
from repo_control.curves import Dividends, ZeroCurve, bootstrap_repo
from repo_control.noise import nested_jump_sd, panel_gap_sd
from repo_control.pricer import DividendSwap, EquityForward, EuropeanOption, Market, StockLoanFee, book_value, sensitivities

T = np.array([1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60]) / 12.0
SCHEDULE = BucketSchedule(np.array([0.0, 0.2164, 1.2164, 2.219, 3.219, 4.219, 5.22]))
PROFILE = Dividends([0.55, 1.55, 2.55, 3.55, 4.55], [1.0, 1.0, 1.0, 1.0, 1.0])
LEVELS = np.array([0.0, 3.5, 3.7, 3.9, 4.1, 4.3])
FO = Market(100.0, ZeroCurve.flat(0.03), ZeroCurve(T, 0.006 + 0 * T), SCHEDULE, LEVELS, PROFILE)
OPTIONS = [EuropeanOption(True, 100, 1.0, 0.25, 10), EuropeanOption(False, 95, 2.0, 0.25, -5), EquityForward(3.0, 95, 2)]


def _lines(book, bump):
    F_totem = FO.forward(T) * (1 - 0.002)
    levels = LEVELS.copy()
    levels[1:3] += bump
    totem = FO.with_levels(levels)
    repo_totem = bootstrap_repo(100.0, FO.rates, totem.dividends, T, F_totem)
    repo_fwd = bootstrap_repo(100.0, FO.rates, FO.dividends, T, F_totem)
    J = implied_repo_jacobian(100.0, FO.rates, SCHEDULE, LEVELS, PROFILE, T, F_totem)
    sr, sd = sensitivities(book, FO)
    lines = control_lines(sr, sd, repo_totem.zeros - FO.repo.zeros, repo_fwd.zeros - FO.repo.zeros, levels - LEVELS, J)
    full = book_value(book, totem.with_repo(repo_totem)) - book_value(book, FO)
    return lines, full, J, sr, sd


@pytest.mark.parametrize("bump", [-0.4, 0.0, 0.5])
def test_forward_only_total_is_dividend_agnostic(bump):
    lines, full, *_ = _lines(OPTIONS, bump)
    # the dividend and repo lines swing by several units; their total stays on the forward line
    tol = 0.01 * abs(lines.div_line) + 0.005
    assert lines.total == pytest.approx(lines.forward_line, abs=tol)
    assert lines.total == pytest.approx(full, abs=tol)
    assert abs(lines.non_forward_div_line) < tol


@pytest.mark.parametrize("extra", [DividendSwap(2, 3.7, 2.3, -8), StockLoanFee(1.0, 3)])
def test_non_forward_exposure_explains_total_move(extra):
    book = OPTIONS + [extra]
    lines, full, *_ = _lines(book, -0.4)
    assert lines.forward_line + lines.non_forward_div_line == pytest.approx(full, abs=0.03)
    assert abs(lines.non_forward_div_line) > 0.5


def test_split_materiality_zero_for_forward_only_book():
    _, _, J, sr, sd = _lines(OPTIONS, 0.0)
    cov = np.diag(np.full(LEVELS.size, 0.05**2))
    m = split_materiality(sd, sr, J, cov)
    assert m.split_sd < 0.02 * m.div_line_sd
    sr2, sd2 = sensitivities(OPTIONS + [DividendSwap(2, 3.7, 2.3, -8)], FO)
    m2 = split_materiality(sd2, sr2, J, cov)
    assert m2.split_sd > 0.2


def test_panel_gap_formula_matches_simulation():
    rng = np.random.default_rng(1)
    T_, D0, sD, nA, nB, S = 1.0, 3.5, 0.25, 15, 10, 100.0
    gaps = []
    for _ in range(4000):
        dA = rng.normal(D0, sD, nA)
        dB = rng.normal(D0, sD, nB)
        bA = rng.normal(0.005, 0.002, nA)
        F = ((S - dA) * np.exp((0.03 - bA) * T_)).mean()
        b_implied = 0.03 - np.log(F / (S - dB.mean())) / T_
        gaps.append(b_implied - bA.mean())
    assert np.std(gaps) == pytest.approx(panel_gap_sd(T_, S, D0, sD, nA, nB), rel=0.05)


def test_nested_jump_variance():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20000, 12))
    jumps = x[:, :7].mean(axis=1) - x.mean(axis=1)
    assert np.std(jumps) == pytest.approx(nested_jump_sd(1.0, 12, 7), rel=0.03)
