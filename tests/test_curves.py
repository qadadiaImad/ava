import numpy as np
import pytest

from repo_control.buckets import BucketSchedule, allocate, allocation_matrix, bucket_levels
from repo_control.curves import Dividends, ZeroCurve, bootstrap_repo, carry_dividends, forward
from repo_control.parity import forward_from_parity, forward_from_parity_fixed_df

T = np.array([1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60]) / 12.0
RATES = ZeroCurve(np.array([0.5, 1, 2, 5]), np.array([0.030, 0.031, 0.032, 0.033]))


@pytest.mark.parametrize(
    "repo, divs",
    [
        (ZeroCurve(T, 0.004 + 0.003 * np.exp(-T)), Dividends([0.05, 0.45, 0.52, 1.45, 2.5, 3.48, 4.47], [0.5, 2.0, 0.3, 2.1, 2.2, 2.3, 2.4])),
        (ZeroCurve(T, -0.01 + 0 * T), Dividends([0.25, 1.0, 2.0], [3.0, 3.0, 3.0])),  # ex-dates on pillars
        (ZeroCurve(T, 0.06 * np.exp(-T / 0.5) + 0.005), Dividends.empty()),  # hard to borrow, no dividends
    ],
)
def test_bootstrap_round_trip(repo, divs):
    F = forward(100.0, RATES, repo, divs, T)
    implied = bootstrap_repo(100.0, RATES, divs, T, F)
    np.testing.assert_allclose(implied.zeros, repo.zeros, atol=1e-12)


def test_forward_without_dividends_is_carry():
    repo = ZeroCurve.flat(0.01)
    F = forward(100.0, RATES, repo, Dividends.empty(), T)
    np.testing.assert_allclose(F, 100 * np.exp(RATES.cum(T) - 0.01 * T))


def test_forward_drops_by_dividend_at_ex_date():
    repo = ZeroCurve.flat(0.0)
    divs = Dividends([0.5], [2.0])
    before, after = forward(100.0, ZeroCurve.flat(0.0), repo, divs, [0.4999, 0.5])
    assert before - after == pytest.approx(2.0, abs=1e-9)


def test_carry_dividends_cumulative():
    divs = Dividends([0.3, 0.8], [1.0, 2.0])
    D = carry_dividends(100.0, ZeroCurve.flat(0.0), ZeroCurve.flat(0.0), divs, [0.2, 0.5, 1.0])
    np.testing.assert_allclose(D, [0.0, 1.0, 3.0])


def test_bucket_allocation_preserves_levels_and_profile():
    schedule = BucketSchedule(np.array([0.0, 1.0, 2.0]))
    profile = Dividends([0.3, 0.7, 1.5], [1.0, 3.0, 2.0])
    divs = allocate(schedule, [8.0, 5.0], profile)
    np.testing.assert_allclose(bucket_levels(divs, schedule), [8.0, 5.0])
    np.testing.assert_allclose(divs.amounts, [2.0, 6.0, 5.0])
    A = allocation_matrix(schedule, profile, [0.5, 1.0, 2.0], ZeroCurve.flat(0.0), ZeroCurve.flat(0.0))
    np.testing.assert_allclose(A, [[0.25, 0.0], [1.0, 0.0], [1.0, 1.0]])


def test_parity_recovers_forward_and_discount():
    K = np.array([80, 90, 100, 110, 120.0])
    F, DF = 103.7, 0.97
    puts = np.array([1.0, 2.5, 5.0, 9.0, 15.0])
    calls = puts + DF * (F - K)
    fit = forward_from_parity(K, calls, puts)
    assert fit.forward == pytest.approx(F)
    assert fit.discount == pytest.approx(DF)
    assert forward_from_parity_fixed_df(K, calls, puts, DF) == pytest.approx(F)
    # a wrong discount factor does not move the zero crossing
    fit2 = forward_from_parity(K, calls * 1.02, puts * 1.02)
    assert fit2.forward == pytest.approx(F)
