"""Synthetic Totem market with explicit contributor panels.

Every name gets a "true" market (dividends, repo), an option panel and a
separate dividend panel of internally consistent contributors, and FO marks.
Totem-style consensus (mean, dispersion, count) is formed per expiry/strike and
per dividend bucket, then run through the real controller pipeline.

Scenarios inject one mechanism each; all random draws of the base market are
taken before the scenario is applied, so the same seed gives paired cases
across scenarios.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
from scipy.special import ndtr

from .buckets import BucketSchedule
from .curves import Dividends, ZeroCurve, forward
from .pipeline import FOSnapshot, ImpliedRepo, TotemSnapshot, imply_totem_repo

VALUATION = date(2026, 10, 14)


def year_fraction(d: date) -> float:
    return (d - VALUATION).days / 365.0


PILLAR_MONTHS = (1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60)
PILLARS = np.array(PILLAR_MONTHS, dtype=float) / 12.0
SCHEDULE = BucketSchedule(np.array([0.0] + [year_fraction(date(y, 1, 1)) for y in range(2027, 2033)]))
BUCKET_LABELS = ("2026 rest", "2027", "2028", "2029", "2030", "2031")
STRIKE_PCT = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
RATES = {
    "EUR": ZeroCurve(np.array([0.25, 1, 2, 3, 5, 7]), np.array([0.0205, 0.0215, 0.0228, 0.0240, 0.0260, 0.0275])),
    "USD": ZeroCurve(np.array([0.25, 1, 2, 3, 5, 7]), np.array([0.0365, 0.0345, 0.0335, 0.0335, 0.0345, 0.0360])),
}

SCENARIOS = (
    "clean",
    "spot_sync",
    "div_bucket",
    "div_timing",
    "season_profile",
    "fwd_outlier",
    "panel_shift",
    "fo_repo_mismark",
    "div_gross_net",
)

SCENARIO_TITLES = {
    "clean": "Clean (null)",
    "spot_sync": "Spot snapshot mismatch",
    "div_bucket": "Totem dividend bucket error",
    "div_timing": "Ex-date timing split in option panel",
    "season_profile": "Stale ex-date profile in allocation",
    "fwd_outlier": "Forward outlier at one expiry",
    "panel_shift": "Biased long-dated sub-panel",
    "fo_repo_mismark": "FO repo mismark (genuine)",
    "div_gross_net": "Gross/net dividend bias (all buckets)",
}

# mechanism kinds the attribution is expected to name for each scenario
EXPECTED_KIND = {
    "spot_sync": {"level"},
    "div_bucket": {"step"},
    "div_timing": {"spike", "season"},
    "season_profile": {"season", "spike"},
    "fwd_outlier": {"spike"},
    "panel_shift": {"step"},
}


@dataclass(frozen=True)
class CategorySpec:
    name: str
    currency: str
    calendar: str
    div_yield: tuple
    div_growth: tuple
    repo_short_bp: tuple
    repo_long_bp: tuple
    repo_tau: tuple
    vol_atm: tuple
    div_disp: tuple
    repo_disp_bp: float
    n_opt: tuple
    n_div: tuple


CATEGORIES = {
    "EU stock": CategorySpec(
        "EU stock", "EUR", "eu_annual", (0.02, 0.06), (-0.02, 0.06), (0, 50), (0, 60), (0.5, 2.0),
        (0.20, 0.35), (0.010, 0.030), 15.0,
        (16, 16, 16, 16, 15, 15, 13, 12, 7, 6, 6), (12, 12, 10, 7, 5, 4),
    ),
    "US stock": CategorySpec(
        "US stock", "USD", "us_quarterly", (0.005, 0.03), (0.00, 0.08), (0, 40), (0, 50), (0.5, 2.0),
        (0.20, 0.35), (0.005, 0.020), 10.0,
        (18, 18, 18, 18, 17, 16, 14, 13, 8, 6, 5), (10, 10, 8, 6, 4, 3),
    ),
    "HTB stock": CategorySpec(
        "HTB stock", "USD", "mixed", (0.0, 0.02), (-0.05, 0.05), (150, 800), (30, 150), (0.3, 1.5),
        (0.45, 0.80), (0.020, 0.050), 60.0,
        (12, 12, 12, 11, 10, 10, 8, 7, 5, 4, 4), (8, 8, 6, 4, 3, 3),
    ),
    "EU index": CategorySpec(
        "EU index", "EUR", "index_eu", (0.028, 0.036), (0.00, 0.05), (-20, 60), (-10, 80), (0.5, 3.0),
        (0.15, 0.22), (0.005, 0.015), 8.0,
        (25, 25, 25, 24, 23, 22, 20, 19, 15, 13, 12), (20, 20, 18, 15, 12, 10),
    ),
}

EU_INDEX_MONTH_WEIGHTS = np.array([1, 2, 4, 14, 24, 18, 6, 4, 3, 4, 3, 2], dtype=float)


@dataclass
class Case:
    seed: int
    category: str
    scenario: str
    truth: dict
    totem: TotemSnapshot
    fo: FOSnapshot
    implied: ImpliedRepo
    true_repo: ZeroCurve
    true_dividends: Dividends
    extra: dict = field(default_factory=dict)


def _calendar(spec: CategorySpec, rng: np.random.Generator, spot: float):
    """True ex-dates and cash amounts up to the end of 2031."""
    q = rng.uniform(*spec.div_yield)
    g = rng.uniform(*spec.div_growth)
    kind = spec.calendar
    if kind == "mixed":
        kind = "eu_annual" if rng.uniform() < 0.5 else "us_quarterly"
    times, amounts = [], []
    if kind == "eu_annual":
        main_offset = int(rng.integers(0, 41))
        interim = rng.uniform() < 0.35
        interim_offset = int(rng.integers(0, 26))
        for y in range(2026, 2032):
            annual = spot * q * (1 + g) ** (y - 2027)
            if y >= 2027:
                times.append(date(y, 4, 20) + timedelta(days=main_offset + int(rng.integers(-3, 4))))
                amounts.append(annual * (0.6 if interim else 1.0))
            if interim:
                times.append(date(y, 11, 1) + timedelta(days=interim_offset + int(rng.integers(-3, 4))))
                amounts.append(annual * 0.4)
    elif kind == "us_quarterly":
        day = int(rng.integers(8, 19))
        for y in range(2026, 2032):
            for mth in (2, 5, 8, 11):
                d = date(y, mth, day) + timedelta(days=int(rng.integers(-3, 4)))
                times.append(d)
                amounts.append(spot * q / 4 * (1 + g) ** max(year_fraction(d), 0.0))
    else:  # index_eu
        w = EU_INDEX_MONTH_WEIGHTS / EU_INDEX_MONTH_WEIGHTS.sum()
        shift = int(rng.integers(-5, 6))
        for y in range(2026, 2032):
            annual = spot * q * (1 + g) ** (y - 2027)
            for mth in range(1, 13):
                times.append(date(y, mth, 15) + timedelta(days=shift))
                amounts.append(annual * w[mth - 1])
    t = np.array([year_fraction(d) for d in times])
    a = np.array(amounts)
    keep = (t > 2 / 365) & (t <= SCHEDULE.edges[-1])
    return Dividends(t[keep], a[keep]), kind


def _black(F, K, df, vol, T, is_call):
    s = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    call = df * (F * ndtr(d1) - K * ndtr(d2))
    return call if is_call else call - df * (F - K)


def _robust_mean(x, active, k=4.0):
    """Mean/std/count over the active contributors after a median/MAD outlier screen (axis 0)."""
    x = np.where(active, x, np.nan)
    med = np.nanmedian(x, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=0)
    ok = active & ((np.abs(x - med) <= k * mad) | (mad == 0))
    xs = np.where(ok, x, np.nan)
    n = ok.sum(axis=0)
    return np.nanmean(xs, axis=0), np.nanstd(xs, axis=0, ddof=1), n


def _views(rng, n, divs: Dividends, disp, rho=0.7):
    """Contributor dividend views: relative errors persistent across ex-dates."""
    z = rng.standard_normal((n, 1))
    e = rng.standard_normal((n, len(divs)))
    sd = disp[0] + disp[1] * divs.ex_times
    sd = np.where(divs.ex_times < 0.15, 0.1 * disp[0], sd)
    return divs.amounts[None, :] * (1 + sd[None, :] * (rho * z + np.sqrt(1 - rho**2) * e))


def scenario_id(name: str) -> int:
    return zlib.crc32(name.encode())


def simulate_case(seed: int, category: str, scenario: str = "clean") -> Case:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    spec = CATEGORIES[category]
    rng = np.random.default_rng([seed, zlib.crc32(category.encode())])
    S = 100.0
    rates = RATES[spec.currency]
    T = PILLARS
    K = T.size

    # ---------------------------------------------------------- base market
    divs, kind = _calendar(spec, rng, S)
    b_short = rng.uniform(*spec.repo_short_bp) * 1e-4
    b_long = rng.uniform(*spec.repo_long_bp) * 1e-4
    tau = rng.uniform(*spec.repo_tau)
    true_repo = ZeroCurve(T, b_long + (b_short - b_long) * np.exp(-T / tau))
    vol_atm = rng.uniform(*spec.vol_atm)

    nA, nB = int(spec.n_opt[0]), int(spec.n_div[0])
    viewsA = _views(rng, nA, divs, spec.div_disp)
    viewsB = _views(rng, nB, divs, spec.div_disp)
    # contributor repo views: smooth curves (level + tilt in ln T), as desks mark a few pillars and interpolate
    tilt = np.log(T / T[0]) / np.log(T[-1] / T[0]) - 0.5
    wA = spec.repo_disp_bp * 1e-4 * (
        rng.standard_normal((nA, 1)) + 0.6 * rng.standard_normal((nA, 1)) * tilt[None, :]
    )
    strikes = np.tile(STRIKE_PCT * S, (K, 1))
    vol_noise = 0.01 * rng.standard_normal((nA, K, STRIKE_PCT.size))
    price_noise = 2e-5 * S * rng.standard_normal((2, nA, K, STRIKE_PCT.size))
    fo_div_err = 0.01 * rng.standard_normal(len(divs))
    fo_repo_err = 3e-4 * rng.standard_normal()

    # -------------------------------------------------------- scenario draws
    srng = np.random.default_rng([seed, zlib.crc32(category.encode()), scenario_id(scenario)])
    sign = 1.0 if srng.uniform() < 0.5 else -1.0
    truth: dict = {"kind": "none"}
    spot_snap = S
    ex_A = np.tile(divs.ex_times, (nA, 1))
    allocation_profile = None
    fo_repo_shift = 0.0
    level_mult = np.ones(SCHEDULE.size)
    level_add = np.zeros(SCHEDULE.size)

    projected = np.flatnonzero(divs.ex_times > 0.15)
    if projected.size:
        dist = np.abs(divs.ex_times[projected][:, None] - T[None, :])
        dist[:, T > 3.01] = np.inf
        flat = int(np.argmin(dist))
        near_div, near_pillar = int(projected[flat // K]), int(flat % K)
    else:
        near_div, near_pillar = -1, -1

    if scenario == "spot_sync":
        eps = sign * srng.uniform(0.0003, 0.0015)
        spot_snap = S * (1 + eps)
        truth = {"kind": "level", "eps": eps}
    elif scenario == "div_bucket":
        y = int(srng.integers(1, 4))
        delta = sign * srng.uniform(0.0015, 0.005) * S
        level_add[y] += delta
        in_bucket = SCHEDULE.index(divs.ex_times) == y
        steps = sorted({int(np.searchsorted(T, t, side="left")) for t in divs.ex_times[in_bucket] if t <= T[-1]})
        truth = {"kind": "step", "bucket": y, "delta": delta, "pillars": steps}
    elif scenario == "div_timing" and near_div >= 0:
        p = srng.uniform(0.35, 0.7)
        movers = srng.uniform(size=nA) < p
        t_i, t_k = divs.ex_times[near_div], T[near_pillar]
        new_t = t_k - 2 / 365 if t_i > t_k else t_k + 2 / 365
        ex_A[movers, near_div] = new_t
        truth = {"kind": "spike", "pillar": near_pillar, "share": float(movers.mean()), "div": near_div}
    elif scenario == "season_profile" and near_div >= 0:
        gap = T[near_pillar] - divs.ex_times[near_div]
        shift = gap + np.sign(gap) * srng.uniform(3, 20) / 365
        allocation_profile = Dividends(divs.ex_times + shift, divs.amounts * (1 + fo_div_err))
        truth = {"kind": "season", "pillar": near_pillar, "shift_days": shift * 365}
    elif scenario == "fwd_outlier":
        truth = {"kind": "spike", "pillar": int(srng.integers(0, K)), "z": sign * srng.uniform(3.0, 6.0)}
    elif scenario == "panel_shift":
        ratios = np.array(spec.n_opt[1:], dtype=float) / np.array(spec.n_opt[:-1], dtype=float)
        k_star = int(np.argmin(ratios)) + 1
        delta = sign * srng.uniform(0.08, 0.20)
        late = divs.ex_times > T[k_star - 1]
        viewsA[: spec.n_opt[k_star], :] *= np.where(late, 1 + delta, 1.0)[None, :]
        truth = {"kind": "step", "pillar": k_star, "delta": delta, "pillars": [k_star]}
    elif scenario == "fo_repo_mismark":
        fo_repo_shift = sign * srng.uniform(15, 50) * 1e-4
        truth = {"kind": "fo", "shift": fo_repo_shift}
    elif scenario == "div_gross_net":
        tau_wht = sign * srng.uniform(0.08, 0.18)
        level_mult *= 1 + tau_wht
        truth = {"kind": "beta", "bias": tau_wht}

    # ------------------------------------------------------- option panel A
    activeA = np.arange(nA)[:, None] < np.array(spec.n_opt)[None, :]
    F_A = np.empty((nA, K))
    for j in range(nA):
        repo_j = ZeroCurve(T, true_repo.zeros + wA[j])
        F_A[j] = forward(spot_snap, rates, repo_j, Dividends(ex_A[j], viewsA[j]), T)
    df = rates.df(T)
    vol_T = vol_atm * (1 + 0.15 * np.exp(-T))
    Kgrid = strikes[None, :, :]
    vols = vol_T[None, :, None] - 0.10 * np.log(Kgrid / F_A[:, :, None]) + vol_noise
    vols = np.clip(vols, 0.03, None)
    calls = _black(F_A[:, :, None], Kgrid, df[None, :, None], vols, T[None, :, None], True) + price_noise[0]
    puts = _black(F_A[:, :, None], Kgrid, df[None, :, None], vols, T[None, :, None], False) + price_noise[1]
    act3 = np.broadcast_to(activeA[:, :, None], calls.shape)
    c_mean, _, _ = _robust_mean(calls, act3)
    p_mean, _, _ = _robust_mean(puts, act3)
    fwd_sd = np.nanstd(np.where(activeA, F_A, np.nan), axis=0, ddof=1)

    # ------------------------------------------------------ dividend panel B
    idx = SCHEDULE.index(divs.ex_times)
    bucket_sums = np.zeros((nB, SCHEDULE.size))
    for y in range(SCHEDULE.size):
        bucket_sums[:, y] = viewsB[:, idx == y].sum(axis=1)
    activeB = np.arange(nB)[:, None] < np.array(spec.n_div)[None, :]
    lvl, lvl_sd, _ = _robust_mean(bucket_sums, activeB)
    lvl = lvl * level_mult + level_add
    lvl_sd = np.where(np.isfinite(lvl_sd), lvl_sd, 0.0)

    totem = TotemSnapshot(
        pillars=T,
        ref_spot=S,
        strikes=strikes,
        call=c_mean,
        put=p_mean,
        n_opt=np.array(spec.n_opt, dtype=float),
        fwd_sd=fwd_sd,
        schedule=SCHEDULE,
        div_level=lvl,
        div_sd=lvl_sd,
        n_div=np.array(spec.n_div, dtype=float),
    )
    fo = FOSnapshot(
        spot=S,
        rates=rates,
        repo=ZeroCurve(T, true_repo.zeros + fo_repo_err + fo_repo_shift),
        dividends=Dividends(divs.ex_times, divs.amounts * (1 + fo_div_err)),
    )
    implied = imply_totem_repo(totem, fo, allocation_profile=allocation_profile)

    if scenario == "fwd_outlier":
        k = truth["pillar"]
        rel = truth["z"] * implied.carry_sd[k]
        call = totem.call.copy()
        call[k] = call[k] + df[k] * implied.forwards[k] * (np.exp(-rel) - 1)
        totem = replace(totem, call=call)
        implied = imply_totem_repo(totem, fo, allocation_profile=allocation_profile)
        truth["ln_fwd_shift"] = -rel

    return Case(seed, category, scenario, truth, totem, fo, implied, true_repo, divs, extra={"calendar": kind})
