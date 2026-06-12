"""Mock market data + mock parsers (adapters module of the engine sheet).

In production the adapters extract (1) the daily surface panel from the
FO system and (2) the vega matrices per book from the risk system. Here
both extracts are SIMULATED, then written in the raw text format such an
extraction would produce, then re-read by the parsers — the engine never
sees anything but parsed extracts, exactly as in production.

Honesty rule: the world is drawn FIRST (surface dynamics + book
compositions from a seed); everything downstream — which fusions
survive, which books pass the variance test, the AVA savings, whether
the engine falls back to majorant mode — is EMERGENT from running the
chain on the parsed data. Nothing is tuned to produce a desired result.

Three simulated regimes (cases):
  ``smooth``  — the surface moves through the level / slope / curvature
                factors of the deformation model plus a small per-cell
                idio: the sandwich hypothesis genuinely holds;
  ``torsion`` — same, plus a genuine wing-asymmetry (kink) factor: the
                T4 automatic decision should fire on its own;
  ``choppy``  — weak common structure drowned in heavy idiosyncratic
                noise: the battery should degrade the underlying to the
                conservative majorant-only mode.

Books are random portfolios of standard vega strategies (directional,
calendar, risk-reversal, butterfly, short wings) with random sizes,
signs and tenor bands — a multi-book inventory, not hand-shaped inputs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "MOCK_TENORS",
    "MOCK_TENOR_YEARS",
    "MOCK_STRIKES",
    "MockEnvironment",
    "generate_surface_extract",
    "generate_book_extracts",
    "parse_surface_extract",
    "parse_book_extracts",
    "load_mock_environment",
]

MOCK_TENORS = ["1M", "3M", "6M", "1Y", "2Y", "5Y"]
MOCK_TENOR_YEARS = [1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0]
MOCK_STRIKES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

_CASES = ("smooth", "torsion", "choppy")


# --------------------------------------------------------------------------- #
# the simulated world
# --------------------------------------------------------------------------- #
def _mode_shapes() -> tuple[np.ndarray, np.ndarray]:
    """Raw (unnormalised) deformation shapes: tenor profiles for the
    level / slope / curvature factors, and the strike shapes 1, x, x^2."""
    y = np.log(np.asarray(MOCK_TENOR_YEARS))
    y = (y - y.mean()) / y.std()
    x = np.asarray(MOCK_STRIKES) - 1.0
    tenor_shapes = np.column_stack([np.ones_like(y), y, y ** 2])
    strike_shapes = np.column_stack([np.ones_like(x), x, x ** 2])
    return tenor_shapes, strike_shapes


def _simulate_shocks(case: str, n_days: int, rng) -> np.ndarray:
    """Daily surface shocks (vol points) for one case. Common part:
    X_t = Y B_t Z' with a 3 x 3 factor matrix (level dominating, slope
    and curvature smaller), plus the per-cell idio; the torsion case adds
    a kink |x| factor; the choppy case shrinks the structure and inflates
    the idio."""
    if case not in _CASES:
        raise ValueError(f"unknown case '{case}', pick one of {_CASES}")
    M, K = len(MOCK_TENORS), len(MOCK_STRIKES)
    yb, zb = _mode_shapes()
    # orthonormalise so the factor amplitudes mean what they say
    yq, _ = np.linalg.qr(yb)
    zq, _ = np.linalg.qr(zb)
    # separable amplitudes (outer product): the common covariance is then
    # exactly SigmaT . SigmaK — the world honestly satisfies the model
    amp_t = np.array([1.00, 0.55, 0.35])
    amp_k = np.array([0.60, 0.45, 0.25])
    amp = amp_t[:, None] @ amp_k[None, :]     # vol points per mode pair
    idio_sd = 0.015 * (1.0 + 2.0 * np.abs(np.asarray(MOCK_STRIKES) - 1.0))
    if case == "choppy":
        amp = 0.25 * amp
        idio_sd = 10.0 * idio_sd
    b = rng.normal(size=(n_days, 3, 3)) * amp[None, :, :]
    shocks = np.stack([yq @ bt @ zq.T for bt in b])
    if case == "torsion":
        # genuine torsion = the part of the kink |x| ORTHOGONAL to the
        # level / slope / curvature span (the in-span part of |x| would
        # just be absorbed by the 9 factors and violate nothing)
        x = np.asarray(MOCK_STRIKES) - 1.0
        kink = np.abs(x)
        kink = kink - zq @ (zq.T @ kink)
        kink /= np.linalg.norm(kink)
        tenor_profile = yq[:, 0]
        tor = rng.normal(0.0, 0.35, n_days)
        shocks += tor[:, None, None] * tenor_profile[None, :, None] * kink[None, None, :]
    shocks += rng.normal(size=(n_days, M, K)) * idio_sd[None, None, :]
    return shocks


def _base_surface() -> np.ndarray:
    """A plausible starting vol surface (in %): downward term structure
    of ATM vols plus a smile per tenor."""
    ten = np.asarray(MOCK_TENOR_YEARS)
    x = np.asarray(MOCK_STRIKES) - 1.0
    atm = 18.0 + 6.0 * np.exp(-ten / 1.5)
    smile = 28.0 * x ** 2 - 6.0 * x
    return atm[:, None] + smile[None, :]


# --------------------------------------------------------------------------- #
# books — random portfolios of standard vega strategies
# --------------------------------------------------------------------------- #
def _strategy_vega(kind: str, rng) -> np.ndarray:
    """(M, K) vega of one strategy in kEUR / vol point. Locations, sizes
    and signs are random — the inventory is not shaped for any result."""
    M, K = len(MOCK_TENORS), len(MOCK_STRIKES)
    ten = np.asarray(MOCK_TENOR_YEARS)
    x = np.asarray(MOCK_STRIKES) - 1.0
    atm_w = np.exp(-0.5 * (x / 0.07) ** 2)
    wing_w = np.clip(np.abs(x) - 0.05, 0.0, None)
    size = float(rng.lognormal(mean=5.5, sigma=0.5))     # ~250 kEUR/pt typical
    sign = rng.choice([-1.0, 1.0])
    a, b = sorted(rng.choice(M, size=2, replace=False))
    band = np.zeros(M)
    band[a : b + 1] = 1.0
    if kind == "directional":
        v = np.exp(-ten / 3.0).reshape(-1, 1) @ atm_w.reshape(1, -1)
    elif kind == "calendar":
        term = np.where(ten <= 0.5, 1.0, -0.8 * np.exp(-(ten - 0.5) / 3.0))
        v = term.reshape(-1, 1) @ atm_w.reshape(1, -1)
    elif kind == "risk_reversal":
        v = band.reshape(-1, 1) @ x.reshape(1, -1)
    elif kind == "butterfly":
        v = band.reshape(-1, 1) @ (wing_w - 0.4 * atm_w).reshape(1, -1)
    elif kind == "short_wings":
        v = -np.exp(-ten / 2.0).reshape(-1, 1) @ wing_w.reshape(1, -1)
    else:
        raise ValueError(f"unknown strategy '{kind}'")
    norm = np.abs(v).max()
    return sign * size * v / (norm if norm > 0 else 1.0)


def _random_book(rng) -> np.ndarray:
    kinds = ["directional", "calendar", "risk_reversal", "butterfly", "short_wings"]
    n_strats = int(rng.integers(2, 5))
    book = sum(_strategy_vega(rng.choice(kinds), rng) for _ in range(n_strats))
    # small residual inventory noise
    book = book + rng.normal(0.0, 0.01 * max(np.abs(book).max(), 1.0), book.shape)
    return book


# --------------------------------------------------------------------------- #
# raw extracts (what the FO / risk systems would hand over)
# --------------------------------------------------------------------------- #
def generate_surface_extract(case: str = "smooth", n_days: int = 400,
                             seed: int = 0) -> str:
    """CSV extract of daily vol SURFACE LEVELS, one row per
    (date, tenor, strike) — the format of an FO export. Levels are the
    base surface plus the cumulated simulated shocks."""
    rng = np.random.default_rng(seed)
    shocks = _simulate_shocks(case, n_days, rng)
    levels = _base_surface()[None, :, :] + np.cumsum(shocks, axis=0)
    levels = np.clip(levels, 1.0, None)
    out = io.StringIO()
    out.write("date,tenor,strike,vol\n")
    for t in range(n_days):
        date = f"D{t + 1:04d}"
        for a, tenor in enumerate(MOCK_TENORS):
            for j, strike in enumerate(MOCK_STRIKES):
                out.write(f"{date},{tenor},{strike:.2f},{levels[t, a, j]:.6f}\n")
    return out.getvalue()


def generate_book_extracts(n_books: int = 4, seed: int = 0) -> str:
    """CSV extract of the vega inventory, one row per
    (book, tenor, strike) — the format of a risk-system sensi report."""
    rng = np.random.default_rng(seed + 1000)
    out = io.StringIO()
    out.write("book,tenor,strike,vega\n")
    for i in range(n_books):
        name = f"BOOK_{i + 1:02d}"
        vega = _random_book(rng)
        for a, tenor in enumerate(MOCK_TENORS):
            for j, strike in enumerate(MOCK_STRIKES):
                out.write(f"{name},{tenor},{strike:.2f},{vega[a, j]:.4f}\n")
    return out.getvalue()


# --------------------------------------------------------------------------- #
# parsers (adapters points 1 and 4 of the sheet)
# --------------------------------------------------------------------------- #
def parse_surface_extract(text: str) -> dict:
    """Parse an FO surface extract into the engine's panel input.

    Returns the fixed grid (tenor labels / years, strikes), the dates and
    the panel X_t of FIRST DIFFERENCES of the vol levels (adapters point
    1: 'Xt = différence première'), shaped (T-1, M, K)."""
    tenor_to_years = dict(zip(MOCK_TENORS, MOCK_TENOR_YEARS))
    dates, tenors, strikes = [], [], []
    cells: dict = {}
    lines = text.strip().splitlines()
    header = lines[0].strip().lower().split(",")
    if header != ["date", "tenor", "strike", "vol"]:
        raise ValueError(f"unexpected surface extract header: {header}")
    for line in lines[1:]:
        date, tenor, strike, vol = line.split(",")
        strike = float(strike)
        if tenor not in tenor_to_years:
            raise ValueError(f"unknown tenor label '{tenor}'")
        if date not in dates:
            dates.append(date)
        if tenor not in tenors:
            tenors.append(tenor)
        if strike not in strikes:
            strikes.append(strike)
        cells[(date, tenor, strike)] = float(vol)
    tenors = sorted(tenors, key=lambda t: tenor_to_years[t])
    strikes = sorted(strikes)
    levels = np.empty((len(dates), len(tenors), len(strikes)))
    for t, date in enumerate(dates):
        for a, tenor in enumerate(tenors):
            for j, strike in enumerate(strikes):
                key = (date, tenor, strike)
                if key not in cells:
                    raise ValueError(f"missing surface cell {key}")
                levels[t, a, j] = cells[key]
    return {
        "tenor_labels": tenors,
        "tenor_years": [tenor_to_years[t] for t in tenors],
        "strikes": strikes,
        "dates": dates,
        "levels": levels,
        "panel": np.diff(levels, axis=0),
    }


def parse_book_extracts(text: str, tenor_labels: list, strikes: list) -> dict:
    """Parse a risk-system sensi report into one (M, K) vega matrix per
    book, on the SAME grid as the surface (adapters point 4) — any cell
    off the grid is an error, not silently dropped."""
    pos_t = {t: i for i, t in enumerate(tenor_labels)}
    pos_k = {round(float(s), 4): j for j, s in enumerate(strikes)}
    lines = text.strip().splitlines()
    header = lines[0].strip().lower().split(",")
    if header != ["book", "tenor", "strike", "vega"]:
        raise ValueError(f"unexpected book extract header: {header}")
    books: dict = {}
    for line in lines[1:]:
        name, tenor, strike, vega = line.split(",")
        strike = round(float(strike), 4)
        if tenor not in pos_t or strike not in pos_k:
            raise ValueError(f"book {name}: cell ({tenor}, {strike}) is off the grid")
        if name not in books:
            books[name] = np.zeros((len(tenor_labels), len(strikes)))
        books[name][pos_t[tenor], pos_k[strike]] += float(vega)
    return books


# --------------------------------------------------------------------------- #
# one call: generate -> write -> parse (the full mock adapters chain)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MockEnvironment:
    case: str
    tenor_labels: list
    tenor_years: list
    strikes: list
    panel: np.ndarray              # (T-1, M, K) daily variations, parsed
    books: dict = field(default_factory=dict)   # name -> (M, K) vega
    surface_extract: str = ""      # the raw texts, kept for audit
    book_extract: str = ""


def load_mock_environment(case: str = "smooth", n_books: int = 4,
                          n_days: int = 400, seed: int = 0) -> MockEnvironment:
    """Simulate the world, write the raw extracts, parse them back —
    the engine consumes only what the parsers produced."""
    surface_text = generate_surface_extract(case, n_days, seed)
    book_text = generate_book_extracts(n_books, seed)
    surf = parse_surface_extract(surface_text)
    books = parse_book_extracts(book_text, surf["tenor_labels"], surf["strikes"])
    return MockEnvironment(
        case=case,
        tenor_labels=surf["tenor_labels"],
        tenor_years=surf["tenor_years"],
        strikes=surf["strikes"],
        panel=surf["panel"],
        books=books,
        surface_extract=surface_text,
        book_extract=book_text,
    )
