"""Data contract and abstract sources.

The application is bank-agnostic: every institution plugs its own market
data by implementing :class:`DataSource` (vega surface + uncertainty model)
and, optionally, :class:`ScenarioSource` (candidate netting partitions).
The exchange format is :class:`MarketDataBundle`, serialisable to JSON.

JSON schema (one document per as-of date / underlying / valuation exposure):

.. code-block:: json

    {
      "meta": {
        "asof": "2026-06-11",
        "underlying": "EURO STOXX 50",
        "desk": "EQD Exotics",
        "vega_unit": "EUR per vol point",
        "uncertainty_unit": "vol points",
        "description": "free text"
      },
      "tenors":       ["1M", "3M", ...],
      "tenor_years":  [0.0833, 0.25, ...],
      "strikes":      [0.80, 0.90, ...],
      "vega":         [[...], ...],
      "s":            [[...], ...],
      "corr_mat":     [[...], ...],
      "corr_strike":  [[...], ...],
      "corr_full":    null
    }

``vega`` and ``s`` are M x K matrices (rows = tenors, columns = strikes,
strikes expressed in moneyness K/F as recommended by the note, sec. 8).
Correlations are given either in decoupled per-axis form (``corr_mat``
M x M and ``corr_strike`` K x K, combined entrywise as
rho[(m,k),(m',k')] = corr_mat[m,m'] * corr_strike[k,k'] — the decoupling
hypothesis of the note's annex) or as a full ``corr_full`` (MK) x (MK)
matrix on the row-major vectorisation; ``corr_full`` takes precedence
when present.

No Kronecker / tensor product is ever assembled: all variances and
covariances are evaluated in matrix form with ordinary matrix products
and a final sum reduction (Property 1 of the note),

    Var(<V, dsigma>) = sum( W * (corr_mat @ W @ corr_strike) ),
    W = V * s  (diagonal scaling by the uncertainties).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import IO, Optional, Union

import numpy as np

__all__ = [
    "MarketDataBundle",
    "DataSource",
    "JSONBundleSource",
    "SyntheticDataSource",
    "ScenarioSource",
    "MatrixScenarioSource",
    "JSONScenarioSource",
    "nearest_correlation",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def nearest_correlation(rho: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the cone of valid correlation matrices
    (eigenvalue clipping + unit-diagonal rescaling)."""
    rho = 0.5 * (rho + rho.T)
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, eps, None)
    fixed = (v * w) @ v.T
    d = np.sqrt(np.clip(np.diag(fixed), eps, None))
    fixed = fixed / d[:, None] / d[None, :]
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _rank1(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Rank-one surface as an ordinary (M, 1) @ (1, K) matrix product."""
    return np.asarray(col, dtype=float).reshape(-1, 1) @ np.asarray(row, dtype=float).reshape(1, -1)


# --------------------------------------------------------------------------- #
# Data contract
# --------------------------------------------------------------------------- #
@dataclass
class MarketDataBundle:
    """Self-contained input set for one valuation exposure.

    Vectorisation convention (only relevant when a full ``corr_full`` is
    supplied): row-major (C order), index ``i = m * K + k``. In the
    decoupled case nothing is ever vectorised — quadratic forms are
    evaluated directly on the (M, K) grid with ordinary matrix products.
    """

    tenors: list[str]
    tenor_years: list[float]
    strikes: list[float]
    vega: np.ndarray                       # (M, K)  EUR / vol point
    s: np.ndarray                          # (M, K)  vol points, > 0
    corr_mat: Optional[np.ndarray] = None      # (M, M)
    corr_strike: Optional[np.ndarray] = None   # (K, K)
    corr_full: Optional[np.ndarray] = None     # (MK, MK), overrides the decoupled pair
    meta: dict = field(default_factory=dict)

    # -- shape ------------------------------------------------------------- #
    @property
    def M(self) -> int:
        return len(self.tenors)

    @property
    def K(self) -> int:
        return len(self.strikes)

    @property
    def n(self) -> int:
        return self.M * self.K

    # -- model ------------------------------------------------------------- #
    def covariance_of(self, expo_a: np.ndarray, expo_b: np.ndarray) -> float:
        """Cov(<U, dsigma>, <V, dsigma>) for two (M, K) exposure matrices
        (Property 1 of the note), with ordinary matrix products only.

        Decoupled correlations (annex of the note): with W = V * s the
        bilinear form is  sum( (U * s) * (corr_mat @ W @ corr_strike) ) —
        a sandwich of usual matrix products followed by a sum reduction.
        The full (MK, MK) covariance matrix is never assembled.
        """
        wa = np.asarray(expo_a, dtype=float) * self.s
        wb = np.asarray(expo_b, dtype=float) * self.s
        if self.corr_full is not None:
            return float(wa.flatten() @ self.corr_full @ wb.flatten())
        if self.corr_mat is None or self.corr_strike is None:
            raise ValueError(
                "Provide either corr_full or both corr_mat and corr_strike."
            )
        return float(np.sum(wa * (self.corr_mat @ wb @ self.corr_strike)))

    def variance_of(self, exposure: np.ndarray) -> float:
        """Var(<exposure, dsigma>) — quadratic case of :meth:`covariance_of`."""
        return self.covariance_of(exposure, exposure)

    def is_decoupled(self) -> bool:
        """True when the correlation is given in per-axis decoupled form."""
        return self.corr_full is None

    # -- validation -------------------------------------------------------- #
    def validate(self) -> list[str]:
        """Return a list of human-readable issues (empty list = clean)."""
        issues: list[str] = []
        if self.vega.shape != (self.M, self.K):
            issues.append(f"vega shape {self.vega.shape} != ({self.M}, {self.K})")
        if self.s.shape != (self.M, self.K):
            issues.append(f"s shape {self.s.shape} != ({self.M}, {self.K})")
        if np.any(self.s <= 0):
            issues.append("uncertainty matrix s must be strictly positive")
        for name, rho, dim in (
            ("corr_mat", self.corr_mat, self.M),
            ("corr_strike", self.corr_strike, self.K),
            ("corr_full", self.corr_full, self.n),
        ):
            if rho is None:
                continue
            if rho.shape != (dim, dim):
                issues.append(f"{name} shape {rho.shape} != ({dim}, {dim})")
                continue
            if not np.allclose(rho, rho.T, atol=1e-8):
                issues.append(f"{name} is not symmetric")
            if not np.allclose(np.diag(rho), 1.0, atol=1e-6):
                issues.append(f"{name} diagonal is not 1")
            if np.linalg.eigvalsh(rho).min() < -1e-8:
                issues.append(f"{name} is not positive semi-definite")
        if self.corr_full is None and (self.corr_mat is None or self.corr_strike is None):
            issues.append("no correlation provided (corr_full or corr_mat+corr_strike)")
        return issues

    # -- (de)serialisation --------------------------------------------------#
    def to_dict(self) -> dict:
        def arr(a):
            return None if a is None else np.asarray(a).tolist()

        return {
            "meta": self.meta,
            "tenors": list(self.tenors),
            "tenor_years": list(self.tenor_years),
            "strikes": list(self.strikes),
            "vega": arr(self.vega),
            "s": arr(self.s),
            "corr_mat": arr(self.corr_mat),
            "corr_strike": arr(self.corr_strike),
            "corr_full": arr(self.corr_full),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "MarketDataBundle":
        def arr(x):
            return None if x is None else np.asarray(x, dtype=float)

        return cls(
            tenors=list(d["tenors"]),
            tenor_years=[float(x) for x in d["tenor_years"]],
            strikes=[float(x) for x in d["strikes"]],
            vega=arr(d["vega"]),
            s=arr(d["s"]),
            corr_mat=arr(d.get("corr_mat")),
            corr_strike=arr(d.get("corr_strike")),
            corr_full=arr(d.get("corr_full")),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "MarketDataBundle":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# Abstract sources — the bank-side integration points
# --------------------------------------------------------------------------- #
class DataSource(ABC):
    """Abstract market-data source.

    Implement :meth:`load` against your golden sources (risk system vegas,
    Totem/Markit dispersions, broker quotes...) and the whole app works
    unchanged. See :class:`JSONBundleSource` for the file-based reference
    implementation.
    """

    @abstractmethod
    def load(self) -> MarketDataBundle:
        ...

    def describe(self) -> str:
        return self.__class__.__name__


class JSONBundleSource(DataSource):
    """Reference implementation reading the JSON contract from a path or
    file-like object."""

    def __init__(self, source: Union[str, IO]):
        self._source = source

    def load(self) -> MarketDataBundle:
        if hasattr(self._source, "read"):
            text = self._source.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
        else:
            with open(self._source, "r", encoding="utf-8") as fh:
                text = fh.read()
        return MarketDataBundle.from_json(text)

    def describe(self) -> str:
        name = getattr(self._source, "name", self._source)
        return f"JSON bundle ({name})"


class ScenarioSource(ABC):
    """Abstract source of candidate netting schemes.

    A scenario is an (M, K) integer label matrix over the (tenor, strike)
    grid: cells sharing a label belong to the same netting set. The bank
    plugs its own provider (e.g. the desk's current netting convention)
    by implementing :meth:`load`.
    """

    @abstractmethod
    def load(self) -> np.ndarray:
        ...


class MatrixScenarioSource(ScenarioSource):
    def __init__(self, labels: np.ndarray):
        self._labels = np.asarray(labels, dtype=int)

    def load(self) -> np.ndarray:
        return self._labels


class JSONScenarioSource(ScenarioSource):
    """Reads ``{"labels": [[...], ...]}`` from a path or file-like object."""

    def __init__(self, source: Union[str, IO]):
        self._source = source

    def load(self) -> np.ndarray:
        if hasattr(self._source, "read"):
            text = self._source.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
        else:
            with open(self._source, "r", encoding="utf-8") as fh:
                text = fh.read()
        return np.asarray(json.loads(text)["labels"], dtype=int)


# --------------------------------------------------------------------------- #
# Synthetic demo source
# --------------------------------------------------------------------------- #
class SyntheticDataSource(DataSource):
    """Realistic equity-vol demo books for development and demos.

    Books
    -----
    ``smile_book``     long ATM gamma vs short wings (fly seller profile)
    ``calendar_book``  long short-dated / short long-dated ATM (calendar)
    ``mixed_desk``     level + risk-reversal + calendar mix, the hard case
    """

    TENORS = ["1M", "2M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"]
    TENOR_YEARS = [1 / 12, 2 / 12, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    STRIKES = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]

    def __init__(self, book: str = "mixed_desk", seed: int = 7):
        self.book = book
        self.seed = seed

    def load(self) -> MarketDataBundle:
        rng = np.random.default_rng(self.seed)
        ten = np.asarray(self.TENOR_YEARS)
        stk = np.asarray(self.STRIKES)
        M, K = len(ten), len(stk)
        x = stk - 1.0                                   # moneyness offset
        atm_w = np.exp(-0.5 * (x / 0.07) ** 2)          # ATM bump
        wing_w = np.clip(np.abs(x) - 0.05, 0.0, None)   # wing weight

        if self.book == "smile_book":
            vega = (
                _rank1(np.exp(-ten / 2.0), 220.0 * atm_w)
                - _rank1(np.exp(-ten / 3.0), 900.0 * wing_w)
            )
        elif self.book == "calendar_book":
            term = np.where(ten <= 0.5, 1.0, -0.8 * np.exp(-(ten - 0.5) / 4.0))
            vega = _rank1(term, 260.0 * atm_w + 80.0 * np.exp(-0.5 * (x / 0.15) ** 2))
        else:  # mixed_desk
            term = np.where(ten <= 0.5, 1.0, -0.7 * np.exp(-(ten - 0.5) / 5.0))
            vega = (
                _rank1(term, 240.0 * atm_w)               # calendar of straddles
                + _rank1(np.exp(-ten / 4.0), 650.0 * x)   # risk-reversal
                - _rank1(np.exp(-ten / 3.0), 520.0 * wing_w)  # short flies
            )
        vega = vega * 1_000.0                              # EUR / vol pt
        vega += rng.normal(0.0, 0.04 * np.abs(vega).mean(), size=(M, K))

        # Consensus dispersion: wider in wings, in short tenors and at the
        # long end (fewer Totem contributors).
        s = (
            0.25
            * (1.0 + 2.2 * np.abs(x))[None, :]
            * (1.0 + 0.55 / np.sqrt(ten / 0.25))[:, None]
            * (1.0 + 0.18 * np.log1p(ten))[:, None]
        )

        # Decoupled per-axis correlations (annex of the note): exponential
        # kernels in log-tenor and in moneyness.
        lt = np.log(ten)
        corr_mat = np.exp(-np.abs(lt[:, None] - lt[None, :]) / 1.4)
        corr_strike = np.exp(-np.abs(stk[:, None] - stk[None, :]) / 0.22)

        return MarketDataBundle(
            tenors=list(self.TENORS),
            tenor_years=list(self.TENOR_YEARS),
            strikes=list(self.STRIKES),
            vega=vega,
            s=s,
            corr_mat=nearest_correlation(corr_mat),
            corr_strike=nearest_correlation(corr_strike),
            meta={
                "asof": "2026-06-11",
                "underlying": "EURO STOXX 50 (synthetic)",
                "desk": "EQD Exotics — demo",
                "vega_unit": "EUR per vol point",
                "uncertainty_unit": "vol points",
                "description": f"Synthetic book '{self.book}', seed {self.seed}",
            },
        )

    def describe(self) -> str:
        return f"Synthetic ({self.book})"
