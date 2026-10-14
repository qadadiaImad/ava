"""Structural detection of bad Totem-implied repo curves.

Works on the Totem-minus-FO difference in *carry space*,
``y_k = B_totem(T_k) - B_FO(T_k)`` with ``B = b T`` (cumulative repo), using
the null covariance ``Sigma`` of the consensus noise (see :mod:`noise`).

Null model (legitimate disagreement): a smooth repo difference
``delta_b(T) = beta0 + beta1 ln T`` -> columns ``T`` and ``T ln T``.

Anomaly templates, each tied to a mechanism:

* ``level``  -- constant in carry (1/T in repo): reference-spot mismatch or
  a dividend before the first pillar;
* ``step k`` -- constant from pillar k on: dividend amount in
  ``(T_{k-1}, T_k]`` or a change of contributor panel at ``T_k``;
* ``spike k`` -- pillar k alone (a dipole in forward repo): forward outlier at
  ``T_k`` or dividend timing across ``T_k``;
* ``season`` -- weighted by the dividend mass near each pillar: systematic
  ex-date / seasonality shift in the allocation of the dividend buckets.

Each template is tested with a generalised likelihood-ratio (matched-filter)
statistic after projecting out the null model; the family-wise critical value
is simulated from the exact null correlation of the statistics, so the false
positive rate is calibrated by construction.  A global chi-square on the
residual catches roughness that no template explains.  Explanations are then
selected greedily (matching pursuit) and refitted jointly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2, norm, t as student_t

from .curves import Dividends

YEAR_DAYS = 365.0


def tenor_label(T: float) -> str:
    months = int(round(T * 12))
    if months < 12 or months % 12:
        return f"{months}M"
    return f"{months // 12}Y"


@dataclass(frozen=True)
class Template:
    kind: str
    pillar: int
    vector: np.ndarray

    @property
    def name(self) -> str:
        return self.kind if self.pillar < 0 else f"{self.kind}@{self.pillar}"


@dataclass(frozen=True)
class DetectionContext:
    """Information used to label explanations in natural units (not to detect)."""

    spot: float
    carry_div: np.ndarray
    dividends: Dividends
    n_opt: np.ndarray
    n_div_at_pillar: np.ndarray
    dispersion_ratio: np.ndarray | None = None


@dataclass(frozen=True)
class Explanation:
    kind: str
    pillar: int
    amplitude: float
    se: float
    label: str

    @property
    def z(self) -> float:
        return self.amplitude / self.se if self.se > 0 else np.inf


@dataclass
class Detection:
    flagged: bool
    p_value: float
    score: float
    q_stat: float
    q_dof: int
    q_p: float
    max_z: float
    max_template: str
    z_crit: float
    explanations: list = field(default_factory=list)
    repo_level: float = 0.0
    repo_level_se: float = 0.0
    repo_slope: float = 0.0
    repo_slope_se: float = 0.0
    fo_mismark: bool = False
    level_crit: float = 0.0
    smooth_fit: np.ndarray | None = None
    anomaly_fit: np.ndarray | None = None

    @property
    def top_kind(self) -> str:
        return self.explanations[0].kind if self.explanations else ("roughness" if self.flagged else "none")


def build_templates(pillars, context: DetectionContext, season_bandwidth: float = 20 / YEAR_DAYS) -> list[Template]:
    T = np.asarray(pillars, dtype=float)
    K = T.size
    idx = np.arange(K)
    out = [Template("level", -1, np.ones(K))]
    out += [Template("step", k, (idx >= k).astype(float)) for k in range(1, K)]
    out += [Template("spike", k, (idx == k).astype(float)) for k in range(K - 1)]
    divs = context.dividends
    if len(divs):
        h = season_bandwidth
        kern = np.exp(-0.5 * ((T[:, None] - divs.ex_times[None, :]) / h) ** 2) / (h * np.sqrt(2 * np.pi))
        season = (kern * divs.amounts[None, :]).sum(axis=1) / (context.spot - context.carry_div)
        if np.max(np.abs(season)) > 0:
            out.append(Template("season", -1, season))
    return out


def nuisance_design(pillars, degree: int = 1) -> np.ndarray:
    """Smooth repo difference ``delta_b(T) = sum_j beta_j (ln T)^j`` expressed in carry space."""
    T = np.asarray(pillars, dtype=float)
    return np.column_stack([T * np.log(T) ** j for j in range(degree + 1)])


class StructuralDetector:
    def __init__(
        self,
        alpha: float = 0.01,
        nuisance_degree: int = 1,
        variance_scale: float = 1.0,
        max_explanations: int = 3,
        season_bandwidth: float = 20 / YEAR_DAYS,
        timing_window: float = 21 / YEAR_DAYS,
        panel_drop_ratio: float = 0.75,
        fo_mismark_min: float = 10e-4,
        n_sim: int = 4000,
        seed: int = 0,
        level_scale: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.nuisance_degree = nuisance_degree
        self.variance_scale = variance_scale
        self.max_explanations = max_explanations
        self.season_bandwidth = season_bandwidth
        self.timing_window = timing_window
        self.panel_drop_ratio = panel_drop_ratio
        self.fo_mismark_min = fo_mismark_min
        self.n_sim = n_sim
        self.seed = seed
        self.level_scale = level_scale

    # ------------------------------------------------------------------ core
    @staticmethod
    def _whitener(cov, scale: float):
        L = np.linalg.cholesky(scale * np.asarray(cov, dtype=float))
        return lambda v: solve_triangular(L, v, lower=True)

    @staticmethod
    def _project_out(Xw: np.ndarray):
        Q, _ = np.linalg.qr(Xw)
        return lambda v: v - Q @ (Q.T @ v)

    def _statistics(self, T, y, cov, context, scale):
        white = self._whitener(cov, scale)
        X0 = nuisance_design(T, self.nuisance_degree)
        X0w = white(X0)
        proj = self._project_out(X0w)
        yw = white(y)
        r = proj(yw)
        templates = build_templates(T, context, self.season_bandwidth)
        raw_w = white(np.column_stack([t.vector for t in templates]))
        tw = proj(raw_w)
        norms = np.sqrt((tw**2).sum(axis=0))
        keep = norms > 1e-8 * norms.max()
        templates = [t for t, k in zip(templates, keep) if k]
        raw_w, tw, norms = raw_w[:, keep], tw[:, keep], norms[keep]
        units = tw / norms
        z = units.T @ r
        # family-wise null distribution of max |z| from the exact correlation of the statistics
        evals, evecs = np.linalg.eigh(units.T @ units)
        root = evecs * np.sqrt(np.clip(evals, 0, None))
        rng = np.random.default_rng(self.seed)
        sims = np.abs(rng.standard_normal((self.n_sim, root.shape[1])) @ root.T).max(axis=1)
        return {
            "white": white,
            "X0": X0,
            "X0w": X0w,
            "yw": yw,
            "q": float(r @ r),
            "dof": int(T.size - X0.shape[1]),
            "templates": templates,
            "raw_w": raw_w,
            "norms": norms,
            "z": z,
            "sims": sims,
            "z_crit": float(np.quantile(sims, 1 - self.alpha / 2)),
        }

    def null_statistics(self, pillars, delta_carry, cov, context: DetectionContext):
        """``(max |z|, z_crit, Q, dof)`` at unit variance scale, for calibration on clean history."""
        st = self._statistics(np.asarray(pillars, float), np.asarray(delta_carry, float), cov, context, 1.0)
        return float(np.max(np.abs(st["z"]))), st["z_crit"], st["q"], st["dof"]

    def calibrate_to_fpr(self, stats, target: float | None = None) -> float:
        """Smallest variance scale whose alarm rate on clean history is at most ``target``.

        ``z`` scales with ``1/sqrt(kappa)`` and ``Q`` with ``1/kappa`` while the
        family-wise critical value does not move, so the scan is exact and cheap.
        """
        target = self.alpha if target is None else target
        z, zc, q, dof = (np.asarray(a, dtype=float) for a in zip(*stats))

        def rate(kappa: float) -> float:
            return float(np.mean((z / np.sqrt(kappa) > zc) | (chi2.sf(q / kappa, dof) < self.alpha / 2)))

        lo, hi = np.log(1e-3), np.log(1e3)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if rate(np.exp(mid)) <= target:
                hi = mid
            else:
                lo = mid
        self.variance_scale = float(np.exp(hi))
        return self.variance_scale

    def calibrate_level_scale(self, level_z, target: float | None = None) -> float:
        """Inflate the standard error of the repo level so that clean history raises at most ``target`` FO-mismark flags.

        ``level_z`` are pairs ``(repo_level / repo_level_se, critical value)`` on clean names, computed with
        ``level_scale = 1``.
        """
        target = self.alpha if target is None else target
        z = np.abs(np.asarray([lz for lz, _ in level_z], dtype=float))
        crit = np.asarray([c for _, c in level_z], dtype=float)
        self.level_scale = max(1.0, float(np.quantile(z / crit, 1 - target)))
        return self.level_scale

    def signal_to_noise(self, pillars, effect, cov) -> float:
        """Norm of a carry-space effect after whitening and projecting out the null model."""
        T = np.asarray(pillars, dtype=float)
        white = self._whitener(cov, self.variance_scale)
        proj = self._project_out(white(nuisance_design(T, self.nuisance_degree)))
        r = proj(white(np.asarray(effect, dtype=float)))
        return float(np.sqrt(r @ r))

    def detect(self, pillars, delta_carry, cov, context: DetectionContext) -> Detection:
        T = np.asarray(pillars, dtype=float)
        y = np.asarray(delta_carry, dtype=float)
        st = self._statistics(T, y, cov, context, self.variance_scale)
        white, X0, X0w, yw = st["white"], st["X0"], st["X0w"], st["yw"]
        templates, raw_w, norms, z, sims = st["templates"], st["raw_w"], st["norms"], st["z"], st["sims"]
        q, dof, z_crit = st["q"], st["dof"], st["z_crit"]
        q_p = float(chi2.sf(q, dof))
        imax = int(np.argmax(np.abs(z)))
        z_max = float(abs(z[imax]))
        p_emp = (1 + np.count_nonzero(sims >= z_max)) / (self.n_sim + 1)
        p_bonf = min(1.0, len(templates) * 2 * norm.sf(z_max))
        p_value = min(1.0, 2 * min(p_emp, p_bonf, q_p))
        flagged = bool(z_max > z_crit or q_p < self.alpha / 2)
        score = float(-np.log10(max(min(2 * p_bonf, 2 * q_p, 1.0), 1e-300)))

        selected: list[int] = []
        if flagged:
            for _ in range(self.max_explanations):
                proj_s = self._project_out(np.column_stack([X0w, raw_w[:, selected]]))
                rs = proj_s(yw)
                V = proj_s(raw_w)
                nv = np.sqrt((V**2).sum(axis=0))
                zs = np.zeros(len(templates))
                ok = nv > 1e-6 * norms.max()
                zs[ok] = np.abs(V[:, ok].T @ rs) / nv[ok]
                zs[selected] = 0.0
                best = int(np.argmax(zs))
                if zs[best] <= z_crit:
                    break
                selected.append(best)

        # joint weighted fit with the selected anomalies
        X = np.column_stack([X0] + [templates[i].vector for i in selected])
        Xw = white(X)
        cov_theta = np.linalg.inv(Xw.T @ Xw)
        theta = cov_theta @ (Xw.T @ yw)
        se = np.sqrt(np.diag(cov_theta))
        p0 = X0.shape[1]
        explanations = [
            self._explain(templates[i], theta[p0 + j], se[p0 + j], T, context) for j, i in enumerate(selected)
        ]
        # the repo-level standard error rests on dispersions estimated from n_A contributors: Student-t quantile
        z_level = student_t.isf(self.alpha / 2, max(float(np.min(context.n_opt[:3])) - 1, 2))
        level, level_se = float(theta[0]), float(se[0]) * self.level_scale
        det = Detection(
            flagged=flagged,
            p_value=p_value,
            score=score,
            q_stat=q,
            q_dof=dof,
            q_p=q_p,
            max_z=z_max,
            max_template=templates[imax].name,
            z_crit=z_crit,
            explanations=explanations,
            repo_level=level,
            repo_level_se=level_se,
            repo_slope=float(theta[1]) if p0 > 1 else 0.0,
            repo_slope_se=float(se[1]) if p0 > 1 else 0.0,
            fo_mismark=bool(abs(level) > z_level * level_se and abs(level) > self.fo_mismark_min),
            level_crit=float(z_level),
            smooth_fit=X0 @ theta[:p0],
            anomaly_fit=X[:, p0:] @ theta[p0:] if selected else np.zeros_like(y),
        )
        return det

    # ------------------------------------------------------------ labelling
    def _explain(self, t: Template, amp: float, se: float, T: np.ndarray, ctx: DetectionContext) -> Explanation:
        S, D = ctx.spot, ctx.carry_div
        divs = ctx.dividends
        if t.kind == "level":
            label = (
                f"Level shift in carry (1/T in repo): reference spot vs FO spot off by {-amp * 100:+.2f}% "
                f"or a dividend before {tenor_label(T[0])} off by {-amp * (S - D[0]):+.3f}"
            )
        elif t.kind == "step":
            k = t.pillar
            lo, hi = T[k - 1], T[k]
            mask = divs.window(lo, hi)
            parts = []
            if mask.any():
                d_int = divs.amounts[mask].sum()
                d_err = -amp * (S - D[k])
                parts.append(
                    f"dividends ex in ({tenor_label(lo)}, {tenor_label(hi)}] look {d_err:+.3f} "
                    f"({d_err / d_int:+.0%}) off in the Totem dividend versus the forwards"
                )
            n0, n1 = ctx.n_opt[k - 1], ctx.n_opt[k]
            m0, m1 = ctx.n_div_at_pillar[k - 1], ctx.n_div_at_pillar[k]
            if n1 <= self.panel_drop_ratio * n0 or m1 <= self.panel_drop_ratio * m0:
                parts.append(f"panel change at {tenor_label(hi)} (option {n0:.0f}->{n1:.0f}, div {m0:.0f}->{m1:.0f})")
            if k == T.size - 1:
                parts.append(f"or a single-pillar outlier: ln F at {tenor_label(hi)} off by {-amp * 1e4:+.0f}bp")
            if not parts:
                parts.append(f"level change from {tenor_label(hi)} on with no dividend or panel change to explain it")
            label = f"Step from {tenor_label(hi)}: " + "; ".join(parts)
        elif t.kind == "spike":
            k = t.pillar
            near = np.abs(divs.ex_times - T[k]) <= self.timing_window
            if near.any():
                i = int(np.flatnonzero(near)[np.argmax(divs.amounts[near])])
                share = -amp * (S - D[k]) / divs.amounts[i]
                days = (divs.ex_times[i] - T[k]) * YEAR_DAYS
                label = (
                    f"Dividend timing at {tenor_label(T[k])}: about {abs(share):.0%} of the {divs.amounts[i]:.2f} "
                    f"dividend ex {days:+.0f}d from expiry is on the other side of the expiry"
                )
            else:
                label = f"Forward outlier at {tenor_label(T[k])}: Totem ln F off by {-amp * 1e4:+.0f}bp"
            if ctx.dispersion_ratio is not None and ctx.dispersion_ratio[k] > 2:
                label += (
                    f" (forward dispersion at {tenor_label(T[k])} is {ctx.dispersion_ratio[k]:.1f}x its neighbours:"
                    " contributors disagree)"
                )
        else:  # season
            label = f"Ex-date profile shift of about {amp * YEAR_DAYS:+.0f} days between the dividend allocation and the forwards"
        return Explanation(t.kind, t.pillar, float(amp), float(se), label)
