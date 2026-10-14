"""Numbers and figures for the PDF report (all computed from the library and the saved benchmark)."""
from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from repo_control.benchmark import DETECTOR_TITLES, DETECTORS, BenchmarkResult  # noqa: E402
from repo_control.buckets import BucketSchedule  # noqa: E402
from repo_control.control import control_lines, implied_repo_jacobian, split_materiality  # noqa: E402
from repo_control.curves import Dividends, ZeroCurve, bootstrap_repo  # noqa: E402
from repo_control.detect import StructuralDetector, tenor_label  # noqa: E402
from repo_control.noise import consensus_covariance, exp_correlation, nested_jump_sd, panel_gap_sd  # noqa: E402
from repo_control.pricer import (  # noqa: E402
    DividendSwap,
    EquityForward,
    EuropeanOption,
    Market,
    StockLoanFee,
    book_value,
    sensitivities,
)
from repo_control.synthetic import (  # noqa: E402
    CATEGORIES,
    EXPECTED_KIND,
    PILLARS,
    SCENARIO_TITLES,
    SCENARIOS,
    simulate_case,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------- styling
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    }
)

TENORS = [tenor_label(t) for t in PILLARS]


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def _marker_kw(color):
    return dict(color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2)


# ---------------------------------------------------------- 1. panel gap
def panel_gap_table(seed: int = 7):
    """Implied-repo noise from the panel gap: closed form versus contributor-level simulation."""
    rng = np.random.default_rng(seed)
    S, q = 100.0, 0.04
    rows = []
    spec = CATEGORIES["EU stock"]
    for T, k in [(0.25, 2), (0.5, 3), (1.0, 5), (2.0, 7), (3.0, 8), (5.0, 10)]:
        D = S * q * T
        sd_D = (0.01 + 0.03 * T) * D
        nA = spec.n_opt[k]
        yidx = min(int(np.ceil(T)), len(spec.n_div) - 1)
        nB = spec.n_div[yidx]
        theory = float(panel_gap_sd(T, S, D, sd_D, nA, nB))
        sims = []
        for _ in range(3000):
            dA = rng.normal(D, sd_D, nA)
            dB = rng.normal(D, sd_D, nB)
            bA = rng.normal(0.004, 0.0015, nA)
            F = ((S - dA) * np.exp((0.025 - bA) * T)).mean()
            sims.append(0.025 - np.log(F / (S - dB.mean())) / T - bA.mean())
        rows.append((tenor_label(T), D, sd_D, nA, nB, theory * 1e4, float(np.std(sims)) * 1e4))
    return rows


def figure_panel_gap(path: Path) -> Path:
    S = 100.0
    T = np.linspace(1 / 12, 5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    configs = [("Index-like: 25 option / 20 div", 25, 20, 0.035, (0.005, 0.015)),
               ("Single stock: 15 / 10", 15, 10, 0.04, (0.01, 0.03)),
               ("Thin panel: 8 / 4", 8, 4, 0.04, (0.01, 0.03))]
    for i, (lab, nA, nB, q, disp) in enumerate(configs):
        D = S * q * T
        sd_D = (disp[0] + disp[1] * T) * D
        ax.plot(T, panel_gap_sd(T, S, D, sd_D, nA, nB) * 1e4, color=SERIES[i], label=lab)
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Implied repo noise (bp)")
    ax.set_title("(a) Panel-gap noise with no repo disagreement", loc="left")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    ax = axes[1]
    for i, eps in enumerate([0.0005, 0.001, 0.003]):
        ax.plot(T, eps / T * 1e4, color=SERIES[i], label=f"spot off by {eps * 100:.2f}%")
    ax.set_yscale("log")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Repo error (bp, log scale)")
    ax.set_title("(b) Spot snapshot mismatch shows up as ε/T", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout(w_pad=2.5)
    return _save(fig, path)


# ------------------------------------------------- 2. control identity demos
def cancellation_table():
    """1Y ATM call, Totem forward 20bp below FO, three dividend panels (single PV dividend)."""
    from scipy.special import ndtr

    S, r, T, K, vol = 100.0, 0.03, 1.0, 100.0, 0.25
    D_fo, b_fo = 3.5, 0.006

    def fwd(D, b):
        return (S - D) * np.exp((r - b) * T)

    def call(D, b):
        F = fwd(D, b)
        s = vol * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * s * s) / s
        return np.exp(-r * T) * (F * ndtr(d1) - K * ndtr(d1 - s))

    F_tot = fwd(D_fo, b_fo) * (1 - 0.002)
    h = 1e-6
    sD = (call(D_fo + h, b_fo) - call(D_fo - h, b_fo)) / (2 * h)
    sb = (call(D_fo, b_fo + h) - call(D_fo, b_fo - h)) / (2 * h)
    rows = []
    for D_tot in (3.5, 3.0, 4.2):
        b_tot = r - np.log(F_tot / (S - D_tot)) / T
        div_line = sD * (D_tot - D_fo)
        repo_line = sb * (b_tot - b_fo)
        full = call(D_tot, b_tot) - call(D_fo, b_fo)
        rows.append((D_tot, b_tot * 1e4, div_line, repo_line, div_line + repo_line, full))
    return rows


def _demo_market():
    T = PILLARS
    sch = BucketSchedule(np.array([0, 0.2164, 1.2164, 2.219, 3.219, 4.219, 5.22]))
    prof = Dividends(np.array([0.55, 1.55, 2.55, 3.55, 4.55]), np.ones(5))
    levels = np.array([0.0, 3.5, 3.7, 3.9, 4.1, 4.3])
    m = Market(100.0, ZeroCurve.flat(0.03), ZeroCurve(T, 0.006 + 0 * T), sch, levels, prof)
    options = [EuropeanOption(True, 100, 1.0, 0.25, 10), EuropeanOption(False, 95, 2.0, 0.25, -5), EquityForward(3.0, 95, 2)]
    books = {
        "A: options and forwards": options,
        "B: A + short 2028 dividend swap": options + [DividendSwap(2, 3.7, 2.3, -8)],
        "C: A + 1Y stock-loan fee": options + [StockLoanFee(1.0, 3)],
    }
    return m, books


def book_lines(bumps=(-0.4, 0.0, 0.5)):
    """Control lines per book when the Totem dividend panel moves 2027-2028 by ``bump`` (cash)."""
    m, books = _demo_market()
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
            out.append((name, bump, cl.div_line, cl.repo_line, cl.total, cl.forward_line, cl.non_forward_div_line, full))
        out.append((name, "materiality", mat.div_line_sd, mat.split_sd, mat.cancellation))
    return out


def figure_cancellation(path: Path) -> Path:
    m, books = _demo_market()
    T = PILLARS
    F_tot = m.forward(T) * (1 - 0.002)
    repo_fwd = bootstrap_repo(m.spot, m.rates, m.dividends, T, F_tot)
    J = implied_repo_jacobian(m.spot, m.rates, m.schedule, m.div_levels, m.profile, T, F_tot)
    bumps = np.linspace(-0.6, 0.6, 25)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, key in zip(axes, ["A: options and forwards", "B: A + short 2028 dividend swap"]):
        book = books[key]
        sr, sd = sensitivities(book, m)
        rows = []
        for bump in bumps:
            lv = m.div_levels.copy()
            lv[1:3] += bump
            mt = m.with_levels(lv)
            rt = bootstrap_repo(m.spot, m.rates, mt.dividends, T, F_tot)
            cl = control_lines(sr, sd, rt.zeros - m.repo.zeros, repo_fwd.zeros - m.repo.zeros, lv - m.div_levels, J)
            full = book_value(book, mt.with_repo(rt)) - book_value(book, m)
            rows.append((cl.div_line, cl.repo_line, cl.total, full))
        rows = np.array(rows)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.plot(bumps, rows[:, 0], color=SERIES[1], label="Dividend line")
        ax.plot(bumps, rows[:, 1], color=SERIES[2], label="Repo line")
        ax.plot(bumps, rows[:, 2], color=SERIES[0], linewidth=2.2, label="Total (div + repo)")
        ax.plot(bumps[::4], rows[::4, 3], linestyle="none", **_marker_kw(SERIES[0]), label="Total, full revaluation")
        ax.set_title(f"Book {key}", loc="left")
    axes[0].set_ylabel("Control line (P&L units)")
    fig.supxlabel("Totem dividend panel minus FO on the 2027 and 2028 buckets (cash per bucket)", fontsize=8.5,
                  color=INK2, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.03, 1, 1))
    return _save(fig, path)


# -------------------------------------------------------- 3. benchmark
def load_benchmark(path: Path) -> BenchmarkResult:
    return BenchmarkResult.from_json(str(path))


def detection_table(res: BenchmarkResult):
    return [(s, [res.detection_rate(s, d) for d in DETECTORS]) for s in SCENARIOS]


def auc_table(res: BenchmarkResult):
    return [(s, [res.auc(s, d) for d in DETECTORS]) for s in SCENARIOS if s != "clean"]


def fpr_by_category(res: BenchmarkResult):
    return [(c, [res.detection_rate("clean", d, c) for d in DETECTORS]) for c in CATEGORIES]


def attribution_table(res: BenchmarkResult):
    rows = []
    for s in list(EXPECTED_KIND) + ["fo_repo_mismark", "div_gross_net"]:
        recs = [r for r in res.records if r["scenario"] == s]
        flagged = [r for r in recs if r["flags"]["structural"]]
        counts = {}
        for r in recs:
            k = r["structural"]["top_kind"]
            counts[k] = counts.get(k, 0) + 1
        rows.append(
            {
                "scenario": s,
                "n": len(recs),
                "detected": len(flagged) / len(recs),
                "attr_cond": (np.mean([r["attribution_ok"] for r in flagged]) if flagged else float("nan"))
                if s in EXPECTED_KIND
                else float("nan"),
                "fo_flag": np.mean([r["structural"]["fo_mismark"] for r in recs]),
                "counts": counts,
            }
        )
    return rows


def figure_detection(res: BenchmarkResult, path: Path) -> Path:
    scen = [s for s in SCENARIOS if s != "clean"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(scen))[::-1]
    offs = np.linspace(-0.28, 0.28, len(DETECTORS))
    for i, d in enumerate(DETECTORS):
        rates = [res.detection_rate(s, d) for s in scen]
        ax.plot(rates, y + offs[i], linestyle="none", markersize=6.5 if d == "structural" else 5.5,
                label=DETECTOR_TITLES[d], **_marker_kw(SERIES[i]))
    for yy in y:
        ax.axhline(yy - 0.5, color=GRID, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_TITLES[s] for s in scen])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"Share of names flagged (all detectors calibrated to {res.fpr_target:.0%} false alarms on clean history)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, fontsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def figure_power(res: BenchmarkResult, path: Path) -> Path:
    edges = (0, 1, 2, 3, 4, 6, 8, 12, np.inf)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    labels = ["0-1", "1-2", "2-3", "3-4", "4-6", "6-8", "8-12", "12+"]
    x = np.arange(len(labels))
    for i, d in enumerate(DETECTORS):
        pc = res.power_curve(d, edges)
        ax.plot(x, [p[2] for p in pc], label=DETECTOR_TITLES[d], linewidth=2.2 if d == "structural" else 1.4,
                **_marker_kw(SERIES[i]))
    counts = [p[3] for p in res.power_curve("structural", edges)]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={c})" for lab, c in zip(labels, counts)])
    ax.set_xlabel("Signal-to-noise of the injected anomaly (whitened, after the null fit)")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, path)


def figure_attribution(res: BenchmarkResult, path: Path) -> Path:
    scen = list(EXPECTED_KIND) + ["fo_repo_mismark", "div_gross_net"]
    kinds = ["level", "step", "spike", "season", "roughness", "none"]
    M = np.zeros((len(scen), len(kinds)))
    for i, s in enumerate(scen):
        recs = [r for r in res.records if r["scenario"] == s]
        for r in recs:
            M[i, kinds.index(r["structural"]["top_kind"])] += 1
        M[i] /= max(len(recs), 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + BLUE_RAMP)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.01:
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(["Level", "Step", "Spike", "Season", "Roughness", "No alarm"])
    ax.set_yticks(range(len(scen)))
    ax.set_yticklabels([SCENARIO_TITLES[s] for s in scen])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("First explanation chosen by the structural detector (share of names per scenario)", loc="left")
    fig.tight_layout()
    return _save(fig, path)


# -------------------------------------------------- 4. example gallery
def _calibrated_detector(res: BenchmarkResult, category: str) -> StructuralDetector:
    return StructuralDetector(alpha=res.fpr_target, variance_scale=res.variance_scale[category],
                              level_scale=res.level_scale.get(category, 1.0))


_PICKS: dict = {}


def pick_examples(res: BenchmarkResult, category: str = "EU stock"):
    """First correctly explained name per scenario whose clean twin stays within 12bp of FO beyond 1Y."""
    key = (id(res), category)
    if key in _PICKS:
        return _PICKS[key]
    wanted = ["spot_sync", "div_bucket", "div_timing", "season_profile", "fwd_outlier", "fo_repo_mismark"]
    picks = {}
    for s in wanted:
        cands = [r for r in res.records if r["category"] == category and r["scenario"] == s and r["attribution_ok"]]
        if s != "fo_repo_mismark":
            cands = [r for r in cands if 6 <= r["snr"] <= 60] or cands
        chosen = cands[0]["seed"] if cands else None
        for r in cands[:25]:
            clean = simulate_case(r["seed"], category, "clean").implied
            if np.max(np.abs(clean.delta_repo[PILLARS > 1.01])) * 1e4 < 12:
                chosen = r["seed"]
                break
        if chosen is not None:
            picks[s] = chosen
    _PICKS[key] = picks
    return picks


def _wrap(text: str, width: int = 58) -> str:
    return "\n".join(textwrap.wrap(text, width))


def figure_gallery(res: BenchmarkResult, path: Path, category: str = "EU stock"):
    picks = pick_examples(res, category)
    det = _calibrated_detector(res, category)
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.4))
    x = np.arange(PILLARS.size)
    notes = {}
    for ax, (scen, seed) in zip(axes.ravel(), picks.items()):
        case = simulate_case(seed, category, scen)
        im = case.implied
        d = det.detect(im.pillars, im.delta_carry, im.cov, im.context())
        b_fo = im.repo_fo * 1e4
        b_tot = im.repo_totem.zeros * 1e4
        repaired = (im.delta_carry - d.anomaly_fit) / im.pillars * 1e4 + b_fo
        band = 2 * np.sqrt(np.diag(im.cov) * det.variance_scale) / im.pillars * 1e4
        ax.fill_between(x, b_fo - band, b_fo + band, color=SERIES[0], alpha=0.10, linewidth=0,
                        label="FO ± 2 sd of consensus noise")
        if d.explanations:
            ax.plot(x, repaired, label="Totem-implied, anomaly removed", linewidth=2.6, color=SERIES[2], alpha=0.9)
        ax.plot(x, b_fo, label="FO repo", **_marker_kw(SERIES[0]))
        ax.plot(x, b_tot, label="Totem-implied repo", **_marker_kw(SERIES[1]))
        lo, hi = min(b_fo.min(), b_tot.min()), max(b_fo.max(), b_tot.max())
        pad = 0.12 * (hi - lo + 1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(TENORS, fontsize=7)
        ax.set_title(SCENARIO_TITLES[scen], loc="left")
        ax.set_ylabel("Zero repo (bp)")
        if d.explanations:
            note = d.explanations[0].label
        elif d.fo_mismark:
            note = (f"No Totem anomaly. FO repo level differs from Totem by {-d.repo_level * 1e4:+.0f}bp "
                    f"(± {d.repo_level_se * 1e4:.0f}bp): FO mismark candidate")
        else:
            note = "No anomaly"
        notes[scen] = (seed, note)
        ax.text(0.0, -0.30, _wrap(note, 62), transform=ax.transAxes, fontsize=6.8, color=INK2, va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(k) for k in ("FO repo", "Totem-implied repo", "Totem-implied, anomaly removed",
                                       "FO ± 2 sd of consensus noise") if k in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), fontsize=7.3)
    fig.tight_layout(h_pad=4.2, w_pad=1.8, rect=(0, 0, 1, 0.975))
    _save(fig, path)
    return path, notes


def figure_decomposition(res: BenchmarkResult, path: Path, category: str = "EU stock", scenario: str = "div_bucket"):
    seed = pick_examples(res, category)[scenario]
    case = simulate_case(seed, category, scenario)
    im = case.implied
    det = _calibrated_detector(res, category)
    d = det.detect(im.pillars, im.delta_carry, im.cov, im.context())
    x = np.arange(PILLARS.size)
    sd = np.sqrt(np.diag(im.cov) * det.variance_scale)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.fill_between(x, d.smooth_fit * 1e4 - 2 * sd * 1e4, d.smooth_fit * 1e4 + 2 * sd * 1e4, color=SERIES[0], alpha=0.10,
                    linewidth=0, label="Null fit ± 2 sd of consensus noise")
    ax.plot(x, d.smooth_fit * 1e4, color=SERIES[0], label="Null fit: smooth repo difference")
    ax.plot(x, (d.smooth_fit + d.anomaly_fit) * 1e4, color=SERIES[1], label="Null fit + selected template")
    ax.plot(x, im.delta_carry * 1e4, linestyle="none", label="Observed Totem − FO carry", **_marker_kw(INK))
    vals = np.concatenate([im.delta_carry, d.smooth_fit + d.anomaly_fit, d.smooth_fit]) * 1e4
    span = vals.max() - vals.min()
    ax.set_ylim(vals.min() - 0.35 * span, vals.max() + 0.35 * span)
    ax.set_xticks(x)
    ax.set_xticklabels(TENORS)
    ax.set_ylabel("ΔB = Δ(b·T) (bp of spot)")
    ax.set_title(f"Carry-space view of the '{SCENARIO_TITLES[scenario]}' example", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    _save(fig, path)
    return path, (seed, d.explanations[0].label if d.explanations else "")


def jump_table():
    rows = []
    for n, m in [(16, 12), (12, 7), (15, 7), (10, 4)]:
        rows.append((n, m, float(nested_jump_sd(1.0, n, m)), 1 / np.sqrt(m)))
    return rows
