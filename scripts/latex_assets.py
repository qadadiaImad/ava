"""Figures (EN/FR), worked examples and generated LaTeX snippets for the theory notes.

Everything is computed from the library and the saved benchmark; nothing is typed in by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import BSpline
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_analysis as ra  # noqa: E402
from report_analysis import AXIS, BLUE_RAMP, GRID, INK, INK2, SERIES, SURFACE, _marker_kw, plt  # noqa: E402

from repo_control.benchmark import DETECTORS, BenchmarkResult  # noqa: E402
from repo_control.curves import Dividends, ZeroCurve, bootstrap_repo, forward  # noqa: E402
from repo_control.detect import StructuralDetector, build_templates, nuisance_design, tenor_label  # noqa: E402
from repo_control.parity import forward_from_parity  # noqa: E402
from repo_control.synthetic import CATEGORIES, EXPECTED_KIND, PILLARS, SCENARIOS, simulate_case  # noqa: E402

# ------------------------------------------------------------------ language
TXT = {
    "en": {
        "tenor_years": "Tenor (years)",
        "gap_noise": "Implied repo noise (bp)",
        "gap_title": "(a) Panel-gap noise, no repo disagreement",
        "cfg_index": "Index-like: 25 option / 20 div",
        "cfg_stock": "Single stock: 15 / 10",
        "cfg_thin": "Thin panel: 8 / 4",
        "spot_title": "(b) Spot snapshot mismatch reads as ε/T",
        "spot_lab": "spot off by {x}%",
        "repo_err": "Repo error (bp, log scale)",
        "div_line": "Dividend line",
        "repo_line": "Repo line",
        "total": "Total (div + repo)",
        "total_full": "Total, full revaluation",
        "book_a": "Book A: options and forwards",
        "book_b": "Book B: A + short 2028 dividend swap",
        "cancel_x": "Totem dividend panel minus FO on the 2027 and 2028 buckets (cash per bucket)",
        "cancel_y": "Control line (P&L units)",
        "det_x": "Share of names flagged (all detectors calibrated to {fpr} false alarms on clean history)",
        "snr_x": "Signal-to-noise of the injected anomaly (whitened, after the null fit)",
        "det_rate": "Detection rate",
        "theory": "Theory: one known template",
        "attr_title": "First explanation chosen by the structural detector (share of names per scenario)",
        "kinds": ["Level", "Step", "Spike", "Season", "Roughness", "No alarm"],
        "fo_repo": "FO repo",
        "tot_repo": "Totem-implied repo",
        "repaired": "Totem-implied, anomaly removed",
        "band": "FO ± 2 sd of consensus noise",
        "zero_repo": "Zero repo (bp)",
        "null_band": "Null fit ± 2 sd of consensus noise",
        "null_fit": "Null fit: smooth repo difference",
        "null_tmpl": "Null fit + selected template",
        "obs": "Observed Totem − FO carry",
        "carry_y": "ΔB = Δ(b·T) (bp of spot)",
        "decomp_title": "Carry-space view: Totem dividend bucket error",
        "detectors": {
            "naive_level": "Naive gap |Δb|",
            "naive_shape": "Naive shape (repo space)",
            "spline_gmm_curve": "B-spline + GMM (curve)",
            "spline_gmm_diff": "B-spline + GMM (Totem − FO)",
            "structural": "Structural GLRT (proposed)",
        },
        "scenarios": {
            "clean": "Clean (null)",
            "spot_sync": "Spot snapshot mismatch",
            "div_bucket": "Totem dividend bucket error",
            "div_timing": "Ex-date timing split in option panel",
            "season_profile": "Stale ex-date profile in allocation",
            "fwd_outlier": "Forward outlier at one expiry",
            "panel_shift": "Biased long-dated sub-panel",
            "fo_repo_mismark": "FO repo mismark (genuine)",
            "div_gross_net": "Gross/net dividend bias (all buckets)",
        },
        "categories": {"EU stock": "EU stock", "US stock": "US stock", "HTB stock": "Hard to borrow", "EU index": "EU index"},
    },
    "fr": {
        "tenor_years": "Maturité (années)",
        "gap_noise": "Bruit du repo implicite (pb)",
        "gap_title": "(a) Bruit d'écart de panel, sans désaccord sur le repo",
        "cfg_index": "Type indice : 25 options / 20 div.",
        "cfg_stock": "Action : 15 / 10",
        "cfg_thin": "Panel réduit : 8 / 4",
        "spot_title": "(b) Un décalage de spot se lit en ε/T",
        "spot_lab": "spot décalé de {x} %",
        "repo_err": "Erreur de repo (pb, échelle log)",
        "div_line": "Ligne dividende",
        "repo_line": "Ligne repo",
        "total": "Total (div. + repo)",
        "total_full": "Total, réévaluation complète",
        "book_a": "Portefeuille A : options et forwards",
        "book_b": "Portefeuille B : A + swap de dividendes 2028 vendu",
        "cancel_x": "Panel dividendes Totem moins FO sur les périodes 2027 et 2028 (montant par période)",
        "cancel_y": "Ligne de contrôle (unités de P&L)",
        "det_x": "Part des sous-jacents signalés (tous les détecteurs calibrés à {fpr} de fausses alertes sur l'historique sain)",
        "snr_x": "Rapport signal sur bruit de l'anomalie injectée (blanchi, après ajustement nul)",
        "det_rate": "Taux de détection",
        "theory": "Théorie : un gabarit connu",
        "attr_title": "Première explication retenue par le détecteur structurel (part des sous-jacents par scénario)",
        "kinds": ["Niveau", "Marche", "Pic", "Saison", "Rugosité", "Pas d'alerte"],
        "fo_repo": "Repo FO",
        "tot_repo": "Repo implicite Totem",
        "repaired": "Implicite Totem, anomalie retirée",
        "band": "FO ± 2 écarts-types du bruit de consensus",
        "zero_repo": "Repo zéro-coupon (pb)",
        "null_band": "Ajustement nul ± 2 écarts-types du bruit",
        "null_fit": "Ajustement nul : écart de repo lisse",
        "null_tmpl": "Ajustement nul + gabarit retenu",
        "obs": "Portage observé Totem − FO",
        "carry_y": "ΔB = Δ(b·T) (pb du spot)",
        "decomp_title": "Vue en espace de portage : erreur sur une période de dividendes Totem",
        "detectors": {
            "naive_level": "Écart naïf |Δb|",
            "naive_shape": "Forme naïve (espace repo)",
            "spline_gmm_curve": "B-spline + GMM (courbe)",
            "spline_gmm_diff": "B-spline + GMM (Totem − FO)",
            "structural": "GLRT structurel (proposé)",
        },
        "scenarios": {
            "clean": "Sain (hypothèse nulle)",
            "spot_sync": "Décalage du spot de référence",
            "div_bucket": "Erreur sur une période de dividendes Totem",
            "div_timing": "Désaccord de date de détachement (panel options)",
            "season_profile": "Profil de détachement périmé dans l'allocation",
            "fwd_outlier": "Forward aberrant à une échéance",
            "panel_shift": "Sous-panel long terme biaisé",
            "fo_repo_mismark": "Erreur de marquage du repo FO (réelle)",
            "div_gross_net": "Biais brut/net des dividendes (toutes périodes)",
        },
        "categories": {"EU stock": "Action UE", "US stock": "Action US", "HTB stock": "Difficile à emprunter", "EU index": "Indice UE"},
    },
}


def tenors(lang: str):
    out = [tenor_label(t) for t in PILLARS]
    return [t.replace("Y", "A") for t in out] if lang == "fr" else out


def _fmt_axis(ax, lang: str, axis: str = "both"):
    """Decimal comma and true minus sign on linear axes for the French figures."""
    if lang != "fr":
        return

    def f(x, pos):
        s = f"{x:g}"
        return s.replace("-", "−").replace(".", ",")

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(f))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(f))


def _pct_fig(x: float, lang: str, digits: int = 0) -> str:
    s = f"{100 * x:.{digits}f}"
    return f"{s.replace('.', ',')} %" if lang == "fr" else f"{s}%"


def _save(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- figures
def fig_panel_gap(path: Path, lang: str) -> Path:
    t = TXT[lang]
    from repo_control.noise import panel_gap_sd

    S = 100.0
    T = np.linspace(1 / 12, 5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    for i, (lab, nA, nB, q, disp) in enumerate([
        (t["cfg_index"], 25, 20, 0.035, (0.005, 0.015)),
        (t["cfg_stock"], 15, 10, 0.04, (0.01, 0.03)),
        (t["cfg_thin"], 8, 4, 0.04, (0.01, 0.03)),
    ]):
        D = S * q * T
        ax.plot(T, panel_gap_sd(T, S, D, (disp[0] + disp[1] * T) * D, nA, nB) * 1e4, color=SERIES[i], label=lab)
    ax.set_xlabel(t["tenor_years"])
    ax.set_ylabel(t["gap_noise"])
    ax.set_title(t["gap_title"], loc="left")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    _fmt_axis(ax, lang)
    ax = axes[1]
    for i, eps in enumerate([0.0005, 0.001, 0.003]):
        lab = t["spot_lab"].format(x=f"{eps * 100:.2f}".replace(".", "," if lang == "fr" else "."))
        ax.plot(T, eps / T * 1e4, color=SERIES[i], label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(t["tenor_years"])
    ax.set_ylabel(t["repo_err"])
    ax.set_title(t["spot_title"], loc="left")
    ax.legend(loc="upper right")
    _fmt_axis(ax, lang, "x")
    fig.tight_layout(w_pad=2.5)
    return _save(fig, path)


def fig_cancellation(path: Path, lang: str) -> Path:
    t = TXT[lang]
    from repo_control.control import control_lines, implied_repo_jacobian
    from repo_control.pricer import book_value, sensitivities

    m, books = ra._demo_market()
    T = PILLARS
    F_tot = m.forward(T) * (1 - 0.002)
    repo_fwd = bootstrap_repo(m.spot, m.rates, m.dividends, T, F_tot)
    J = implied_repo_jacobian(m.spot, m.rates, m.schedule, m.div_levels, m.profile, T, F_tot)
    bumps = np.linspace(-0.6, 0.6, 25)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, key, title in zip(axes, ["A: options and forwards", "B: A + short 2028 dividend swap"], [t["book_a"], t["book_b"]]):
        book = books[key]
        sr, sd = sensitivities(book, m)
        rows = []
        for bump in bumps:
            lv = m.div_levels.copy()
            lv[1:3] += bump
            mt = m.with_levels(lv)
            rt = bootstrap_repo(m.spot, m.rates, mt.dividends, T, F_tot)
            cl = control_lines(sr, sd, rt.zeros - m.repo.zeros, repo_fwd.zeros - m.repo.zeros, lv - m.div_levels, J)
            rows.append((cl.div_line, cl.repo_line, cl.total, book_value(book, mt.with_repo(rt)) - book_value(book, m)))
        rows = np.array(rows)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.plot(bumps, rows[:, 0], color=SERIES[1], label=t["div_line"])
        ax.plot(bumps, rows[:, 1], color=SERIES[2], label=t["repo_line"])
        ax.plot(bumps, rows[:, 2], color=SERIES[0], linewidth=2.2, label=t["total"])
        ax.plot(bumps[::4], rows[::4, 3], linestyle="none", **_marker_kw(SERIES[0]), label=t["total_full"])
        ax.set_title(title, loc="left")
        _fmt_axis(ax, lang)
    axes[0].set_ylabel(t["cancel_y"])
    fig.supxlabel(t["cancel_x"], fontsize=8.5, color=INK2, y=0.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.04, 1, 1))
    return _save(fig, path)


def fig_detection(res: BenchmarkResult, path: Path, lang: str) -> Path:
    t = TXT[lang]
    scen = [s for s in SCENARIOS if s != "clean"]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    y = np.arange(len(scen))[::-1]
    offs = np.linspace(-0.28, 0.28, len(DETECTORS))
    for i, d in enumerate(DETECTORS):
        ax.plot([res.detection_rate(s, d) for s in scen], y + offs[i], linestyle="none",
                markersize=6.5 if d == "structural" else 5.5, label=t["detectors"][d], **_marker_kw(SERIES[i]))
    for yy in y:
        ax.axhline(yy - 0.5, color=GRID, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([t["scenarios"][s] for s in scen])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(t["det_x"].format(fpr=_pct_fig(res.fpr_target, lang)))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.16), ncol=3, fontsize=7.5)
    _fmt_axis(ax, lang, "x")
    fig.tight_layout()
    return _save(fig, path)


def theoretical_power(lam, c):
    lam = np.asarray(lam, dtype=float)
    return norm.cdf(lam - c) + norm.cdf(-lam - c)


def fig_power(res: BenchmarkResult, path: Path, lang: str, z_crit: float) -> Path:
    t = TXT[lang]
    edges = (0, 1, 2, 3, 4, 6, 8, 12, np.inf)
    labels = ["0–1", "1–2", "2–3", "3–4", "4–6", "6–8", "8–12", "12+"]
    mids = np.array([0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 14.0])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(labels))
    ax.plot(x, theoretical_power(mids, z_crit), color="#898781", linewidth=1.2, label=t["theory"])
    for i, d in enumerate(DETECTORS):
        pc = res.power_curve(d, edges)
        ax.plot(x, [p[2] for p in pc], label=t["detectors"][d], linewidth=2.2 if d == "structural" else 1.4,
                **_marker_kw(SERIES[i]))
    counts = [p[3] for p in res.power_curve("structural", edges)]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={c})" for lab, c in zip(labels, counts)])
    ax.set_xlabel(t["snr_x"])
    ax.set_ylabel(t["det_rate"])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=7.2)
    _fmt_axis(ax, lang, "y")
    fig.tight_layout()
    return _save(fig, path)


def fig_attribution(res: BenchmarkResult, path: Path, lang: str) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    t = TXT[lang]
    scen = list(EXPECTED_KIND) + ["fo_repo_mismark", "div_gross_net"]
    kinds = ["level", "step", "spike", "season", "roughness", "none"]
    M = np.zeros((len(scen), len(kinds)))
    for i, s in enumerate(scen):
        recs = [r for r in res.records if r["scenario"] == s]
        for r in recs:
            M[i, kinds.index(r["structural"]["top_kind"])] += 1
        M[i] /= max(len(recs), 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + BLUE_RAMP), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.01:
                ax.text(j, i, _pct_fig(M[i, j], lang), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(t["kinds"])
    ax.set_yticks(range(len(scen)))
    ax.set_yticklabels([t["scenarios"][s] for s in scen])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(t["attr_title"], loc="left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


GALLERY = ["spot_sync", "div_bucket", "div_timing", "season_profile", "fwd_outlier", "fo_repo_mismark"]
GALLERY_TITLES = {
    "en": ["Spot snapshot mismatch", "Dividend bucket error", "Ex-date timing split", "Stale ex-date profile",
           "Forward outlier", "FO repo mismark (genuine)"],
    "fr": ["Décalage du spot", "Erreur de période de dividendes", "Désaccord de date de détachement",
           "Profil de détachement périmé", "Forward aberrant", "Erreur de marquage FO (réelle)"],
}


def gallery_cases(res: BenchmarkResult, category: str = "EU stock"):
    picks = ra.pick_examples(res, category)
    det = ra._calibrated_detector(res, category)
    out = []
    for scen in GALLERY:
        case = simulate_case(picks[scen], category, scen)
        im = case.implied
        out.append((scen, case, det.detect(im.pillars, im.delta_carry, im.cov, im.context())))
    return det, out


def fig_gallery(res: BenchmarkResult, path: Path, lang: str, cases) -> Path:
    t = TXT[lang]
    det, items = cases
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.4))
    x = np.arange(PILLARS.size)
    for n, (ax, (scen, case, d)) in enumerate(zip(axes.ravel(), items)):
        im = case.implied
        b_fo = im.repo_fo * 1e4
        b_tot = im.repo_totem.zeros * 1e4
        repaired = (im.delta_carry - d.anomaly_fit) / im.pillars * 1e4 + b_fo
        band = 2 * np.sqrt(np.diag(im.cov) * det.variance_scale) / im.pillars * 1e4
        ax.fill_between(x, b_fo - band, b_fo + band, color=SERIES[0], alpha=0.10, linewidth=0, label=t["band"])
        if d.explanations:
            ax.plot(x, repaired, label=t["repaired"], linewidth=2.6, color=SERIES[2], alpha=0.9)
        ax.plot(x, b_fo, label=t["fo_repo"], **_marker_kw(SERIES[0]))
        ax.plot(x, b_tot, label=t["tot_repo"], **_marker_kw(SERIES[1]))
        lo, hi = min(b_fo.min(), b_tot.min()), max(b_fo.max(), b_tot.max())
        pad = 0.12 * (hi - lo + 1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(tenors(lang), fontsize=7)
        ax.set_title(f"({'abcdef'[n]}) {GALLERY_TITLES[lang][n]}", loc="left", fontsize=8.5)
        ax.set_ylabel(t["zero_repo"])
        _fmt_axis(ax, lang, "y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(k) for k in (t["fo_repo"], t["tot_repo"], t["repaired"], t["band"]) if k in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
    fig.tight_layout(h_pad=1.6, w_pad=1.8)
    return _save(fig, path)


def fig_decomposition(res: BenchmarkResult, path: Path, lang: str, cases) -> Path:
    t = TXT[lang]
    det, items = cases
    scen, case, d = [it for it in items if it[0] == "div_bucket"][0]
    im = case.implied
    x = np.arange(PILLARS.size)
    sd = np.sqrt(np.diag(im.cov) * det.variance_scale)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.fill_between(x, (d.smooth_fit - 2 * sd) * 1e4, (d.smooth_fit + 2 * sd) * 1e4, color=SERIES[0], alpha=0.10,
                    linewidth=0, label=t["null_band"])
    ax.plot(x, d.smooth_fit * 1e4, color=SERIES[0], label=t["null_fit"])
    ax.plot(x, (d.smooth_fit + d.anomaly_fit) * 1e4, color=SERIES[1], label=t["null_tmpl"])
    ax.plot(x, im.delta_carry * 1e4, linestyle="none", label=t["obs"], **_marker_kw(INK))
    vals = np.concatenate([im.delta_carry, d.smooth_fit + d.anomaly_fit, d.smooth_fit]) * 1e4
    span = vals.max() - vals.min()
    ax.set_ylim(vals.min() - 0.35 * span, vals.max() + 0.35 * span)
    ax.set_xticks(x)
    ax.set_xticklabels(tenors(lang))
    ax.set_ylabel(t["carry_y"])
    ax.set_title(t["decomp_title"], loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    _fmt_axis(ax, lang, "y")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------- explanations
def explanation_text(expl, T, ctx, det: StructuralDetector, lang: str) -> str:
    """Natural-language explanation (EN uses the detector's own label; FR mirrors its formulas)."""
    if lang == "en":
        return expl.label
    S, D, divs, amp = ctx.spot, ctx.carry_div, ctx.dividends, expl.amplitude

    def n(x, d=2):
        return f"{x:+.{d}f}".replace(".", ",").replace("-", "−")

    tl = [tenor_label(x).replace("Y", "A") for x in T]
    if expl.kind == "level":
        return (f"Décalage de niveau en portage (1/T en repo) : spot de référence décalé de {n(-amp * 100)} % "
                f"par rapport au spot FO, ou dividende avant {tl[0]} décalé de {n(-amp * (S - D[0]), 3)}")
    if expl.kind == "step":
        k = expl.pillar
        parts = []
        mask = divs.window(T[k - 1], T[k])
        if mask.any():
            d_int = divs.amounts[mask].sum()
            d_err = -amp * (S - D[k])
            parts.append(f"dividendes détachés dans ({tl[k - 1]}, {tl[k]}] décalés de {n(d_err, 3)} "
                         f"({n(100 * d_err / d_int, 0)} %) dans le dividende Totem par rapport aux forwards")
        n0, n1 = ctx.n_opt[k - 1], ctx.n_opt[k]
        m0, m1 = ctx.n_div_at_pillar[k - 1], ctx.n_div_at_pillar[k]
        if n1 <= det.panel_drop_ratio * n0 or m1 <= det.panel_drop_ratio * m0:
            parts.append(f"changement de panel à {tl[k]} (options {n0:.0f}→{n1:.0f}, dividendes {m0:.0f}→{m1:.0f})")
        if k == T.size - 1:
            parts.append(f"ou forward aberrant isolé : ln F à {tl[k]} décalé de {n(-amp * 1e4, 0)} pb")
        if not parts:
            parts.append(f"changement de niveau à partir de {tl[k]} sans dividende ni changement de panel")
        return f"Marche à partir de {tl[k]} : " + " ; ".join(parts)
    if expl.kind == "spike":
        k = expl.pillar
        near = np.abs(divs.ex_times - T[k]) <= det.timing_window
        if near.any():
            i = int(np.flatnonzero(near)[np.argmax(divs.amounts[near])])
            share = abs(-amp * (S - D[k]) / divs.amounts[i])
            days = (divs.ex_times[i] - T[k]) * 365
            out = (f"Date de détachement à {tl[k]} : environ {100 * share:.0f} % du dividende de "
                   f"{divs.amounts[i]:.2f}".replace(".", ",")
                   + f" détaché à {n(days, 0)} j de l'échéance est placé de l'autre côté de l'échéance")
        else:
            out = f"Forward aberrant à {tl[k]} : ln F Totem décalé de {n(-amp * 1e4, 0)} pb"
        if ctx.dispersion_ratio is not None and ctx.dispersion_ratio[k] > 2:
            out += (f" (dispersion des forwards à {tl[k]} égale à {ctx.dispersion_ratio[k]:.1f}".replace(".", ",")
                    + " fois celle des voisins : les contributeurs divergent)")
        return out
    return (f"Décalage du profil de détachement d'environ {n(amp * 365, 0)} jours entre l'allocation des dividendes "
            "et les forwards")


def fo_mismark_text(d, lang: str) -> str:
    if lang == "en":
        return (f"No Totem anomaly. FO repo level differs from Totem by {-d.repo_level * 1e4:+.0f}bp "
                f"(± {d.repo_level_se * 1e4:.0f}bp): FO mismark candidate")
    return (f"Pas d'anomalie Totem. Le niveau du repo FO diffère de Totem de {-d.repo_level * 1e4:+.0f} pb "
            f"(± {d.repo_level_se * 1e4:.0f} pb) : erreur de marquage FO possible").replace("-", "−")


# ------------------------------------------------------- worked examples
def example_bootstrap():
    """Two dividends, three pillars: forwards from a known repo curve, then the bootstrap with its Newton steps."""
    S = 100.0
    rates = ZeroCurve.flat(0.03)
    T = np.array([0.25, 0.5, 1.0])
    true = ZeroCurve(T, np.array([0.0080, 0.0070, 0.0060]))
    divs = Dividends([0.30, 0.80], [1.5, 2.0])
    F = forward(S, rates, true, divs, T)
    implied = bootstrap_repo(S, rates, divs, T, F)
    # Newton steps on pillar 3 (T = 1), reproducing the solver's iteration explicitly
    R = rates.cum(T)
    B = true.cumulative
    t_prev, b_prev = T[1], B[1]
    early = divs.ex_times <= t_prev
    b_early = np.interp(divs.ex_times[early], [0, T[0], T[1]], [0, B[0], B[1]])
    early_terms = divs.amounts[early] * np.exp(R[2] - rates.cum(divs.ex_times[early]) + b_early)
    mid = divs.window(t_prev, T[2])
    w = (divs.ex_times[mid] - t_prev) / (T[2] - t_prev)
    mid_amt = divs.amounts[mid]
    mid_base = R[2] - rates.cum(divs.ex_times[mid])

    def model(x):
        e_spot = S * np.exp(R[2] - x)
        e_early = early_terms * np.exp(-x)
        e_mid = mid_amt * np.exp(mid_base - (1 - w) * (x - b_prev))
        v = e_spot - e_early.sum() - e_mid.sum()
        dv = -e_spot + e_early.sum() + ((1 - w) * e_mid).sum()
        return v, dv

    x = R[2] - np.log(F[2] / (S - early_terms.sum() * np.exp(-R[2]) - (mid_amt * np.exp(-rates.cum(divs.ex_times[mid]) + b_prev)).sum()))
    steps = []
    for _ in range(4):
        v, dv = model(x)
        g = np.log(v) - np.log(F[2])
        steps.append((x, g))
        x = x - g / (dv / v)
    return {"T": T, "F": F, "true": true.zeros, "implied": implied.zeros, "steps": steps, "divs": divs}


def example_parity():
    K = np.array([90.0, 100.0, 110.0])
    F, DF = 102.3, 0.97
    puts = np.array([2.1, 5.4, 10.9])
    calls = puts + DF * (F - K)
    fit = forward_from_parity(K, calls, puts)
    fit2 = forward_from_parity(K, 1.02 * calls, 1.02 * puts)
    return {"K": K, "calls": calls, "puts": puts, "F": fit.forward, "DF": fit.discount, "F2": fit2.forward, "DF2": fit2.discount}


def example_toy_glrt(alpha: float = 0.02, n_sim: int = 200_000, seed: int = 11):
    """Four pillars, independent noise, nuisance = repo level only; a 12bp spike at 6M."""
    T = np.array([0.25, 0.5, 1.0, 2.0])
    sd = np.array([1.0, 2.0, 4.0, 8.0]) * 1e-4
    beta0 = 30e-4
    spike = np.array([0.0, 12e-4, 0.0, 0.0])
    noise = np.array([0.4, -1.1, 2.6, -3.0]) * 1e-4
    y = beta0 * T + spike + noise
    W = np.diag(1 / sd)
    X0 = T[:, None]
    X0w = W @ X0
    Q, _ = np.linalg.qr(X0w)
    P = np.eye(4) - Q @ Q.T
    r = P @ (W @ y)
    names, vecs = [], []
    names.append("level")
    vecs.append(np.ones(4))
    for k in range(1, 4):
        names.append(f"step@{k}")
        vecs.append((np.arange(4) >= k).astype(float))
    for k in range(3):
        names.append(f"spike@{k}")
        vecs.append((np.arange(4) == k).astype(float))
    tw = np.column_stack([P @ (W @ v) for v in vecs])
    norms = np.sqrt((tw**2).sum(axis=0))
    z = (tw.T @ r) / norms
    amp = (tw.T @ r) / norms**2
    se = 1 / norms
    C = (tw / norms).T @ (tw / norms)
    ev, evec = np.linalg.eigh(C)
    root = evec * np.sqrt(np.clip(ev, 0, None))
    sims = np.abs(np.random.default_rng(seed).standard_normal((n_sim, len(vecs))) @ root.T).max(axis=1)
    c = float(np.quantile(sims, 1 - alpha / 2))
    m = len(vecs)
    c_sidak = float(norm.isf((1 - (1 - alpha / 2) ** (1 / m)) / 2))
    q = float(r @ r)
    beta_hat = float(np.linalg.lstsq(X0w, W @ y, rcond=None)[0][0])
    return {"T": T, "sd": sd, "y": y, "names": names, "z": z, "amp": amp, "se": se, "c": c, "c_sidak": c_sidak,
            "q": q, "beta_hat": beta_hat, "m": m}


def example_leverage(n_basis: int = 6):
    x = np.log(PILLARS)
    degree = 3
    n_inner = n_basis - degree - 1
    inner = np.linspace(x[0], x[-1], n_inner + 2)[1:-1]
    knots = np.concatenate([[x[0]] * (degree + 1), inner, [x[-1]] * (degree + 1)])
    B = BSpline.design_matrix(x, knots, degree).toarray()
    H = B @ np.linalg.pinv(B)
    return np.diag(H)


def typical_z_crit(res: BenchmarkResult, category: str = "EU stock", n: int = 30) -> float:
    det = ra._calibrated_detector(res, category)
    vals = []
    for i in range(n):
        im = simulate_case(10_000 + i, category, "clean").implied
        vals.append(det.null_statistics(im.pillars, im.delta_carry, im.cov, im.context())[1])
    return float(np.median(vals))


def template_count() -> int:
    im = simulate_case(10_000, "EU stock", "clean").implied
    return len(build_templates(im.pillars, im.context()))


def nuisance_columns() -> int:
    return nuisance_design(PILLARS).shape[1]
