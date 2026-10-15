"""Worked examples and generated text for the methodology notes (French and English).

Everything is computed from the library and the saved benchmark; nothing is typed in by hand. Figures live in
:mod:`note_figures`.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import note_figures as nf  # noqa: E402
import note_examples as ne  # noqa: E402

from repo_control.benchmark import BenchmarkResult  # noqa: E402
from repo_control.curves import Dividends, ZeroCurve, bootstrap_repo, forward  # noqa: E402
from repo_control.detect import build_templates, tenor_label  # noqa: E402
from repo_control.parity import forward_from_parity  # noqa: E402
from repo_control.synthetic import PILLARS, VALUATION, simulate_case  # noqa: E402

MONTHS = {
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
           "novembre", "décembre"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"],
}

NAMES = {
    "fr": {
        "scenarios": {"clean": "Sain", **nf.TXT["fr"]["scenarios"]},
        "categories": {"EU stock": "Action UE", "US stock": "Action US", "HTB stock": "Difficile à emprunter",
                       "EU index": "Indice UE"},
    },
    "en": {
        "scenarios": {"clean": "Clean", **nf.TXT["en"]["scenarios"]},
        "categories": {"EU stock": "EU stock", "US stock": "US stock", "HTB stock": "Hard to borrow",
                       "EU index": "EU index"},
    },
}


# ----------------------------------------------------------------- formatting
def num(x: float, d: int = 2, sign: bool = False) -> str:
    """A siunitx number (the decimal separator follows the document locale)."""
    if sign:
        return f"\\num[retain-explicit-plus]{{{x:+.{d}f}}}"
    return f"\\num{{{x:.{d}f}}}"


def pct(x: float, d: int = 0, sign: bool = False) -> str:
    """A percentage from a fraction."""
    if sign:
        return f"\\pct[retain-explicit-plus]{{{100 * x:+.{d}f}}}"
    return f"\\pct{{{100 * x:.{d}f}}}"


def tenor(T: float, lang: str) -> str:
    s = tenor_label(T)
    return s.replace("Y", "A") if lang == "fr" else s


def date_text(t: float, lang: str) -> str:
    d = VALUATION + timedelta(days=int(round(t * 365)))
    if lang == "fr":
        day = "1\\textsuperscript{er}" if d.day == 1 else str(d.day)
        return f"{day}~{MONTHS['fr'][d.month - 1]} {d.year}"
    return f"{d.day}~{MONTHS['en'][d.month - 1]} {d.year}"


# ------------------------------------------------------------ running example
def running_example(res: BenchmarkResult) -> dict:
    """Numbers of the running example (one synthetic European stock under every scenario, same seed)."""
    det, cases = nf.running(res)
    case, d = cases["clean"]
    im = case.implied
    T = im.pillars
    sd = np.sqrt(np.diag(im.cov) * det.variance_scale)
    divs = case.true_dividends
    k9 = int(np.argmin(np.abs(T - 0.75)))
    k6 = int(np.argmin(np.abs(T - 0.5)))
    return {
        "det": det,
        "cases": cases,
        "pillars": T,
        "div_mean": float(np.mean(divs.amounts)),
        "first_ex": float(divs.ex_times[0]),
        "n_opt": im.n_opt,
        "n_div": case.totem.n_div[1:],
        "q_fo": im.repo_fo,
        "q_tot": im.repo_totem.zeros,
        "delta": im.delta_carry,
        "sd": sd,
        "sd_repo": sd / T,
        "k6": k6,
        "k9": k9,
        "div_gap_9m": float(im.delta_carry[k9] * (im.spot - im.carry_div[k9])),
        "clean": d,
    }


def running_rows(run: dict, lang: str) -> list[tuple[str, str, str]]:
    """(situation, what is injected, what the detector says) for every scenario of the running example."""
    det, cases = run["det"], run["cases"]
    fr = lang == "fr"
    sc = NAMES[lang]["scenarios"]
    rows = []
    for scen, (case, d) in cases.items():
        im, tr = case.implied, case.truth
        T = im.pillars
        ctx = im.context()
        S, D = ctx.spot, ctx.carry_div
        ex = d.explanations[0] if d.explanations else None

        if scen == "clean":
            inj = "aucune anomalie" if fr else "no anomaly"
        elif scen == "spot_sync":
            inj = (f"contributeurs options sur un spot décalé de {pct(tr['eps'], 3, True)}" if fr
                   else f"option contributors price off a spot shifted by {pct(tr['eps'], 3, True)}")
        elif scen == "div_bucket":
            p = tr["bucket"]
            y = 2026 + p
            lvl, base = case.totem.div_level[p], case.totem.div_level[p] - tr["delta"]
            inj = (f"niveau Totem {y} : {num(lvl)} au lieu de {num(base)}" if fr
                   else f"Totem {y} level: {num(lvl)} instead of {num(base)}")
        elif scen == "div_timing":
            k, i = tr["pillar"], tr["div"]
            days = (case.true_dividends.ex_times[i] - T[k]) * 365
            inj = (f"{pct(tr['share'])} du panel options placent avant l'échéance {tenor(T[k], lang)} le dividende "
                   f"détaché {days:.0f}~jours après" if fr
                   else f"{pct(tr['share'])} of the option panel put the dividend going ex {days:.0f}~days after the "
                        f"{tenor(T[k], lang)} expiry before it")
        elif scen == "season_profile":
            inj = (f"profil d'allocation décalé de {num(tr['shift_days'], 0, True)}~jours" if fr
                   else f"allocation profile shifted by {num(tr['shift_days'], 0, True)}~days")
        elif scen == "fwd_outlier":
            k = tr["pillar"]
            shift = num(tr["ln_fwd_shift"] * 1e4, 0, True)
            inj = (f"$\\ln F$ de consensus en {tenor(T[k], lang)} décalé de {shift}~pb "
                   f"({num(abs(tr['z']), 1)} écarts-types)" if fr
                   else f"consensus $\\ln F$ at {tenor(T[k], lang)} shifted by {shift}~bp "
                        f"({num(abs(tr['z']), 1)} standard deviations)")
        elif scen == "panel_shift":
            k = tr["pillar"]
            n_left = int(case.totem.n_opt[k])
            inj = (f"à partir de {tenor(T[k], lang)}, les {n_left} contributeurs options restants surestiment les "
                   f"dividendes suivants de {pct(tr['delta'])}" if fr
                   else f"from {tenor(T[k], lang)} on, the {n_left} remaining option contributors overstate later "
                        f"dividends by {pct(tr['delta'])}")
        elif scen == "fo_repo_mismark":
            inj = (f"repo FO décalé de {num(tr['shift'] * 1e4, 0, True)}~pb, Totem intact" if fr
                   else f"FO repo shifted by {num(tr['shift'] * 1e4, 0, True)}~bp, Totem untouched")
        else:  # div_gross_net
            inj = (f"toutes les périodes Totem multipliées par {num(1 + tr['bias'])}" if fr
                   else f"every Totem period multiplied by {num(1 + tr['bias'])}")

        if not d.flagged:
            if d.fo_mismark:
                out = (f"pas d'anomalie Totem ; niveau $\\hat\\beta_0={num(d.repo_level * 1e4, 1)}$~pb "
                       f"(écart-type {num(d.repo_level_se * 1e4, 1)}), au-delà du seuil : "
                       "erreur de marquage FO possible"
                       if fr else
                       f"no Totem anomaly; level $\\hat\\beta_0={num(d.repo_level * 1e4, 1)}$~bp "
                       f"(standard error {num(d.repo_level_se * 1e4, 1)}), beyond the threshold: possible FO mismark")
            else:
                out = (f"pas d'alerte ($\\max_g|z_g|={num(d.max_z)}<c_\\alpha={num(d.z_crit)}$)" if fr
                       else f"no alarm ($\\max_g|z_g|={num(d.max_z)}<c_\\alpha={num(d.z_crit)}$)")
        else:
            z = f"$z={num(ex.z, 1)}$"
            k = ex.pillar
            if ex.kind == "level":
                out = (f"niveau, {z} : spot décalé de {pct(-ex.amplitude, 2, True)}" if fr
                       else f"level, {z}: spot off by {pct(-ex.amplitude, 2, True)}")
            elif ex.kind == "step":
                err = -ex.amplitude * (S - D[k])
                lo, hi = tenor(T[k - 1], lang), tenor(T[k], lang)
                mask = ctx.dividends.window(T[k - 1], T[k])
                rel = err / ctx.dividends.amounts[mask].sum()
                out = (f"marche en {hi}, {z} : dividende détaché dans $\\intoc{{\\text{{{lo}}}}}{{\\text{{{hi}}}}}$ "
                       f"plus bas de {num(-err)} ({pct(rel, 0, True)}) côté dividendes que dans les forwards" if fr
                       else f"step at {hi}, {z}: dividend going ex in $\\intoc{{\\text{{{lo}}}}}{{\\text{{{hi}}}}}$ "
                            f"lower by {num(-err)} ({pct(rel, 0, True)}) in the dividend service than in the forwards")
            elif ex.kind == "spike":
                near = np.abs(ctx.dividends.ex_times - T[k]) <= det.timing_window
                if near.any():
                    i = int(np.flatnonzero(near)[np.argmax(ctx.dividends.amounts[near])])
                    share = abs(ex.amplitude * (S - D[k]) / ctx.dividends.amounts[i])
                    ratio = ctx.dispersion_ratio[k]
                    out = (f"pic en {tenor(T[k], lang)}, {z} : environ {pct(share)} du dividende placé de l'autre côté "
                           f"de l'échéance ; dispersion des forwards {num(ratio, 0)} fois celle des voisines" if fr
                           else f"spike at {tenor(T[k], lang)}, {z}: about {pct(share)} of the dividend on the other "
                                f"side of the expiry; forward dispersion {num(ratio, 0)} times its neighbours'")
                else:
                    shift = num(-ex.amplitude * 1e4, 0, True)
                    out = (f"pic en {tenor(T[k], lang)}, {z} : $\\ln F$ décalé de {shift}~pb"
                           if fr else
                           f"spike at {tenor(T[k], lang)}, {z}: $\\ln F$ off by {shift}~bp")
            else:
                out = (f"saison, {z} : profil décalé d'environ {num(ex.amplitude * 365, 0, True)}~jours" if fr
                       else f"season, {z}: profile shifted by about {num(ex.amplitude * 365, 0, True)}~days")
            if len(d.explanations) > 1:
                e2 = d.explanations[1]
                kind2 = {"fr": {"spike": "pic", "step": "marche", "level": "niveau", "season": "saison"},
                         "en": {"spike": "spike", "step": "step", "level": "level", "season": "season"}}[lang][e2.kind]
                where = f" en {tenor(T[e2.pillar], lang)}" if fr and e2.pillar >= 0 else (
                    f" at {tenor(T[e2.pillar], lang)}" if e2.pillar >= 0 else "")
                out += (f" ; puis {kind2}{where} ($z={num(e2.z, 1)}$)" if fr
                        else f"; then {kind2}{where} ($z={num(e2.z, 1)}$)")
        rows.append((sc[scen], inj, out))
    return rows


# ------------------------------------------------------- worked examples
def example_panel_gap(n_sim: int = 4000, seed: int = 7):
    """Implied-repo noise from the panel gap alone: closed form versus a contributor-level simulation.

    Dividend views follow the running example's model (:func:`note_figures.gap_dividends`); contributors also disagree
    on the repo (15bp), which the implied repo must not pick up. Panels are disjoint, with the running example's sizes.
    """
    rng = np.random.default_rng(seed)
    S, r, q = 100.0, 0.03, 0.001
    t_ex = nf.GAP_EX
    d_pv = nf.GAP_DIV * np.exp(-nf.GAP_RATE * t_ex)
    rel_sd = nf.GAP_DISP[0] + nf.GAP_DISP[1] * t_ex
    rho = nf.GAP_RHO
    rows = []
    for T in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        D, sd = (float(x[0]) for x in nf.gap_dividends(T))
        nA, nB = (int(x[0]) for x in nf.running_panels(T))
        theory = float(sd * np.sqrt(1 / nA + 1 / nB) / (T * (S - D)))
        m = t_ex <= T

        def views(n):
            z = rng.standard_normal((n_sim, n, 1))
            e = rng.standard_normal((n_sim, n, m.sum()))
            return (d_pv[m] * (1 + rel_sd[m] * (rho * z + np.sqrt(1 - rho**2) * e))).sum(axis=2)

        DA, DB = views(nA), views(nB)
        qA = q + 15e-4 * rng.standard_normal((n_sim, nA))
        F = ((S - DA) * np.exp((r - qA) * T)).mean(axis=1)
        implied = r - np.log(F / (S - DB.mean(axis=1))) / T
        sims = implied - qA.mean(axis=1)
        rows.append((T, D, sd, nA, nB, theory * 1e4, float(np.std(sims)) * 1e4))
    return rows


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

    x = R[2] - np.log(F[2] / (S - early_terms.sum() * np.exp(-R[2])
                              - (mid_amt * np.exp(-rates.cum(divs.ex_times[mid]) + b_prev)).sum()))
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
    return {"K": K, "calls": calls, "puts": puts, "F": fit.forward, "DF": fit.discount, "F2": fit2.forward,
            "DF2": fit2.discount}


def example_toy_glrt(alpha: float = 0.02, n_sim: int = 200_000, seed: int = 11):
    """Four pillars, independent noise, null model = repo level only; a 12bp spike at 6M."""
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
    names, vecs = ["level"], [np.ones(4)]
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
    # after selecting the spike at T_2
    best = int(np.argmax(np.abs(z)))
    Xs = np.column_stack([X0w, W @ vecs[best]])
    Qs, _ = np.linalg.qr(Xs)
    Ps = np.eye(4) - Qs @ Qs.T
    rs = Ps @ (W @ y)
    ts = Ps @ np.column_stack([W @ v for v in vecs])
    ns = np.sqrt((ts**2).sum(axis=0))
    ok = ns > 1e-9
    z_after = np.zeros(m)
    z_after[ok] = np.abs(ts[:, ok].T @ rs) / ns[ok]
    return {"T": T, "sd": sd, "y": y, "names": names, "z": z, "amp": amp, "se": se, "c": c, "c_sidak": c_sidak,
            "q": q, "beta_hat": beta_hat, "m": m, "best": best, "max_z_after": float(z_after.max()),
            "q_after": float(rs @ rs)}


def example_leverage(n_basis: int = 6):
    x = np.log(PILLARS)
    degree = 3
    n_inner = n_basis - degree - 1
    inner = np.linspace(x[0], x[-1], n_inner + 2)[1:-1]
    knots = np.concatenate([[x[0]] * (degree + 1), inner, [x[-1]] * (degree + 1)])
    B = BSpline.design_matrix(x, knots, degree).toarray()
    H = B @ np.linalg.pinv(B)
    return np.diag(H)


def theoretical_power(lam, c):
    lam = np.asarray(lam, dtype=float)
    return norm.cdf(lam - c) + norm.cdf(-lam - c)


def typical_z_crit(res: BenchmarkResult, category: str = "EU stock", n: int = 30) -> float:
    det = ne.calibrated_detector(res, category)
    vals = []
    for i in range(n):
        im = simulate_case(10_000 + i, category, "clean").implied
        vals.append(det.null_statistics(im.pillars, im.delta_carry, im.cov, im.context())[1])
    return float(np.median(vals))


def template_count() -> int:
    im = simulate_case(10_000, "EU stock", "clean").implied
    return len(build_templates(im.pillars, im.context()))
