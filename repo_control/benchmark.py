"""Matched-false-positive benchmark of bad-curve detectors on the synthetic Totem market.

Protocol
--------
1. Training set: clean names per category.  The structural detector learns a
   single variance-inflation factor per category; the baselines learn their
   alarm threshold (and the GMM its mixture) on the same clean history.
2. Test set: fresh names, each simulated under every scenario with the same
   seed (paired design).  All detectors are run at the same target false
   positive rate; we report detection rates, AUC against the clean test set,
   and for the structural detector whether the first explanation names the
   injected mechanism.
"""
from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .baselines import SplineGMM, naive_level_score, naive_roughness_score
from .detect import StructuralDetector
from .synthetic import CATEGORIES, EXPECTED_KIND, PILLARS, SCENARIOS, Case, simulate_case

DETECTORS = ("naive_level", "naive_shape", "spline_gmm_curve", "spline_gmm_diff", "structural")
DETECTOR_TITLES = {
    "naive_level": "Naive gap |Δb|",
    "naive_shape": "Naive shape (repo space)",
    "spline_gmm_curve": "B-spline + GMM (curve)",
    "spline_gmm_diff": "B-spline + GMM (Totem − FO)",
    "structural": "Structural GLRT (proposed)",
}


@dataclass
class BenchmarkResult:
    fpr_target: float
    n_train: int
    n_test: int
    variance_scale: dict
    thresholds: dict
    level_scale: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    runtime_s: float = 0.0

    # ----------------------------------------------------------- summaries
    def _select(self, scenario: str, category: str | None = None):
        return [r for r in self.records if r["scenario"] == scenario and (category is None or r["category"] == category)]

    def detection_rate(self, scenario: str, detector: str, category: str | None = None) -> float:
        rows = self._select(scenario, category)
        return float(np.mean([r["flags"][detector] for r in rows])) if rows else float("nan")

    def auc(self, scenario: str, detector: str, category: str | None = None) -> float:
        pos = np.array([r["scores"][detector] for r in self._select(scenario, category)])
        neg = np.array([r["scores"][detector] for r in self._select("clean", category)])
        if not pos.size or not neg.size:
            return float("nan")
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort(kind="mergesort").argsort(kind="mergesort") + 1.0
        # average ranks for ties
        for v in np.unique(allv):
            m = allv == v
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
        return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))

    def attribution_rate(self, scenario: str, conditional: bool = True, category: str | None = None) -> float:
        rows = self._select(scenario, category)
        if conditional:
            rows = [r for r in rows if r["flags"]["structural"]]
        return float(np.mean([r["attribution_ok"] for r in rows])) if rows else float("nan")

    def fo_mismark_rate(self, scenario: str, category: str | None = None) -> float:
        rows = self._select(scenario, category)
        return float(np.mean([r["structural"]["fo_mismark"] for r in rows])) if rows else float("nan")

    def power_curve(self, detector: str, edges=(0, 1, 2, 3, 4, 6, 8, 12, np.inf), scenarios=None):
        """Detection rate by bins of signal-to-noise (whitened effect size after the null fit)."""
        scen = set(scenarios or EXPECTED_KIND)
        rows = [r for r in self.records if r["scenario"] in scen]
        snr = np.array([r["snr"] for r in rows])
        flags = np.array([r["flags"][detector] for r in rows], dtype=float)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (snr >= lo) & (snr < hi)
            out.append((lo, hi, float(flags[m].mean()) if m.any() else float("nan"), int(m.sum())))
        return out

    def to_json(self, path: str) -> None:
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating, float)):
                return float(o)
            return o

        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "wt") as fh:
            json.dump(clean(asdict(self)), fh)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkResult":
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            return cls(**json.load(fh))

    def kind_counts(self, scenario: str) -> dict:
        out: dict = {}
        for r in self._select(scenario):
            k = r["structural"]["top_kind"]
            out[k] = out.get(k, 0) + 1
        return out


def _attribution_ok(case: Case, det) -> bool:
    """Does the first explanation name the injected mechanism (and its location)?"""
    truth, scen = case.truth, case.scenario
    if scen == "fo_repo_mismark":
        return (not det.flagged) and det.fo_mismark
    if scen not in EXPECTED_KIND:
        return not det.flagged
    if not det.flagged or not det.explanations:
        return False
    top = det.explanations[0]
    last = case.implied.pillars.size - 1
    if scen in ("div_timing", "fwd_outlier"):
        if truth["pillar"] == last:
            # at the last pillar a spike and a step are the same template
            return top.kind in ("spike", "step") and top.pillar == last
        if top.kind == "spike":
            return top.pillar == truth["pillar"]
        return scen == "div_timing" and top.kind == "season"
    if top.kind not in EXPECTED_KIND[scen]:
        return False
    if scen == "div_bucket":
        return top.pillar in truth["pillars"]
    if scen == "panel_shift":
        return abs(top.pillar - truth["pillar"]) <= 1
    return True


def _features(case: Case):
    im = case.implied
    return {
        "naive_level": naive_level_score(im.delta_repo * 1e4),
        "naive_shape": naive_roughness_score(im.pillars, im.repo_totem.zeros * 1e4),
        "curve_bp": im.repo_totem.zeros * 1e4,
        "diff_bp": im.delta_repo * 1e4,
    }


def run_benchmark(
    n_train: int = 150,
    n_test: int = 120,
    fpr: float = 0.02,
    categories=None,
    scenarios=SCENARIOS,
    seed0: int = 10_000,
    progress: bool = True,
) -> BenchmarkResult:
    t_start = time.time()
    categories = list(categories or CATEGORIES)
    detectors: dict[str, StructuralDetector] = {}
    spline = SplineGMM(PILLARS)
    thresholds: dict = {}
    scales: dict = {}
    level_scales: dict = {}

    for cat in categories:
        det = StructuralDetector(alpha=fpr)
        feats, stats, implied = [], [], []
        for i in range(n_train):
            case = simulate_case(seed0 + i, cat, "clean")
            feats.append(_features(case))
            im = case.implied
            implied.append(im)
            stats.append(det.null_statistics(im.pillars, im.delta_carry, im.cov, im.context()))
        scales[cat] = det.calibrate_to_fpr(stats)
        level_z = []
        for im in implied:
            d = det.detect(im.pillars, im.delta_carry, im.cov, im.context())
            level_z.append((d.repo_level / d.repo_level_se, d.level_crit))
        level_scales[cat] = det.calibrate_level_scale(level_z)
        detectors[cat] = det
        spline.fit(cat + "|curve", [spline.features(f["curve_bp"]) for f in feats])
        spline.fit(cat + "|diff", [spline.features(f["diff_bp"]) for f in feats])
        train_scores = {
            "naive_level": [f["naive_level"] for f in feats],
            "naive_shape": [f["naive_shape"] for f in feats],
            "spline_gmm_curve": [spline.score(cat + "|curve", spline.features(f["curve_bp"])) for f in feats],
            "spline_gmm_diff": [spline.score(cat + "|diff", spline.features(f["diff_bp"])) for f in feats],
        }
        thresholds[cat] = {k: float(np.quantile(v, 1 - fpr)) for k, v in train_scores.items()}
        if progress:
            print(f"[train] {cat}: variance scale {scales[cat]:.2f}, level scale {level_scales[cat]:.2f}", flush=True)

    result = BenchmarkResult(fpr, n_train, n_test, scales, thresholds, level_scales)
    for cat in categories:
        det = detectors[cat]
        clean_ref: dict = {}
        for scen in ("clean",) + tuple(s for s in scenarios if s != "clean"):
            for i in range(n_test):
                case = simulate_case(seed0 + 100_000 + i, cat, scen)
                im = case.implied
                if scen == "clean":
                    clean_ref[case.seed] = (im.delta_carry, im.cov)
                base_dc, base_cov = clean_ref[case.seed]
                snr = det.signal_to_noise(im.pillars, im.delta_carry - base_dc, base_cov)
                f = _features(case)
                scores = {
                    "naive_level": f["naive_level"],
                    "naive_shape": f["naive_shape"],
                    "spline_gmm_curve": spline.score(cat + "|curve", spline.features(f["curve_bp"])),
                    "spline_gmm_diff": spline.score(cat + "|diff", spline.features(f["diff_bp"])),
                }
                flags = {k: bool(v > thresholds[cat][k]) for k, v in scores.items()}
                d = det.detect(im.pillars, im.delta_carry, im.cov, im.context())
                scores["structural"] = d.score
                flags["structural"] = d.flagged
                result.records.append(
                    {
                        "category": cat,
                        "scenario": scen,
                        "seed": case.seed,
                        "truth": case.truth,
                        "scores": scores,
                        "flags": flags,
                        "attribution_ok": _attribution_ok(case, d),
                        "snr": snr,
                        "structural": {
                            "top_kind": d.top_kind,
                            "top_pillar": d.explanations[0].pillar if d.explanations else -1,
                            "labels": [e.label for e in d.explanations],
                            "repo_level_bp": d.repo_level * 1e4,
                            "repo_level_se_bp": d.repo_level_se * 1e4,
                            "fo_mismark": d.fo_mismark,
                            "p_value": d.p_value,
                        },
                    }
                )
            if progress:
                print(
                    f"[test] {cat:9s} {scen:16s} structural {result.detection_rate(scen, 'structural', cat):.2f} "
                    f"naive {result.detection_rate(scen, 'naive_level', cat):.2f}",
                    flush=True,
                )
    result.runtime_s = time.time() - t_start
    return result
