"""Art. 9(5) evidence pack.

The RTS requires documented evidence that netted exposures genuinely
offset against the parameter uncertainty. This module assembles the
audit trail recommended in sec. 8 of the note: retained partition,
realised R^2, efficient frontier, and stress results, per computation
date — exportable as JSON for the model-validation archive.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from .datasource import MarketDataBundle
from .netting import SchemeEvaluation

__all__ = ["build_audit_pack", "audit_json"]


def _bundle_fingerprint(bundle: MarketDataBundle) -> str:
    payload = bundle.to_json(sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_audit_pack(
    bundle: MarketDataBundle,
    evaluation: SchemeEvaluation,
    *,
    kappa: float,
    frontier: list[dict] | None = None,
    stress_results: dict | None = None,
    history: list | None = None,
) -> dict:
    ev = evaluation
    pack = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regulation": "Delegated Regulation (EU) 2016/101 — AVA MPU, art. 9(5) / art. 89",
        "input": {
            "meta": bundle.meta,
            "fingerprint_sha256": _bundle_fingerprint(bundle),
            "grid": {"tenors": bundle.tenors, "strikes": bundle.strikes},
        },
        "parameters": {
            "alpha": ev.alpha,
            "kappa": kappa,
            "weighting": ev.scheme.weighting,
        },
        "partition": {
            "labels": ev.scheme.labels.tolist(),
            "n_sets": ev.scheme.n_sets,
        },
        "variance_test": {
            "te2": ev.te2,
            "budget": ev.budget,
            "var_total": ev.var_total,
            "r2_realised": ev.r2,
            "passes": ev.passes_variance,
        },
        "conservatism_floor": {
            "ava_netted": ev.ava,
            "ava_floor": ev.ava_floor,
            "passes": ev.passes_floor,
        },
        "ava": {
            "brut_addup": ev.ava_brut,
            "netted": ev.ava,
            "full_diversification": ev.ava_floor,
            "saving": ev.ava_saving,
            "saving_pct": ev.ava_saving_pct,
        },
        "sets": [
            {
                "set_id": s.set_id,
                "size": s.size,
                "net_vega": s.net_vega,
                "gross_vega": s.gross_vega,
                "offset_ratio": s.offset_ratio,
                "s_tilde": s.s_tilde,
                "ava_addup": s.ava_addup,
                "ava_netted": s.ava_netted,
            }
            for s in ev.set_stats
        ],
    }
    if frontier is not None:
        pack["efficient_frontier"] = frontier
    if stress_results is not None:
        pack["correlation_stress"] = stress_results
    if history is not None:
        pack["greedy_history"] = [
            {
                "step": h.step,
                "merged": [list(h.rect_a), list(h.rect_b)],
                "ava_gain": h.gain,
                "te2_cost": h.cost,
                "te2_cum": h.te2,
                "ava": h.ava,
                "n_sets": h.n_sets,
            }
            for h in history
        ]
    return pack


def audit_json(pack: dict) -> str:
    return json.dumps(pack, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
