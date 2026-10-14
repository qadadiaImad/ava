"""Run the synthetic benchmark and save the per-case records to results/benchmark.json.gz."""
import argparse
import warnings
from pathlib import Path

from repo_control.benchmark import DETECTORS, run_benchmark
from repo_control.synthetic import SCENARIOS

warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--fpr", type=float, default=0.02)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "benchmark.json.gz"))
    args = ap.parse_args()
    res = run_benchmark(n_train=args.n_train, n_test=args.n_test, fpr=args.fpr)
    res.to_json(args.out)
    print(f"\nsaved {args.out} ({len(res.records)} cases, {res.runtime_s:.0f}s)")
    for s in SCENARIOS:
        print(f"{s:16s} " + " ".join(f"{d}={res.detection_rate(s, d):.2f}" for d in DETECTORS))
