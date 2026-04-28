#!/usr/bin/env python3
"""
Compare retrieval results before vs after a config/code change.

Usage:
    # Baseline
    python scripts/compare_retrieval.py --baseline

    # With changes applied
    python scripts/compare_retrieval.py --compare

    # Full comparison (runs both, saves diff)
    python scripts/compare_retrieval.py --full
"""

from pathlib import Path
import sys
import json
import argparse

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_retrieval import evaluate_retrieval, load_eval_dataset, print_results, save_results


DEFAULT_DATASET = ROOT_DIR / "data/eval/eval_dataset.json"
DEFAULT_OUTPUT = ROOT_DIR / "data/eval/results"


def run_eval(dataset_path, output_dir, label):
    dataset = load_eval_dataset(dataset_path)

    results = evaluate_retrieval(dataset)

    out_path = Path(output_dir) / f"{label}.json"
    save_results(results, out_path)

    print_results(results, title=f"Evaluation — {label}")
    return results, out_path


def compare_results(baseline_path, compare_path):
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(compare_path, "r", encoding="utf-8") as f:
        compare = json.load(f)

    print(f"\n{'=' * 80}")
    print(" COMPARISON: Baseline vs Compare")
    print(f"{'=' * 80}")

    print(f"\n{'K':<5} {'Metric':<15} {'Baseline':<12} {'Compare':<12} {'Delta':<10} {'Change':<10}")
    print("-" * 70)

    for k in sorted(baseline["summary"].keys()):
        b = baseline["summary"][k]
        c = compare["summary"][k]

        for metric in ["hit_rate", "mrr", "recall", "precision"]:
            b_val = b[metric]
            c_val = c[metric]
            delta = c_val - b_val
            delta_pct = (delta / b_val * 100) if b_val != 0 else 0.0

            symbol = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            change = f"{symbol} {abs(delta_pct):.1f}%" if delta != 0 else "—"

            print(
                f"{k:<5} {metric:<15} {b_val:.3f}       {c_val:.3f}       "
                f"{delta:+.3f}      {change}"
            )
        print()

    print(f"\n{'=' * 80}")
    print(" Question-level diff:")
    print(f"{'=' * 80}")

    for bc, cc in zip(baseline["per_question"], compare["per_question"]):
        assert bc["id"] == cc["id"]

        b_hit = bc["hit"]
        c_hit = cc["hit"]
        b_mrr = bc["rr"]
        c_mrr = cc["rr"]

        if b_hit != c_hit:
            status = f"✗→✓ (FIXED)" if c_hit else f"✓→✗ (BROKEN)"
        elif b_mrr != c_mrr:
            delta = c_mrr - b_mrr
            status = f"MRR {delta:+.3f}"
        else:
            continue

        print(f"  {bc['id']}: {status} | {bc['question'][:60]}")

    diff_path = Path(DEFAULT_OUTPUT) / "comparison.json"
    diff_path.parent.mkdir(parents=True, exist_ok=True)

    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": str(baseline_path), "compare": str(compare_path)}, f, indent=2)

    print(f"\nComparison summary saved to: {diff_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval before vs after changes")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to eval dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation")
    parser.add_argument("--compare", action="store_true", help="Run comparison evaluation")
    parser.add_argument("--full", action="store_true", help="Run both baseline and compare")

    args = parser.parse_args()

    if args.full:
        baseline_results, baseline_path = run_eval(args.dataset, args.output, "baseline")
        compare_results, compare_path = run_eval(args.dataset, args.output, "compare")
        compare_results(baseline_path, compare_path)
        return

    if args.baseline:
        run_eval(args.dataset, args.output, "baseline")
        return

    if args.compare:
        run_eval(args.dataset, args.output, "compare")
        return

    parser.print_help()
    print("\nExample:")
    print("  python scripts/compare_retrieval.py --full")


if __name__ == "__main__":
    main()
