#!/usr/bin/env python3
"""Verify AUTHOR'S published 7B extraction CSV, not new model inference."""
import argparse
import csv
import json
from pathlib import Path

TARGETS = {
    ("S2_1P", "final_token", 24): 0.9565,
    ("S2_3P", "final_token", 24): 0.9650,
    ("S2_1P", "mean", 26): 0.9570,
    ("S2_3P", "mean", 26): 0.9400,
}


def verify(directory: Path) -> None:
    with (directory / "layer_curves.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = {(r["dataset"], r["extraction"], int(r["layer"])): r for r in rows}
    with (directory / "summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["best_layer_final_token"] == 24
    assert summary["best_layer_mean"] == 26
    for key, expected in TARGETS.items():
        actual = float(rows[key]["auc_vs_all_controls"])
        assert abs(actual - expected) < 1e-6, (key, actual, expected)
        print(f"PASS official CSV {key}: AUC={actual:.4f}")
    print("Validated official reference CSV; this is NOT an independent GPU rerun.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    args = ap.parse_args()
    verify(args.directory)


if __name__ == "__main__":
    main()
