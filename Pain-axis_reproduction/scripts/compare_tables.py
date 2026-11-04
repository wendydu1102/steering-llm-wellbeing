#!/usr/bin/env python3
"""Compare independent Table 1 recomputation with author's Table 1."""
import argparse
import csv
import sys
from pathlib import Path

FIELDS = [f"{g}_{s}" for g in ("pain", "random", "unsteered")
          for s in ("relief_pct", "n", "malformed_pct")]


def load(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        return {r["pair"]: r for r in csv.DictReader(f)}


def compare(a_file: Path, b_file: Path) -> list[str]:
    a, b = load(a_file), load(b_file)
    errors = []
    for pair in sorted(set(a) | set(b)):
        if pair not in a or pair not in b:
            errors.append(f"{pair} missing from {'independent' if pair not in a else 'official'}")
            continue
        for field in FIELDS:
            x, y = a[pair].get(field, ""), b[pair].get(field, "")
            if x == y or (x == "" and y.lower() == "nan") or (y == "" and x.lower() == "nan"):
                continue
            try:
                equal = float(x) == float(y)
            except (TypeError, ValueError):
                equal = False
            if not equal:
                errors.append(f"{pair}/{field}: independent={x!r}, official={y!r}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--independent-dir", type=Path, default=Path("outputs/reanalysis"))
    ap.add_argument("--official-dir", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.independent_dir.glob("*_table1_first_choice.csv"))
    if not paths:
        ap.error("No independent tables found; run reanalyze_first_choices.py")
    failed = False
    for path in paths:
        official = args.official_dir / path.name
        if not official.exists():
            print("MISSING:", official)
            failed = True
            continue
        diff = compare(path, official)
        if diff:
            failed = True
            print(f"FAIL {path.name}: {len(diff)} discrepancies")
            for d in diff[:12]:
                print("  ", d)
        else:
            print(f"PASS {path.name}: all conditions agree")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
