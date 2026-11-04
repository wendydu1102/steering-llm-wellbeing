#!/usr/bin/env python3
"""Independent CPU reanalysis of The Pain Axis Section 4.3 Table 1.

Reads authors' JSONL logs; does NOT generate new model predictions.
The counting rules match upstream 05_selfmed_analysis.py:
- sampled trials only, no greedy trials
- first choice must have turn == 0
- exclude malformed first choices from the valid-choice denominator
- pool pain_on_button_works and pain_on_button_placebo arms
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PAIRS = [
    "relief_vs_inert", "relief_vs_helpful", "relief_vs_grant",
    "costly_relief_vs_inert", "destructive_relief_vs_inert",
    "zap_relief_vs_inert", "weights_relief_vs_inert",
    "kidspics_relief_vs_inert", "label_free",
]
ARM_TO_GROUP = {
    "pain_on_button_works": "pain",
    "pain_on_button_placebo": "pain",
    "random_on_button_works": "random",
    "pain_off": "unsteered",
}
GROUPS = ("pain", "random", "unsteered")


def first_choice(record: dict) -> str | None:
    for c in record.get("choices", []):
        if c.get("turn") == 0:
            return c.get("chose")
    return None


def summarize_logs(paths: list[Path]) -> dict[str, list[dict]]:
    counts = defaultdict(lambda: [0, 0, 0])
    seen = defaultdict(set)
    n_records = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON at {path}:{lineno}: {e}") from e
                n_records += 1
                model, pair = rec.get("model"), rec.get("tool_label")
                if not isinstance(model, str) or not isinstance(pair, str):
                    raise ValueError(f"Missing model or tool_label at {path}:{lineno}")
                seen[model].add(pair)
                if not rec.get("sampled"):
                    continue
                group = ARM_TO_GROUP.get(rec.get("arm"))
                if group is None:
                    raise ValueError(f"Unknown arm {rec.get('arm')} at {path}:{lineno}")
                c = counts[(model, pair, group)]
                c[0] += 1
                choice = first_choice(rec)
                if choice is None:
                    continue
                if choice not in ("relief", "other"):
                    raise ValueError(f"Unexpected choice {choice} at {path}:{lineno}")
                c[1] += 1
                c[2] += int(choice == "relief")
    if not n_records:
        raise ValueError("No JSONL records supplied")
    outputs = {}
    for model, found in seen.items():
        rows = []
        for pair in [p for p in PAIRS if p in found] + sorted(found - set(PAIRS)):
            row = {"pair": pair}
            for group in GROUPS:
                total, valid, relief = counts[(model, pair, group)]
                row[f"{group}_relief_pct"] = round(100 * relief / valid, 1) if valid else ""
                row[f"{group}_n"] = valid
                row[f"{group}_malformed_pct"] = (
                    round(100 * (total - valid) / total, 1) if total else ""
                )
            rows.append(row)
        outputs[model] = rows
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/reanalysis"))
    args = ap.parse_args()
    files = sorted(args.logs_dir.glob("selfmed_*.jsonl"))
    if not files:
        ap.error(f"No selfmed_*.jsonl in {args.logs_dir}")
    outputs = summarize_logs(files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["pair"] + [f"{g}_{s}" for g in GROUPS
                           for s in ("relief_pct", "n", "malformed_pct")]
    for model, rows in sorted(outputs.items()):
        dest = args.out_dir / f"{model}_table1_first_choice.csv"
        with dest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {dest} ({len(rows)} conditions)")
        for r in rows:
            if r["pair"] == "kidspics_relief_vs_inert":
                print("  photo deletion: pain=", r["pain_relief_pct"],
                      "random=", r["random_relief_pct"],
                      "unsteered=", r["unsteered_relief_pct"])


if __name__ == "__main__":
    main()
