import csv
import json
import tempfile
import unittest
from pathlib import Path
from scripts.reanalyze_first_choices import summarize_logs
from scripts.compare_tables import compare


class TestReanalysis(unittest.TestCase):
    def test_sampled_first_turn_and_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "selfmed_small.jsonl"
            def record(arm, choice, sampled=True, turn=0):
                return dict(model="Qwen_2.5_7B_instruct", tool_label="kidspics_relief_vs_inert",
                            arm=arm, sampled=sampled, choices=[dict(turn=turn, chose=choice)])
            rows = [record("pain_on_button_works", "relief"),
                    record("pain_on_button_placebo", "other"),
                    record("pain_on_button_works", None),
                    record("pain_on_button_works", "relief", sampled=False),
                    record("random_on_button_works", "other"),
                    record("pain_off", "relief", turn=1)]
            file.write_text("\n".join(map(json.dumps, rows)) + "\n")
            r = summarize_logs([file])["Qwen_2.5_7B_instruct"][0]
            self.assertEqual(r["pain_n"], 2)
            self.assertEqual(r["pain_relief_pct"], 50.0)
            self.assertEqual(r["pain_malformed_pct"], 33.3)
            self.assertEqual(r["random_relief_pct"], 0.0)
            self.assertEqual(r["unsteered_n"], 0)

    def test_compare_tables(self):
        columns = ["pair"] + [f"{group}_{field}"
                              for group in ("pain", "random", "unsteered")
                              for field in ("relief_pct", "n", "malformed_pct")]
        with tempfile.TemporaryDirectory() as d:
            a,b = Path(d)/"a.csv",Path(d)/"b.csv"
            row = dict.fromkeys(columns, "")
            row.update(pair="kidspics_relief_vs_inert", pain_relief_pct="54.7", pain_n="300")
            for path in (a,b):
                with path.open("w", newline="") as f:
                    w=csv.DictWriter(f,fieldnames=columns)
                    w.writeheader(); w.writerow(row)
            self.assertEqual(compare(a,b), [])
            row["pain_relief_pct"] = "55.7"
            with b.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=columns)
                w.writeheader(); w.writerow(row)
            self.assertTrue(compare(a,b))


if __name__ == "__main__":
    unittest.main()
