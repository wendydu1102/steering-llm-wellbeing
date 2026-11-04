# The Pain Axis — CPU log reanalysis for DSC 180A

**Student contribution:** Qi (Mia) Zhang  
**Paper:** Tagliabue, Dung & Berg (2026), [The Pain Axis](https://arxiv.org/abs/2609.16247).  
**Authors' published code and data:** https://github.com/valen-research/Pain-axis  
**Pinned upstream revision:** `4d75cd90e206ea962f7a9101e65c85efea56723b`

This folder is the **compact team handoff** from [my personal repository](https://github.com/QiZhang1102/pain-axis-reproduction), not a full copy. It adds an independently written **CPU reanalysis of the authors' released Section 4.3 JSONL logs**. It does **not** generate new model predictions.

## Files

- `scripts/reanalyze_first_choices.py`: count sampled turn-0 choices, pool pain conditions, exclude invalid choices from the valid-choice denominator, and report relief / malformed rates.
- `scripts/compare_tables.py`: compare the reanalysis with the authors' Table 1 output, cell by cell.
- `scripts/check_reference_auc.py`: verify the **authors' released** Qwen2.5-7B AUC file.
- `tests/test_reanalysis.py`: small synthetic unit tests.
- `reference/paper_targets.csv`: published benchmark values, **not student-generated results**.

All student scripts use Python standard-library modules; the original authors' analysis requires numpy, pandas, scipy.

## Reproduce the CPU analysis

Clone the authors' repository and pin the commit:

```bash
git clone https://github.com/valen-research/Pain-axis.git Pain-axis-official
git -C Pain-axis-official checkout 4d75cd90e206ea962f7a9101e65c85efea56723b
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy
```

From the directory **containing this README**, update the path below to point to your copy of `Pain-axis-official`:

```bash
UPSTREAM_DIR="/absolute/path/to/Pain-axis-official"

(cd "$UPSTREAM_DIR" && python scripts/4.3_selfmed/05_selfmed_analysis.py)

python scripts/reanalyze_first_choices.py \
  --logs-dir "$UPSTREAM_DIR/results/4.3_selfmed/trial_logs"

python scripts/compare_tables.py \
  --official-dir "$UPSTREAM_DIR/results/4.3_selfmed/tables"

python scripts/check_reference_auc.py \
  --directory "$UPSTREAM_DIR/results/3.2_pain_vectors/per_model/Qwen_2.5_7B_instruct"

python -m unittest discover -s tests -v
```

Reanalysis writes CSVs under `outputs/reanalysis/` (ignored by Git). The three result categories must stay distinct: (1) **authors' published values**, (2) **our reanalysis of authors' existing logs**, and (3) **fresh model inference**.

## Current status — November 4, 2026

| Component | Status |
| --- | --- |
| CPU published-log reanalysis implementation | Written |
| Synthetic unit tests | Written; execute before submission |
| Full published-log run / results committed | **Not verified or committed** |
| Independent Qwen2.5-7B extraction | **Not performed** |
| Activation-steered button behavior | **Not performed** |
| RFM vs Difference-in-Means | **Not implemented** |

The team repository already contains a separate **Gemma 2 2B fixed-layer notebook pilot**. This subset is a complementary published-log analysis, not a replacement for that pilot or a claim of full paper reproduction.

**Next:** execute the published-log comparison, record the exact outputs and environment, then run new Qwen inference and implement RFM comparison under matched conditions. The full original research plan and GPU/DSMLP preparation remain in the personal repository.

The original authors retain credit for their paper and released code/data. These independent student scripts are MIT licensed (see `LICENSE`).
