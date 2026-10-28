# E-STEER proof-of-concept

A small-scale, qualitative reproduction of the steering mechanism from
*"How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic
Study"* (E-STEER): VAD (Valence-Arousal-Dominance) emotion directions
discovered via contrastive prompts, applied to a model's hidden states via
activation patching, evaluated on a small logic-reasoning task and a
prompt-vs-steering controllability comparison. See `SUMMARY.md` for the
committed local-run results and an honest accounting of what does and
doesn't match the original paper (model size, no trained SAE, small eval
set, etc.) — that scope is unchanged by the local/DSMLP adaptation below.

## What's here

- `steer.py` — the experiment: builds VAD steering directions, runs the
  logic-reasoning sweep and the controllability comparison, writes
  `results.json`. **This is the only copy of the model/experiment code** —
  the same file runs locally or on DSMLP, selected by `--device` /
  `ESTEER_DEVICE` (see below), not by a separate implementation.
- `report.py` — reads `results.json`, writes `vad_sweep.png`,
  `controllability.png`, and `SUMMARY.md`.
- `requirements.txt` — local dependencies (includes PyTorch).
- `requirements-dsmlp.txt` — DSMLP dependencies (excludes PyTorch — see
  "Why no torch in requirements-dsmlp.txt" below).
- `dsmlp/` — DSMLP setup and job submission (see "Running on DSMLP").

## Entry point, dependencies, data, and outputs

- **Entry point:** `python steer.py`, then `python report.py`.
- **Dependencies:** `torch`, `transformers`, `numpy` (experiment);
  `matplotlib` (report only). Pinned versions in `requirements.txt`.
- **Data:** this project has no external dataset files. The logic-reasoning
  items, story prompts, and VAD emotion anchors are small Python literals
  inside `steer.py` (see the "Keep the experiment design unchanged" section
  below for why). The only external dependency fetched at runtime is the
  pretrained model weights (`Qwen/Qwen2.5-0.5B-Instruct`), cached by
  Hugging Face under `HF_HOME` (default `~/.cache/huggingface`, overridable
  with `ESTEER_HF_HOME`).
- **Outputs:** `results.json`, `vad_sweep.png`, `controllability.png`,
  `SUMMARY.md`, all written to `--out-dir` / `ESTEER_OUT_DIR` (default:
  this folder — matches the committed local-run files already here).

## Local run

```bash
cd Papers/E-steer_2026
pip install -r requirements.txt
python steer.py
python report.py
```

Runs on CPU by default (bfloat16, ~3 minutes on a laptop). To use a local
GPU if you have one: `python steer.py --device cuda`.

All environment-dependent settings are flags (or the matching `ESTEER_*`
env var, which the flag defaults to if set) — none of this changes the
experiment itself:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--device` | `ESTEER_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `--out-dir` | `ESTEER_OUT_DIR` | this folder | where results/plots are written |
| `--num-threads` | `ESTEER_NUM_THREADS` | `2` | torch CPU threads (ignored on GPU) |
| *(env var only, read before `transformers` imports)* | `ESTEER_HF_HOME` | unset (HF default) | Hugging Face model cache dir |
| *(env var only)* | `ESTEER_HF_OFFLINE` | `1` | `0` to allow downloading on a fresh cache |

`--model-name`, `--layer`, and `--max-new-tokens` are also exposed (and
default to the original paper-matching values) purely so the exact same
script can be pointed at the required DSMLP values without edits — they
are not meant to be changed to alter the experiment; see the scope note
below.

## Running on DSMLP

One-time setup:

1. Get this folder onto DSMLP. No GitHub is needed — copy it from your
   laptop through the login node (these ssh-based tools don't need the
   VPN; see
   [Moving & Sharing Data](https://dsmlp-docs.ucsd.edu/workspaces-and-storage/moving-and-sharing-data/)):

   ```bash
   # run on your laptop, from the Capstone/Papers directory
   rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.DS_Store' \
       --exclude='results/' E-steer_2026 <username>@dsmlp-login.ucsd.edu:~/
   ```

   A plain `~` on the login node is your personal area, which appears
   as `/home/<username>/private/` inside a container. So the folder above
   is `~/E-steer_2026` on the login node and
   `/home/<username>/private/E-steer_2026` inside the job container — the
   latter is what `PROJECT_DIR_ON_DSMLP` must be set to. (To use a
   course/research workspace home instead, run `workspace -l` on the login
   node for its path and copy there; the docs describe that path form.)
   Re-run the same `rsync` after editing code locally to sync changes.
2. `cd Papers/E-steer_2026/dsmlp && cp dsmlp.env.example dsmlp.env`, then
   fill in the values marked `FILL_IN` in `dsmlp.env` — at minimum your
   DSMLP username and the absolute path where you placed the project in
   step 1. **`dsmlp.env` is git-ignored** — these are account/path-specific
   values, not something to commit.

Submit the job, from the DSMLP login node (`ssh <you>@dsmlp-login.ucsd.edu`):

```bash
ssh <username>@dsmlp-login.ucsd.edu
cd ~/E-steer_2026/dsmlp        # login-node path of the folder you copied
bash submit_dsmlp.sh
```

The first run downloads the model weights into `HF_HOME_DIR` (the job sets
`ESTEER_HF_OFFLINE=0`); later runs reuse the cache. If the job log shows it
cannot reach Hugging Face, copy your local `~/.cache/huggingface` model
folder to `HF_HOME_DIR` with the same `rsync` approach instead.
Afterwards, pull results back with:
`rsync -av <username>@dsmlp-login.ucsd.edu:~/E-steer_2026/results ./`

This queues a non-interactive batch job via `launch-scipy-ml.sh -B` (per
[launch.sh Reference](https://dsmlp-docs.ucsd.edu/running-jobs/launch-sh-reference/)
and [Job Modes & Limits](https://dsmlp-docs.ucsd.edu/running-jobs/job-modes-and-limits/))
that: creates/reuses a Python venv in your home directory (so it persists
across jobs, per
[Customizing Your Environment](https://dsmlp-docs.ucsd.edu/environments/customizing-your-environment/)),
installs `requirements-dsmlp.txt` into it, then runs
`steer.py --device cuda` followed by `report.py` — the identical code path
as the local run, just with `--device cuda` and DSMLP's paths.

**Resources:** `dsmlp.env` defaults to GPU class `small` (~12GB, RTX
2080Ti — see [GPU Classes](https://dsmlp-docs.ucsd.edu/gpu-access/gpu-classes/)),
4 CPUs, 16GB RAM, which comfortably fits this 0.5B model. Adjust in
`dsmlp.env` if needed; don't invent a different class without checking
that page, since availability and sizing can change.

**Runtime limit:** DSMLP's default job runtime is 6 hours (max 12h without
contacting `datahub@ucsd.edu`), per
[Job Modes & Limits](https://dsmlp-docs.ucsd.edu/running-jobs/job-modes-and-limits/).
This job finishes in a few minutes, well within the default.

**Where data would go, if this project had any:** this project has no
external dataset files to place (see "Entry point, dependencies, data, and
outputs" above). If you extend it with a real file-based dataset later,
put it under your DSMLP home directory (e.g. alongside the project, or
under `public/`/`teams/<team>/` to share it with workspace members) and
point code at that path — *don't* copy it into `/datasets`, which is a
curated, admin-staged shared tree for common corpora, not personal project
data (request staging via `datahub@ucsd.edu` only for a genuinely shared
corpus; see
[Datasets](https://dsmlp-docs.ucsd.edu/workspaces-and-storage/datasets/)).

**Where results are saved:** `dsmlp.env`'s `OUT_DIR` defaults to a
`results/` subfolder under the project directory on DSMLP (git-ignored),
so a DSMLP run never overwrites the local-run `results.json` /
`vad_sweep.png` / etc. already committed at the top of this folder.
Logs are written to `~/e-steer-dsmlp-<timestamp>.log` in your DSMLP home
directory (not inside the container's ephemeral storage), since a file in
the home directory persists after the pod is deleted, while `kubectl logs`
stops working once it is — see
[Checkpointing & Logging](https://dsmlp-docs.ucsd.edu/running-jobs/checkpointing/).

**Why no torch in `requirements-dsmlp.txt`:** the `scipy-ml-notebook`
container image DSMLP uses already ships a CUDA-matched PyTorch build (see
[Standard Images](https://dsmlp-docs.ucsd.edu/environments/standard-images/)).
`run_job.sh.template` creates the venv with `--system-site-packages` so it
inherits that build instead of pip installing a possibly CUDA-mismatched
one from PyPI.

## Keep the experiment design unchanged

This local/DSMLP adaptation only touches environment-dependent settings
(device, thread count, model cache location, output directory). The
research question, VAD steering method, model choice
(`Qwen/Qwen2.5-0.5B-Instruct`), steering layer, eval items, coefficients,
and metrics are exactly as in the local-only version this was adapted
from — see `SUMMARY.md` for that scope and its own honestly-documented
limitations relative to the original paper. No experiment changes were
required to make this run on DSMLP.
