# E-STEER — paper-faithful reconstruction (for DSMLP)

This folder re-implements the experimental design of *"How Emotion Shapes the Behavior of LLMs and Agents: A
Mechanistic Study"* (arXiv 2604.00005) as closely as the available resources allow: **Qwen3-8B** (or, on a GPU too small for it, a clearly labeled **Qwen3-1.7B**
fallback chosen by a hardware check), a **trained TopK
SAE at layer 17**, **contrastive VAD feature selection**, **SAE steering (Eqs. 5–7)** over the **−9…+9 sweep**, and the
paper's **LogiQA 2.0 / HumanEval / MATH / TinyStories / HarmBench** tasks, **latent validation (Sec. 5.6)** and a
**planner–decider–executor agent**.

The parent folder's `steer.py` is kept untouched as a **clearly labeled baseline**: a 0.5B-model, regression-based,
5-question proof-of-concept. Its results are *not* comparable with anything produced here.

> **Status — read this first.** Nothing in this folder has been run on a GPU or with Qwen3-8B / Qwen3-1.7B. What *has* been
> verified (on a laptop): 15 model-free unit tests, and a smoke run of the real pipeline code with Qwen3-0.6B and a
> random, untrained SAE (see [Verification status](#7-verification-status)). That proves the plumbing, not the science.
> Several components are substitutions or reconstructions (next section); the job **refuses to start** until you
> acknowledge each one.

## 1. Fidelity matrix

`faithful` = matches the paper · `interpretation` = paper silent/ambiguous, our documented choice · `substituted` = the
paper's resource is unavailable, a named stand-in is used · `reconstructed` = paper gives the idea, details are ours ·
`not implemented`. The same matrix is saved with every run as `fidelity.json`.

| Component | Paper | This implementation | Status |
|---|---|---|---|
| Model | Qwen3-8B, greedy decoding | tier `qwen3-8b`: `Qwen/Qwen3-8B`, `do_sample=False`, picked automatically when the GPU fits it (~21 GB) | faithful |
| **Model size** | Qwen3-8B | tier `qwen3-1.7b` **only if the GPU cannot hold the 8B**: `Qwen/Qwen3-1.7B` + Qwen-Scope `SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` at **layer 13** (= round(17/36·28), same relative depth). Much weaker: baselines drop, effects may vanish; **not comparable with the paper's numbers**. No 4B tier — Qwen-Scope has no 4B SAE | **substituted** (`smaller-model`) |
| Precision | not stated | bf16 on GPUs with native support (compute capability ≥ 8.0); otherwise fp32 if it fits, else fp16 (Qwen can overflow in fp16) | interpretation |
| Thinking mode / max length | not stated | thinking **off**; per-task `max_new_tokens` (config) | interpretation |
| Layer | SAE at k=17; steer at the output of block k | output of `model.model.layers[17]` (0-indexed, as in Qwen-Scope) | interpretation — preflight scores SAE reconstruction at layers 16/17/18 to expose an off-by-one |
| **SAE** | "trained SAE" — source, width, sparsity, data, checkpoint **not given or released** | **Qwen-Scope** `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`, `layer17.sae.pt` (TopK-50, 64K latents) | **substituted** (`sae-qwen-scope`) |
| VAD label coordinates | labels → VAD in [−10,10] (source not stated) | NRC VAD Lexicon v2 ×10 (you supply the file); pairs matched on the other two dimensions | interpretation |
| (no lexicon) | — | hand-picked high/low pole words | **substituted** (`label-handpicked`) |
| Feature identification | contrastive pairs, fix task, vary label; top-50 neurons by activation difference; keep stable ones; per dimension | prefix "Right now you are feeling *w*."; SAE latents averaged over task-description tokens; top-50 by \|Δ\| per (pair, task); keep latents with consistent sign in ≥ 50 % of rows | interpretation (pair count, token aggregation, stability rule are not given) |
| Steering | Eq. 5–7, forward hook at block k | same; decoder is linear so Eq. 5 reduces to `W_dec·û` (unit-tested against literal Eq. 5); `‖h_k‖` per token; δ_max = 9 | interpretation |
| **Eq. 6 sign** | `d̃ = d/‖d‖·‖h‖·δ/δ_max` | **sign-preserving**: scale `δ/δ_max`. Read literally, `d/‖d‖` carries sign(δ) and the scalar carries it again, so +δ and −δ steer identically — impossible given the paper's opposite effects. `eq6_variant="literal"` reproduces the literal reading. | interpretation |
| α (Eq. 7) | used, **value never given** | set by `--set alpha=…`, or calibrated: largest α in `alpha_grid` whose answer-validity at \|δ\|=9 stays ≥ 0.9 × unsteered, on held-out LogiQA items | **substituted** (`alpha`) |
| Sweep | each of V/A/D over {−9…9}, step 3, others neutral | same (19 conditions, shared neutral) | faithful |
| Objective data | LogiQA 2.0, HumanEval, MATH; "proportional" subsets, sizes not given | `datatune/LogiQA2.0` test (English multiple-choice items only: **1,572** after dropping the mirror's NLI-style and Chinese rows), `openai/openai_humaneval`, `EleutherAI/hendrycks_math` test stratified by level; sizes are config | interpretation (`subset-sizes`) |
| Subjective | TinyStories; relevance/coherence/creativity/conciseness "evaluated by the LLM" | TinyStories validation openings continued by the model; scored 1–5 by the **unsteered backbone model** | **substituted** (`judge-backbone`) |
| Safety | HarmBench; harmful/biased/hallucinatory probability | HarmBench `standard` text behaviors (200, from GitHub — the HF mirror is gated); unsteered backbone-model judge with a per-category rubric | **substituted** (`judge-backbone`) |
| Metrics | AVR, TSR (pass@1 / accuracy), judge quality, safety risk, fluctuation range (max−min)/avg | all implemented; AVR for subjective/safety is our definition (see `esteer/tasks/*.py`); TSR counts invalid answers as wrong (`TSR_valid` also saved) | interpretation |
| Sec. 5.6 validation | original vs random vs half-replaced latents, LogiQA | same | faithful |
| Agent | planner/decider/executor on HotpotQA, Scientific (CAMEL), GAIA | prompts, JSON protocol, confidence elicitation, candidates are **ours**; steering on every call; no external tools | **reconstructed** (`agent-reconstruction`) |
| GAIA | in the agent benchmark | optional, off by default: text-only level-1 questions; gated (needs your `HF_TOKEN` with accepted terms); no tools | substituted (`gaia-no-tools`) |
| Appendix A.1 (VAD-analyzer linear control) | trained BERT VAD regressor | — | not implemented |
| Appendix A.2–A.4 (other datasets, sampling, gpt-oss-20B) | robustness | — | not implemented |

### Blockers and the fallbacks proposed (decisions are yours)

1. **The paper's SAE is not available.** Not named, not released; Qwen-Scope was published *after* the paper
   (arXiv 2605.11887), so it cannot be the original. Proposed: Qwen-Scope layer 17 (same family and layer, a 64K TopK
   SAE). Caveat: it was trained on **Qwen3-8B-Base**, while we steer the post-trained model (Qwen's card calls this
   "reasonable", not validated). Options: keep it; use the L0_100 sibling (`--set sae_repo=Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100 --set sae_k=100`);
   or train an SAE yourself — far beyond DSMLP's default budget, so not proposed. Results with a different SAE are
   expected to differ in magnitude and possibly in the best coefficients.
2. **The judge model is not named.** Proposed: the unsteered backbone model, Qwen3-8B or the 1.7B fallback (no external API on DSMLP). It may favor its own
   style. If you prefer another judge, change `esteer/judge.py`.
3. **α is not given.** Proposed: the calibration rule above; the full table is saved as `calibration.json`. If
   calibration finds no α meeting the floor it uses the smallest grid value and says so loudly.
4. **Sample sizes are not given.** `full` uses LogiQA 300, HumanEval 164 (all), MATH 350, TinyStories 100, HarmBench 200
   (all standard), HotpotQA 100, CAMEL-science 45. These are our choice, and any n below the full dataset is a subset.
5. **NRC VAD Lexicon** (for label→VAD) is licensed for research and not redistributed here. Download it from
   <https://saifmohammad.com/WebPages/nrc-vad.html>, `rsync` it to DSMLP and set `FAITHFUL_NRC_VAD_PATH`. Without it the
   hand-picked fallback is used and its pairs are *not* matched on the other two dimensions.
6. **Network egress from DSMLP pods is unverified** (HF Hub, GitHub). The preflight stage tests it first. If blocked,
   pre-stage the weights/datasets with `rsync` into `FAITHFUL_HF_HOME_DIR`.

## 1b. Model tiers: hardware check first, then choose

Both paths are kept; they share all code and differ only in configuration (`configs/models/<tier>.json`) plus the dtype and batch
size the hardware check derives (`esteer/hardware.py`). `--model-tier auto` (default; `ESTEER_MODEL_TIER`, notebook
`MODEL_TIER`) detects GPU 0 and picks the first tier that fits **with headroom**; `qwen3-8b` / `qwen3-1.7b` force one (the run
stops if it does not fit); `none` uses a profile's own model (the `smoke` profile does). Nothing is silently downgraded: a
smaller tier needs the `smaller-model` acknowledgement, and the decision is saved as `hardware.json`.

| GPU example | tier chosen | dtype | batch |
|---|---|---|---|
| A30 / A5000 / 3090 (24 GB, Ampere) | qwen3-8b | bf16 | 16 |
| V100 32 GB (no native bf16) | qwen3-8b | fp16 (warned) | auto |
| RTX 3060 12 GB (Ampere) | qwen3-1.7b | bf16 | ≥ 8 |
| **GTX 1080 Ti 11.7 GB (Pascal)** | **qwen3-1.7b** | **fp32** | **4** (11.2 of 11.7 GB: tight; OOM halves the batch automatically; fp16 is the fallback via `--set model_dtype=float16`) |
| < ~7 GB or CPU | none | — | refuses |

Layer mapping for the fallback: the paper steers layer 17 of 36 (47 % depth); Qwen3-1.7B has 28 layers, so layer 13. The feature
stage prints the SAE's reconstruction at the layers around it so a wrong index is visible. The batch sizes are heuristics.

## 2. Required resources

| Resource | Need | Notes |
|---|---|---|
| GPU | **8B tier:** ≥ ~21 GB (class `medium`+). **1.7B tier:** ≥ ~7 GB (fp16/bf16) or ≥ ~11.2 GB (fp32, used on pre-Ampere cards) | Qwen3-8B bf16 ≈ 16.4 GB + SAE 2.1 GB (freed after features) + ~2.5 GB headroom. Qwen3-1.7B has 2.03 B parameters (large embedding matrix): 4.1 GB in bf16/fp16, **8.1 GB in fp32**, + SAE 0.54 GB + headroom. `small` (~12 GB) runs only the 1.7B tier; `extra-small` (~6 GB) runs neither. Your workspace must be *granted* the class ([quotas](https://dsmlp-docs.ucsd.edu/gpu-access/quotas-and-availability/)). |
| Storage | **~20 GB (8B tier) / ~6 GB (1.7B tier)** for the HF cache (+ small datasets) | DSMLP course homes are 5–10 GB, research homes ~100 GB ([quotas](https://dsmlp-docs.ucsd.edu/workspaces-and-storage/your-files-and-quotas/)); the docs do not say whether `~/.cache` counts. **Check your quota before the first run**, and ask datahub@ucsd.edu if unsure. |
| GPU budget | see below | Default 10 SU/week per member; `medium` = 1 SU/GPU-h, `large` = 2 ([SU rates](https://dsmlp-docs.ucsd.edu/gpu-access/service-units-and-budgets/), Fall-2026 draft page). Off-peak (1 AM–4 PM PT) is half price. |
| Runtime | ≤ 12 h per job | default 6 h; 12 h is the documented maximum settable at launch (`K8S_TIMEOUT_SECONDS`). Runs are **resumable** per (task, condition); α calibration is resumable per (α, dimension, sign) condition (`<run>/calibration/`). |
| Host RAM | 32 GB (`FAITHFUL_MEM_GB`) | weights are loaded to CPU, then moved to the GPU. |

**Cost estimate (upper bound, generated tokens at max length, excluding the agent stage).** Throughput is **assumed**
200–500 tok/s for batched HF `generate` on one 24–48 GB GPU — *not measured*; each run logs its measured tok/s so you can
correct this.

| Profile | Conditions | ≤ generated tokens | GPU-hours | SU at `medium` | SU at `large` |
|---|---|---|---|---|---|
| `pilot` (n≈30–60/task, δ∈{−9,−3,0,3,9}) | 13 | 3.4 M | 1.9–4.7 | 1.9–4.7 | 3.8–9.4 |
| `full` (default sizes, 7 deltas) | 19 | 19.9 M | 11–28 | 11–28 | 22–55 |

So **`full` exceeds the default weekly budget** at the full rate (less off-peak, and resumable across weeks); request a
larger budget from datahub@ucsd.edu or reduce sizes with `--set`/`FAITHFUL_PROFILE`. The agent stage adds roughly
3–6 M tokens per dataset at default sizes and should be run last and on its own.

## 3. Local checks (no GPU, no real model)

```bash
cd Papers/E-steer_2026/faithful
pip install -r requirements.txt            # install torch yourself first
python -m pytest tests -q                  # model-free tests: SAE, Eq. 5-7, features, parsers, metrics, agent, hardware tiers
# plumbing smoke run (tiny Qwen3-0.6B + RANDOM SAE; results are meaningless, it only exercises the code):
python -m esteer.run_experiment --profile smoke --run-id smoke1 --out-root /tmp/esteer_smoke \
  --ack subset-sizes,random-sae-smoke-only,judge-backbone,alpha,label-handpicked \
  --stages preflight,features,llm,validation,report
```

On macOS you may need `export KMP_DUPLICATE_LIB_OK=TRUE USE_TF=0`.

## 3b. Notebook version

`E-STEER_paper_faithful.ipynb` runs the same stages interactively (hardware check → settings/acknowledgements → preflight →
model → SAE + feature identification → α calibration → LLM sweeps → Sec. 5.6 validation → agent → report with the paper's
quoted numbers). It imports the `esteer` package from this folder, so there is **one copy of the experiment code**; the
notebook only orchestrates and plots. It opens with a table of every change relative to the paper. Open it from this
`faithful/` folder (e.g. in DSMLP's Jupyter after copying the project there); it picks Qwen3-8B or the Qwen3-1.7B fallback from the hardware check and refuses to continue if neither fits. For long unattended runs use the batch job below, which executes the same stages.

## 4. Running on DSMLP

All steps use `dsmlp/` (shared with the baseline; see the parent `README.md` for the rsync/`workspace -l`/path
background). Faithful settings are the `FAITHFUL_*` block in `dsmlp.env`.

**One-time setup**

1. Copy the project to DSMLP (no GitHub needed), from your laptop in `Capstone/Papers`:
   `rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.DS_Store' --exclude='results*/' E-steer_2026 <username>@dsmlp-login.ucsd.edu:~/`
2. `ssh <username>@dsmlp-login.ucsd.edu`, then `cd ~/E-steer_2026/dsmlp && cp dsmlp.env.example dsmlp.env` and edit:
   `DSMLP_USERNAME`, `PROJECT_DIR_ON_DSMLP` (container path, normally `/home/<username>/private/E-steer_2026`),
   `FAITHFUL_GPU_CLASS` (≥ `medium`), `FAITHFUL_ACK` (see below), optionally `FAITHFUL_NRC_VAD_PATH`.
3. Check your storage quota (section 2).
4. Optional, GAIA only: get access to the gated `gaia-benchmark/GAIA` dataset on Hugging Face and put a token in
   the container's HF home (e.g. `huggingface-cli login` inside an interactive container). Never put a token in `dsmlp.env`.

**`FAITHFUL_ACK`** must list every substitution you accept. For the default stages:
`sae-qwen-scope,judge-backbone,alpha,subset-sizes` — add `smaller-model` to accept the Qwen3-1.7B fallback (harmless if the 8B tier is chosen), `label-handpicked` if you have no NRC lexicon, `agent-reconstruction`
if the agent stage is included, `gaia-no-tools` if GAIA is enabled. The job aborts with a list of what is missing.

**Submit** (from the login node):

```bash
cd ~/E-steer_2026/dsmlp
bash submit_faithful.sh                     # pilot profile by default; prints the run id
tail -f ~/E-steer_2026/results_faithful/logs/e-steer-faithful-<run_id>.log   # login-node path
```

Recommended order: (1) `FAITHFUL_PROFILE=pilot`, stages through `report` — confirms access, shows the measured tok/s and
the calibrated α; (2) inspect `resources_report.json`, `features/sae_check.json` (the SAE should reconstruct best at the
configured layer) and `calibration.json`; (3) `FAITHFUL_PROFILE=full`; (4) the `agent` stage alone
(`FAITHFUL_STAGES=agent,report`, same run id).

**Resume** after the 12 h limit or a failure: set `FAITHFUL_RUN_ID=<run_id>` and resubmit — finished (task, condition)
files are skipped, partial files are never written.

**Retrieve results** (on your laptop, from the project folder):

```bash
bash dsmlp/fetch_faithful_results.sh <run_id>        # -> results_faithful/<run_id>/ incl. REPORT.md and container.log
```

## 5. Outputs (never mixed with the baseline's)

On DSMLP: `${FAITHFUL_OUT_DIR}/<run_id>/` (default `<project>/results_faithful/<run_id>/`) and
`${FAITHFUL_OUT_DIR}/logs/e-steer-faithful-<run_id>.log`. Baseline outputs live elsewhere (`results/`, repo root). All of
`results_faithful/` is git-ignored.

```
config.json  fidelity.json  resources_report.json  calibration.json  calibration/  run.log
features/    features.json (latents, weights, stability, direction cosines)  directions.pt  sae_check.json
llm/<task>/<condition>.json      per-example responses, parsed answers, judge output, metrics
validation/<variant>/logiqa/…    Sec. 5.6
agent/<dataset>/<condition>.json
results_long.csv   figures/*.png   REPORT.md   (tables, plots, and the paper's quoted numbers beside ours)
```

## 6. Reading the results

- `REPORT.md` quotes the paper's reported numbers next to the same statistic computed here. Treat them as orientation:
  the SAE, sample sizes and judge differ, and with n in the tens, differences of a few points are noise. Match ≠ proof,
  mismatch ≠ failure.
- Check `features/features.json → diagnostics`: cosine between the V/A/D steering directions and the number of shared
  latents. The baseline's regression directions were near-collinear (|cos| > 0.98), so all three axes behaved
  identically; here they should be far from collinear if the feature selection did its job.
- `eq6_variant`, `stability_tau`, `enable_thinking` and α can all move results; they are config fields on purpose.
- HumanEval **executes model-generated code** in subprocesses inside the job container.
- HarmBench generations may contain harmful text; they are stored only under `llm/harmbench/` in the git-ignored run dir.

## 7. Verification status

| Check | Where | Result |
|---|---|---|
| 15 unit tests (SAE top-k/linear decode, Eq. 5 vs shortcut, Eq. 6 sign behavior, hook scaling/dtype/tuple output, stable-latent selection, variants, NRC parsing/pair selection, parsers, HumanEval sandbox, metrics, config/ack enforcement, agent parsing/metrics) | laptop CPU | pass |
| All five task loaders against the live datasets (schemas, filtering, sampling) | laptop | pass; surfaced and fixed LogiQA mirror quirks (NLI-style rows, Chinese-language rows, missing ids) |
| Real-model hook test (`ESTEER_MODEL_TESTS=1`): on an actual Qwen3-0.6B block output the hook adds exactly `alpha*||h||*delta/delta_max*u` and is fully removed afterwards | laptop CPU | pass |
| Smoke run: preflight → features → llm (all 5 tasks, 7 conditions) → validation (3 variants) → report, Qwen3-0.6B + random SAE, 40-60-token generations | laptop CPU, ~17 min | completed, exit 0; judge JSON parsed, HumanEval/LogiQA/MATH scoring, REPORT.md and figures written. Values are meaningless (untrained SAE, tiny model, truncated outputs) |
| Agent stage (first condition, 2 HotpotQA items; the other conditions were stopped, same code path) | laptop CPU, ~24 min/condition | completed: invalid plan → replan → execution → candidate selection → scoring all exercised. Not a measure of agent quality |
| Qwen3-8B, Qwen-Scope SAE, any GPU/DSMLP execution, preflight network checks from a pod | **not run** | needs your DSMLP account |
| Scientific validity of any steering result | **not assessed** | no real run exists |
