# Activation steering for decision advice

Research question: **Can activation steering make an LLM less likely to simply endorse a user's preferred decision?**

Open [advice_steering_pilot.ipynb](advice_steering_pilot.ipynb) in Jupyter or VS Code. This pilot uses Qwen2.5-0.5B-Instruct, 38 fictional scenarios across four domains, a contrastive activation direction, a development dose sweep, held-out controls, blinded manual rating export, and analysis plots.

## Start

Keep this folder intact, including `data/`. From this folder, install dependencies into the environment used by the notebook kernel:

```bash
python -m pip install -r requirements.txt
```

Select that environment's kernel; an existing `steering` kernel is suitable if these packages are installed in its environment. On DSMLP, preserve the site's CUDA-compatible PyTorch environment. The notebook does not install packages itself.

1. Run sections 1-4 in order to load the model and generate eight exploratory baseline responses. A first load may download weights; set `ALLOW_DOWNLOAD=False` before importing Transformers for a cached offline run.
2. Set `BUILD_DIRECTION=True`, rerun the configuration cell, and continue through sections 5-7 to construct the direction and verify the hook.
3. Set `RUN_DEVELOPMENT=True` and rerun configuration before section 8. The full sweep generates 60 responses. Choose a dose using reasoning and quality, then set `CHOSEN_DOSE` and `SELECTION_NOTES` to freeze the evaluation configuration. An ineffective direction is a valid outcome.
4. Set `RUN_FINAL=True` only after freezing. The final comparison generates 144 cached responses. Complete `outputs/ratings.csv` using the shuffled packet before running section 12.

The default output cap remains 192 tokens to match the saved pilot. Many saved answers reached that cap. Test a larger cap, such as `ADVICE_MAX_NEW_TOKENS=384`, on development cases before interpreting missing conclusions, and apply the same budget across compared conditions. CPU generation can be slow; individual completed responses are cached for interrupted reruns.

The notebook has no prefilled human ratings or claimed final outcomes. Routine generated outputs are ignored by Git. The examples are AI-assisted fictional data, not an externally validated benchmark. Agreement can be justified, and a small model may produce generic or poorly grounded advice.

## Preliminary pilot evidence

A [development-results note and preserved run](results/2026-11-06-development/README.md) document all 60 responses and a qualitative roommate-payment example. Fifty answers reached the token limit, grounding problems remain, and human ratings and held-out controls are pending. These are preliminary observations, not evidence of overall effectiveness.

The snapshot includes matching data, vector, metadata, and the run's archived algorithm source. Its prose is polished, but the algorithm and recorded outcomes are preserved. Use the main notebook above for continued work. The archive's code predates some of the current notebook's error-message improvements; the manifest records both identities. Routine `outputs/` files stay ignored.

## Portability

Optional environment variables: `ADVICE_PROJECT_DIR`, `ADVICE_OUTPUT_DIR`, `ADVICE_HF_HOME`, `ADVICE_DEVICE` (`auto`, `cpu`, `cuda`), `ADVICE_MODEL_ID`, `ADVICE_MODEL_REVISION`, `ADVICE_ALLOW_DOWNLOAD` (`1`/`0`), `ADVICE_CPU_THREADS`, `ADVICE_MAX_NEW_TOKENS`, and `ADVICE_LAYER`.

Keep notebook and data together. Configuration/output folders distinguish model and dataset identities. Changing a frozen configuration requires a separate output directory and, if prior results informed tuning, a fresh final test set. DSMLP account setup/job submission is not included.

## Mechanical validation

Install the optional validation packages, then run from the repository root:

```bash
python -m pip install pytest nbformat
python -m pytest Advice-steering_experiment/tests -q
```
These tests load notebook definitions and use tiny tensors; no model download is needed. They verify hook coverage/cleanup, split isolation, cache identity, and incomplete-rating handling. The notebook also contains real-model indexing and zero-dose checks.

## Included files

- `advice_steering_pilot.ipynb`: runnable experiment and method explanations.
- `data/`: exploratory, construction, development, and final-test JSON files.
- `requirements.txt`: experiment dependencies.
- `tests/`: mechanical validation without model downloads.
- `results/2026-11-06-development/`: selected preliminary evidence and provenance.

The steering method is a CAA-inspired adaptation using mean assistant-response activations. It is separate from the team's E-STEER and Pain Axis experiments.
