# AI use: important prompts

Important prompts used while building this project, from two tools: **Claude Code** (writing code and the report) and
**ChatGPT** (shaping the replication and the scope). This is a structured selection, not a transcript of every message.

How to read it:

- **Verbatim** items are quoted from the saved Claude Code session transcripts. Chinese prompts are given with an English
  translation. `[...]` marks text shortened for length. `P<n>` is the prompt's position in the Claude Code transcript.
- **Paraphrased** items come from the ChatGPT history, which kept exact wording only for some short prompts and
  summarized the longer exchanges. They are labeled so they are not mistaken for verbatim prompts.
- Models: Claude Code with `claude-opus-5-5` and `claude-sonnet-5-5`; the ChatGPT model is not recorded in the history.

## 1. Important prompts

### 1.1 Shaping the replication (ChatGPT, October 28-30, 2026) — paraphrased

- Proposed using SAE, RFM, and difference-in-means to derive or compare steering directions for reproducing E-STEER.
- Asked whether dataset and material collection was required, and how to find the inputs needed for the experiment.
- Asked for the adapted implementation's differences from the paper and an explicit account of its scope cuts.
- Directed that the DSMLP experiment stay as close to the paper as feasible, and that blockers and fallback choices be
  explained rather than silently substituting unavailable models, SAE checkpoints, datasets or evaluation methods. (The
  same requirement appears verbatim in P17 below.)

### 1.2 Baseline and repository setup (Claude Code, October 28, 2026) — verbatim

**P0**, the project's first prompt, with the paper PDF (arXiv 2604.00005) attached:
> replicate this paper's result

**P4-P5**, repository setup (abridged; the original has a stray character after each sentence):
> Please initialize Git in the current capstone root folder, using main as the default branch. Check the existing files
> and create a suitable .gitignore for this Python research project, excluding virtual environments, caches, secrets, and
> large generated outputs. Keep all existing files unchanged, and don't create a commit yet. [...]
>
> Then check git status and review the files to be committed, making sure there are no secrets, virtual environments, or
> large generated files. [...] create the initial commit with the message Initial project setup.

**P6**, local and DSMLP (abridged):
> Please inspect the paper replication project under Papers/ and adapt it to run both locally and on UCSD DSMLP. First
> identify its current entry point, dependencies, data paths, and output paths, then consult the official DSMLP
> documentation or existing project configuration. Don't guess the queue, partition, job scheduler commands, or storage
> paths. Keep the model and experiment code shared between local and DSMLP runs [...] Don't create a separate copy of the
> model implementation. [...] If you need my DSMLP account, queue, or data storage path, expose each as a configuration
> setting and clearly mark where I need to fill it in rather than inventing a value.

### 1.3 Paper-faithful version (Claude Code, October 30 - November 5, 2026) — verbatim

**P17 (October 30)**, the prompt that created `Papers/E-steer_2026/faithful/`:
> Please revise this project so the version intended to run on DSMLP reproduces the E-STEER paper as closely as
> feasible. The current code is a small PoC that uses Qwen2.5-0.5B, regression over a few emotion-labeled prompts, and
> five logic questions; simply making that same experiment run on a GPU is not enough for this goal.
>
> First inspect the paper and the existing project files. Then identify what is needed to match the paper's
> experimental design, including the Qwen3-8B model and layer, the trained SAE and its VAD feature-selection procedure,
> the steering method and coefficient sweep, and the reported evaluation tasks and metrics. Implement the closest
> faithful version that can run within DSMLP's available resources.
>
> Do not silently replace unavailable models, SAE checkpoints, datasets, or evaluation methods with toy substitutes.
> Check whether the required resources are available; if anything is missing or infeasible, explain the blocker,
> distinguish implemented components from paper-faithful components, and propose the closest transparent fallback before
> using it. [...] Preserve the current PoC as a clearly labeled baseline if practical.

Why it matters: this prompt is the source of the "no silent substitutions" design. The fidelity matrix, the
acknowledgement flags and the "not run" markers in `faithful/README.md` exist because of it.

**P18 (November 5)**, notebook and fidelity (translated from Chinese and English):
> Can you write the code in a Jupyter notebook? Use this format of hardware check [code block pasted]. Reproduce the
> experiments in the paper as closely as possible, don't use the local version, and tell me what you changed.

### 1.4 Replication obstacles (ChatGPT, November 2, 2026) — paraphrased

- Identified an obstacle: no ready-to-use ICE was available for the model and setup, so the options were to construct
  one or bypass it, while recognizing that bypassing it could reduce fidelity.
- Considered whether steering could be used for cognitive reappraisal, and how to reproduce the RESORT framework's six
  appraisal dimensions.

### 1.5 Hardware and resumability (Claude Code, November 5-6, 2026) — verbatim

**P19**, after pasting the hardware report (GTX 1080 Ti, 11.7 GB, "VRAM check for Qwen3-8B (need >= 22 GB): TOO SMALL"):
> If this is the hardware that is most available to me, can we choose a smaller model that can fit in the resources? How
> will the result be impacted by this? How long will it take to run? State your plan for now, do not modify anything

**P21** (translated from Chinese):
> Can you check the hardware first and then pick the model by the GPU that is available? Keep the code for both models.

**P32** (translated from Chinese):
> Change it so calibration saves each alpha separately and can resume after an interruption.

### 1.6 Q1 scope (ChatGPT and Claude Code, November 5-6, 2026)

ChatGPT, paraphrased:

- Asked to distinguish the original E-STEER method from the implementation actually reproduced, including its
  limitations.
- Reported the DSMLP constraints that affect implementation choices: the paper's SAE was unavailable, Qwen3-8B exceeded
  the available GPU memory, and alpha calibration was slow. These were described as solvable but time-consuming, with no
  need to resolve every issue for the checkpoint.
- Left open whether the group should continue with E-STEER or choose a different paper for the Q1 project.

Claude Code, verbatim:

**P38 (November 6)**, with the course's Q1 assignment page linked:
> for the purpose of Q1 project, how can we scope down the reproduction

**P42**, the human pasted a proposed scope (Qwen3-1.7B, one VAD dimension, LogiQA only, delta in {-9,-3,0,3,9}, alpha
fixed on a separate sample) and asked (translated from Chinese):
> How about this?

**P43-P44**, the human added the five-step method chain to keep and set the key constraint (translated from Chinese):
> Now I want to keep the five steps I just described while cutting the other experiments and steps in the paper. Right
> now we need to define the scope, not change the code. Use the lexicon, not hand-picked words.

**P46** (translated from Chinese), which produced the alpha provenance section of `SCOPE_Q1.md`:
> I'm running the pilot now. Use the calibration result to set alpha. The scope can either fix alpha or calibrate it;
> default to a fixed value from a run that has been done once (but it has to be run once with the lexicon, right?). Put
> this in the scope and state where the fixed alpha comes from.

**P48-P49**, the human checking what the data actually was (translated from Chinese): "How do I know whether it is the
hand-picked word list? The pilot is the current version." Then, after the assistant's explanation: "It is hand-picked."

**P51**, confirming the scope (translated from Chinese): "I agree with the first five items in section 3. The sixth
can be decided after we have the lexicon." This is why the alpha rule is marked deferred in `SCOPE_Q1.md`.

### 1.7 Report (Claude Code, November 6, 2026) — verbatim

**P54:** `write a checkpoint report`

**P58**, with the checkpoint requirements pasted in:
> leave the details, only write what was asked here

**P62, P66:** add the mentors and authors to the `.tex`; `你来吧` ("go ahead") to the offer to verify the `[verify]`
citations against their sources.

**P68:** `make sure that table 1 appears before references`

## Coverage note

This is a focused selection of prompts, not a transcript of every message in every project conversation. The ChatGPT
history retained exact wording for some short prompts only, so those entries are paraphrases. The Claude Code
entries are quoted from the saved session transcripts; about 40 of the roughly 70 messages are left out, mostly
short follow-ups and administrative ones (commit, push, "explain", screenshots, pasted logs).
