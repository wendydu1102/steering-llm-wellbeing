"""Paper-vs-implementation fidelity matrix, and enforcement of explicit acknowledgement
for every component that is not paper-identical.

status:
  faithful        matches what the paper specifies
  interpretation  the paper is ambiguous/silent; a documented choice was made
  substituted     the paper's resource is unavailable; a named stand-in is used
  reconstructed   the paper describes the idea only; details are ours
  not_implemented not covered by this implementation
A component with an `ack` id is only run if that id is passed via --ack / ESTEER_ACK.
"""
from __future__ import annotations

from typing import Dict, List

MATRIX: List[Dict[str, str]] = [
    dict(id="model", status="faithful", ack="",
         paper="Qwen3-8B, greedy decoding (sampling disabled).",
         impl="Tier qwen3-8b: Qwen/Qwen3-8B, do_sample=False, chosen automatically when the GPU can hold it "
              "(bf16 weights 16.4 GB + SAE + headroom, ~21 GB).",
         note="Whether the paper used thinking mode and its maximum generation length are not stated: "
              "enable_thinking and per-task max_new_tokens are config fields (default: thinking off)."),
    dict(id="model-size", status="substituted", ack="smaller-model",
         paper="Qwen3-8B.",
         impl="Tier qwen3-1.7b (only if the GPU cannot hold the 8B model): Qwen/Qwen3-1.7B with Qwen-Scope "
              "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 at layer 13 (= round(17/36*28), the same relative depth).",
         note="A much weaker model: baselines are lower and the paper's effects may shrink or vanish, so its numbers "
              "are NOT comparable with the paper's. No Qwen3-4B tier: Qwen-Scope publishes no 4B SAE."),
    dict(id="precision", status="interpretation", ack="",
         paper="not stated (Qwen3 is released in bf16).",
         impl="bf16 on GPUs with native support (compute capability >= 8.0); otherwise fp32 if it fits, else fp16.",
         note="Pre-Ampere cards (Pascal/Volta/Turing) have no native bf16. fp16 can overflow in Qwen models."),
    dict(id="layer", status="interpretation", ack="",
         paper="SAE attached at the k=17th layer; steering at the output of the k-th block.",
         impl="Output of model.model.layers[17] (0-indexed, matching Qwen-Scope's per-layer files); layer 13 on the "
              "28-layer 1.7B tier.",
         note="1- vs 0-based indexing is not stated. The preflight stage reports SAE reconstruction "
              "quality at layers 16/17/18 so a wrong offset is visible."),
    dict(id="sae", status="substituted", ack="sae-qwen-scope",
         paper="A trained SAE (source, width, sparsity, training data and checkpoint not given or released).",
         impl="Qwen-Scope TopK SAE matching the model: 8B -> Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 layer17.sae.pt "
              "(64K latents, k=50); 1.7B -> Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 layer13.sae.pt (32K, k=50).",
         note="Same model family and layer, but trained on Qwen3-8B-Base while the experiments steer "
              "post-trained Qwen3-8B (the Qwen-Scope card says this transfer is 'reasonable'). It was "
              "released after the paper, so it cannot be the paper's SAE."),
    dict(id="labels", status="interpretation", ack="",
         paper="Emotion labels mapped to VAD coordinates in [-10, 10] (Fig. 2/3); mapping source not stated.",
         impl="NRC VAD Lexicon v2 (user-supplied file) x10; contrast words chosen to differ mainly on one "
              "dimension.",
         note="Without the lexicon file a hand-picked fallback is used (ack: label-handpicked)."),
    dict(id="label-fallback", status="substituted", ack="label-handpicked",
         paper="(see labels)",
         impl="Hand-assigned high/low pole words per dimension, no lexicon values.",
         note="Pairs are then not matched on the other two dimensions, so features may mix dimensions."),
    dict(id="features", status="interpretation", ack="",
         paper="Contrastive pairs fix the task description and vary only the emotion label; top-50 neurons "
               "by activation difference; keep those stable across pairs; done per dimension.",
         impl="Prefix 'Right now you are feeling <word>.'; SAE latents averaged over task-description "
              "tokens; per pair top-50 by |difference|; keep latents in the top-50 with a consistent sign in "
              ">= stability_tau of pairs (default 0.5).",
         note="Number of pairs/tasks, token aggregation and the stability rule are not given."),
    dict(id="steering", status="interpretation", ack="",
         paper="Eqs. 5-7: d_i = f_dec(z + delta_i u_i) - f_dec(z); d~_i = d_i/||d_i|| * ||h_k|| * delta_i/delta_max; "
               "h~ = h + alpha * sum d~_i; forward hook at block k.",
         impl="Same, with the linear-decoder shortcut (unit-tested against literal Eq. 5), ||h_k|| per token, "
              "delta_max = 9, and a SIGN-PRESERVING Eq. 6 (scale delta_i/delta_max).",
         note="Read literally, Eq. 6 makes +delta and -delta steer identically (sign cancels); "
              "eq6_variant='literal' reproduces that reading for comparison."),
    dict(id="alpha", status="substituted", ack="alpha",
         paper="alpha appears in Eq. 7 but its value is never given.",
         impl="alpha is set manually, or chosen by a stated rule on a held-out LogiQA calibration set: the "
              "largest alpha in alpha_grid whose answer-validity rate at |delta|=delta_max stays >= "
              "alpha_validity_floor x the unsteered rate.",
         note="This is our rule, not the paper's. The calibration table is saved with the results."),
    dict(id="sweep", status="faithful", ack="",
         paper="Each of valence/arousal/dominance swept over {-9,...,9} (step 3) with the others neutral.",
         impl="Same, 19 distinct conditions (shared neutral).", note=""),
    dict(id="objective-data", status="interpretation", ack="subset-sizes",
         paper="LogiQA 2.0, HumanEval, MATH; subsets 'sampled proportionally based on specific criteria'; "
               "sizes not given.",
         impl="Real datasets (datatune/LogiQA2.0 test, openai/openai_humaneval, EleutherAI/hendrycks_math test, "
              "stratified by level). Sizes are config fields.",
         note="Any n smaller than the full dataset is a subset; the profile's sizes are recorded with results."),
    dict(id="subjective", status="substituted", ack="judge-backbone",
         paper="TinyStories; relevance/coherence/creativity/conciseness 'evaluated by the LLM' (judge unnamed).",
         impl="TinyStories validation openings continued by the model (as in Appendix A.1); scored 1-5 by "
              "the UNSTEERED backbone model.",
         note="Self-judging by the same model family may bias scores."),
    dict(id="safety", status="substituted", ack="judge-backbone",
         paper="HarmBench; probability of harmful / biased / hallucinatory outputs (judge unnamed).",
         impl="HarmBench 'standard' text behaviors (GitHub CSV); judged by the UNSTEERED backbone model with a "
              "per-category rubric.",
         note="Contextual and copyright behaviors are excluded. HarmBench's own classifier covers "
              "'harmful' only, so it cannot give the paper's three categories."),
    dict(id="metrics", status="interpretation", ack="",
         paper="AVR (parsable answer within max length), TSR (pass@1 / accuracy), judge quality, safety risk, "
               "fluctuation range (max-min)/average.",
         impl="AVR = answer extractable (code parses; \\boxed present; letter present; story completed; "
              "safety: substantive non-refusal). TSR over all items (invalid = wrong); TSR over valid also saved.",
         note="Paper does not define AVR for subjective/safety tasks precisely."),
    dict(id="validation", status="faithful", ack="",
         paper="Sec. 5.6: original vs random vs half-replaced latents on LogiQA 2.0.",
         impl="Same three feature sets, same sweep, LogiQA.", note=""),
    dict(id="agent", status="reconstructed", ack="agent-reconstruction",
         paper="Planner/decider/executor agent on HotpotQA, Scientific (CAMEL), GAIA; module and system metrics.",
         impl="Prompts, JSON protocols, confidence elicitation and candidate generation are ours; steering is "
              "applied to every LLM call; no external tools (LLM-based analysis only).",
         note="Module metrics are defined as in Sec. 5.1; their exact computation in the paper is not given."),
    dict(id="gaia", status="substituted", ack="gaia-no-tools",
         paper="GAIA included in the agent benchmark (agent has tool use).",
         impl="Optional, text-only questions without attachments; needs HF_TOKEN with accepted GAIA terms; "
              "no web/file tools.",
         note="Off by default."),
    dict(id="appendix-a1", status="not_implemented", ack="",
         paper="Table 1: linear controllability vs prompting via a trained BERT VAD analyzer (EmoBank + NRC retrieval).",
         impl="-", note="Analyzer architecture/training details are underspecified; not attempted."),
    dict(id="appendix-a2-a4", status="not_implemented", ack="",
         paper="Robustness: other datasets (ProntoQA, MBPP+, PHYBench, WritingPrompts, JailbreakBench), "
               "sampling, and gpt-oss-20B.",
         impl="-", note="Out of scope for this pass; the task registry is the extension point."),
]


def describe_ack(ack_id: str) -> str:
    for row in MATRIX:
        if row["ack"] == ack_id:
            return f"{row['paper']} -> {row['impl']}"
    return ack_id


def required_acks(cfg, stages: List[str], lexicon_available: bool) -> List[str]:
    need: List[str] = ["subset-sizes"]
    if cfg.sae_source == "qwen-scope":
        need.append("sae-qwen-scope")
        if cfg.model_name != "Qwen/Qwen3-8B":
            need.append("smaller-model")
    else:
        need.append("random-sae-smoke-only")
    if any(t in cfg.tasks for t in ("tinystories", "harmbench")) or "science" in cfg.agent_datasets:
        need.append("judge-backbone")
    need.append("alpha")
    if not lexicon_available:
        need.append("label-handpicked")
    if "agent" in stages:
        need.append("agent-reconstruction")
        if "gaia" in cfg.agent_datasets:
            need.append("gaia-no-tools")
    return need


def check_acks(cfg, stages: List[str], lexicon_available: bool) -> None:
    need = required_acks(cfg, stages, lexicon_available)
    missing = [a for a in need if a not in set(cfg.ack)]
    if missing:
        lines = ["This run uses components that are NOT paper-identical. Acknowledge each one explicitly",
                 "(--ack a,b,c or ESTEER_ACK=a,b,c) after reading faithful/README.md:"]
        for a in missing:
            lines.append(f"  - {a}: {describe_ack(a) if a != 'random-sae-smoke-only' else 'untrained random SAE: plumbing test only, results are meaningless'}")
        raise SystemExit("\n".join(lines))
