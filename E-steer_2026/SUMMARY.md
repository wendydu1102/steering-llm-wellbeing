# E-STEER proof-of-concept — results

A small-scale, CPU-only reproduction of the core mechanism in *"How Emotion
Shapes the Behavior of LLMs and Agents: A Mechanistic Study"* (E-STEER).
Model: `Qwen/Qwen2.5-0.5B-Instruct` (bf16), steering applied additively at
decoder layer 12/24. Typical residual-stream norm at that layer: 113.7.

**What matches the paper's method:** VAD as the emotion representation,
direction discovery via contrastive emotion-labeled prompts, steering via
direct activation patching (not just prompting), and measuring task
accuracy as a function of steering coefficient.

**What's simplified:** no Sparse Autoencoder (directions are fit by linear
regression of 7 emotion-centroid activations onto their VAD coordinates,
in raw residual space); one small 5-item logic-reasoning subset instead of
LogiQA 2.0; 3 coefficient levels instead of a dense sweep; no SAE
validation, no safety/creative/agent tasks, no multi-dataset robustness
checks.

## 1. Logic-reasoning accuracy under VAD steering

| axis | coeff −6 | coeff 0 | coeff +6 |
|---|---|---|---|
| valence | 40% | 80% | 40% |
| arousal | 40% | 80% | 40% |
| dominance | 40% | 80% | 40% |

(5-item subset, so accuracy only resolves in 20-point steps.)

**Qualitative echo of the paper:** every axis shows an inverted-U —
accuracy peaks at no steering and *drops* when pushed to either extreme.
This is broadly consistent with the paper's repeated finding that
moderate/neutral emotional states often outperform strongly-steered ones
on objective reasoning tasks, and that steering too hard in either
direction degrades performance.

**An honest limitation this run surfaced:** the three per-axis directions
turned out to be nearly collinear — cosine similarity |cos| > 0.98 between
every pair (valence vs. arousal: −0.997; valence vs. dominance: +0.987;
arousal vs. dominance: −0.994). That's why all three axes produce *exactly*
the same per-item right/wrong pattern (`YNNNY` at ±6, `YYYYN` at 0) despite
the generated text differing. In other words, this simplified method
mostly recovered a single "how hard are we perturbing the model" axis, not
three independently-controllable emotional dimensions. The paper avoids
this by training a Sparse Autoencoder with many more contrastive pairs and
much higher-dimensional sparse features — the method this proof-of-concept
swapped in (least-squares regression from 7 emotion-centroid points in an
896-dim space) is underdetermined and can't guarantee orthogonality the
way a trained SAE's dictionary can. This is the single biggest
methodological gap between this demo and the real result.

## 2. Prompt-level vs. activation-steering linear controllability

(Analogous to the paper's Table 1 comparison.)

| valence coeff | −6 | 0 | +6 | correlation (r) |
|---|---|---|---|---|
| prompt-level | +0.00 | +0.00 | +0.00 | 0.000 |
| activation steering | +0.00 | +0.00 | +0.00 | 0.000 |

This metric came back flat for both methods — inconclusive, not a
negative result. The lexicon-based valence scorer (a ~14-word positive/
negative word list) found zero matches in the generated two-sentence story
continuations; three short continuations per coefficient is too little
text, and a hand-written word list is too sparse, to pick up a reliable
signal. The paper's own version of this measurement uses a trained
BERT-based VAD regressor over much larger generations — reproducing that
faithfully would need a sentiment/VAD model, not a word list.

## Bottom line

This demo shows the *mechanism* works end-to-end (contrastive-pair
direction discovery → activation patching → measurable behavioral shift)
and reproduces the paper's qualitative shape (inverted-U, degradation at
emotional extremes) on a toy reasoning set. It does **not** demonstrate
that the paper's three VAD dimensions are independently controllable with
this simplified approach — that specifically requires the trained-SAE
machinery the paper uses, which this CPU proof-of-concept could not
replicate.
