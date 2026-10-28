"""
E-STEER proof-of-concept (small-scale reproduction; runs on CPU or GPU).

This is a deliberately scaled-down, qualitative reproduction of the core
mechanism in "How Emotion Shapes the Behavior of LLMs and Agents: A
Mechanistic Study" (E-STEER), using a 0.5B model instead of Qwen3-8B + a
trained Sparse Autoencoder. The experiment design below (model, steering
mechanism, tasks, metrics) is unchanged from the original local-only
version; only environment-dependent settings (device, thread count, model
cache location, output directory) were made configurable so the same code
runs unmodified on a laptop CPU or on a DSMLP GPU pod. See README.md in
this folder for local and DSMLP run instructions.

What's faithfully reproduced:
  - VAD (Valence-Arousal-Dominance) as the emotion representation.
  - Representation-level steering via activation patching (additive
    intervention on a mid-layer residual stream), not just prompting.
  - Direction discovery via contrastive emotion pairs (the paper's
    "positive-negative contrastive procedure"), though we regress
    directly in residual space instead of a trained SAE's sparse latent
    space (we have no SAE checkpoint and no budget to train one here).
  - Evaluation across the valence / arousal / dominance axes independently,
    and a prompt-level vs. activation-steering linear-controllability
    comparison analogous to the paper's Table 1.

What's NOT reproduced (out of scope for this proof-of-concept):
  - Qwen3-8B, trained SAEs, the full 8-dataset battery, the 3-module agent.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Both a plain laptop install and DSMLP's scipy-ml-notebook image ship
# TensorFlow alongside PyTorch; letting `transformers` probe for TF at
# import time triggered a real protobuf/absl mutex deadlock during local
# testing. Disabling the TF/Flax backends avoids that in both environments.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
# Offline by default so a cached model always reproduces deterministically;
# set ESTEER_HF_OFFLINE=0 for a first run that needs to download weights
# (e.g. the first run on a fresh DSMLP home directory).
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("ESTEER_HF_OFFLINE", "1"))
# Where Hugging Face caches downloaded model weights. Left unset by default
# (falls back to HF's own ~/.cache/huggingface); set ESTEER_HF_HOME to park
# the cache under a DSMLP home directory instead, since container storage
# outside the home directory does not persist between jobs.
if os.environ.get("ESTEER_HF_HOME"):
    os.environ.setdefault("HF_HOME", os.environ["ESTEER_HF_HOME"])

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Defaults (identical to the original local-only values). Each can be
# overridden via CLI flag or ESTEER_* environment variable at runtime; see
# parse_args() below and README.md for the full list. ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 12            # mid-layer residual stream (model has 24 layers)
MAX_NEW_TOKENS = 32
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "auto"        # "auto" picks cuda if available, else cpu
NUM_THREADS = 2        # torch CPU thread count; ignored when device is cuda

torch.manual_seed(0)


def resolve_device(requested):
    """"auto" -> cuda if available else cpu; otherwise use the requested
    device as-is (lets a user force "cpu" on a GPU box, etc.)."""
    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def parse_args(argv=None):
    """CLI flags for the environment/infra knobs only (device, cache
    location, output directory, thread count). The experiment design
    itself -- model choice as a research setting, steering layer, VAD
    anchors, eval items, coefficients -- is not exposed here, to keep the
    experiment identical across environments; see README.md if you need
    to intentionally change the research design."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=os.environ.get("ESTEER_DEVICE", DEVICE),
                    help='"cpu", "cuda", or "auto" (default: %(default)s)')
    p.add_argument("--model-name", default=os.environ.get("ESTEER_MODEL_NAME", MODEL_NAME),
                    help="Hugging Face model id (default: %(default)s)")
    p.add_argument("--layer", type=int, default=int(os.environ.get("ESTEER_LAYER", LAYER)),
                    help="decoder layer index to steer (default: %(default)s)")
    p.add_argument("--max-new-tokens", type=int,
                    default=int(os.environ.get("ESTEER_MAX_NEW_TOKENS", MAX_NEW_TOKENS)),
                    help="generation length cap (default: %(default)s)")
    p.add_argument("--out-dir", default=os.environ.get("ESTEER_OUT_DIR", OUT_DIR),
                    help="where to write results.json etc. (default: this script's folder)")
    p.add_argument("--num-threads", type=int,
                    default=int(os.environ.get("ESTEER_NUM_THREADS", NUM_THREADS)),
                    help="torch CPU thread count, ignored on GPU (default: %(default)s)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 1. Emotion anchors in VAD space (approximate circumplex coordinates,
#    consistent with the positions sketched in the paper's Figure 2).
#    Scale: each axis in [-10, 10].
# ---------------------------------------------------------------------------
EMOTION_VAD = {
    "joy":      (8, 5, 5),
    "surprise": (4, 7, -2),
    "anger":    (-6, 6, 5),
    "disgust":  (-7, 3, 2),
    "fear":     (-7, 6, -6),
    "sadness":  (-6, -4, -6),
    "neutral":  (0, 0, 0),
}

DIAGNOSTIC_SENTENCES = [
    "The weather today is cloudy with occasional rain.",
    "I need to finish writing this report before tomorrow.",
    "The train was delayed by about twenty minutes.",
    "She picked up the book and started reading quietly.",
]


def load_model(model_name=None, device=None, num_threads=None):
    model_name = model_name or MODEL_NAME
    device = resolve_device(device or DEVICE)
    if device == "cpu":
        torch.set_num_threads(num_threads or NUM_THREADS)
    tok = AutoTokenizer.from_pretrained(model_name)
    # bfloat16 halves the resident footprint vs. float32, which matters a lot
    # on a memory-constrained laptop and keeps the same dtype (so the same
    # numbers) when moved to a GPU.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def get_layer_activation(tok, model, text, layer=None):
    """Mean-pooled residual-stream activation at `layer` for a prompt."""
    layer = LAYER if layer is None else layer
    device = next(model.parameters()).device
    ids = tok(text, return_tensors="pt").to(device)
    out = model(**ids, output_hidden_states=True)
    h = out.hidden_states[layer][0]  # [seq, hidden]
    return h.mean(dim=0).float().cpu().numpy()


@torch.no_grad()
def build_emotion_centroids(tok, model):
    """For each emotion label, average activation over diagnostic sentences
    prefixed with an emotional framing (contrastive-pairs procedure)."""
    centroids = {}
    for label in EMOTION_VAD:
        vecs = []
        for sent in DIAGNOSTIC_SENTENCES:
            prefix = f"Current emotional state: {label}. " if label != "neutral" else ""
            vecs.append(get_layer_activation(tok, model, prefix + sent))
        centroids[label] = np.mean(vecs, axis=0)
    return centroids


def fit_vad_directions(centroids):
    """Linear regression: activation ~ a + b_v*V + b_a*A + b_d*D.
    Returns unit directions for valence, arousal, dominance, and the
    typical residual-stream norm (for picking steering magnitudes)."""
    labels = list(centroids.keys())
    X = np.array([[1.0, *EMOTION_VAD[l]] for l in labels])   # [n, 4]
    Y = np.array([centroids[l] for l in labels])              # [n, hidden]
    # least squares solve for coefficients [a, b_v, b_a, b_d] per hidden dim
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)              # [4, hidden]
    b_v, b_a, b_d = coef[1], coef[2], coef[3]
    norms = {k: np.linalg.norm(v) for k, v in centroids.items()}
    typical_norm = float(np.mean(list(norms.values())))
    directions = {
        "valence": b_v / (np.linalg.norm(b_v) + 1e-8),
        "arousal": b_a / (np.linalg.norm(b_a) + 1e-8),
        "dominance": b_d / (np.linalg.norm(b_d) + 1e-8),
    }
    return directions, typical_norm


class Steerer:
    """Registers a forward hook on one decoder layer to additively patch
    the residual stream: h <- h + coeff * scale * unit_direction."""

    def __init__(self, model, layer, direction_vec, coeff, scale):
        self.handle = None
        self.layer = layer
        self.model = model
        self.vec = torch.tensor(direction_vec, dtype=torch.float32)
        self.coeff = coeff
        self.scale = scale

    def _hook(self, module, inputs, output):
        if self.coeff == 0:
            return output
        if isinstance(output, tuple):
            hs = output[0]
            delta = (self.coeff * self.scale * self.vec).to(dtype=hs.dtype, device=hs.device)
            hs = hs + delta
            return (hs,) + output[1:]
        else:
            delta = (self.coeff * self.scale * self.vec).to(dtype=output.dtype, device=output.device)
            return output + delta

    def __enter__(self):
        target = self.model.model.layers[self.layer]
        self.handle = target.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


@torch.no_grad()
def generate(tok, model, prompt, steerer=None, max_new_tokens=None):
    max_new_tokens = MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    device = next(model.parameters()).device
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
    attn_mask = torch.ones_like(ids)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.eos_token_id, attention_mask=attn_mask)
    if steerer is not None:
        with steerer:
            out = model.generate(ids, **gen_kwargs)
    else:
        out = model.generate(ids, **gen_kwargs)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 2. A small hand-written logical-reasoning set (LogiQA-style True/False),
#    standing in for the paper's LogiQA 2.0 (kept small + self-contained
#    for a CPU proof-of-concept rather than pulling the full benchmark).
# ---------------------------------------------------------------------------
LOGIC_ITEMS = [
    ("All engineers at the firm know Python. Maria is an engineer at the firm. "
     "Therefore, Maria knows Python.", True),
    ("No reptiles are warm-blooded. A snake is a reptile. Therefore, snakes are warm-blooded.", False),
    ("If it rains, the game is cancelled. It did not rain. Therefore, the game was cancelled.", False),
    ("Every student who passed the exam studied hard. Tom studied hard. Therefore, Tom passed the exam.", False),
    ("All squares are rectangles. All rectangles have four sides. Therefore, all squares have four sides.", True),
    ("Some birds cannot fly. Penguins are birds. Therefore, penguins cannot fly.", False),
    ("If a number is divisible by 6, it is divisible by 3. 18 is divisible by 6. Therefore, 18 is divisible by 3.", True),
    ("No mammals lay eggs, except monotremes. A platypus is a monotreme. Therefore, a platypus may lay eggs.", True),
    ("All members of the committee voted yes. Priya is not on the committee. Therefore, Priya voted yes.", False),
    ("If X is true then Y is true. Y is true. Therefore, X must be true.", False),
    ("All cats are mammals. All mammals are animals. Therefore, all cats are animals.", True),
    ("Either the package was lost or it was delivered late. It was not lost. Therefore, it was delivered late.", True),
    ("Every prime number greater than 2 is odd. 7 is a prime number greater than 2. Therefore, 7 is odd.", True),
    ("No fish can breathe air directly. A lungfish can breathe air directly. Therefore, a lungfish is not a fish in this strict sense.", True),
    ("All employees who are late get a warning. Sam got a warning. Therefore, Sam was late.", False),
    ("If the alarm is broken, it will not ring. The alarm rang. Therefore, the alarm is not broken.", True),
    ("All A are B. All B are C. Some C are D. Therefore, all A are D.", False),
    ("Any triangle with two equal sides is isosceles. Triangle T has two equal sides. Therefore, T is isosceles.", True),
    ("Only citizens can vote. Lee voted. Therefore, Lee is a citizen.", True),
    ("All metals conduct electricity. Rubber does not conduct electricity. Therefore, rubber is not a metal.", True),
]

LOGIC_PROMPT_TMPL = (
    "Determine whether the following argument's conclusion logically follows. "
    "Answer with exactly one word, True or False, then a one-sentence reason.\n\n{stmt}"
)


def parse_true_false(text):
    t = text.strip().lower()
    if t.startswith("true"):
        return True
    if t.startswith("false"):
        return False
    if "true" in t[:20] and "false" not in t[:20]:
        return True
    if "false" in t[:20] and "true" not in t[:20]:
        return False
    return None  # unparseable


def run_logic_eval(tok, model, directions, typical_norm, axis, coeffs, items=None, scale_frac=0.02):
    """Sweep one VAD axis (others held at 0) and measure accuracy on `items`."""
    items = items if items is not None else LOGIC_ITEMS
    vec = directions[axis]
    scale = typical_norm * scale_frac
    results = []
    for c in coeffs:
        steerer = Steerer(model, LAYER, vec, c, scale)
        correct = 0
        unparsed = 0
        raw = []
        for i, (stmt, label) in enumerate(items):
            t_item = time.time()
            resp = generate(tok, model, LOGIC_PROMPT_TMPL.format(stmt=stmt), steerer=steerer)
            pred = parse_true_false(resp)
            raw.append({"stmt": stmt, "gold": label, "resp": resp, "pred": pred})
            if pred is None:
                unparsed += 1
            elif pred == label:
                correct += 1
            print(f"    item {i+1}/{len(items)} done in {time.time()-t_item:.1f}s", flush=True)
        n_valid = len(items) - unparsed
        acc = correct / n_valid if n_valid else 0.0
        results.append({"axis": axis, "coeff": c, "accuracy": acc,
                         "n_unparsed": unparsed, "raw": raw})
        print(f"  [{axis} = {c:+d}] accuracy={acc:.3f} (unparsed={unparsed}/{len(items)})", flush=True)
    return results


# ---------------------------------------------------------------------------
# 3. Table-1-style comparison: prompt-level vs. activation-steering linear
#    controllability, measured by correlating the *intended* valence
#    coefficient with a simple lexicon-based valence score of the output.
# ---------------------------------------------------------------------------
POS_WORDS = {"happy", "joy", "great", "wonderful", "love", "glad", "excited",
             "hope", "bright", "good", "nice", "beautiful", "delight", "smile"}
NEG_WORDS = {"sad", "angry", "fear", "bad", "terrible", "awful", "hate",
             "worried", "dark", "cry", "pain", "upset", "hopeless", "afraid"}


def lexicon_valence(text):
    words = [w.strip(".,!?\"'").lower() for w in text.split()]
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


STORY_PROMPTS = [
    "Write two sentences continuing this story: 'The old dog walked into the kitchen and'",
    "Write two sentences continuing this story: 'On her way to school, Mia noticed'",
    "Write two sentences continuing this story: 'The letter arrived on a quiet afternoon, and'",
]

VALENCE_COEFFS = [-6, 0, 6]


def run_controllability_comparison(tok, model, directions, typical_norm, scale_frac=0.02):
    scale = typical_norm * scale_frac
    vec = directions["valence"]

    prompt_scores, steer_scores = [], []
    for c in VALENCE_COEFFS:
        # --- prompt-level induction ---
        label = {-6: "very sad and gloomy", -3: "a little sad", 0: "neutral",
                  3: "a little happy", 6: "very happy and joyful"}[c]
        prompt_vals = []
        for p in STORY_PROMPTS:
            resp = generate(tok, model, f"You are feeling {label}. " + p)
            prompt_vals.append(lexicon_valence(resp))
        prompt_scores.append(float(np.mean(prompt_vals)))

        # --- activation steering ---
        steerer = Steerer(model, LAYER, vec, c, scale)
        steer_vals = []
        for p in STORY_PROMPTS:
            resp = generate(tok, model, p, steerer=steerer)
            steer_vals.append(lexicon_valence(resp))
        steer_scores.append(float(np.mean(steer_vals)))
        print(f"  valence={c:+d}  prompt_score={prompt_scores[-1]:+.3f}  steer_score={steer_scores[-1]:+.3f}", flush=True)

    def corr(xs, ys):
        if np.std(ys) == 0:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])

    r_prompt = corr(VALENCE_COEFFS, prompt_scores)
    r_steer = corr(VALENCE_COEFFS, steer_scores)
    return {
        "coeffs": VALENCE_COEFFS,
        "prompt_scores": prompt_scores,
        "steer_scores": steer_scores,
        "r_prompt": r_prompt,
        "r_steer": r_steer,
    }


def main(args=None):
    global MODEL_NAME, LAYER, MAX_NEW_TOKENS, OUT_DIR

    args = args if args is not None else parse_args()
    MODEL_NAME = args.model_name
    LAYER = args.layer
    MAX_NEW_TOKENS = args.max_new_tokens
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    device = resolve_device(args.device)

    t0 = time.time()
    print(f"Loading model ({MODEL_NAME}) on device={device} ...")
    tok, model = load_model(model_name=MODEL_NAME, device=device, num_threads=args.num_threads)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nBuilding emotion centroids + fitting VAD directions...", flush=True)
    centroids = build_emotion_centroids(tok, model)
    directions, typical_norm = fit_vad_directions(centroids)
    print(f"  typical residual norm at layer {LAYER}: {typical_norm:.2f}", flush=True)

    coeffs = [-6, 0, 6]
    items = LOGIC_ITEMS[:5]
    all_logic_results = {}
    for axis in ["valence", "arousal", "dominance"]:
        print(f"\n=== Logic eval sweep: {axis} ===", flush=True)
        all_logic_results[axis] = run_logic_eval(tok, model, directions, typical_norm, axis, coeffs, items=items)

    print("\n=== Controllability comparison (prompt vs. steering) ===")
    ctrl = run_controllability_comparison(tok, model, directions, typical_norm)

    out = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "typical_norm": typical_norm,
        "logic_results": all_logic_results,
        "controllability": ctrl,
    }
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
