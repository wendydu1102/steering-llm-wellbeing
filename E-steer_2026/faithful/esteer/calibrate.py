"""alpha calibration (alpha in Eq. 7 is not given in the paper) and the Sec. 5.6 latent validation."""
from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from typing import Dict

from .evaluate import cond_key, evaluate_condition, run_llm_stage
from .features import build_directions
from .steering import DIMS, EmotionSteering
from .tasks import REGISTRY


def calibration_examples(cfg):
    """LogiQA items disjoint from the evaluation sample, so calibration never touches reported items."""
    task = REGISTRY["logiqa"]
    eval_ids = {e.id for e in task.load(cfg.tasks.get("logiqa", 0), cfg.seed)} if "logiqa" in cfg.tasks else set()
    pool = [e for e in task.load(10 ** 9, 0) if e.id not in eval_ids]
    return task, random.Random(cfg.seed + 7).sample(pool, min(cfg.calibration_n, len(pool)))


def _fingerprint(cfg, examples, alpha=None) -> dict:
    """What a cached calibration condition depends on. A file whose fingerprint differs is recomputed, so
    changing calibration_n, delta_max, the model or the sampled items never reuses stale results."""
    return {"ids": [e.id for e in examples], "alpha": alpha, "delta_max": cfg.delta_max,
            "eq6_variant": cfg.eq6_variant, "model": getattr(cfg, "model_name", None),
            "layer": getattr(cfg, "layer", None), "max_new_tokens": cfg.max_new_tokens.get("logiqa")}


def _cached_condition(path, fp, resume, compute, log):
    """Run compute() unless a valid cached result exists; write atomically so a killed job leaves no half file."""
    if resume and os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("fingerprint") == fp:
                return d["summary"], True
        except (OSError, ValueError, KeyError):
            pass                                              # unreadable/stale -> recompute
    res = compute()
    out = {"fingerprint": fp, "summary": res["summary"], "records": res["records"]}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    return res["summary"], False


def calibrate_alpha(bundle, cfg, directions: Dict, run_dir: str, log=print) -> float:
    """Our rule (NOT the paper's): the largest alpha in alpha_grid such that, at |delta| = delta_max on every
    dimension and sign, the answer-validity rate stays >= alpha_validity_floor x the unsteered rate.

    Resumable: every condition (the unsteered baseline and each alpha x dimension x sign) is saved to
    <run_dir>/calibration/ as soon as it finishes; a rerun skips conditions whose saved fingerprint (items,
    alpha, delta_max, model, layer, token budget) still matches. calibration.json is written at the end."""
    task, examples = calibration_examples(cfg)
    cdir = os.path.join(run_dir, "calibration")
    os.makedirs(cdir, exist_ok=True)
    total = 1 + len(cfg.alpha_grid) * len(DIMS) * 2
    done = 0

    def step(name, fp, compute):
        nonlocal done
        t0 = time.time()
        summary, cached = _cached_condition(os.path.join(cdir, name + ".json"), fp, cfg.resume, compute, log)
        done += 1
        log(f"  calib [{done}/{total}] {name:<24} AVR={summary.get('AVR', float('nan')):.3f} "
            + ("(cached)" if cached else f"{time.time() - t0:.0f}s"))
        return summary

    bundle.steering = None
    base = step("neutral", _fingerprint(cfg, examples),
                lambda: evaluate_condition(bundle, task, examples, "neutral", 0.0, cfg))["AVR"]
    rows = []
    for alpha in cfg.alpha_grid:
        avr, tsr = {}, {}
        for dim in DIMS:
            for sign in (+1, -1):
                d = sign * cfg.delta_max

                def compute(dim=dim, d=d, alpha=alpha):
                    bundle.steering = EmotionSteering(directions, alpha, cfg.delta_max, cfg.eq6_variant)
                    return evaluate_condition(bundle, task, examples, dim, d, cfg)

                s = step(f"a{alpha:g}_{cond_key(dim, d)}", _fingerprint(cfg, examples, alpha), compute)
                avr[cond_key(dim, d)], tsr[cond_key(dim, d)] = s["AVR"], s.get("TSR")
        rows.append({"alpha": alpha, "AVR": avr, "TSR": tsr, "min_AVR": min(avr.values())})
        log(f"  alpha={alpha}: min AVR at |delta|={cfg.delta_max:g} = {rows[-1]['min_AVR']:.3f} "
            f"(unsteered {base:.3f})")
    bundle.steering = None
    ok = [r["alpha"] for r in rows if r["min_AVR"] >= cfg.alpha_validity_floor * base]
    chosen = max(ok) if ok else min(cfg.alpha_grid)
    out = {"rule": f"largest alpha with min AVR >= {cfg.alpha_validity_floor} x unsteered AVR",
           "neutral_AVR": base, "n_examples": len(examples), "rows": rows, "chosen_alpha": chosen,
           "no_alpha_met_floor": not ok}
    with open(os.path.join(run_dir, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    if not ok:
        log("  WARNING: no alpha met the validity floor; using the smallest grid value. "
            "Inspect calibration.json before trusting the sweep.")
    return chosen


def run_validation(bundle, sae, features: dict, cfg, alpha: float, run_dir: str, log=print) -> None:
    """Sec. 5.6: LogiQA accuracy under original / randomly selected / half-replaced latents."""
    vcfg = dataclasses.replace(cfg, tasks={"logiqa": cfg.validation_n})
    for variant in ("original", "random", "half"):
        log(f"validation: {variant} latents")
        dirs = build_directions(sae, features, variant, seed=cfg.seed)
        bundle.steering = EmotionSteering(dirs, alpha, cfg.delta_max, cfg.eq6_variant)
        run_llm_stage(bundle, vcfg, run_dir, log=log, subdir=os.path.join("validation", variant))
    bundle.steering = None
