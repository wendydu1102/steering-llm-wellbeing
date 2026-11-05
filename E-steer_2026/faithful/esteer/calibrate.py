"""alpha calibration (alpha in Eq. 7 is not given in the paper) and the Sec. 5.6 latent validation."""
from __future__ import annotations

import dataclasses
import json
import os
import random
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


def calibrate_alpha(bundle, cfg, directions: Dict, run_dir: str, log=print) -> float:
    """Our rule (NOT the paper's): the largest alpha in alpha_grid such that, at |delta| = delta_max on every
    dimension and sign, the answer-validity rate stays >= alpha_validity_floor x the unsteered rate."""
    task, examples = calibration_examples(cfg)
    bundle.steering = None
    base = evaluate_condition(bundle, task, examples, "neutral", 0.0, cfg)["summary"]["AVR"]
    rows = []
    for alpha in cfg.alpha_grid:
        bundle.steering = EmotionSteering(directions, alpha, cfg.delta_max, cfg.eq6_variant)
        avr, tsr = {}, {}
        for dim in DIMS:
            for sign in (+1, -1):
                d = sign * cfg.delta_max
                s = evaluate_condition(bundle, task, examples, dim, d, cfg)["summary"]
                avr[cond_key(dim, d)], tsr[cond_key(dim, d)] = s["AVR"], s["TSR"]
        rows.append({"alpha": alpha, "AVR": avr, "TSR": tsr, "min_AVR": min(avr.values())})
        log(f"  alpha={alpha}: min AVR at |delta|={cfg.delta_max:g} = {rows[-1]['min_AVR']:.3f} "
            f"(unsteered {base:.3f})")
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
    bundle.steering = None
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
