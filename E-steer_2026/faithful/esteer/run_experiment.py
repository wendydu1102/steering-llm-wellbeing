"""Orchestrates the paper-faithful E-STEER pipeline in resumable stages.

  python -m esteer.run_experiment --profile pilot --stages preflight,features,calibrate,llm,validation,agent,report \\
      --ack sae-qwen-scope,judge-backbone,alpha,subset-sizes,label-handpicked,agent-reconstruction

Outputs go to <out_root>/<run_id>/ (never next to the PoC results). Re-running with the same --run-id resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from .config import load_config
from .fidelity import MATRIX, check_acks, required_acks
from .resources import blocked, run_preflight

STAGES = ["preflight", "features", "calibrate", "llm", "validation", "agent", "report"]
MODEL_STAGES = {"features", "calibrate", "llm", "validation", "agent"}
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SANITY_TEXTS = [
    "The committee met on Tuesday to review the annual budget and approve the schedule for next year.",
    "To install the package, first download the archive and then run the setup script from a terminal.",
    "Water boils at a lower temperature at high altitude because the surrounding air pressure is reduced.",
    "She walked to the market, bought some vegetables, and returned home before the rain started.",
    "The function takes a list of integers and returns the sum of every second element.",
    "In 1998 the company opened its first overseas office and began hiring local engineers.",
    "A triangle with two equal sides is called isosceles, and its base angles are equal.",
    "The museum's new exhibition features paintings from several European artists of the nineteenth century.",
]


def make_logger(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "a", buffering=1)

    def log(msg: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
    return log


def load_sae(cfg, device):
    from .sae import TopKSAE
    if cfg.sae_source == "random":
        from transformers import AutoConfig
        d_model = AutoConfig.from_pretrained(cfg.model_name).hidden_size
        return TopKSAE.random(d_model, cfg.sae_random_d_sae, cfg.sae_k, seed=cfg.seed, device=device)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(cfg.sae_repo, cfg.sae_file_pattern.format(layer=cfg.layer))
    return TopKSAE.from_file(path, cfg.sae_k, device=device)


def stage_features(bundle, cfg, run_dir, log):
    from .features import direction_diagnostics, build_directions, run_feature_identification, sae_layer_check
    from .tasks import REGISTRY
    fdir = os.path.join(run_dir, "features")
    os.makedirs(fdir, exist_ok=True)
    sae = load_sae(cfg, bundle.device)
    check = sae_layer_check(bundle, sae, SANITY_TEXTS)
    log(f"SAE reconstruction (fraction of variance explained) by hooked layer: {check} "
        f"(configured layer {cfg.layer}; the SAE's true hook point should score highest)")
    with open(os.path.join(fdir, "sae_check.json"), "w") as f:
        json.dump({"fve_by_layer": check, "configured_layer": cfg.layer, "sae_source": cfg.sae_source}, f, indent=2)
    feats = run_feature_identification(bundle, sae, cfg, REGISTRY, log)
    dirs = build_directions(sae, feats, "original")
    feats["diagnostics"] = direction_diagnostics(dirs, feats)
    log(f"direction cosines: {feats['diagnostics']['cosine']}; shared latents: {feats['diagnostics']['shared_latents']}")
    with open(os.path.join(fdir, "features.json"), "w") as f:
        json.dump(feats, f, indent=2)
    torch.save(dirs, os.path.join(fdir, "directions.pt"))
    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_directions(run_dir):
    return torch.load(os.path.join(run_dir, "features", "directions.pt"), map_location="cpu", weights_only=True)


def get_alpha(cfg, run_dir):
    if cfg.alpha is not None:
        return cfg.alpha
    p = os.path.join(run_dir, "calibration.json")
    if os.path.exists(p):
        return json.load(open(p))["chosen_alpha"]
    raise SystemExit("alpha is unset and no calibration.json exists: run the 'calibrate' stage or pass --set alpha=<value>")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=os.environ.get("ESTEER_PROFILE", "pilot"),
                    help="profile name in configs/ (smoke|pilot|full) or a path to a JSON file")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--run-id", default=os.environ.get("ESTEER_RUN_ID"))
    ap.add_argument("--out-root", default=os.environ.get("ESTEER_OUT_ROOT"))
    ap.add_argument("--ack", default=os.environ.get("ESTEER_ACK", ""), help="comma-separated acknowledged fallbacks")
    ap.add_argument("--nrc-vad-path", default=os.environ.get("ESTEER_NRC_VAD_PATH"))
    ap.add_argument("--device", default=os.environ.get("ESTEER_DEVICE"))
    ap.add_argument("--model-tier", default=os.environ.get("ESTEER_MODEL_TIER", "auto"),
                    help="auto (choose by GPU: Qwen3-8B if it fits, else Qwen3-1.7B) | qwen3-8b | qwen3-1.7b | none "
                         "(use the profile's own model). Profiles that name their own model, e.g. smoke, ignore auto.")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override any config field")
    args = ap.parse_args(argv)

    stages = [s for s in args.stages.split(",") if s]
    bad = set(stages) - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stages: {sorted(bad)}; valid: {STAGES}")
    prof = args.profile if os.path.exists(args.profile) else os.path.join(PKG_DIR, "..", "configs", f"{args.profile}.json")
    from .hardware import resolve
    pre_log = []                                    # hardware decisions are printed now and re-logged into run.log
    tier, tier_kw, hw, rec = resolve(args.model_tier, "model_name" in json.load(open(prof)),
                                     bool(MODEL_STAGES & set(stages)), lambda m: (print(m, flush=True), pre_log.append(m)))
    cfg = load_config(prof, args.set, model_tier=tier, run_id=args.run_id, out_root=args.out_root, device=args.device,
                      nrc_vad_path=args.nrc_vad_path, ack=[a for a in args.ack.split(",") if a], **tier_kw)
    os.makedirs(cfg.run_dir, exist_ok=True)
    log = make_logger(os.path.join(cfg.run_dir, "run.log"))
    log(f"run dir: {os.path.abspath(cfg.run_dir)}  profile={cfg.profile}  stages={stages}")
    for m in pre_log:
        log(m)
    log(f"model: {cfg.model_name} ({cfg.model_dtype}), layer {cfg.layer}, SAE {cfg.sae_repo if cfg.sae_source == 'qwen-scope' else 'random'}, "
        f"batch {cfg.batch_size}" + (f", tier {cfg.model_tier}" if cfg.model_tier else ""))
    with open(os.path.join(cfg.run_dir, "hardware.json"), "w") as f:
        json.dump({"hardware": hw, "requested_tier": args.model_tier, "chosen_tier": tier,
                   "recommendation": None if rec is None else {**rec.__dict__}}, f, indent=2, default=str)

    if MODEL_STAGES & set(stages):
        check_acks(cfg, stages, lexicon_available=bool(cfg.nrc_vad_path))
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(cfg.run_dir, "fidelity.json"), "w") as f:
        json.dump({"matrix": MATRIX, "acknowledged": cfg.ack,
                   "required": required_acks(cfg, stages, bool(cfg.nrc_vad_path))}, f, indent=2)
    torch.manual_seed(cfg.seed)

    if "preflight" in stages:
        log("preflight:")
        report = run_preflight(cfg, log)
        bl = blocked(report)
        if bl and any(s in MODEL_STAGES for s in stages):
            raise SystemExit("preflight found blockers; fix them or drop the affected tasks:\n"
                             + "\n".join(f"  - {i['name']}: {i['detail']}" for i in bl))

    bundle = None
    if MODEL_STAGES & set(stages):
        from .model import ModelBundle
        log(f"loading {cfg.model_name} ({cfg.model_dtype}) ...")
        t0 = time.time()
        bundle = ModelBundle(cfg)
        log(f"  loaded on {bundle.device} in {time.time() - t0:.0f}s")

    if "features" in stages:
        log("stage: features")
        stage_features(bundle, cfg, cfg.run_dir, log)

    from .steering import EmotionSteering
    dirs = None
    if MODEL_STAGES & set(stages) - {"features"}:
        dirs = load_directions(cfg.run_dir)

    if "calibrate" in stages:
        from .calibrate import calibrate_alpha
        if cfg.alpha is not None:
            log(f"alpha fixed by config: {cfg.alpha} (calibration skipped)")
        else:
            log("stage: calibrate (alpha)")
            alpha = calibrate_alpha(bundle, cfg, dirs, cfg.run_dir, log)
            log(f"  chosen alpha = {alpha}")

    def arm():
        alpha = get_alpha(cfg, cfg.run_dir)
        bundle.steering = EmotionSteering(dirs, alpha, cfg.delta_max, cfg.eq6_variant)
        return alpha

    if "llm" in stages:
        from .evaluate import run_llm_stage
        log(f"stage: llm (alpha={arm()})")
        run_llm_stage(bundle, cfg, cfg.run_dir, log)
    if "validation" in stages:
        from .calibrate import run_validation
        log("stage: validation (Sec. 5.6)")
        sae = load_sae(cfg, "cpu")
        feats = json.load(open(os.path.join(cfg.run_dir, "features", "features.json")))
        run_validation(bundle, sae, feats, cfg, get_alpha(cfg, cfg.run_dir), cfg.run_dir, log)
    if "agent" in stages:
        from .agent import run_agent_stage
        log(f"stage: agent (alpha={arm()})")
        run_agent_stage(bundle, cfg, cfg.run_dir, log)
    if "report" in stages:
        from .report import build_report
        log("stage: report")
        build_report(cfg.run_dir, log)
    if bundle is not None:
        log(f"measured generation throughput: {bundle.throughput():.0f} tok/s "
            f"({bundle.tokens_generated:,} tokens)")
    log("done")


if __name__ == "__main__":
    main(sys.argv[1:])
