"""Preflight: verify, before any GPU time is spent, that every resource the run needs is reachable,
and estimate its cost. Writes resources_report.json. Never prints secrets (only whether HF_TOKEN is set)."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import urllib.request
from typing import Any, Dict, List

# GPU budget figures from https://dsmlp-docs.ucsd.edu/gpu-access/service-units-and-budgets/ (Fall 2026 draft
# page: "Confirm any figure with datahub@ucsd.edu"). SU per GPU-hour, full rate.
SU_PER_GPU_HOUR = {"extra-small": 0.25, "small": 0.5, "medium": 1.0, "large": 2.0, "extra-large": 4.0}
DEFAULT_WEEKLY_SU = 10
# Approximate class memory from https://dsmlp-docs.ucsd.edu/gpu-access/gpu-classes/
GPU_CLASS_GB = {"extra-small": 6, "small": 12, "medium": 24, "large": 48, "extra-large": 96}
GITHUB_HARMBENCH = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/"
                    "behavior_datasets/harmbench_behaviors_text_all.csv")
DATASETS = {"logiqa": "datatune/LogiQA2.0", "humaneval": "openai/openai_humaneval",
            "math": "EleutherAI/hendrycks_math", "tinystories": "roneneldan/TinyStories",
            "hotpotqa": "hotpotqa/hotpot_qa", "science": "camel-ai/physics", "gaia": "gaia-benchmark/GAIA"}
# ASSUMED aggregate decode throughput range for batched HF generate with an 8B model on one 24-48 GB GPU.
# These are planning guesses, NOT measurements; the run logs measured tok/s so the estimate can be corrected.
ASSUMED_TOK_PER_S = (200, 500)


def _item(items: List[dict], name: str, status: str, detail: str) -> None:
    items.append({"name": name, "status": status, "detail": detail})


def _http_ok(url: str, method: str = "HEAD") -> bool:
    try:
        req = urllib.request.Request(url, method=method, headers={"User-Agent": "esteer-preflight"})
        with urllib.request.urlopen(req, timeout=15):
            return True
    except Exception:
        return False


def estimate_cost(cfg) -> Dict[str, Any]:
    from .evaluate import conditions
    n_cond = len(conditions(cfg))
    gen = sum(n * cfg.max_new_tokens[t] for t, n in cfg.tasks.items()) * n_cond
    judge = sum(n for t, n in cfg.tasks.items() if t in ("tinystories", "harmbench")) * n_cond * cfg.judge_max_new_tokens
    calib = cfg.calibration_n * cfg.max_new_tokens.get("logiqa", 512) * (1 + 6 * len(cfg.alpha_grid))
    valid = 3 * cfg.validation_n * cfg.max_new_tokens.get("logiqa", 512) * n_cond
    total = gen + judge + calib + valid                      # agent stage not included (needs dataset-specific counts)
    hours = [total / r / 3600 for r in ASSUMED_TOK_PER_S[::-1]]
    return {"conditions": n_cond, "generated_tokens_upper_bound": total,
            "assumed_tok_per_s": list(ASSUMED_TOK_PER_S),
            "gpu_hours_range_upper_bound": [round(h, 1) for h in hours],
            "su_range_by_class": {c: [round(h * r, 1) for h in hours] for c, r in SU_PER_GPU_HOUR.items()
                                  if c in ("medium", "large")},
            "default_weekly_su": DEFAULT_WEEKLY_SU,
            "note": "Upper bound at max_new_tokens for every generation, excluding the agent stage; "
                    "throughputs are assumptions, not measurements."}


def run_preflight(cfg, log=print) -> Dict[str, Any]:
    import torch
    import transformers
    items: List[dict] = []
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "cuda_available": torch.cuda.is_available(),
           "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))}
    from .hardware import native_bf16, requirement_for
    req = requirement_for(cfg.model_name, cfg.model_dtype)          # None for models without sizing data
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        vram_gb = p.total_memory / 1e9
        env["gpu"] = {"name": p.name, "vram_gb": round(vram_gb, 1), "count": torch.cuda.device_count(),
                      "compute_capability": f"{p.major}.{p.minor}", "native_bf16": p.major >= 8}
        if req:
            _item(items, "gpu", "ok" if vram_gb >= req["vram_gb"] else "blocked",
                  f"{p.name}, {vram_gb:.1f} GB (needs >= {req['vram_gb']:.1f} GB for {cfg.model_name} in "
                  f"{cfg.model_dtype})")
        else:
            _item(items, "gpu", "warn", f"{p.name}, {vram_gb:.1f} GB (no sizing data for {cfg.model_name})")
        if cfg.model_dtype == "bfloat16" and p.major < 8:
            _item(items, "dtype", "warn", f"{cfg.model_dtype} without native bf16 on {p.name} (cc {p.major}.{p.minor}); "
                                          "slow/emulated. Use model_dtype=float32 or let the hardware stage choose.")
    else:
        _item(items, "gpu", "blocked" if req else "warn",
              "no CUDA device visible (expected on a laptop; on DSMLP request a GPU class large enough for the tier)")
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    probe = hf_home
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 2 ** 30
    need_gb = req["disk_gb"] if req else 2
    _item(items, "model cache disk", "ok" if free_gb >= need_gb else "warn",
          f"{hf_home}: {free_gb:.0f} GB free reported by the filesystem; ~{need_gb} GB needed "
          "(DSMLP quotas may be lower than the filesystem shows; see README)")

    hf_ok = _http_ok("https://huggingface.co")
    gh_ok = _http_ok("https://raw.githubusercontent.com")
    _item(items, "network: huggingface.co", "ok" if hf_ok else "unreachable", "needed to download model/SAE/datasets")
    _item(items, "network: raw.githubusercontent.com", "ok" if gh_ok else "unreachable", "needed for HarmBench CSV")

    if hf_ok:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.model_info(cfg.model_name)
            from transformers import AutoConfig
            mc = AutoConfig.from_pretrained(cfg.model_name)
            _item(items, f"model {cfg.model_name}", "ok", f"{mc.num_hidden_layers} layers, hidden {mc.hidden_size}")
        except Exception as e:
            _item(items, f"model {cfg.model_name}", "blocked", str(e)[:200])
        if cfg.sae_source == "qwen-scope":
            fname = cfg.sae_file_pattern.format(layer=cfg.layer)
            try:
                ok = api.file_exists(cfg.sae_repo, fname)
                _item(items, f"SAE {cfg.sae_repo}/{fname}", "substituted" if ok else "blocked",
                      "Qwen-Scope stand-in for the paper's unreleased SAE (ack: sae-qwen-scope)" if ok
                      else "file not found")
            except Exception as e:
                _item(items, "SAE", "blocked", str(e)[:200])
        else:
            _item(items, "SAE", "warn", "random untrained SAE: plumbing test only")
        wanted = list(cfg.tasks) + list(cfg.agent_datasets)
        for key in wanted:
            if key == "harmbench":
                continue
            try:
                info = api.dataset_info(DATASETS[key])
                gated = bool(getattr(info, "gated", False))
                if gated and not env["hf_token_set"]:
                    _item(items, f"dataset {DATASETS[key]}", "blocked", "gated: needs HF_TOKEN with accepted terms")
                else:
                    _item(items, f"dataset {DATASETS[key]}", "ok", "gated, token present" if gated else "open")
            except Exception as e:
                _item(items, f"dataset {DATASETS[key]}", "blocked", str(e)[:200])
    if "harmbench" in cfg.tasks:
        _item(items, "dataset HarmBench (GitHub CSV)", "ok" if gh_ok and _http_ok(GITHUB_HARMBENCH) else "unreachable",
              "HF mirror is gated; the GitHub CSV is open")
    if cfg.nrc_vad_path:
        exists = os.path.exists(cfg.nrc_vad_path)
        _item(items, "NRC VAD lexicon", "ok" if exists else "blocked", cfg.nrc_vad_path)
    else:
        _item(items, "NRC VAD lexicon", "optional-missing",
              "no nrc_vad_path: hand-picked pole words will be used (ack: label-handpicked)")

    report = {"environment": env, "items": items, "cost_estimate": estimate_cost(cfg)}
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "resources_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    width = max(len(i["name"]) for i in items)
    for i in items:
        log(f"  [{i['status']:>16}] {i['name']:<{width}}  {i['detail']}")
    ce = report["cost_estimate"]
    log(f"  cost: <= {ce['generated_tokens_upper_bound']:,} generated tokens over {ce['conditions']} conditions; "
        f"{ce['gpu_hours_range_upper_bound'][0]}-{ce['gpu_hours_range_upper_bound'][1]} GPU-h at the ASSUMED "
        f"{ASSUMED_TOK_PER_S[0]}-{ASSUMED_TOK_PER_S[1]} tok/s (unmeasured); default budget is {DEFAULT_WEEKLY_SU} SU/week")
    return report


def blocked(report: dict) -> List[dict]:
    return [i for i in report["items"] if i["status"] in ("blocked",)]
