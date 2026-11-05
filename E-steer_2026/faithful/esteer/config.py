"""Run configuration. Defaults describe the 'full' profile; configs/*.json override them.

Anything the paper does not specify is a field here (not a hard-coded constant),
and is listed in fidelity.py with the reason it had to be chosen.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


def _tasks_default() -> Dict[str, int]:
    # Sample sizes are NOT given in the paper ("subsets sampled proportionally").
    return {"logiqa": 300, "humaneval": 164, "math": 350, "tinystories": 100, "harmbench": 200}


def _max_new_default() -> Dict[str, int]:
    # Paper: "a fixed maximum generation length" (value not given).
    return {"logiqa": 512, "humaneval": 768, "math": 1024, "tinystories": 256, "harmbench": 384}


@dataclass
class RunConfig:
    profile: str = "full"

    # --- model (paper: Qwen3-8B, SAE at k = 17) ---
    model_tier: str = ""                   # "qwen3-8b" | "qwen3-1.7b" ("" = model_name/layer set directly)
    model_name: str = "Qwen/Qwen3-8B"
    model_dtype: str = "bfloat16"
    device: str = "auto"
    enable_thinking: bool = False          # paper does not say; see fidelity matrix
    layer: int = 17                        # output of model.model.layers[layer]

    # --- SAE ---
    sae_source: str = "qwen-scope"         # "qwen-scope" | "random" (smoke tests only)
    sae_repo: str = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
    sae_file_pattern: str = "layer{layer}.sae.pt"
    sae_k: int = 50
    sae_random_d_sae: int = 4096           # random SAE only

    # --- VAD labels / contrastive pairs (Sec. 4.2) ---
    nrc_vad_path: Optional[str] = None     # NRC VAD Lexicon file; see README
    n_label_pairs: int = 8
    n_feature_tasks: int = 8               # task descriptions per task family
    n_top: int = 50                        # "top 50 neurons" (paper)
    stability_tau: float = 0.5             # "stable across multiple pairs" (threshold not given)
    min_features: int = 5
    feature_batch_size: int = 16

    # --- steering (Sec. 4.3) ---
    deltas: List[float] = field(default_factory=lambda: [-9, -6, -3, 0, 3, 6, 9])
    delta_max: float = 9.0
    alpha: Optional[float] = None          # not given in paper; set explicitly or calibrated
    alpha_grid: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4, 0.8])
    alpha_validity_floor: float = 0.9
    calibration_n: int = 64
    eq6_variant: str = "sign_preserving"

    # --- LLM tasks (Sec. 5.2-5.4) ---
    tasks: Dict[str, int] = field(default_factory=_tasks_default)
    max_new_tokens: Dict[str, int] = field(default_factory=_max_new_default)
    batch_size: int = 16
    judge_batch_size: int = 16
    judge_max_new_tokens: int = 80
    validation_n: int = 150                # Sec. 5.6 latent validation (LogiQA)

    # --- agent (Sec. 5.5) ---
    agent_datasets: Dict[str, int] = field(default_factory=lambda: {"hotpotqa": 100, "science": 45})
    agent_max_new_tokens: int = 384

    # --- run bookkeeping ---
    seed: int = 0
    out_root: str = "results_faithful"
    run_id: str = ""
    resume: bool = True
    ack: List[str] = field(default_factory=list)   # acknowledged fallbacks/interpretations

    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_root, self.run_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if 0 not in [float(d) for d in self.deltas]:
            raise ValueError("deltas must include 0 (the neutral baseline)")
        if max(abs(float(d)) for d in self.deltas) > self.delta_max + 1e-9:
            raise ValueError("|delta| must not exceed delta_max")
        if self.sae_source not in ("qwen-scope", "random"):
            raise ValueError("sae_source must be 'qwen-scope' or 'random'")
        unknown = set(self.tasks) - {"logiqa", "humaneval", "math", "tinystories", "harmbench"}
        if unknown:
            raise ValueError(f"unknown LLM tasks: {sorted(unknown)}")
        unknown = set(self.agent_datasets) - {"hotpotqa", "science", "gaia"}
        if unknown:
            raise ValueError(f"unknown agent datasets: {sorted(unknown)}")


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "models")


def load_config(profile_path: Optional[str] = None, overrides: Optional[List[str]] = None,
                model_tier: Optional[str] = None, **kw) -> RunConfig:
    """Defaults <- profile JSON <- model-tier overlay <- keyword args <- 'key=json_value' overrides.

    The profile sets the experiment size (tasks, sweep); the tier overlay (configs/models/<tier>.json) sets the
    model-specific fields (model, SAE, layer, batch sizes). Keyword args are for the hardware-derived dtype/batch."""
    data: Dict[str, Any] = {}
    if profile_path:
        with open(profile_path) as f:
            data.update(json.load(f))
        data.setdefault("profile", os.path.splitext(os.path.basename(profile_path))[0])
    if model_tier:
        overlay = os.path.join(MODELS_DIR, f"{model_tier}.json")
        if not os.path.exists(overlay):
            raise ValueError(f"unknown model tier {model_tier!r}: no {overlay}")
        with open(overlay) as f:
            data.update(json.load(f))
    data.update({k: v for k, v in kw.items() if v is not None})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        try:
            data[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            data[key.strip()] = raw
    known = {f.name for f in fields(RunConfig)}
    bad = set(data) - known
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    cfg = RunConfig(**data)
    if not cfg.run_id:
        cfg.run_id = f"{cfg.profile}-{time.strftime('%Y%m%d-%H%M%S')}"
    cfg.validate()
    return cfg
