"""Hardware detection and model-tier selection.

Two model tiers are kept side by side; the code path is identical, only configuration differs
(configs/models/<tier>.json plus the dtype and batch size derived here):

  qwen3-8b    the paper's model (Qwen3-8B, SAE at layer 17 of 36)             -> paper-faithful
  qwen3-1.7b  smaller fallback for GPUs that cannot hold the 8B model          -> NOT the paper's model
              (Qwen3-1.7B, Qwen-Scope W32K SAE at layer 13 of 28 = round(17/36*28), same relative depth)

Why not Qwen3-4B: Qwen-Scope publishes SAEs for Qwen3-1.7B and -8B (and larger/other families) but none for
Qwen3-4B, so a 4B tier would need a self-trained or unvetted SAE.

Sizes are exact (Hugging Face safetensors metadata): 8B = 8,190,735,360 params, 1.7B = 2,031,739,904 params
(the 1.7B model is larger than its name because of its 151,936-token embedding matrix).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GB = 1e9
HEADROOM_GB = 2.5          # beyond weights + SAE: CUDA context, activations, a minimal KV cache
KV_RESERVE_GB = 1.5        # subtracted before sizing the generation batch
AVG_CTX_TOKENS = 1500      # prompt + generation length assumed per sequence when sizing the batch
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

TIERS: Dict[str, dict] = {
    "qwen3-8b": dict(model_name="Qwen/Qwen3-8B", params=8_190_735_360, sae_bytes=2_147_764_603,
                     layers=36, kv_heads=8, head_dim=128, disk_gb=20.0, paper_faithful=True),
    "qwen3-1.7b": dict(model_name="Qwen/Qwen3-1.7B", params=2_031_739_904, sae_bytes=537_012_603,
                       layers=28, kv_heads=8, head_dim=128, disk_gb=6.0, paper_faithful=False),
}
PREFERENCE = ["qwen3-8b", "qwen3-1.7b"]          # most paper-faithful first


def tier_for_model(model_name: str) -> Optional[str]:
    for k, v in TIERS.items():
        if v["model_name"] == model_name:
            return k
    return None


def detect() -> dict:
    """GPU 0 and host RAM. Native bf16 needs compute capability >= 8.0 (Ampere or newer); Pascal/Volta/Turing
    cards can only emulate it slowly, so they use fp32/fp16 instead."""
    import torch
    hw = {"cuda": bool(torch.cuda.is_available()), "torch": torch.__version__, "ram_gb": None, "n_gpus": 0}
    try:
        hw["ram_gb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except (ValueError, OSError, AttributeError):
        pass
    if hw["cuda"]:
        p = torch.cuda.get_device_properties(0)
        hw.update(name=p.name, vram_bytes=int(p.total_memory), cc=[p.major, p.minor], n_gpus=torch.cuda.device_count())
    return hw


def native_bf16(hw: dict) -> bool:
    return bool(hw.get("cuda")) and hw.get("cc", [0, 0])[0] >= 8


def need_bytes(tier: str, dtype: str) -> float:
    t = TIERS[tier]
    return t["params"] * DTYPE_BYTES[dtype] + t["sae_bytes"] + HEADROOM_GB * GB


def requirement_for(model_name: str, dtype: str) -> Optional[dict]:
    """{'vram_gb', 'disk_gb'} for a known tier model, else None (unknown model: no sizing data)."""
    tier = tier_for_model(model_name)
    if tier is None or dtype not in DTYPE_BYTES:
        return None
    return {"vram_gb": need_bytes(tier, dtype) / GB, "disk_gb": TIERS[tier]["disk_gb"], "tier": tier}


def _dtype_candidates(hw: dict) -> List[str]:
    # bf16 where natively supported (the dtype the models/SAEs were trained in); otherwise fp32 first
    # (numerically safest) and fp16 only as a fallback (Qwen models can overflow in fp16).
    return ["bfloat16"] if native_bf16(hw) else ["float32", "float16"]


def _batch_size(tier: str, dtype: str, vram: float) -> int:
    t = TIERS[tier]
    kv_per_tok = t["layers"] * 2 * t["kv_heads"] * t["head_dim"] * DTYPE_BYTES[dtype]
    free = vram - t["params"] * DTYPE_BYTES[dtype] - KV_RESERVE_GB * GB
    b = int(0.8 * free / (kv_per_tok * AVG_CTX_TOKENS)) if free > 0 else 1
    b = max(1, min(16, b))
    return 1 << (b.bit_length() - 1)                       # floor to a power of two


@dataclass
class Recommendation:
    tier: Optional[str]
    model_name: Optional[str] = None
    dtype: Optional[str] = None
    batch_size: Optional[int] = None
    fits: bool = False
    paper_faithful: bool = False
    candidates: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def overrides(self) -> dict:
        """Keyword overrides for load_config (take effect after the tier's JSON overlay)."""
        if not self.tier or not self.dtype:
            return {}
        return {"model_dtype": self.dtype, "batch_size": self.batch_size,
                "feature_batch_size": min(self.batch_size, 16), "judge_batch_size": self.batch_size}

    def lines(self) -> List[str]:
        out = []
        for c in self.candidates:
            out.append(f"  {c['tier']:<11} {c['dtype']:<9} needs {c['need_gb']:5.1f} GB  "
                       f"({'fits' if c['fits'] else 'does NOT fit'} in {c['vram_gb']:.1f} GB)")
        if self.tier:
            out.append(f"  -> {self.tier} / {self.dtype} / batch {self.batch_size}"
                       + ("" if self.fits else "  (DOES NOT FIT: forced choice)")
                       + ("  [paper's model]" if self.paper_faithful else "  [smaller model: NOT the paper's]"))
        else:
            out.append("  -> no tier fits this hardware")
        out += [f"  warning: {w}" for w in self.warnings]
        return out


def recommend(hw: dict, prefer: Optional[str] = "auto") -> Recommendation:
    """prefer='auto': first tier (8B, then 1.7B) that fits. prefer=<tier>: evaluate only that tier
    (the result carries fits=False if it does not fit, so callers can refuse)."""
    if prefer not in (None, "auto") and prefer not in TIERS:
        raise ValueError(f"unknown tier {prefer!r}; choose auto or one of {sorted(TIERS)}")
    if not hw.get("cuda"):
        return Recommendation(None, warnings=["no CUDA GPU detected; neither tier is usable on CPU"])
    vram = float(hw["vram_bytes"])
    rec = Recommendation(None)
    tiers = PREFERENCE if prefer in (None, "auto") else [prefer]
    for tier in tiers:
        for dtype in _dtype_candidates(hw):
            need = need_bytes(tier, dtype)
            ok = need <= vram
            rec.candidates.append(dict(tier=tier, dtype=dtype, need_gb=need / GB, vram_gb=vram / GB, fits=ok))
            if ok and rec.tier is None:
                rec.tier, rec.dtype, rec.fits = tier, dtype, True
        if rec.fits:
            break
    if rec.tier is None and prefer not in (None, "auto"):          # forced tier that does not fit
        rec.tier = prefer
        rec.dtype = _dtype_candidates(hw)[0]
    if rec.tier:
        t = TIERS[rec.tier]
        rec.model_name, rec.paper_faithful = t["model_name"], t["paper_faithful"]
        rec.batch_size = _batch_size(rec.tier, rec.dtype, vram)
        if rec.dtype == "float16":
            rec.warnings.append("fp16 fallback: Qwen models can overflow in fp16; check AVR/outputs on the pilot run")
        if not native_bf16(hw):
            rec.warnings.append("GPU has no native bf16 (compute capability < 8.0): "
                                f"using {rec.dtype} instead of the bf16 the paper's setup implies")
        if not t["paper_faithful"]:
            rec.warnings.append("smaller model: results are NOT comparable with the paper's Qwen3-8B numbers "
                                "(ack: smaller-model)")
        if hw.get("n_gpus", 1) > 1:
            rec.warnings.append(f"{hw['n_gpus']} GPUs visible; only GPU 0 is used")
    return rec


def resolve(requested: Optional[str], profile_pins_model: bool, needs_model: bool, log=print):
    """Hardware check -> model tier. Returns (tier or None, load_config keyword overrides, hw, recommendation).

    requested: 'auto' (pick by hardware), a tier name (force it), or 'none' (use the profile's own model).
    A profile that pins its own model (the 'smoke' profile) is left alone unless a tier is requested explicitly.
    Raises RuntimeError when a model stage will run but nothing fits (no silent CPU fallback)."""
    hw = detect()
    requested = requested or "auto"
    if requested == "none" or (profile_pins_model and requested == "auto"):
        return None, {}, hw, None
    rec = recommend(hw, requested)
    if hw.get("cuda"):
        log(f"hardware: {hw.get('name')}, {hw['vram_bytes'] / GB:.1f} GB, compute capability "
            f"{hw['cc'][0]}.{hw['cc'][1]}, native bf16: {native_bf16(hw)}, RAM {hw['ram_gb'] and round(hw['ram_gb'])} GB")
    else:
        log("hardware: no CUDA GPU")
    for line in rec.lines():
        log(line)
    if needs_model and not rec.fits:
        why = "; ".join(rec.warnings) or "see the table above"
        raise RuntimeError(
            f"requested tier {requested!r} does not fit this hardware ({why}). Use a larger GPU class, choose "
            f"another tier, or set the tier to 'none' with a profile that names its own model.")
    return rec.tier, rec.overrides(), hw, rec
