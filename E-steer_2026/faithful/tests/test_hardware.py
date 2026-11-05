"""Hardware -> model-tier selection, with simulated GPUs (no real hardware needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esteer.config import load_config                          # noqa: E402
from esteer.fidelity import check_acks, required_acks          # noqa: E402
from esteer.hardware import (TIERS, native_bf16, recommend, requirement_for, resolve)   # noqa: E402


def gpu(name, gb, cc):
    return {"cuda": True, "name": name, "vram_bytes": int(gb * 1e9), "cc": list(cc), "n_gpus": 1, "ram_gb": 64}


def test_ampere_24gb_gets_the_paper_model_in_bf16():
    r = recommend(gpu("A30", 25.2, (8, 0)))
    assert (r.tier, r.dtype, r.fits, r.paper_faithful) == ("qwen3-8b", "bfloat16", True, True)
    assert r.batch_size >= 8 and not any("smaller" in w for w in r.warnings)


def test_users_gtx_1080_ti_gets_1p7b_in_fp32_with_small_batch():
    r = recommend(gpu("NVIDIA GeForce GTX 1080 Ti", 11.7, (6, 1)))
    assert (r.tier, r.dtype, r.fits, r.paper_faithful) == ("qwen3-1.7b", "float32", True, False)
    assert 1 <= r.batch_size <= 4                                   # ~8.1 GB of fp32 weights leaves little for KV cache
    assert any("no native bf16" in w for w in r.warnings) and any("smaller model" in w for w in r.warnings)
    eight = [c for c in r.candidates if c["tier"] == "qwen3-8b"]
    assert eight and not any(c["fits"] for c in eight)             # the 8B candidates were evaluated and rejected


def test_12gb_ampere_uses_bf16_on_the_small_tier():
    r = recommend(gpu("RTX 3060", 12.0, (8, 6)))
    assert (r.tier, r.dtype) == ("qwen3-1.7b", "bfloat16") and r.batch_size >= 8


def test_pre_ampere_with_enough_memory_falls_back_to_fp16_with_a_warning():
    r = recommend(gpu("V100", 32.0, (7, 0)))                       # 8B fp32 (32.8 GB) cannot fit, fp16 can
    assert (r.tier, r.dtype) == ("qwen3-8b", "float16")
    assert any("fp16 fallback" in w for w in r.warnings)


def test_too_small_or_no_gpu_has_no_tier():
    assert recommend(gpu("tiny", 6.0, (8, 0))).tier is None
    assert recommend({"cuda": False}).tier is None


def test_forced_tier_that_does_not_fit_is_reported_not_silently_changed():
    r = recommend(gpu("GTX 1080 Ti", 11.7, (6, 1)), prefer="qwen3-8b")
    assert r.tier == "qwen3-8b" and r.fits is False
    with pytest.raises(ValueError):
        recommend(gpu("x", 24, (8, 0)), prefer="qwen3-4b")          # no Qwen-Scope SAE exists for 4B -> no such tier


def test_native_bf16_follows_compute_capability():
    assert native_bf16(gpu("a", 24, (8, 0))) and not native_bf16(gpu("a", 24, (7, 5))) and not native_bf16({"cuda": False})


def test_requirement_lookup_and_unknown_models():
    r8, r17 = requirement_for("Qwen/Qwen3-8B", "bfloat16"), requirement_for("Qwen/Qwen3-1.7B", "float32")
    assert 20 < r8["vram_gb"] < 23 and 10 < r17["vram_gb"] < 12
    assert requirement_for("Qwen/Qwen3-0.6B", "bfloat16") is None


def test_tier_overlay_sets_model_sae_layer_and_hardware_overrides_win():
    cfg = load_config(None, [], model_tier="qwen3-1.7b", run_id="t", model_dtype="float32", batch_size=4)
    assert (cfg.model_name, cfg.layer, cfg.model_tier) == ("Qwen/Qwen3-1.7B", 13, "qwen3-1.7b")
    assert cfg.sae_repo.endswith("1.7B-Base-W32K-L0_50") and cfg.model_dtype == "float32" and cfg.batch_size == 4
    cfg8 = load_config(None, [], model_tier="qwen3-8b", run_id="t")
    assert (cfg8.model_name, cfg8.layer) == ("Qwen/Qwen3-8B", 17) and cfg8.sae_repo.endswith("8B-Base-W64K-L0_50")
    # layer mapping rule documented in hardware.py: same relative depth
    assert round(17 / 36 * 28) == 13


def test_smaller_model_requires_its_own_acknowledgement():
    small = load_config(None, [], model_tier="qwen3-1.7b", run_id="t")
    big = load_config(None, [], model_tier="qwen3-8b", run_id="t")
    assert "smaller-model" in required_acks(small, ["llm"], True) and "smaller-model" not in required_acks(big, ["llm"], True)
    with pytest.raises(SystemExit) as e:
        check_acks(small, ["llm"], True)
    assert "smaller-model" in str(e.value)


def test_resolve_leaves_smoke_style_profiles_alone_and_refuses_misfit(monkeypatch):
    import esteer.hardware as hwmod
    monkeypatch.setattr(hwmod, "detect", lambda: gpu("GTX 1080 Ti", 11.7, (6, 1)))
    logs = []
    tier, kw, hw, rec = resolve("auto", profile_pins_model=True, needs_model=True, log=logs.append)
    assert tier is None and kw == {} and rec is None                 # profile names its own (tiny) model
    tier, kw, hw, rec = resolve("auto", profile_pins_model=False, needs_model=True, log=logs.append)
    assert tier == "qwen3-1.7b" and kw["model_dtype"] == "float32" and kw["batch_size"] <= 4
    with pytest.raises(RuntimeError):
        resolve("qwen3-8b", profile_pins_model=False, needs_model=True, log=logs.append)
    monkeypatch.setattr(hwmod, "detect", lambda: {"cuda": False})
    with pytest.raises(RuntimeError):
        resolve("auto", profile_pins_model=False, needs_model=True, log=logs.append)
    assert resolve("auto", profile_pins_model=False, needs_model=False, log=logs.append)[0] is None   # report-only run
    assert set(TIERS) == {"qwen3-8b", "qwen3-1.7b"}
