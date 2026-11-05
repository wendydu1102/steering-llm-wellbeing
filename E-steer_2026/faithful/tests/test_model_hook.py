"""Opt-in test against a real (tiny) HF model: verifies the steering hook adds exactly alpha*||h||*delta/delta_max*u
at the configured block's output in the installed transformers version, and that it is a no-op when removed.

  ESTEER_MODEL_TESTS=1 python -m pytest tests/test_model_hook.py -q      (downloads Qwen/Qwen3-0.6B, ~1.2 GB)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytestmark = pytest.mark.skipif(os.environ.get("ESTEER_MODEL_TESTS") != "1", reason="set ESTEER_MODEL_TESTS=1")


def test_hook_adds_expected_shift_at_block_output():
    os.environ.setdefault("USE_TF", "0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from esteer.steering import EmotionSteering

    name = os.environ.get("ESTEER_MODEL_TEST_NAME", "Qwen/Qwen3-0.6B")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).eval()
    layer_idx = 5
    layer = model.model.layers[layer_idx]
    d = model.config.hidden_size
    u = torch.nn.functional.normalize(torch.randn(d, generator=torch.Generator().manual_seed(0)), dim=0)
    st = EmotionSteering({"valence": u}, alpha=0.1, delta_max=9.0)
    enc = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")

    def capture(store):
        return layer.register_forward_hook(lambda m, a, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float()))

    base, steered = {}, {}
    with torch.no_grad():
        h = capture(base)
        model.model(**enc)
        h.remove()
        with st.attach(layer, {"valence": 6.0}):
            h = capture(steered)                     # registered after the steering hook, so it sees the modified output
            model.model(**enc)
            h.remove()
        again = {}
        h = capture(again)
        model.model(**enc)
        h.remove()
    expected = 0.1 * base["h"].norm(dim=-1, keepdim=True) * (6.0 / 9.0) * u
    diff = steered["h"] - base["h"]
    rel = (diff - expected).norm() / expected.norm()
    assert rel < 0.05, f"relative error {rel:.3f} (bf16 rounding allowed)"
    assert torch.equal(again["h"], base["h"]), "hook must be fully removed after the context exits"
