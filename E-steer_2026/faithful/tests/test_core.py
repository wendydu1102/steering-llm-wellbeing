"""Model-free unit tests. Run:  cd faithful && python -m pytest tests -q"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esteer.agent.data import exact_match, f1                                  # noqa: E402
from esteer.agent.pipeline import parse_plan, summarize                        # noqa: E402
from esteer.config import load_config                                          # noqa: E402
from esteer.emotions import load_nrc_vad, select_pairs                         # noqa: E402
from esteer.features import build_directions, identify_features                # noqa: E402
from esteer.fidelity import check_acks, required_acks                          # noqa: E402
from esteer.metrics import aggregate, fluctuation_range                        # noqa: E402
from esteer.sae import TopKSAE                                                 # noqa: E402
from esteer.steering import (EmotionSteering, decoder_unit_direction, eq5_direction,   # noqa: E402
                             latent_unit_vector)
from esteer.tasks.common import (answer_letter, last_boxed, last_code_block, math_equal,   # noqa: E402
                                 normalize_math, parse_json_obj, stratified_sample)
from esteer.tasks.objective import run_python                                  # noqa: E402


# ---- SAE & steering ---------------------------------------------------------------------------
def test_topk_sparsity_and_linear_decoder():
    sae = TopKSAE.random(16, 64, 5, seed=1)
    x = torch.randn(3, 4, 16)
    z = sae.encode(x)
    assert z.shape == (3, 4, 64)
    assert ((z != 0).sum(-1) <= 5).all()
    a, b = torch.randn(64), torch.randn(64)
    assert torch.allclose(sae.decode(a + b), sae.decode(a) + sae.decode(b) - sae.b_dec, atol=1e-5)


def test_eq5_matches_linear_shortcut_and_is_z_independent():
    sae = TopKSAE.random(16, 64, 5, seed=2)
    u = latent_unit_vector([3, 9, 20], [1.0, -2.0, 0.5], 64)
    d1 = eq5_direction(sae, sae.encode(torch.randn(16)), u, 4.0)
    d2 = eq5_direction(sae, sae.encode(torch.randn(16)), u, 4.0)
    assert torch.allclose(d1, d2, atol=1e-5)                       # f_dec is linear: z cancels
    assert torch.allclose(d1 / d1.norm(), decoder_unit_direction(sae, u), atol=1e-5)
    assert torch.allclose(eq5_direction(sae, torch.zeros(64), u, -4.0) / -1,
                          eq5_direction(sae, torch.zeros(64), u, 4.0), atol=1e-5)


def _steer(variant="sign_preserving"):
    dirs = {"valence": torch.tensor([1.0, 0, 0, 0]), "arousal": torch.tensor([0, 1.0, 0, 0])}
    return EmotionSteering(dirs, alpha=0.5, delta_max=9.0, eq6_variant=variant)


def test_eq6_sign_preserving_vs_literal():
    sp, lit = _steer(), _steer("literal")
    assert torch.allclose(sp.unit_shift({"valence": 6.0}), -sp.unit_shift({"valence": -6.0}))
    # the literal reading cannot distinguish +delta from -delta (documented defect)
    assert torch.allclose(lit.unit_shift({"valence": 6.0}), lit.unit_shift({"valence": -6.0}))


def test_hook_scales_with_token_norm_and_delta_and_multi_dim():
    st = _steer()
    layer = torch.nn.Identity()
    h = torch.tensor([[[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 8.0]]])   # token norms 5 and 10
    expect = h.clone()
    expect[0, 0] += torch.tensor([0.5 * 5 * 1.0, 0.5 * 5 * -0.5, 0, 0])
    expect[0, 1] += torch.tensor([0.5 * 10 * 1.0, 0.5 * 10 * -0.5, 0, 0])
    with st.attach(layer, {"valence": 9.0, "arousal": -4.5}):
        assert torch.allclose(layer(h), expect, atol=1e-5)
    assert torch.equal(layer(h), h)                                  # hook removed on exit
    with st.attach(layer, {"valence": 0.0}):                         # all-zero deltas: no intervention
        assert torch.equal(layer(h), h)
    assert st.make_hook({"valence": 0.0}) is None and st.make_hook({}) is None


def test_hook_handles_tuple_outputs_and_dtype():
    st = _steer()
    h = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    res = st.make_hook({"valence": 3.0})(None, (), (h, "extra"))
    assert isinstance(res, tuple) and res[0].dtype == torch.bfloat16 and res[1] == "extra"


# ---- feature identification -----------------------------------------------------------------
def test_identify_features_finds_stable_signed_latents():
    g = torch.Generator().manual_seed(0)
    rows = 0.05 * torch.randn(40, 200, generator=g)
    rows[:, 7] += 2.0                                              # consistently positive
    rows[:, 11] -= 3.0                                             # consistently negative
    rows[:20, 50] += 2.0                                           # only half the rows
    f = identify_features(rows, n_top=5, tau=0.9, min_features=1)
    assert set(f["indices"]) == {7, 11} and not f["relaxed_threshold"]
    w = dict(zip(f["indices"], f["weights"]))
    assert w[7] > 0 > w[11]


def test_identify_features_relaxes_threshold_when_too_strict():
    rows = 0.05 * torch.randn(10, 100, generator=torch.Generator().manual_seed(1))
    f = identify_features(rows, n_top=3, tau=0.99, min_features=4)
    assert len(f["indices"]) == 4 and f["relaxed_threshold"]


def test_build_directions_variants():
    sae = TopKSAE.random(16, 128, 5, seed=3)
    feats = {"dims": {d: {"indices": list(range(10 * i, 10 * i + 10)), "weights": [1.0] * 10}
                      for i, d in enumerate(("valence", "arousal", "dominance"))}}
    orig = build_directions(sae, feats, "original")
    rnd = build_directions(sae, feats, "random", seed=1)
    half = build_directions(sae, feats, "half", seed=1)
    for d in orig:
        assert abs(float(orig[d].norm()) - 1) < 1e-5
        assert float(torch.dot(orig[d], rnd[d])) < 0.99          # random set points elsewhere
        assert float(torch.dot(orig[d], half[d])) > float(torch.dot(orig[d], rnd[d])) - 1e-6


# ---- emotions / labels ----------------------------------------------------------------------
def test_nrc_loader_and_pair_selection(tmp_path):
    p = tmp_path / "lex.txt"
    rows = ["term\tvalence\tarousal\tdominance", "joyful\t0.9\t0.5\t0.4", "miserable\t-0.9\t0.5\t-0.4",
            "delighted\t0.8\t0.4\t0.3", "gloomy\t-0.8\t0.3\t-0.3", "calm\t0.3\t-0.8\t0.1", "excited\t0.4\t0.8\t0.2"]
    p.write_text("\n".join(rows))
    lex = load_nrc_vad(str(p))
    assert abs(lex["joyful"][0] - 9.0) < 1e-6                     # x10 -> [-10, 10]
    pairs = select_pairs(lex, "valence", 2, min_abs=5.0)
    assert {pairs[0]["pos"], pairs[1]["pos"]} <= {"joyful", "delighted"}
    assert {pairs[0]["neg"], pairs[1]["neg"]} <= {"miserable", "gloomy"}
    pa = select_pairs(lex, "arousal", 1, min_abs=3.0)
    assert pa[0]["pos"] == "excited" and pa[0]["neg"] == "calm"


# ---- parsers & metrics ----------------------------------------------------------------------
def test_parsers():
    assert last_boxed(r"so \boxed{\frac{1}{2}} and finally \boxed{x^{2}+1}") == "x^{2}+1"
    assert last_boxed("no box") is None
    assert math_equal(r"\dfrac{1}{2}", r"\frac{1}{2}") and math_equal("5.0", "5") and not math_equal("4", "5")
    assert normalize_math(r"\text{ 12 } \%") == "12"
    assert answer_letter("blah\nAnswer: c") == "C" and answer_letter("Answer: (B)") == "B" and answer_letter("none") is None
    assert last_code_block("x\n```python\nprint(1)\n```\ny\n```python\nprint(2)\n```") == "print(2)\n"
    assert parse_json_obj('noise {"a": 1} tail') == {"a": 1} and parse_json_obj("{bad") is None


def test_stratified_sample_is_proportional_and_deterministic():
    items = [("a", i) for i in range(80)] + [("b", i) for i in range(20)]
    s1 = stratified_sample(items, lambda x: x[0], 10, 0)
    assert s1 == stratified_sample(items, lambda x: x[0], 10, 0)
    assert sum(1 for x in s1 if x[0] == "a") == 8 and len(s1) == 10


def test_run_python_pass_fail_timeout():
    assert run_python("assert 1 + 1 == 2", 10)
    assert not run_python("assert 1 + 1 == 3", 10)
    assert not run_python("import time\ntime.sleep(5)", 1)


def test_aggregate_and_fluctuation():
    recs = [{"valid": True, "correct": True}, {"valid": True, "correct": False},
            {"valid": False, "correct": False}, {"valid": True, "correct": True}]
    a = aggregate(recs, True)
    assert a["AVR"] == 0.75 and a["TSR"] == 0.5 and abs(a["TSR_valid"] - 2 / 3) < 1e-9
    s = aggregate([{"valid": True, "harmful": True, "biased": False, "hallucination": False},
                   {"valid": False, "harmful": False, "biased": False, "hallucination": False}], False)
    assert s["safety_risk_any"] == 0.5 and s["harmful"] == 0.5
    assert abs(fluctuation_range([0.4, 0.6, 0.5]) - 0.4) < 1e-9 and fluctuation_range([None]) is None


# ---- config, acknowledgements, agent --------------------------------------------------------
def test_config_validation_and_ack_enforcement():
    with pytest.raises(ValueError):
        load_config(None, ["deltas=[-9,3,9]"])
    with pytest.raises(KeyError):
        load_config(None, ["nonexistent=1"])
    cfg = load_config(None, [], run_id="t")
    assert "sae-qwen-scope" in required_acks(cfg, ["llm"], lexicon_available=True)
    with pytest.raises(SystemExit) as e:
        check_acks(cfg, ["llm"], lexicon_available=False)
    assert "label-handpicked" in str(e.value) and "sae-qwen-scope" in str(e.value)
    cfg.ack = required_acks(cfg, ["llm"], lexicon_available=False)
    check_acks(cfg, ["llm"], lexicon_available=False)             # now passes


def test_agent_plan_parsing_metrics_and_qa():
    good = '{"steps": [{"action": "extract", "description": "find x"}, {"action": "reason", "description": "infer"}]}'
    assert len(parse_plan(good)) == 2
    assert parse_plan('{"steps": [{"action": "browse", "description": "x"}]}') is None
    assert parse_plan("not json") is None
    assert exact_match("The Eiffel Tower.", "eiffel tower") and f1("x y z", "x y w") == pytest.approx(2 / 3)
    recs = [dict(plan1_valid=True, replan_triggered=False, decider_conf=0.9, decider_feasible=True, conf1=0.8,
                 conf2=None, steps1=[2, 2], steps2=None, rational_selection=True, correct=True),
            dict(plan1_valid=False, replan_triggered=True, decider_conf=0.7, decider_feasible=False, conf1=0.4,
                 conf2=0.6, steps1=[1, 2], steps2=[2, 2], rational_selection=False, correct=False)]
    s = summarize(recs)
    assert s["plan_validity_rate"] == 0.5 and s["replan_frequency"] == 0.5
    assert s["replan_trigger_confidence"] == 0.7 and s["replan_improvement"] == pytest.approx(0.2)
    assert s["execution_completion_rate"] == pytest.approx(5 / 6) and s["rational_selection_rate"] == 0.5
    assert s["overall_success_rate"] == 0.5
