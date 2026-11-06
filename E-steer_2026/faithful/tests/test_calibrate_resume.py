"""Calibration is resumable per condition and never reuses stale results."""
import dataclasses
import json
import os

import esteer.calibrate as cal
from esteer.config import load_config
from esteer.steering import DIMS


class _Ex:
    def __init__(self, i):
        self.id = i


def _setup(monkeypatch, tmp_path, calls, fail_after=None):
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "smoke.json"))
    cfg = dataclasses.replace(cfg, alpha_grid=[0.1, 0.2], calibration_n=3, resume=True)
    exs = [_Ex(f"x{i}") for i in range(3)]
    monkeypatch.setattr(cal, "calibration_examples", lambda c: (object(), exs))

    def fake_eval(bundle, task, examples, dim, delta, c):
        if fail_after is not None and len(calls) >= fail_after:
            raise KeyboardInterrupt
        calls.append((dim, delta, bundle.steering.alpha if bundle.steering else None))
        return {"summary": {"AVR": 1.0, "TSR": 0.5}, "records": []}

    monkeypatch.setattr(cal, "evaluate_condition", fake_eval)

    class B:
        steering = None
    return cfg, B(), str(tmp_path)


def test_resume_skips_finished_conditions(monkeypatch, tmp_path):
    calls = []
    cfg, b, rd = _setup(monkeypatch, tmp_path, calls, fail_after=5)
    try:
        cal.calibrate_alpha(b, cfg, {}, rd, log=lambda *_: None)
    except KeyboardInterrupt:
        pass
    assert len(calls) == 5 and not os.path.exists(os.path.join(rd, "calibration.json"))
    calls2 = []
    cfg, b, rd = _setup(monkeypatch, tmp_path, calls2)
    alpha = cal.calibrate_alpha(b, cfg, {}, rd, log=lambda *_: None)
    total = 1 + 2 * len(DIMS) * 2
    assert len(calls2) == total - 5                       # only the unfinished ones ran
    assert alpha == 0.2 and os.path.exists(os.path.join(rd, "calibration.json"))


def test_changed_settings_invalidate_cache(monkeypatch, tmp_path):
    calls = []
    cfg, b, rd = _setup(monkeypatch, tmp_path, calls)
    cal.calibrate_alpha(b, cfg, {}, rd, log=lambda *_: None)
    calls2 = []
    cfg, b, rd = _setup(monkeypatch, tmp_path, calls2)
    cfg = dataclasses.replace(cfg, delta_max=cfg.delta_max + 1)
    cal.calibrate_alpha(b, cfg, {}, rd, log=lambda *_: None)
    assert len(calls2) == len(calls)                      # nothing reused


def test_resume_off_recomputes(monkeypatch, tmp_path):
    calls = []
    cfg, b, rd = _setup(monkeypatch, tmp_path, calls)
    cal.calibrate_alpha(b, cfg, {}, rd, log=lambda *_: None)
    calls2 = []
    cfg, b, rd = _setup(monkeypatch, tmp_path, calls2)
    cal.calibrate_alpha(b, dataclasses.replace(cfg, resume=False), {}, rd, log=lambda *_: None)
    assert len(calls2) == len(calls)
