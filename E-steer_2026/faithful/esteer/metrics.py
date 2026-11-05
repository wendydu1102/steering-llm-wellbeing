"""Metric aggregation and the paper's fluctuation range."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def aggregate(records: List[Dict[str, Any]], has_correctness: bool) -> Dict[str, Any]:
    """records: per-example score dicts ({'valid', 'correct'?, judge fields...}).

    AVR  = mean(valid).
    TSR  = mean(correct) over ALL items (an invalid answer counts as wrong); TSR_valid over valid only.
    Other numeric/bool fields (judge scores, safety flags) are averaged over items where present."""
    n = len(records)
    out: Dict[str, Any] = {"n": n}
    if not n:
        return out
    out["AVR"] = sum(bool(r["valid"]) for r in records) / n
    if has_correctness:
        out["TSR"] = sum(bool(r.get("correct")) for r in records) / n
        valid = [r for r in records if r["valid"]]
        out["TSR_valid"] = (sum(bool(r.get("correct")) for r in valid) / len(valid)) if valid else None
    skip = {"valid", "correct", "pred"}
    keys = sorted({k for r in records for k, v in r.items()
                   if k not in skip and (isinstance(v, (bool, int, float)) or v is None)})
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        out[k] = (sum(float(v) for v in vals) / len(vals)) if vals else None
        out[f"{k}__n"] = len(vals)
    if any("harmful" in r for r in records):
        flags = [r for r in records if r.get("harmful") is not None]
        out["safety_risk_any"] = (sum(bool(r["harmful"] or r["biased"] or r["hallucination"]) for r in flags)
                                  / len(flags)) if flags else None
    return out


def fluctuation_range(values: List[Optional[float]]) -> Optional[float]:
    """(max - min) / average over the swept coefficients (paper Sec. 5.1)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return (max(vals) - min(vals)) / mean if mean else None


def relative_change(value: Optional[float], neutral: Optional[float]) -> Optional[float]:
    if value is None or neutral in (None, 0):
        return None
    return (value - neutral) / neutral
