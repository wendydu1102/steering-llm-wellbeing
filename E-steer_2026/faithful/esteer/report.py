"""Aggregate a run into tables, figures and REPORT.md, with the paper's reported numbers alongside.

The paper's numbers are quoted from its text (Sec. 5.2-5.5) for orientation only. They come from a
different SAE, sample sizes and judge, so a mismatch is not by itself a failure to reproduce, and a match
is not proof of reproduction; with small n, differences of a few points are within noise.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from .evaluate import load_summaries
from .metrics import fluctuation_range, relative_change
from .steering import DIMS

METRICS = {"logiqa": ["AVR", "TSR"], "humaneval": ["AVR", "TSR"], "math": ["AVR", "TSR"],
           "tinystories": ["AVR", "relevance", "coherence", "creativity", "conciseness"],
           "harmbench": ["AVR", "harmful", "biased", "hallucination", "safety_risk_any"]}
OBJECTIVE = ("logiqa", "humaneval", "math")
AGENT_METRICS = ["plan_validity_rate", "replan_improvement", "replan_frequency", "replan_trigger_confidence",
                 "rational_selection_rate", "execution_completion_rate", "overall_success_rate"]


def _curve(summ: Dict[str, dict], dim: str, metric: str) -> List[tuple]:
    """[(delta, value)] sorted by delta, with the shared neutral point at 0."""
    pts = []
    for key, d in summ.items():
        if d["dim"] == dim or key == "neutral":
            pts.append((0.0 if key == "neutral" else d["delta"], d["summary"].get(metric)))
    return sorted(set(pts), key=lambda p: p[0])


def _val(summ, dim, delta, metric) -> Optional[float]:
    for d, v in _curve(summ, dim, metric):
        if abs(d - delta) < 1e-9:
            return v
    return None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def paper_comparison(S: Dict[str, Dict[str, dict]]) -> List[dict]:
    rows = []

    def add(claim, paper, ours, note=""):
        rows.append({"claim": claim, "paper": paper, "ours": ours, "note": note})

    obj = [t for t in OBJECTIVE if t in S]
    if obj:
        pos = _mean([_mean([_val(S[t], "valence", d, "AVR") for d in (3, 6, 9)]) for t in obj])
        neg = _mean([_mean([_val(S[t], "valence", d, "AVR") for d in (-3, -6, -9)]) for t in obj])
        add("Objective AVR: positive vs negative valence (relative)", "+33.1%",
            relative_change(pos, neg), "mean over objective tasks present")
        for dim, ref in (("valence", "+3.4% (avg over tasks)"), ("arousal", "+4.7% at +3"), ("dominance", "task-dependent")):
            best = []
            for t in obj:
                neu = _val(S[t], dim, 0, "TSR")
                vals = [v for d, v in _curve(S[t], dim, "TSR") if d != 0 and v is not None]
                if vals and neu:
                    best.append(relative_change(max(vals), neu))
            add(f"Objective TSR: best {dim} setting vs neutral", ref, _mean(best), "relative change of the best non-zero delta")
        interior, total = 0, 0
        for t in obj:
            for dim in DIMS:
                c = [(d, v) for d, v in _curve(S[t], dim, "TSR") if v is not None]
                if len(c) >= 3:
                    total += 1
                    peak = max(c, key=lambda p: p[1])[0]
                    interior += int(c[0][0] < peak < c[-1][0])
        add("TSR peaks at an interior (non-extreme) coefficient (inverted-U)", "all three dimensions",
            (interior / total) if total else None, f"{interior}/{total} task-dimension curves")
        for dim in DIMS:
            add(f"TSR fluctuation range, {dim}", {"valence": "71.2%", "arousal": "-17.8% vs valence",
                                                  "dominance": "-25.6% vs valence"}[dim],
                _mean([fluctuation_range([v for _, v in _curve(S[t], dim, 'TSR')]) for t in obj]),
                "(max-min)/mean across coefficients")
    if "harmbench" in S:
        h = S["harmbench"]
        for dim, delta, ref in (("valence", -3, "-52.7%"), ("arousal", -3, "-21.7%"), ("dominance", 6, "-68.3%")):
            add(f"Safety-risk probability at {dim} {delta:+d} vs neutral", ref,
                relative_change(_val(h, dim, delta, "safety_risk_any"), _val(h, dim, 0, "safety_risk_any")),
                "any of harmful/biased/hallucination; paper's aggregation not specified")
    if "tinystories" in S:
        t = S["tinystories"]
        add("Creativity at valence +3 vs neutral", "+6.5%",
            relative_change(_val(t, "valence", 3, "creativity"), _val(t, "valence", 0, "creativity")))
        add("Conciseness: negative (-3) vs positive (+3) valence", "+23.3%",
            relative_change(_val(t, "valence", -3, "conciseness"), _val(t, "valence", 3, "conciseness")))
    return rows


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:+.1%}" if isinstance(x, float) and abs(x) < 10 else str(x))


def _plot_task(S_task, name, outdir) -> Optional[str]:
    # Object-oriented Figure (no pyplot): saving a PNG must not switch matplotlib's global backend, or later
    # plt.show() calls in a notebook would stop displaying.
    from matplotlib.figure import Figure
    metrics = [m for m in METRICS[name] if any(m in d["summary"] for d in S_task.values())]
    if not metrics:
        return None
    fig = Figure(figsize=(12, 2.6 * len(metrics)))
    axes = fig.subplots(len(metrics), 3, squeeze=False, sharex=True)
    for r, m in enumerate(metrics):
        for c, dim in enumerate(DIMS):
            pts = [(d, v) for d, v in _curve(S_task, dim, m) if v is not None]
            ax = axes[r][c]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
            ax.axvline(0, color="gray", lw=0.6, ls="--")
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(dim.capitalize())
            if c == 0:
                ax.set_ylabel(m)
            if r == len(metrics) - 1:
                ax.set_xlabel("coefficient delta")
    fig.suptitle(f"{name}")
    fig.tight_layout()
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=130)
    return path


def _load_agent(run_dir) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    root = os.path.join(run_dir, "agent")
    if not os.path.isdir(root):
        return out
    for ds in sorted(os.listdir(root)):
        for fn in sorted(os.listdir(os.path.join(root, ds))):
            if fn.endswith(".json"):
                d = json.load(open(os.path.join(root, ds, fn)))
                out.setdefault(ds, {})[fn[:-5]] = {"dim": d["dim"], "delta": d["delta"], "summary": d["summary"]}
    return out


def build_report(run_dir: str, log=print) -> None:
    figdir = os.path.join(run_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    S = load_summaries(run_dir)
    A = _load_agent(run_dir)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    fid = json.load(open(os.path.join(run_dir, "fidelity.json")))

    with open(os.path.join(run_dir, "results_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "task", "dim", "delta", "metric", "value"])
        for fam, data in (("llm", S), ("agent", A)):
            for task, conds in data.items():
                for key, d in conds.items():
                    for m, v in d["summary"].items():
                        if isinstance(v, (int, float)) or v is None:
                            w.writerow([fam, task, d["dim"], d["delta"], m, v])

    L = [f"# E-STEER paper-faithful run: {cfg['run_id']}", "",
         f"profile `{cfg['profile']}`; model `{cfg['model_name']}` layer {cfg['layer']}; SAE `{cfg['sae_source']}`"
         f"{' (' + cfg['sae_repo'] + ')' if cfg['sae_source'] == 'qwen-scope' else ' (UNTRAINED: plumbing test only)'}; "
         f"alpha from calibration/config; seed {cfg['seed']}.", "",
         "## What is and is not paper-identical", "",
         "Acknowledged non-identical components for this run: " + ", ".join(f"`{a}`" for a in fid["acknowledged"]) + ".",
         "See `fidelity.json` (full matrix) and `faithful/README.md`. Numbers below are NOT directly comparable "
         "to the paper's: different SAE, sample sizes and judge.", ""]
    if os.path.exists(os.path.join(run_dir, "calibration.json")):
        cal = json.load(open(os.path.join(run_dir, "calibration.json")))
        L += [f"alpha = **{cal['chosen_alpha']}** chosen by: {cal['rule']} (calibration.json).", ""]
    chk = os.path.join(run_dir, "features", "sae_check.json")
    if os.path.exists(chk):
        L += [f"SAE reconstruction FVE by hooked layer: `{json.load(open(chk))['fve_by_layer']}`.", ""]
    fe = os.path.join(run_dir, "features", "features.json")
    if os.path.exists(fe):
        ft = json.load(open(fe))
        L += [f"Stable latents per dimension: " + ", ".join(f"{d}={len(ft['dims'][d]['indices'])}" for d in DIMS)
              + f"; label source `{ft['label_source']}`; direction cosines `{ft['diagnostics']['cosine']}`.", ""]

    L += ["## LLM tasks", ""]
    for task, conds in S.items():
        png = _plot_task(conds, task, figdir)
        n = next(iter(conds.values()))["summary"]["n"]
        L += [f"### {task} (n={n} per condition)", "", f"![{task}](figures/{task}.png)", ""]
        metrics = [m for m in METRICS[task] if any(m in d["summary"] for d in conds.values())]
        deltas = sorted({0.0} | {d["delta"] for d in conds.values()})
        for m in metrics:
            L += [f"**{m}**", "", "| dimension | " + " | ".join(f"{d:+g}" for d in deltas) + " | fluctuation |",
                  "|---|" + "---|" * (len(deltas) + 1)]
            for dim in DIMS:
                vals = {d: v for d, v in _curve(conds, dim, m)}
                cells = ["n/a" if vals.get(d) is None else f"{vals[d]:.3f}" for d in deltas]
                fr = fluctuation_range([v for _, v in _curve(conds, dim, m)])
                L.append(f"| {dim} | " + " | ".join(cells) + f" | {_fmt(fr) if fr is not None else 'n/a'} |")
            L.append("")
    if A:
        L += ["## Agent (reconstructed; see fidelity matrix)", ""]
        for ds, conds in A.items():
            L += [f"### {ds}", ""]
            deltas = sorted({0.0} | {d["delta"] for d in conds.values()})
            for m in AGENT_METRICS:
                L += [f"**{m}**", "", "| dimension | " + " | ".join(f"{d:+g}" for d in deltas) + " |",
                      "|---|" + "---|" * len(deltas)]
                for dim in DIMS:
                    vals = {d: v for d, v in _curve(conds, dim, m)}
                    L.append(f"| {dim} | " + " | ".join("n/a" if vals.get(d) is None else f"{vals[d]:.3f}" for d in deltas) + " |")
                L.append("")
    vroot = os.path.join(run_dir, "validation")
    if os.path.isdir(vroot):
        L += ["## Sec. 5.6 latent validation (LogiQA TSR)", ""]
        for variant in sorted(os.listdir(vroot)):
            vs = load_summaries(run_dir, os.path.join("validation", variant)).get("logiqa", {})
            L += [f"**{variant}**", "", "| dimension | " + " | ".join(f"{d:+g}" for d in sorted({0.0} | {d['delta'] for d in vs.values()})) + " |", "|---|" + "---|" * len(sorted({0.0} | {d['delta'] for d in vs.values()}))]
            for dim in DIMS:
                cv = _curve(vs, dim, "TSR")
                L.append(f"| {dim} | " + " | ".join("n/a" if v is None else f"{v:.3f}" for _, v in cv) + " |")
            L.append("")
    cmp_rows = paper_comparison(S)
    if cmp_rows:
        L += ["## Paper's reported numbers vs this run", "",
              "| quantity | paper | this run | note |", "|---|---|---|---|"]
        for r in cmp_rows:
            L.append(f"| {r['claim']} | {r['paper']} | {_fmt(r['ours'])} | {r['note']} |")
        L.append("")
    with open(os.path.join(run_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    log(f"wrote {os.path.join(run_dir, 'REPORT.md')}")
