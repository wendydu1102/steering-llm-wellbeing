"""LLM-task evaluation over the VAD coefficient sweep (paper Sec. 5.2-5.4), resumable per condition."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

from .judge import run_judge
from .metrics import aggregate
from .steering import DIMS
from .tasks import REGISTRY


def conditions(cfg) -> List[Tuple[str, float]]:
    """One shared neutral condition plus each dimension swept with the others at 0."""
    conds: List[Tuple[str, float]] = [("neutral", 0.0)]
    for dim in DIMS:
        conds += [(dim, float(d)) for d in cfg.deltas if float(d) != 0]
    return conds


def cond_key(dim: str, delta: float) -> str:
    return "neutral" if dim == "neutral" else f"{dim}_{delta:+g}"


def deltas_for(dim: str, delta: float) -> Dict[str, float]:
    return {} if dim == "neutral" else {dim: delta}


def evaluate_condition(bundle, task, examples, dim: str, delta: float, cfg) -> Dict[str, Any]:
    prompts = [e.prompt for e in examples]
    outs = bundle.generate(prompts, cfg.max_new_tokens[task.name], deltas=deltas_for(dim, delta))
    jprompts = [task.judge_prompt(e, o.text) if o.text.strip() else None for e, o in zip(examples, outs)]
    judged = (run_judge(bundle, jprompts, cfg.judge_max_new_tokens, cfg.judge_batch_size)
              if any(p is not None for p in jprompts) else [None] * len(outs))
    records = []
    for e, o, j in zip(examples, outs, judged):
        rec = {"id": e.id, "response": o.text, "n_new_tokens": o.n_new_tokens, "finished": o.finished,
               "judge_raw": j}
        rec.update(task.score(e, o.text, o.finished, j))
        records.append(rec)
    return {"task": task.name, "dim": dim, "delta": delta,
            "alpha": bundle.steering.alpha if bundle.steering else None,
            "summary": aggregate([{k: v for k, v in r.items() if k not in ("response", "judge_raw", "id",
                                                                           "n_new_tokens", "finished")}
                                  for r in records], task.has_correctness),
            "records": records}


def run_llm_stage(bundle, cfg, run_dir: str, log=print, subdir: str = "llm", only_tasks=None) -> None:
    root = os.path.join(run_dir, subdir)
    conds = conditions(cfg)
    for name, n in cfg.tasks.items():
        if only_tasks and name not in only_tasks:
            continue
        task = REGISTRY[name]
        tdir = os.path.join(root, name)
        os.makedirs(tdir, exist_ok=True)
        examples = task.load(n, cfg.seed)
        with open(os.path.join(tdir, "examples.json"), "w") as f:
            json.dump([{"id": e.id, "prompt": e.prompt} for e in examples], f)
        for i, (dim, delta) in enumerate(conds):
            path = os.path.join(tdir, f"{cond_key(dim, delta)}.json")
            if cfg.resume and os.path.exists(path):
                continue
            t0 = time.time()
            res = evaluate_condition(bundle, task, examples, dim, delta, cfg)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f)
            os.replace(tmp, path)                              # a killed job never leaves a half-written file
            s = res["summary"]
            log(f"[{name}] {cond_key(dim, delta):>14} ({i + 1}/{len(conds)}) "
                f"AVR={s.get('AVR', float('nan')):.3f}"
                + (f" TSR={s['TSR']:.3f}" if "TSR" in s else "")
                + f"  {time.time() - t0:.0f}s  {bundle.throughput():.0f} tok/s")


def load_summaries(run_dir: str, subdir: str = "llm") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{task: {cond_key: {'dim','delta','summary'}}} from the saved condition files."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    root = os.path.join(run_dir, subdir)
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        tdir = os.path.join(root, name)
        if not os.path.isdir(tdir):
            continue
        for fn in sorted(os.listdir(tdir)):
            if fn.endswith(".json") and fn != "examples.json":
                with open(os.path.join(tdir, fn)) as f:
                    d = json.load(f)
                out.setdefault(name, {})[fn[:-5]] = {"dim": d["dim"], "delta": d["delta"],
                                                     "summary": d["summary"]}
    return out
