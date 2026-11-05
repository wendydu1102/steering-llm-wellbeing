"""Planner / decider / executor agent (paper Sec. 5.1, 5.5).

RECONSTRUCTION (ack: agent-reconstruction): the paper describes the three modules and their metrics but
not prompts, protocols, tools or how confidence is obtained. Everything below is our concrete reading:
  planner  -> JSON plan of 2-5 steps; can replan with the decider's feedback
  decider  -> judges plan feasibility (+confidence), diagnoses, and selects the final answer
  executor -> carries out each step by LLM analysis (no external tools) and reports confidence
Emotion steering (the same hook as the LLM tasks) is applied to every planner/decider/executor call.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from ..evaluate import cond_key, conditions, deltas_for
from ..judge import run_judge
from ..tasks.common import parse_json_obj
from .data import LOADERS, AgentItem, exact_match, f1

ACTIONS = {"extract", "compute", "reason", "verify"}


def _conf(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if 1 < x <= 100:
        x /= 100.0
    return x if 0.0 <= x <= 1.0 else None


def _ctx(item: AgentItem) -> str:
    return f"Context:\n{item.context}\n\n" if item.context else ""


def parse_plan(text: str) -> Optional[List[dict]]:
    obj = parse_json_obj(text)
    steps = obj.get("steps") if obj else None
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        return None
    for s in steps:
        if not isinstance(s, dict) or s.get("action") not in ACTIONS or not str(s.get("description", "")).strip():
            return None
    return steps


def planner_prompt(item: AgentItem, feedback: Optional[str] = None, prev: Optional[list] = None) -> str:
    p = ("You are the PLANNER of a problem-solving agent. Break the task into 2-5 concrete steps that an "
         "executor can carry out using only reasoning over the given information (no external tools).\n"
         "Allowed step actions: extract (pull out relevant facts), compute (calculate or derive), "
         "reason (draw an inference), verify (check a result).\n\n"
         f"{_ctx(item)}Task: {item.question}\n\n")
    if feedback is not None:
        p += (f"Your previous plan was rejected. Previous plan: {json.dumps(prev) if prev else '(invalid)'}\n"
              f"Feedback: {feedback}\nProduce an improved plan.\n\n")
    return p + ("Respond with ONLY a JSON object: "
                "{\"steps\": [{\"action\": \"extract|compute|reason|verify\", \"description\": \"...\"}]}")


def decider_plan_prompt(item: AgentItem, plan: list) -> str:
    return ("You are the DECIDER of a problem-solving agent. Judge whether the plan below is feasible and "
            "sufficient to answer the task using only the given information.\n\n"
            f"{_ctx(item)}Task: {item.question}\n\nPlan: {json.dumps(plan)}\n\n"
            "Respond with ONLY a JSON object: {\"feasible\": true/false, \"confidence\": <number from 0 to 1 "
            "for how sure you are of this judgement>, \"reason\": \"<one sentence>\"}")


def step_prompt(item: AgentItem, plan: list, results: List[str], j: int) -> str:
    done = "\n".join(f"{k + 1}. {r}" for k, r in enumerate(results)) or "(none yet)"
    s = plan[j]
    return ("You are the EXECUTOR of a problem-solving agent. Carry out ONE step of the plan.\n\n"
            f"{_ctx(item)}Task: {item.question}\n\nPlan: {json.dumps(plan)}\n\nResults so far:\n{done}\n\n"
            f"Current step {j + 1} ({s['action']}): {s['description']}\n\n"
            "Respond with ONLY a JSON object: {\"result\": \"<what you found or derived>\", "
            "\"confidence\": <number from 0 to 1>}")


def final_prompt(item: AgentItem, plan: Optional[list], results: Optional[List[str]]) -> str:
    body = ""
    if plan is not None:
        done = "\n".join(f"{k + 1}. {r}" for k, r in enumerate(results or []))
        body = f"Plan: {json.dumps(plan)}\n\nStep results:\n{done}\n\n"
    return ("You are the EXECUTOR of a problem-solving agent. Give the final answer to the task. "
            "Keep it short (a few words or one sentence).\n\n"
            f"{_ctx(item)}Task: {item.question}\n\n{body}"
            "Respond with ONLY a JSON object: {\"answer\": \"<final answer>\", "
            "\"confidence\": <number from 0 to 1 that the answer is correct>}")


def select_prompt(item: AgentItem, cands: List[dict]) -> str:
    lines = "\n".join(f"Candidate {k + 1} (executor confidence {c['conf']:.2f}): {c['answer']}"
                      for k, c in enumerate(cands))
    return ("You are the DECIDER of a problem-solving agent. Several candidate answers were produced for the "
            f"task. Select the best one.\n\n{_ctx(item)}Task: {item.question}\n\n{lines}\n\n"
            "Respond with ONLY a JSON object: {\"selected\": <candidate number>}")


def science_judge_prompt(item: AgentItem, answer: str) -> str:
    return ("Compare a candidate answer with the reference solution. Does the candidate's final answer agree "
            "with the reference's final answer (same result, allowing different wording or rounding)?\n\n"
            f"Question:\n{item.question}\n\nReference solution:\n{item.gold[:2500]}\n\n"
            f"Candidate answer:\n{answer[:800]}\n\nRespond with ONLY a JSON object: {{\"correct\": true/false}}")


class AgentRunner:
    def __init__(self, bundle, cfg, deltas: Dict[str, float]):
        self.bundle, self.cfg, self.deltas = bundle, cfg, deltas

    def llm(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        return [o.text for o in self.bundle.generate(prompts, self.cfg.agent_max_new_tokens, deltas=self.deltas)]

    def execute(self, items: List[AgentItem], plans: Dict[int, list]) -> Dict[int, dict]:
        st = {i: {"results": [], "ok": 0, "planned": len(p)} for i, p in plans.items()}
        for j in range(max((len(p) for p in plans.values()), default=0)):
            act = [i for i, p in plans.items() if j < len(p)]
            for i, t in zip(act, self.llm([step_prompt(items[i], plans[i], st[i]["results"], j) for i in act])):
                o = parse_json_obj(t)
                if o and o.get("result") is not None:
                    st[i]["ok"] += 1
                    st[i]["results"].append(str(o["result"])[:600])
                else:
                    st[i]["results"].append("(step failed)")
        idx = list(plans)
        out = {}
        for i, t in zip(idx, self.llm([final_prompt(items[i], plans[i], st[i]["results"]) for i in idx])):
            o = parse_json_obj(t) or {}
            out[i] = {"answer": (str(o["answer"]).strip() if o.get("answer") is not None else None),
                      "conf": _conf(o.get("confidence")), "completed_steps": st[i]["ok"],
                      "planned_steps": st[i]["planned"]}
        return out

    def run(self, items: List[AgentItem]) -> List[Dict[str, Any]]:
        N = len(items)
        plan1 = [parse_plan(t) for t in self.llm([planner_prompt(it) for it in items])]
        valid = [i for i in range(N) if plan1[i]]
        decision: Dict[int, tuple] = {}
        for i, t in zip(valid, self.llm([decider_plan_prompt(items[i], plan1[i]) for i in valid])):
            o = parse_json_obj(t) or {}
            decision[i] = (o.get("feasible") if isinstance(o.get("feasible"), bool) else None,
                           _conf(o.get("confidence")), str(o.get("reason", ""))[:300])
        trig = [i for i in range(N) if plan1[i] is None or decision.get(i, (None,))[0] is False]
        plan2: Dict[int, Optional[list]] = {}
        for i, t in zip(trig, self.llm([planner_prompt(items[i], feedback=(decision.get(i) or (0, 0, ""))[2]
                                                       or "The plan was not a valid executable plan.",
                                                       prev=plan1[i]) for i in trig])):
            plan2[i] = parse_plan(t)
        exec1 = self.execute(items, {i: plan1[i] for i in valid})
        exec2 = self.execute(items, {i: p for i, p in plan2.items() if p})
        dtexts = self.llm([final_prompt(items[i], None, None) for i in range(N)])
        direct = {}
        for i, t in enumerate(dtexts):
            o = parse_json_obj(t) or {}
            direct[i] = {"answer": (str(o["answer"]).strip() if o.get("answer") is not None else None),
                         "conf": _conf(o.get("confidence"))}

        cands: Dict[int, List[dict]] = {i: [] for i in range(N)}
        for src, ex in (("plan1", exec1), ("plan2", exec2), ("direct", direct)):
            for i, e in ex.items():
                if e["answer"]:
                    cands[i].append({"source": src, "answer": e["answer"], "conf": e["conf"] or 0.0})
        multi = [i for i in range(N) if len(cands[i]) >= 2]
        picked: Dict[int, Optional[int]] = {}
        for i, t in zip(multi, self.llm([select_prompt(items[i], cands[i]) for i in multi])):
            s = (parse_json_obj(t) or {}).get("selected")
            picked[i] = int(s) - 1 if isinstance(s, (int, float)) and 1 <= int(s) <= len(cands[i]) else None

        finals, recs = {}, []
        for i in range(N):
            c = cands[i]
            if not c:
                chosen, sel_ok, rational = None, False, None
            elif len(c) == 1:
                chosen, sel_ok, rational = 0, True, None
            else:
                chosen = picked.get(i)
                sel_ok = chosen is not None
                chosen = 0 if chosen is None else chosen
                rational = bool(sel_ok and c[chosen]["conf"] >= max(x["conf"] for x in c))
            finals[i] = c[chosen]["answer"] if chosen is not None else ""
            recs.append({"id": items[i].id, "plan1_valid": plan1[i] is not None,
                         "replan_triggered": i in trig, "decider_feasible": decision.get(i, (None,))[0],
                         "decider_conf": decision.get(i, (None, None))[1],
                         "plan2_valid": (plan2.get(i) is not None) if i in plan2 else None,
                         "conf1": exec1.get(i, {}).get("conf"), "conf2": exec2.get(i, {}).get("conf"),
                         "steps1": [exec1[i]["completed_steps"], exec1[i]["planned_steps"]] if i in exec1 else None,
                         "steps2": [exec2[i]["completed_steps"], exec2[i]["planned_steps"]] if i in exec2 else None,
                         "n_candidates": len(c), "rational_selection": rational, "selection_valid": sel_ok,
                         "final_answer": finals[i]})
        # success
        if items and items[0].dataset == "science":
            j = run_judge(self.bundle, [science_judge_prompt(items[i], finals[i]) if finals[i] else None
                                        for i in range(N)], self.cfg.judge_max_new_tokens, self.cfg.judge_batch_size)
            for r, jr in zip(recs, j):
                r["correct"] = bool(jr and jr.get("correct") is True)
        else:
            for i, r in enumerate(recs):
                r["correct"] = exact_match(finals[i], items[i].gold) if finals[i] else False
                r["f1"] = f1(finals[i], items[i].gold) if finals[i] else 0.0
        return recs


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = [s for r in recs for s in (r["steps1"], r["steps2"]) if s]
    both = [r["conf2"] - r["conf1"] for r in recs if r["conf1"] is not None and r["conf2"] is not None]
    return {"n": len(recs),
            "plan_validity_rate": _mean([float(r["plan1_valid"]) for r in recs]),
            "replan_frequency": _mean([float(r["replan_triggered"]) for r in recs]),
            "replan_trigger_confidence": _mean([r["decider_conf"] for r in recs
                                                if r["replan_triggered"] and r["decider_feasible"] is False]),
            "replan_improvement": _mean(both), "replan_improvement_n": len(both),
            "rational_selection_rate": _mean([float(r["rational_selection"]) for r in recs
                                              if r["rational_selection"] is not None]),
            "execution_completion_rate": (sum(a for a, _ in steps) / sum(b for _, b in steps)) if steps else None,
            "overall_success_rate": _mean([float(r["correct"]) for r in recs])}


def run_agent_stage(bundle, cfg, run_dir: str, log=print) -> None:
    for name, n in cfg.agent_datasets.items():
        items = LOADERS[name](n, cfg.seed)
        adir = os.path.join(run_dir, "agent", name)
        os.makedirs(adir, exist_ok=True)
        conds = conditions(cfg)
        for k, (dim, delta) in enumerate(conds):
            path = os.path.join(adir, f"{cond_key(dim, delta)}.json")
            if cfg.resume and os.path.exists(path):
                continue
            t0 = time.time()
            recs = AgentRunner(bundle, cfg, deltas_for(dim, delta)).run(items)
            res = {"dataset": name, "dim": dim, "delta": delta, "summary": summarize(recs), "records": recs}
            with open(path + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(path + ".tmp", path)
            s = res["summary"]
            log(f"[agent/{name}] {cond_key(dim, delta):>14} ({k + 1}/{len(conds)}) "
                f"success={s['overall_success_rate']:.3f} plan_valid={s['plan_validity_rate']:.3f} "
                f"{time.time() - t0:.0f}s")
