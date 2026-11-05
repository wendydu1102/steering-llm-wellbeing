"""Objective tasks (paper Sec. 5.2): LogiQA 2.0 (logic), HumanEval (code), MATH (quantitative)."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from .common import (Example, Task, answer_letter, last_boxed, last_code_block, math_equal, sample,
                     stratified_sample)

LETTERS = "ABCD"


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


class LogiQA(Task):
    """datatune/LogiQA2.0 (test split). Each row's `text` is a JSON string with the item. The mirror also holds
    Chinese-language items; only English multiple-choice items are used."""
    name = "logiqa"

    def load(self, n: int, seed: int) -> List[Example]:
        from datasets import load_dataset
        ds = load_dataset("datatune/LogiQA2.0", split="test")
        rows = []
        for idx, r in enumerate(ds):
            try:
                d = json.loads(r["text"])
            except json.JSONDecodeError:                   # a few rows in the mirror are blank/truncated
                continue
            # The mirror interleaves the multiple-choice items (options/answer) with LogiQA 2.0's NLI-style
            # items (premise/hypothesis/label); only the multiple-choice form is used here.
            if (isinstance(d, dict) and len(d.get("options", [])) == 4 and d.get("answer") in range(4)
                    and "text" in d and "question" in d
                    and not _has_cjk(d["text"] + d["question"] + "".join(d["options"]))):
                d["_row"] = idx                          # ids are not unique/present in the mirror; use row index
                rows.append(d)
        out = []
        for d in sample(rows, n, seed):
            opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(d["options"]))
            prompt = ("Read the passage and answer the multiple-choice question.\n\n"
                      f"Passage: {d['text']}\n\nQuestion: {d['question']}\n\nOptions:\n{opts}\n\n"
                      "Think through the problem step by step, then finish with a final line of the form "
                      "\"Answer: X\" where X is one of A, B, C, D.")
            out.append(Example(f"logiqa-test-{d['_row']}", prompt, {"gold": LETTERS[d["answer"]]}))
        return out

    def score(self, ex, text, finished, judge=None) -> Dict[str, Any]:
        pred = answer_letter(text)
        return {"valid": pred is not None, "correct": pred == ex.meta["gold"], "pred": pred}


class HumanEval(Task):
    """openai/openai_humaneval; pass@1 under greedy decoding via executing the unit tests.
    NOTE: this runs model-generated code in a subprocess (timeout, temp dir); do it only in the
    job container, never on a machine with data you care about."""
    name = "humaneval"
    TIMEOUT_S = 15

    def load(self, n: int, seed: int) -> List[Example]:
        from datasets import load_dataset
        ds = list(load_dataset("openai/openai_humaneval", split="test"))
        out = []
        for d in sample(ds, n, seed):
            prompt = ("Complete the following Python function. Respond with the complete function "
                      "(including its signature) inside a single ```python code block and nothing else.\n\n"
                      f"```python\n{d['prompt']}```")
            out.append(Example(d["task_id"], prompt, {"header": d["prompt"], "test": d["test"],
                                                      "entry_point": d["entry_point"]}))
        return out

    def score(self, ex, text, finished, judge=None) -> Dict[str, Any]:
        code = last_code_block(text)
        if code is None:
            return {"valid": False, "correct": False}
        try:
            ast.parse(code)
        except SyntaxError:
            return {"valid": False, "correct": False}
        imports = "\n".join(l for l in ex.meta["header"].splitlines() if l.startswith(("import ", "from ")))
        program = f"{imports}\n{code}\n\n{ex.meta['test']}\n\ncheck({ex.meta['entry_point']})\n"
        return {"valid": True, "correct": run_python(program, self.TIMEOUT_S)}


def run_python(program: str, timeout: int) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, path], cwd=tmp, capture_output=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


class MathQA(Task):
    """EleutherAI/hendrycks_math (all 7 subjects, test), sampled proportionally across difficulty level."""
    name = "math"
    SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]

    def load(self, n: int, seed: int) -> List[Example]:
        from datasets import load_dataset
        rows = []
        for subj in self.SUBJECTS:
            for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", subj, split="test")):
                gold = last_boxed(r["solution"])
                if gold is not None and r["level"].startswith("Level ") and r["level"][-1].isdigit():
                    rows.append((f"math-{subj}-{i}", r["problem"], gold, r["level"]))
        out = []
        for ex_id, problem, gold, level in stratified_sample(rows, lambda r: r[3], n, seed):
            prompt = ("Solve the following math problem. Reason step by step, then give the final answer "
                      f"in the form \\boxed{{...}}.\n\nProblem: {problem}")
            out.append(Example(ex_id, prompt, {"gold": gold, "level": level}))
        return out

    def score(self, ex, text, finished, judge=None) -> Dict[str, Any]:
        pred = last_boxed(text)
        if pred is None:
            return {"valid": False, "correct": False}
        return {"valid": True, "correct": math_equal(pred, ex.meta["gold"]), "pred": pred}
