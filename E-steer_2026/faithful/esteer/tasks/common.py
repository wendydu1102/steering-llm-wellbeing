"""Shared task types and answer-parsing helpers."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Example:
    id: str
    prompt: str                       # user prompt without any emotion framing
    meta: Dict[str, Any] = field(default_factory=dict)


class Task:
    name = "task"
    has_correctness = True            # contributes a task-success rate (TSR)

    def load(self, n: int, seed: int) -> List[Example]:
        raise NotImplementedError

    def descriptions(self, n: int, seed: int) -> List[str]:
        """Task descriptions (prompts only, no labels) for contrastive feature identification."""
        return [e.prompt for e in self.load(n, seed)]

    def judge_prompt(self, ex: Example, text: str) -> Optional[str]:
        return None                   # tasks scored by an LLM judge override this

    def score(self, ex: Example, text: str, finished: bool, judge: Optional[dict] = None) -> Dict[str, Any]:
        raise NotImplementedError


def sample(items: List[Any], n: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    items = list(items)
    if n >= len(items):
        return items
    return rng.sample(items, n)


def stratified_sample(items: List[Any], key: Callable[[Any], Any], n: int, seed: int) -> List[Any]:
    """Proportional sampling across strata (e.g. difficulty level), deterministic in `seed`."""
    if n >= len(items):
        return list(items)
    groups: Dict[Any, List[Any]] = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    rng = random.Random(seed)
    total, out = len(items), []
    keys = sorted(groups, key=str)
    quotas = {k: int(n * len(groups[k]) / total) for k in keys}
    for k in sorted(keys, key=lambda k: -len(groups[k]))[: n - sum(quotas.values())]:
        quotas[k] += 1
    for k in keys:
        out.extend(rng.sample(groups[k], min(quotas[k], len(groups[k]))))
    return out


# ---- answer parsing -------------------------------------------------------------------------
def last_boxed(s: str) -> Optional[str]:
    """Content of the last \\boxed{...} (brace-matched), or None."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def normalize_math(s: str) -> str:
    s = s.strip().replace("\n", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\s+", "", s).rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def math_equal(pred: str, gold: str) -> bool:
    if normalize_math(pred) == normalize_math(gold):
        return True
    try:                              # optional symbolic check
        from math_verify import parse, verify
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        return False


_LETTER = re.compile(r"answer\s*[:：]\s*\**\(?\s*([A-D])\b", re.IGNORECASE)


def answer_letter(text: str) -> Optional[str]:
    m = _LETTER.findall(text)
    return m[-1].upper() if m else None


_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def last_code_block(text: str) -> Optional[str]:
    m = _CODE.findall(text)
    return m[-1] if m else None


def parse_json_obj(text: str) -> Optional[dict]:
    """First balanced-looking {...} in text parsed as JSON, else None."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
