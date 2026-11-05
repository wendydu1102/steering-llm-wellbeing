"""Agent benchmarks (paper Sec. 5.1): HotpotQA, 'Scientific' (CAMEL science datasets, Li et al. 2023), GAIA."""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..tasks.common import sample, stratified_sample


@dataclass
class AgentItem:
    id: str
    dataset: str
    question: str
    context: str
    gold: str
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_qa(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_qa(pred) == normalize_qa(gold)


def f1(pred: str, gold: str) -> float:
    p, g = normalize_qa(pred).split(), normalize_qa(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)


def load_hotpotqa(n: int, seed: int) -> List[AgentItem]:
    from datasets import load_dataset
    ds = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    out = []
    for r in stratified_sample(ds, lambda r: r["level"], n, seed):
        ctx = "\n".join(f"[{t}] {' '.join(s)}" for t, s in zip(r["context"]["title"], r["context"]["sentences"]))
        out.append(AgentItem(f"hotpotqa-{r['id']}", "hotpotqa", r["question"], ctx, r["answer"],
                             {"level": r["level"]}))
    return out


def load_science(n: int, seed: int) -> List[AgentItem]:
    """CAMEL physics/chemistry/biology problem-solution pairs; the reference solution is the gold."""
    from datasets import load_dataset
    subjects = ["physics", "chemistry", "biology"]
    per = max(1, n // len(subjects))
    out = []
    for subj in subjects:
        ds = load_dataset(f"camel-ai/{subj}", split="train", streaming=True)
        pool = []
        for i, r in enumerate(ds):
            if i >= 3000:
                break
            if 40 <= len(r["message_1"]) <= 600 and len(r["message_2"]) <= 2500:
                pool.append((i, r["message_1"], r["message_2"]))
        for i, q, a in sample(pool, per, seed):
            out.append(AgentItem(f"science-{subj}-{i}", "science", q, "", a, {"subject": subj}))
    return out


def load_gaia(n: int, seed: int) -> List[AgentItem]:
    """GAIA level-1 validation questions WITHOUT attachments (text-only). Gated: needs HF_TOKEN."""
    from datasets import load_dataset
    ds = [r for r in load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation") if not r["file_name"]]
    return [AgentItem(f"gaia-{r['task_id']}", "gaia", r["Question"], "", r["Final answer"], {})
            for r in sample(ds, n, seed)]


LOADERS = {"hotpotqa": load_hotpotqa, "science": load_science, "gaia": load_gaia}
