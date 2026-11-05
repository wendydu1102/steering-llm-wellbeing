"""Subjective task (paper Sec. 5.3): TinyStories open-ended generation, scored by an LLM judge
on relevance, coherence, creativity and conciseness (1-5)."""
from __future__ import annotations

import itertools
import re
from typing import Any, Dict, List, Optional

from .common import Example, Task, sample

CRITERIA = ("relevance", "coherence", "creativity", "conciseness")


class TinyStories(Task):
    name = "tinystories"
    has_correctness = False

    def load(self, n: int, seed: int) -> List[Example]:
        from datasets import load_dataset
        stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        pool = []
        for i, r in enumerate(itertools.islice(stream, 4000)):
            sents = re.split(r"(?<=[.!?])\s+", r["text"].strip())
            if len(sents) >= 6 and 300 <= len(r["text"]) <= 1800:
                pool.append((i, " ".join(sents[:3])))
        out = []
        for i, opening in sample(pool, n, seed):
            prompt = ("Here is the beginning of a children's story. Continue the story in about 100 words. "
                      f"Write only the continuation.\n\nBeginning: {opening}")
            out.append(Example(f"tinystories-{i}", prompt, {"opening": opening}))
        return out

    def judge_prompt(self, ex: Example, text: str) -> Optional[str]:
        return ("You are a careful evaluator of children's story continuations. Rate the continuation on "
                "four criteria, each an integer from 1 (very poor) to 5 (excellent):\n"
                "- relevance: stays consistent with the characters, setting and plot of the beginning\n"
                "- coherence: logically and grammatically coherent and well structured\n"
                "- creativity: novel, imaginative content\n"
                "- conciseness: free of redundancy and padding for its length\n\n"
                f"Beginning:\n{ex.meta['opening']}\n\nContinuation:\n{text[:2500]}\n\n"
                "Respond with only a JSON object: "
                "{\"relevance\": n, \"coherence\": n, \"creativity\": n, \"conciseness\": n}")

    def score(self, ex, text, finished, judge: Optional[dict] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"valid": bool(finished and len(text.split()) >= 20)}
        for c in CRITERIA:
            v = (judge or {}).get(c)
            out[c] = float(v) if isinstance(v, (int, float)) and 1 <= v <= 5 else None
        return out
