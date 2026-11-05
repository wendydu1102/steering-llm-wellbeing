from .objective import HumanEval, LogiQA, MathQA
from .safety import HarmBench
from .subjective import TinyStories

REGISTRY = {t.name: t for t in (LogiQA(), HumanEval(), MathQA(), TinyStories(), HarmBench())}


def get_task(name):
    return REGISTRY[name]
