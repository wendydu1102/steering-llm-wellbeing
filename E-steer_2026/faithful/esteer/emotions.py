"""Emotion words, VAD coordinates and contrastive-pair selection (paper Sec. 3.1, 4.2).

The paper represents emotion as VAD in [-10, 10] and builds positive/negative pairs by
varying only the emotion label. NRC VAD Lexicon v2 (Mohammad, 2025; scores in [-1, 1])
scaled by 10 matches that range, so we use it when the user supplies the file (it is
licensed for research use and is not redistributed here). Without it, a hand-picked
fallback is available but must be acknowledged (label-handpicked).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .steering import DIMS

# Candidate emotion words (no numeric values here: coordinates always come from the lexicon).
CANDIDATES: List[str] = """
joyful delighted cheerful happy glad pleased content grateful hopeful proud elated ecstatic amused
relieved optimistic serene calm relaxed peaceful tranquil satisfied confident powerful assertive
brave triumphant excited thrilled enthusiastic energetic eager playful loving affectionate admiring
inspired surprised amazed astonished curious sad miserable depressed gloomy despairing heartbroken
lonely grieving hopeless ashamed guilty embarrassed disgusted resentful bitter jealous angry furious
enraged irritated annoyed hostile afraid terrified anxious nervous worried panicked alarmed stressed
helpless powerless weak submissive timid overwhelmed vulnerable intimidated bored tired sleepy drowsy
sluggish apathetic numb exhausted frustrated agitated restless tense shocked dominant commanding
fearless bold authoritative humiliated insecure
""".split()

# Fallback poles, used only without the lexicon (ack: label-handpicked). They are NOT matched on the
# other two dimensions (e.g. 'joyful' is also high-arousal), which is why this is a weaker substitute.
HAND_POLES: Dict[str, Tuple[List[str], List[str]]] = {
    "valence": (["joyful", "delighted", "cheerful", "grateful", "hopeful", "content", "elated", "pleased"],
                ["miserable", "devastated", "despairing", "gloomy", "bitter", "ashamed", "disgusted", "resentful"]),
    "arousal": (["excited", "thrilled", "frantic", "agitated", "energized", "alarmed", "restless", "exhilarated"],
                ["calm", "relaxed", "sleepy", "drowsy", "serene", "tranquil", "sluggish", "placid"]),
    "dominance": (["confident", "powerful", "assertive", "commanding", "authoritative", "fearless", "empowered", "bold"],
                  ["helpless", "powerless", "submissive", "overwhelmed", "timid", "meek", "vulnerable", "intimidated"]),
}


def load_nrc_vad(path: str) -> Dict[str, Tuple[float, float, float]]:
    """Parse NRC-VAD-Lexicon (tab-separated: term, valence, arousal, dominance) into [-10, 10].

    v2 scores are in [-1, 1] (x10); v1 scores are in [0, 1] (mapped to [-10, 10])."""
    rows: Dict[str, Tuple[float, float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                v, a, d = (float(x) for x in parts[1:4])
            except ValueError:
                continue                                       # header row
            rows[parts[0].strip().lower()] = (v, a, d)
    if not rows:
        raise ValueError(f"no VAD rows parsed from {path}")
    lo = min(min(r) for r in rows.values())
    if lo >= 0.0:                                              # v1-style [0, 1]
        return {w: tuple((x * 2 - 1) * 10 for x in r) for w, r in rows.items()}
    return {w: tuple(x * 10 for x in r) for w, r in rows.items()}


def select_pairs(lex: Dict[str, Tuple[float, float, float]], dim: str, n_pairs: int,
                 min_abs: float = 3.0) -> List[dict]:
    """High-vs-low word pairs on `dim`, matched to be as close as possible on the other two dims."""
    di = DIMS.index(dim)
    others = [i for i in range(3) if i != di]
    cands = [(w, lex[w]) for w in CANDIDATES if w in lex]
    pos = [(w, v) for w, v in cands if v[di] >= min_abs]
    neg = [(w, v) for w, v in cands if v[di] <= -min_abs]
    scored = []
    for pw, pv in pos:
        for nw, nv in neg:
            other = math.sqrt(sum((pv[i] - nv[i]) ** 2 for i in others))
            gap = pv[di] - nv[di]
            scored.append((other - 0.25 * gap, pw, nw, pv, nv, other, gap))
    scored.sort(key=lambda t: t[0])
    used, out = set(), []
    for _, pw, nw, pv, nv, other, gap in scored:
        if pw in used or nw in used:
            continue
        used.update((pw, nw))
        out.append(dict(pos=pw, neg=nw, vad_pos=list(pv), vad_neg=list(nv),
                        gap=round(gap, 3), other_dims_distance=round(other, 3)))
        if len(out) == n_pairs:
            break
    if len(out) < min(2, n_pairs):
        raise ValueError(f"fewer than 2 usable contrast pairs for {dim}; check the lexicon file")
    return out


def handpicked_pairs(dim: str, n_pairs: int) -> List[dict]:
    pos, neg = HAND_POLES[dim]
    return [dict(pos=p, neg=n, vad_pos=None, vad_neg=None, gap=None, other_dims_distance=None)
            for p, n in list(zip(pos, neg))[:n_pairs]]


def build_pairs(nrc_vad_path: Optional[str], n_pairs: int) -> Tuple[Dict[str, List[dict]], str]:
    """Returns ({dim: pairs}, source) where source is 'nrc-vad' or 'handpicked'."""
    if nrc_vad_path:
        lex = load_nrc_vad(nrc_vad_path)
        return {d: select_pairs(lex, d, n_pairs) for d in DIMS}, "nrc-vad"
    return {d: handpicked_pairs(d, n_pairs) for d in DIMS}, "handpicked"
