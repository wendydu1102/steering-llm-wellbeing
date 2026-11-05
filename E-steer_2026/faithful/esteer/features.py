"""VAD feature identification (paper Sec. 4.2): positive-negative contrastive pairs that fix the
task description and vary only the assigned emotion label, compared in the SAE latent space.

Per dimension: for every (pair, task) take the difference of task-token-averaged SAE activations
(positive word minus negative word), keep the 50 latents with the largest |difference| per row, and
retain those that appear with a consistent sign in at least `stability_tau` of the rows.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .emotions import build_pairs
from .sae import fraction_variance_explained
from .steering import DIMS, decoder_unit_direction, latent_unit_vector

FRAME = "Right now you are feeling {label}.\n\n{task}"


@torch.no_grad()
def task_latent_means(bundle, sae, label: str, tasks: List[str], batch_size: int) -> torch.Tensor:
    """(n_tasks, d_sae): SAE activations averaged over the task-description tokens only, with the
    emotion label in the prompt prefix. Returned on CPU, float32."""
    out = torch.zeros(len(tasks), sae.d_sae)
    for s in range(0, len(tasks), batch_size):
        chunk = tasks[s:s + batch_size]
        chats = [bundle.chat_text(FRAME.format(label=label, task=t)) for t in chunk]
        spans = []
        for chat, t in zip(chats, chunk):
            a = chat.rfind(t)
            if a < 0:
                raise RuntimeError("task text not found verbatim in the chat-templated prompt; "
                                   "the chat template altered it, so task-token spans cannot be located")
            spans.append((a, a + len(t)))
        h, enc, offsets = bundle.block_outputs(chats)
        attn = enc["attention_mask"].bool()
        for b, (a, z) in enumerate(spans):
            sel = attn[b] & (offsets[b, :, 0] >= a) & (offsets[b, :, 1] <= z) & (offsets[b, :, 1] > offsets[b, :, 0])
            if not sel.any():
                raise RuntimeError("no task tokens located; chat template/offset mismatch")
            out[s + b] = sae.encode(h[b][sel.to(h.device)]).mean(0).cpu()
    return out


def identify_features(delta_rows: torch.Tensor, n_top: int, tau: float, min_features: int) -> dict:
    """delta_rows: (n_rows, d_sae) positive-minus-negative latent differences."""
    n, d = delta_rows.shape
    top = delta_rows.abs().topk(min(n_top, d), dim=1).indices
    signs = torch.sign(delta_rows.gather(1, top)).flatten()
    flat = top.flatten()
    appear = torch.zeros(d).scatter_add_(0, flat, torch.ones_like(signs))
    pos = torch.zeros(d).scatter_add_(0, flat, (signs > 0).float())
    neg = appear - pos
    stability = torch.maximum(pos, neg) / n
    sign = torch.where(pos >= neg, 1.0, -1.0)
    keep = stability >= tau
    relaxed = False
    if int(keep.sum()) < min_features:                         # threshold too strict: take the best few
        relaxed = True
        keep = torch.zeros(d, dtype=torch.bool)
        keep[stability.topk(min_features).indices] = True
    idx = keep.nonzero().flatten()
    mean_delta = delta_rows.mean(0)
    weights = sign[idx] * mean_delta[idx].abs().clamp_min(1e-12)
    return dict(indices=idx.tolist(), weights=weights.tolist(), stability=stability[idx].tolist(),
                n_rows=n, tau=tau, relaxed_threshold=relaxed)


def run_feature_identification(bundle, sae, cfg, task_registry, log=print) -> dict:
    pairs_by_dim, label_source = build_pairs(cfg.nrc_vad_path, cfg.n_label_pairs)
    tasks: List[str] = []
    for name in cfg.tasks:
        tasks += task_registry[name].descriptions(cfg.n_feature_tasks, cfg.seed + 1000)
    log(f"feature identification: {len(tasks)} task descriptions, label source = {label_source}")
    result = {"label_source": label_source, "n_task_descriptions": len(tasks), "dims": {}}
    for dim in DIMS:
        pairs = pairs_by_dim[dim]
        labels = sorted({p["pos"] for p in pairs} | {p["neg"] for p in pairs})
        means = {lab: task_latent_means(bundle, sae, lab, tasks, cfg.feature_batch_size) for lab in labels}
        rows = torch.cat([means[p["pos"]] - means[p["neg"]] for p in pairs], 0)
        feats = identify_features(rows, cfg.n_top, cfg.stability_tau, cfg.min_features)
        feats["pairs"] = pairs
        result["dims"][dim] = feats
        log(f"  {dim}: {len(feats['indices'])} stable latents from {rows.shape[0]} rows"
            + (" (threshold relaxed to min_features)" if feats["relaxed_threshold"] else ""))
        del means, rows
    return result


def build_directions(sae, features: dict, variant: str = "original", seed: int = 0) -> Dict[str, torch.Tensor]:
    """Decoder-space unit direction per dimension. variant: 'original' | 'random' | 'half'
    (Sec. 5.6: originally identified / randomly selected / half replaced by random latents)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for dim in DIMS:
        f = features["dims"][dim]
        idx = torch.as_tensor(f["indices"])
        w = torch.as_tensor(f["weights"])
        if variant != "original":
            n_rand = len(idx) if variant == "random" else len(idx) // 2
            keep = torch.randperm(len(idx), generator=g)[: len(idx) - n_rand]
            pool = torch.ones(sae.d_sae, dtype=torch.bool)
            pool[idx] = False
            cand = pool.nonzero().flatten()
            new = cand[torch.randperm(len(cand), generator=g)[:n_rand]]
            idx = torch.cat([idx[keep], new])
            w = torch.cat([w[keep], w[torch.randperm(len(w), generator=g)[:n_rand]]])
        u = latent_unit_vector(idx.tolist(), w.tolist(), sae.d_sae)
        out[dim] = decoder_unit_direction(sae, u).cpu()
    return out


def direction_diagnostics(directions: Dict[str, torch.Tensor], features: dict) -> dict:
    """Cosine similarity between the three steering directions and overlap of the latent sets.
    (The local PoC's regression directions were near-collinear; this makes that visible here.)"""
    cos, overlap = {}, {}
    for i, a in enumerate(DIMS):
        for b in DIMS[i + 1:]:
            cos[f"{a}~{b}"] = float(torch.dot(directions[a], directions[b]))
            sa, sb = set(features["dims"][a]["indices"]), set(features["dims"][b]["indices"])
            overlap[f"{a}~{b}"] = len(sa & sb)
    return {"cosine": cos, "shared_latents": overlap}


@torch.no_grad()
def sae_layer_check(bundle, sae, texts: List[str], layers: Optional[List[int]] = None) -> dict:
    """Fraction of variance the layer-k SAE explains when fed block outputs from layers k-1, k, k+1.
    The best-matching layer is the SAE's true hook point; a peak away from cfg.layer means an index offset.
    Position 0 (attention sink, huge norm) is excluded."""
    layers = layers or [bundle.cfg.layer - 1, bundle.cfg.layer, bundle.cfg.layer + 1]
    out = {}
    for L in layers:
        if not 0 <= L < bundle.n_layers:
            continue
        h, enc, _ = bundle.block_outputs(texts, layer=L)
        attn = enc["attention_mask"].bool().clone()
        attn[:, 0] = False
        x = h[attn.to(h.device)].float()
        out[str(L)] = fraction_variance_explained(x, sae.decode(sae.encode(x)))
    return out
