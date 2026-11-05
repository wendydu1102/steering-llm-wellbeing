"""Model loading and (optionally steered) batched greedy generation."""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

# Both a bare laptop install and DSMLP's scipy-ml image ship TensorFlow next to PyTorch; probing for it at
# `transformers` import time deadlocked locally (protobuf/absl mutex), so keep the TF/Flax backends off.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


@dataclass
class GenOut:
    text: str
    n_new_tokens: int
    finished: bool            # emitted an end-of-sequence token before max_new_tokens


def resolve_device(requested: str) -> str:
    if requested in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class ModelBundle:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        dtype = getattr(torch, cfg.model_dtype)
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        gc = self.model.generation_config.eos_token_id
        self.eos_ids = set(gc if isinstance(gc, (list, tuple)) else [gc]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)
        self.n_layers = self.model.config.num_hidden_layers
        if not 0 <= cfg.layer < self.n_layers:
            raise ValueError(f"layer {cfg.layer} out of range for {cfg.model_name} ({self.n_layers} layers)")
        self.layer_module = self.model.model.layers[cfg.layer]
        self.steering = None            # set by the runner once directions/alpha are known
        self.tokens_generated = 0
        self.seconds_generating = 0.0

    # ---- prompting -------------------------------------------------------------------------
    def chat_text(self, user_text: str, system: Optional[str] = None, thinking: Optional[bool] = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.cfg.enable_thinking if thinking is None else thinking)

    # ---- generation ------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, user_texts: List[str], max_new_tokens: int, deltas: Optional[Dict[str, float]] = None,
                 batch_size: Optional[int] = None, system: Optional[str] = None,
                 thinking: Optional[bool] = None) -> List[GenOut]:
        """Greedy generation for chat prompts. `deltas` = {'valence': d, ...} enables steering;
        None / all-zero leaves the model untouched (also used for the unsteered judge)."""
        texts = [self.chat_text(t, system, thinking) for t in user_texts]
        bs = batch_size or self.cfg.batch_size
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: List[Optional[GenOut]] = [None] * len(texts)
        for start in range(0, len(order), bs):
            idxs = order[start:start + bs]
            for i, out in zip(idxs, self._generate_chunk([texts[i] for i in idxs], max_new_tokens, deltas)):
                results[i] = out
        return results  # type: ignore[return-value]

    def _generate_chunk(self, batch: List[str], max_new_tokens: int, deltas) -> List[GenOut]:
        try:
            return self._generate_once(batch, max_new_tokens, deltas)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(batch) // 2
            return (self._generate_chunk(batch[:half], max_new_tokens, deltas)
                    + self._generate_chunk(batch[half:], max_new_tokens, deltas))

    def _generate_once(self, batch: List[str], max_new_tokens: int, deltas) -> List[GenOut]:
        enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        ctx = (self.steering.attach(self.layer_module, deltas)
               if (self.steering is not None and deltas) else contextlib.nullcontext())
        t0 = time.time()
        with ctx:
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        res = []
        for row in new.tolist():
            n, finished = len(row), False
            for j, tid in enumerate(row):
                if tid in self.eos_ids:
                    n, finished = j, True
                    break
            res.append(GenOut(self.tok.decode(row[:n], skip_special_tokens=True).strip(), n, finished))
        self.tokens_generated += sum(r.n_new_tokens for r in res)
        self.seconds_generating += time.time() - t0
        return res

    def throughput(self) -> float:
        return self.tokens_generated / self.seconds_generating if self.seconds_generating else 0.0

    # ---- hidden-state capture (feature identification / SAE sanity) --------------------------
    @torch.no_grad()
    def block_outputs(self, texts: List[str], layer: Optional[int] = None):
        """Run texts (right-padded) and return (hidden[B,T,D] at the output of block `layer`,
        encoding with offset mappings). Used by feature identification and the SAE preflight."""
        layer = self.cfg.layer if layer is None else layer
        saved = {}

        def grab(module, args, output):
            saved["h"] = (output[0] if isinstance(output, tuple) else output).detach()

        handle = self.model.model.layers[layer].register_forward_hook(grab)
        side = self.tok.padding_side
        self.tok.padding_side = "right"
        try:
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                           return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            self.model.model(**{k: v.to(self.device) for k, v in enc.items()})   # base model: skips the LM head/logits
        finally:
            handle.remove()
            self.tok.padding_side = side
        return saved["h"], enc, offsets
