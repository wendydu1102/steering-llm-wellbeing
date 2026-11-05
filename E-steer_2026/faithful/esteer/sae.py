"""TopK sparse autoencoder (paper Sec. 3.2: z = f_enc(h), h' = f_dec(z); the
decoder is a single linear layer).

Checkpoint format follows the Qwen-Scope release
(Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 and _L0_100): a torch dict with
W_enc (d_sae, d_model), b_enc (d_sae,), W_dec (d_model, d_sae), b_dec (d_model,).
Encoding is the documented  TopK(x @ W_enc.T + b_enc)  with no b_dec subtraction.
"""
from __future__ import annotations

import torch


class TopKSAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, k: int):
        self.W_enc, self.b_enc = W_enc, b_enc
        self.W_dec, self.b_dec = W_dec, b_dec
        self.k = int(k)

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[1]

    @classmethod
    def from_file(cls, path: str, k: int, device="cpu", dtype=torch.float32) -> "TopKSAE":
        sd = torch.load(path, map_location="cpu", weights_only=True)
        missing = {"W_enc", "b_enc", "W_dec", "b_dec"} - set(sd)
        if missing:
            raise KeyError(f"{path}: SAE checkpoint is missing keys {sorted(missing)}")
        t = {n: sd[n].to(device=device, dtype=dtype) for n in ("W_enc", "b_enc", "W_dec", "b_dec")}
        return cls(t["W_enc"], t["b_enc"], t["W_dec"], t["b_dec"], k)

    @classmethod
    def random(cls, d_model: int, d_sae: int, k: int, seed: int = 0, device="cpu") -> "TopKSAE":
        """Untrained random SAE. ONLY for plumbing/smoke tests: it has no
        interpretable features and must never be used for reported results."""
        g = torch.Generator().manual_seed(seed)
        W_enc = torch.randn(d_sae, d_model, generator=g) / d_model ** 0.5
        W_dec = torch.randn(d_model, d_sae, generator=g) / d_sae ** 0.5
        return cls(W_enc.to(device), torch.zeros(d_sae, device=device),
                   W_dec.to(device), torch.zeros(d_model, device=device), k)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        """x: (..., d_model) -> sparse activations (..., d_sae); exactly k non-zeros per row."""
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        outs = []
        for s in range(0, flat.shape[0], chunk):
            xs = flat[s:s + chunk].to(self.W_enc.device, torch.float32)
            pre = xs @ self.W_enc.T + self.b_enc
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, idx, vals)
            outs.append(z)
        return torch.cat(outs, 0).reshape(*lead, self.d_sae)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.to(self.W_dec.device, torch.float32) @ self.W_dec.T + self.b_dec

    @torch.no_grad()
    def decoder_direction(self, latent_vec: torch.Tensor) -> torch.Tensor:
        """Linear part of f_dec applied to a latent-space vector: W_dec @ v (b_dec cancels)."""
        return self.W_dec @ latent_vec.to(self.W_dec.device, torch.float32)


def fraction_variance_explained(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """1 - ||x - x_hat||^2 / ||x - mean(x)||^2 over rows of (n, d)."""
    x, x_hat = x.float().reshape(-1, x.shape[-1]), x_hat.float().reshape(-1, x.shape[-1])
    num = ((x - x_hat) ** 2).sum()
    den = ((x - x.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - num / den)
