"""SAE-based emotion steering (paper Sec. 4.3, Eqs. 5-7).

Eq. 5  d_i  = f_dec(z + delta_i * u_i) - f_dec(z)
Eq. 6  d~_i = d_i / ||d_i|| * ||h_k|| * delta_i / delta_max
Eq. 7  h~_k = h_k + alpha * sum_i d~_i

Because f_dec is linear, d_i = delta_i * (W_dec @ u_i) regardless of z, so the
unit direction d_i/||d_i|| is a fixed vector per dimension (times sign(delta_i)).
We precompute it once; ||h_k|| is evaluated per token at hook time.

INTERPRETATION (Eq. 6 sign): read literally, d_i/||d_i|| contributes sign(delta_i)
and the scalar delta_i/delta_max contributes another sign(delta_i), so the product
is |delta_i|/delta_max and +delta / -delta would steer identically. That cannot
be the paper's intent (it reports opposite effects for opposite signs), so the
default ("sign_preserving") uses the signed scale delta_i/delta_max. The literal
reading is kept as eq6_variant="literal" and is covered by a unit test.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

import torch

DIMS = ("valence", "arousal", "dominance")


def latent_unit_vector(indices: Iterable[int], weights: Iterable[float], d_sae: int) -> torch.Tensor:
    """u_i: unit vector in latent space spanned by the selected neurons."""
    u = torch.zeros(d_sae)
    idx = torch.as_tensor(list(indices), dtype=torch.long)
    w = torch.as_tensor(list(weights), dtype=torch.float32)
    u[idx] = w
    n = u.norm()
    if n == 0:
        raise ValueError("selected feature weights are all zero")
    return u / n


def decoder_unit_direction(sae, u_hat: torch.Tensor) -> torch.Tensor:
    """d_i / ||d_i|| for delta_i > 0 (Eq. 5 with the linear decoder)."""
    d = sae.decoder_direction(u_hat)
    return d / d.norm().clamp_min(1e-12)


def eq5_direction(sae, z: torch.Tensor, u_hat: torch.Tensor, delta: float) -> torch.Tensor:
    """Literal Eq. 5, used to verify the linear shortcut in tests."""
    return sae.decode(z + delta * u_hat.to(z.device)) - sae.decode(z)


class EmotionSteering:
    """Holds one unit direction per VAD dimension and applies Eq. 6-7 in a forward hook."""

    def __init__(self, directions: Dict[str, torch.Tensor], alpha: float,
                 delta_max: float = 9.0, eq6_variant: str = "sign_preserving"):
        if eq6_variant not in ("sign_preserving", "literal"):
            raise ValueError("eq6_variant must be 'sign_preserving' or 'literal'")
        unknown = set(directions) - set(DIMS)
        if unknown:
            raise ValueError(f"unknown dimensions: {sorted(unknown)}")
        self.directions = {k: v.detach().float() for k, v in directions.items()}
        self.alpha = float(alpha)
        self.delta_max = float(delta_max)
        self.eq6_variant = eq6_variant

    def unit_shift(self, deltas: Dict[str, float]) -> torch.Tensor:
        """sum_i s_i * u_i with s_i = delta_i/delta_max (or |.| for the literal reading).
        Multiplying by alpha * ||h_k|| gives the Eq. 7 update."""
        total = None
        for dim, delta in deltas.items():
            if delta == 0:
                continue
            s = delta / self.delta_max
            if self.eq6_variant == "literal":
                s = abs(s)
            term = s * self.directions[dim]
            total = term if total is None else total + term
        return total

    def make_hook(self, deltas: Dict[str, float]):
        unit = self.unit_shift(deltas)
        if unit is None:
            return None
        alpha = self.alpha

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            hf = h.float()
            norm = hf.norm(dim=-1, keepdim=True)               # ||h_k|| per token
            h2 = (hf + alpha * norm * unit.to(hf.device)).to(h.dtype)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

        return hook

    @contextmanager
    def attach(self, layer_module, deltas: Dict[str, float]):
        hook = self.make_hook(deltas)
        if hook is None:                                       # all deltas zero: no intervention
            yield
            return
        handle = layer_module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
