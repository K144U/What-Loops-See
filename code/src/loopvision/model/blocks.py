"""Transformer building blocks.

Pre-norm RMSNorm, multi-head attention with RoPE, SwiGLU at 8/3 expansion,
no dropout. Head dimension is fixed at 64, so the head count follows the
model width.

Deliberately plain. The scientific question is about loop counts, and every
architectural flourish is a confound to rule out later, so the block is the
least interesting thing in the repository and should stay that way.

Positional information is RoPE over a flattened token index, plus a learned
segment embedding distinguishing image patches from query tokens
(IMPLEMENTATION.md Section 5). RoPE is applied inside attention rather than
added to the residual stream, so the recurrent state carries no absolute
position of its own and repeated core applications cannot drift positionally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_DIM = 64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computed in fp32 regardless of autocast: the recurrent state norm
        # grows across iterations (up to 316x in the text case), and a bf16
        # reciprocal square root at that scale loses meaningful precision.
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * rms).to(dtype) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, device, dtype, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of shape (seq_len, head_dim // 2)."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is (B, heads, N, head_dim). Rotates the two halves pairwise."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        if d % HEAD_DIM != 0:
            raise ValueError(f"width {d} is not a multiple of head dim {HEAD_DIM}")
        self.heads = d // HEAD_DIM
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Bidirectional. Nothing here is autoregressive, so there is no mask.
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class SwiGLU(nn.Module):
    def __init__(self, d: int, expansion: float = 8 / 3) -> None:
        super().__init__()
        hidden = int(expansion * d)
        hidden = ((hidden + 63) // 64) * 64  # keep matmuls on a friendly size
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One pre-norm transformer block."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scale_shift=None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        if scale_shift is not None:
            # adaLN loop conditioning, milestone 3. Unused at the gate.
            scale, shift = scale_shift
            h = h * (1 + scale) + shift
        x = x + self.attn(h, cos, sin)
        return x + self.mlp(self.norm2(x))
