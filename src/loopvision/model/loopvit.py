"""The looped vision transformer: prelude, tied core, coda.

Deliberately mirrors the Huginn tripartite layout so that Stage 2
comparisons at milestone 10 are not confounded by architecture
differences (IMPLEMENTATION.md Section 5).

    image (3,32,32) --conv 4x4 stride 4--> 64 patch tokens
    query token ids --embed-->             L query tokens
                    concat, + segment embedding
                              |
                    prelude: 2 untied blocks -> e
                              |
        s_0 = randn * d**-0.5   (ablate: zeros)
        repeat k times:  s = Core(adapter([s ; e]))
                              |
                    coda: 2 untied blocks -> head

The adapter concatenation is the load-bearing detail. Without input
injection at every iteration the core drifts away from the input and the
loop count stops meaning anything, which is why ``inject=False`` exists
only as an ablation and is not a supported training configuration.

State capture for the M4 and M2 instruments goes through ``forward``'s
``return_states`` flag rather than through a separate code path. If the
instrument path and the training path can silently disagree, every
measurement downstream is suspect, which is what
tests/test_model_invariance.py exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from loopvision.model.blocks import HEAD_DIM, Block, RMSNorm, build_rope_cache


@dataclass
class LoopViTConfig:
    d_model: int = 384
    prelude_blocks: int = 2
    core_blocks: int = 2
    coda_blocks: int = 2
    num_classes: int = 48
    vocab_size: int = 75
    query_len: int = 8
    canvas: int = 32
    patch: int = 4
    inject: bool = True            # ablation control: False removes injection
    state_init: str = "randn"      # randn | zeros
    conditioning: str = "none"     # none | embed | adaln, milestone 3
    k_max_conditioning: int = 64

    @property
    def num_patches(self) -> int:
        side = self.canvas // self.patch
        return side * side

    @property
    def seq_len(self) -> int:
        return self.num_patches + self.query_len


class LoopViT(nn.Module):
    def __init__(self, cfg: LoopViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.patchify = nn.Conv2d(3, d, kernel_size=cfg.patch, stride=cfg.patch)
        self.query_embed = nn.Embedding(cfg.vocab_size, d)
        # Distinguishes image patches from query tokens. Position itself is
        # carried by RoPE inside attention.
        self.segment = nn.Embedding(2, d)

        self.prelude = nn.ModuleList([Block(d) for _ in range(cfg.prelude_blocks)])
        self.core = nn.ModuleList([Block(d) for _ in range(cfg.core_blocks)])
        self.coda = nn.ModuleList([Block(d) for _ in range(cfg.coda_blocks)])

        self.adapter = nn.Linear(2 * d, d, bias=False)
        self.norm_out = RMSNorm(d)
        self.head = nn.Linear(d, cfg.num_classes, bias=False)

        if cfg.conditioning == "embed":
            self.loop_embed = nn.Embedding(cfg.k_max_conditioning, d)
        elif cfg.conditioning == "adaln":
            self.loop_mlp = nn.Sequential(
                nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2 * d * cfg.core_blocks)
            )

        self.apply(self._init_weights)
        self._scale_core_output()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_core_output(self) -> None:
        """Shrink the core's output projections at initialisation.

        Scaled by 1/sqrt(2 * core_depth * k_mean) per IMPLEMENTATION.md
        Section 6.3. Applying the same weights k times compounds their
        effect on the state norm, so without this the state explodes in the
        first few hundred steps at larger k. This and the input injection
        are the two things that most often decide whether a small looped
        model trains at all.
        """
        k_mean = 8
        scale = 1.0 / math.sqrt(2 * self.cfg.core_blocks * k_mean)
        with torch.no_grad():
            for block in self.core:
                block.attn.proj.weight.mul_(scale)
                block.mlp.down.weight.mul_(scale)

    def embed_input(self, image: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        patches = self.patchify(image).flatten(2).transpose(1, 2)  # (B, P, d)
        tokens = self.query_embed(query)  # (B, L, d)
        x = torch.cat([patches, tokens], dim=1)

        segment_ids = torch.zeros(x.shape[1], dtype=torch.long, device=x.device)
        segment_ids[patches.shape[1] :] = 1
        return x + self.segment(segment_ids)[None, :, :]

    def init_state(self, e: torch.Tensor) -> torch.Tensor:
        if self.cfg.state_init == "zeros":
            return torch.zeros_like(e)
        return torch.randn_like(e) * (self.cfg.d_model**-0.5)

    def _core_step(self, s, e, cos, sin, iteration: int):
        if self.cfg.inject:
            h = self.adapter(torch.cat([s, e], dim=-1))
        else:
            # Ablation: the core sees only its own state. Rules out the
            # possibility that looping merely re-reads the input.
            h = s

        scale_shift_per_block = None
        if self.cfg.conditioning == "embed":
            idx = torch.tensor(
                min(iteration, self.cfg.k_max_conditioning - 1), device=s.device
            )
            h = h + self.loop_embed(idx)[None, None, :]
        elif self.cfg.conditioning == "adaln":
            t = _sinusoidal(iteration, self.cfg.d_model, s.device, s.dtype)
            params = self.loop_mlp(t).view(self.cfg.core_blocks, 2, -1)
            scale_shift_per_block = [
                (params[i, 0][None, None, :], params[i, 1][None, None, :])
                for i in range(self.cfg.core_blocks)
            ]

        for i, block in enumerate(self.core):
            ss = scale_shift_per_block[i] if scale_shift_per_block else None
            h = block(h, cos, sin, scale_shift=ss)
        return h

    def forward(
        self,
        image: torch.Tensor,
        query: torch.Tensor,
        k: int,
        return_states: bool = False,
        bptt_window: int | None = None,
        s0: torch.Tensor | None = None,
    ):
        """Run the model for k loop iterations.

        ``bptt_window`` truncates backpropagation to the last w iterations,
        per Section 6.2. None means full backpropagation through the loop.
        ``return_states`` additionally yields every s_i, which is what the
        M2, M3 and M4 instruments consume.

        ``s0`` supplies the initial state explicitly. This matters for M2:
        with the default ``randn`` initialisation every call starts from a
        fresh random state, so a clean run and its corrupted twin would
        differ in their starting point as well as in the patched position,
        and the patching effect would be measured against that noise. The
        instrument passes one s0 to both runs.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        x = self.embed_input(image, query)
        cos, sin = build_rope_cache(x.shape[1], HEAD_DIM, x.device, x.dtype)

        e = x
        for block in self.prelude:
            e = block(e, cos, sin)

        s = self.init_state(e) if s0 is None else s0
        states = [s] if return_states else None

        n_nograd = 0 if bptt_window is None else max(0, k - bptt_window)
        if n_nograd:
            with torch.no_grad():
                for i in range(n_nograd):
                    s = self._core_step(s, e, cos, sin, i)
                    if return_states:
                        states.append(s)
            s = s.detach()

        for i in range(n_nograd, k):
            s = self._core_step(s, e, cos, sin, i)
            if return_states:
                states.append(s)

        logits = self.decode(s, cos, sin)
        return (logits, states) if return_states else logits

    def decode(self, s: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """Run the coda and head on a state.

        Exposed separately because M3, the coda lens, decodes every
        intermediate s_i through exactly this path. Sharing it means the
        lens cannot drift from the real output head.
        """
        if cos is None or sin is None:
            cos, sin = build_rope_cache(s.shape[1], HEAD_DIM, s.device, s.dtype)
        h = s
        for block in self.coda:
            h = block(h, cos, sin)
        # Pool over the query tokens: they are where the answer is assembled.
        pooled = self.norm_out(h)[:, self.cfg.num_patches :, :].mean(dim=1)
        return self.head(pooled)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _sinusoidal(t: int, dim: int, device, dtype) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / half
    )
    angles = torch.tensor(float(t), device=device) * freqs
    return torch.cat([angles.sin(), angles.cos()]).to(dtype)


class FeedforwardBaseline(nn.Module):
    """Matched-compute non-looped baseline: depth k * core_blocks, untied.

    Gate G2's headline comparison, and gate G1.1 needs it too: depth 3 must
    be unsolvable at k=1 by a looped model *and* by a matched non-looped
    one, otherwise the task is too easy and there is no loop-count curve to
    analyse.
    """

    def __init__(self, cfg: LoopViTConfig, k: int) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.patchify = nn.Conv2d(3, d, kernel_size=cfg.patch, stride=cfg.patch)
        self.query_embed = nn.Embedding(cfg.vocab_size, d)
        self.segment = nn.Embedding(2, d)
        total = cfg.prelude_blocks + k * cfg.core_blocks + cfg.coda_blocks
        self.blocks = nn.ModuleList([Block(d) for _ in range(total)])
        self.norm_out = RMSNorm(d)
        self.head = nn.Linear(d, cfg.num_classes, bias=False)
        self.apply(LoopViT._init_weights.__get__(self))

    def forward(self, image: torch.Tensor, query: torch.Tensor, k: int = 0):
        B = image.shape[0]
        patches = self.patchify(image).flatten(2).transpose(1, 2)
        tokens = self.query_embed(query)
        x = torch.cat([patches, tokens], dim=1)
        segment_ids = torch.zeros(x.shape[1], dtype=torch.long, device=x.device)
        segment_ids[patches.shape[1] :] = 1
        x = x + self.segment(segment_ids)[None, :, :]

        cos, sin = build_rope_cache(x.shape[1], HEAD_DIM, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        pooled = self.norm_out(x)[:, self.cfg.num_patches :, :].mean(dim=1)
        return self.head(pooled)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
