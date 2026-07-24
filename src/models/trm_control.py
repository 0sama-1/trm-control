"""Tiny Recursive Model (TRM) trunks, adapted for continuous-control RL.

Ports the recursive-reasoning mechanism from:

    "Less is More: Recursive Reasoning with Tiny Networks"
    Alexia Jolicoeur-Martineau, 2025. https://arxiv.org/abs/2510.04871

for use as a drop-in trunk in the off-policy actor/critic networks
(``algos/ddpg.py``, ``algos/td3.py``, ``algos/sac.py``), following the same
contract as ``models.dnn.DNNTrunk``: ``instantiate(model_cfg, in_dim=...)``
returns a module with an ``.out_dim`` property and a ``.forward(x) -> Tensor``
that the caller feeds into its own output head (see ``models.dnn.make_head``).

The reference implementation this is adapted from lives in
``src/models/trm/models/recursive_reasoning/{trm,trm_hier6,trm_singlez}.py``
(vendored from the paper's official repo, https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
That code solves token-grid puzzles (ARC/Sudoku/Maze): a sequence of discrete
cell tokens, refined through attention + MLP blocks. Two things don't
transfer to a flat continuous-control observation and are deliberately
dropped here, rather than faked:

1. **Token-sequence attention.** The puzzle models mix information across
   sequence positions (grid cells) with self-attention (or an MLP-mixer
   variant, ``mlp_t``). An RL observation is a single flat vector — a
   "sequence" of length 1 — so attention over it degenerates to a per-token
   linear projection with nothing to attend to. Rather than carry dead
   weight, the recursive cell here keeps only the channel-mixing half of
   each block (the SwiGLU MLP + RMSNorm residual), which is exactly what
   the ``mlp_t`` ablation in the source models is doing in spirit.
2. **ACT halting (the Q-head / adaptive compute steps).** In the source
   models, one ``forward()`` call is a single fixed-depth recursion
   (``H_cycles`` x ``L_cycles``); the *adaptive* part happens outside it, in
   the training loop (see ``trm/pretrain.py::train_batch``): the same
   latent carry is fed back through ``forward()`` for up to
   ``halt_max_steps`` calls, and a learned Q-head decides when to stop,
   with the loss applied at each halted step (deep supervision). That
   machinery assumes a persistent carry that is revisited across many
   optimizer steps on the *same* input. Off-policy RL here calls the
   trunk once per transition with no reuse of the same observation across
   steps, so there is nothing for a learned halting policy to act on — it
   would either always run to the max or never learn a useful signal.
   What *is* preserved, and is the actual object of comparison against the
   plain MLP trunk, is the recursive latent-refinement mechanism itself and
   its truncated-backprop training trick (below).

What's preserved faithfully from the source:

- A single tiny shared cell (``_ReasoningCell``, analogous to ``L_level`` in
  the source) is recursed for *both* the fast/low-level latent update and
  the slow/high-level update — TRM's central simplification over HRM's two
  separate networks.
- The exact recursion structure and 1-step-gradient truncation: the first
  ``H_cycles - 1`` outer cycles run under ``torch.no_grad()`` (cheap,
  unrolled "thinking"), and only the final outer cycle is tracked for
  backprop. This is what makes deep recursion (many effective refinement
  steps) affordable — gradients never have to be carried through the full
  unroll.
- Three variants matching the three source files, all selectable as
  ``model=`` Hydra config groups:

    * ``trm``          -> ``TRMTrunk``          (2-latent z_H / z_L recursion)
    * ``trm_hier6``     -> ``TRMHier6Trunk``     (6 parallel low-level latents)
    * ``trm_singlez``   -> ``TRMSingleZTrunk``   (single latent, no z_H/y split)

Every forward pass starts recursion from fixed (non-trainable) learned-at-
init random states, exactly like the source models' ``H_init``/``L_init``
buffers — there is no persistent carry between calls, since each RL
transition is an independent decision, not a step in an iterative refinement
of the *same* input.

Everything runs in plain float32 (no bf16 casting, no puzzle/vocab
embeddings, no RoPE) since these don't apply to a small continuous vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x.to(dtype)


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """JAX/flax-style truncated normal init (ported from ``models/trm/models/common.py``).

    Plain ``nn.init.trunc_normal_`` does not actually reproduce the requested
    std dev on a truncated normal; this applies the same correction the TRM
    reference code uses.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
            return tensor

        import math
        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2

        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower ** 2)
        pdf_l = c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


class SwiGLU(nn.Module):
    """Gated MLP (channel-mixing only — see module docstring point 1)."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        trunc_normal_init_(self.gate_up_proj.weight, std=1.0 / (hidden_size ** 0.5))
        trunc_normal_init_(self.down_proj.weight, std=1.0 / (inter ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class _ReasoningCell(nn.Module):
    """The single tiny network recursed over every latent update.

    A stack of ``num_layers`` SwiGLU blocks with post-norm residual
    connections, matching the source ``ReasoningModule`` minus the dropped
    attention/token-mixing sub-layer (module docstring point 1). Instances
    of this cell are reused (called multiple times per forward pass, and
    for more than one role — e.g. both the z_L and z_H updates in
    ``TRMTrunk``) rather than stacked into a deep feedforward network —
    that weight sharing across recursion steps is the "tiny" in TRM.
    """

    def __init__(self, hidden_size: int, expansion: float, num_layers: int, rms_norm_eps: float):
        super().__init__()
        self.blocks = nn.ModuleList([SwiGLU(hidden_size, expansion) for _ in range(num_layers)])
        self.eps = rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, injection: torch.Tensor | None = None) -> torch.Tensor:
        if injection is not None:
            hidden_states = hidden_states + injection
        for block in self.blocks:
            hidden_states = rms_norm(hidden_states + block(hidden_states), self.eps)
        return hidden_states


def _input_embedding(in_dim: int, hidden_size: int) -> tuple[nn.Linear, float]:
    """Linear input projection with the source models' embedding-scale trick:
    init the projection to unit variance, then rescale by sqrt(hidden_size)
    so the embedding's contribution to the latent stays scale-consistent
    across hidden sizes (mirrors ``embed_scale`` / ``embed_init_std`` in
    ``TinyRecursiveReasoningModel_ACTV1_Inner``, applied to a continuous
    input instead of a token-embedding lookup)."""
    proj = nn.Linear(in_dim, hidden_size, bias=False)
    embed_scale = hidden_size ** 0.5
    trunc_normal_init_(proj.weight, std=1.0 / embed_scale)
    return proj, embed_scale


class TRMTrunk(nn.Module):
    """Two-latent TRM trunk — recursion over a low-level latent z_L (input-
    conditioned "thinking") and a high-level latent z_H ("acting" on z_L).
    Mirrors ``trm.py``'s ``TinyRecursiveReasoningModel_ACTV1_Inner.forward``.

    Config (all optional, sensible tiny-network defaults given):
        hidden_size:   latent width. Also the trunk's out_dim.
        expansion:     SwiGLU hidden expansion ratio.
        L_layers:      number of SwiGLU blocks in the shared reasoning cell.
        H_cycles:      outer recursion depth (T). Only the last cycle is
                       backpropped; H_cycles - 1 run under no_grad.
        L_cycles:      inner latent-refinement steps per outer cycle (n).
        rms_norm_eps:  epsilon for the residual RMSNorm.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 128,
        expansion: float = 4.0,
        L_layers: int = 2,
        H_cycles: int = 3,
        L_cycles: int = 6,
        rms_norm_eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        assert H_cycles >= 1 and L_cycles >= 1

        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.input_proj, self.embed_scale = _input_embedding(in_dim, hidden_size)
        self.reasoning = _ReasoningCell(hidden_size, expansion, L_layers, rms_norm_eps)

        self.register_buffer("H_init", trunc_normal_init_(torch.empty(hidden_size), std=1.0))
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(hidden_size), std=1.0))

        self._out_dim = hidden_size

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        input_emb = self.embed_scale * self.input_proj(x)

        z_H = self.H_init.expand(batch, -1)
        z_L = self.L_init.expand(batch, -1)

        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z_L = self.reasoning(z_L, injection=z_H + input_emb)
                z_H = self.reasoning(z_H, injection=z_L)

        for _ in range(self.L_cycles):
            z_L = self.reasoning(z_L, injection=z_H + input_emb)
        z_H = self.reasoning(z_H, injection=z_L)

        return z_H


class TRMHier6Trunk(nn.Module):
    """Hierarchical-latent TRM trunk — 6 parallel low-level latents whose sum
    is read as context each inner step, with only one of the 6 written per
    step. Mirrors ``trm_hier6.py``. An ablation on ``TRMTrunk`` testing
    whether splitting the low-level latent into several parallel "slots"
    helps over a single z_L.

    Adds ``n_latents`` (default 6, hence "hier6") to the ``TRMTrunk`` config.
    ``L_cycles`` steps are distributed over the ``n_latents`` slots
    round-robin (``L_cycles == n_latents`` reproduces the source exactly:
    each of the 6 cycles updates a different slot).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 128,
        expansion: float = 4.0,
        L_layers: int = 2,
        H_cycles: int = 3,
        L_cycles: int = 6,
        n_latents: int = 6,
        rms_norm_eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        assert H_cycles >= 1 and L_cycles >= 1 and n_latents >= 1

        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.n_latents = n_latents

        self.input_proj, self.embed_scale = _input_embedding(in_dim, hidden_size)
        self.reasoning = _ReasoningCell(hidden_size, expansion, L_layers, rms_norm_eps)

        self.register_buffer("H_init", trunc_normal_init_(torch.empty(hidden_size), std=1.0))
        for i in range(n_latents):
            self.register_buffer(f"L{i}_init", trunc_normal_init_(torch.empty(hidden_size), std=1.0))

        self._out_dim = hidden_size

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def _l_inits(self):
        return [getattr(self, f"L{i}_init") for i in range(self.n_latents)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        input_emb = self.embed_scale * self.input_proj(x)

        z_H = self.H_init.expand(batch, -1)
        z_L = [init.expand(batch, -1) for init in self._l_inits()]

        def l_step():
            for step in range(self.L_cycles):
                idx = step % self.n_latents
                z_sum = sum(z_L)
                z_L[idx] = self.reasoning(z_sum, injection=z_H + input_emb)

        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                l_step()
                z_H = self.reasoning(z_H, injection=sum(z_L))

        l_step()
        z_H = self.reasoning(z_H, injection=sum(z_L))

        return z_H


class TRMSingleZTrunk(nn.Module):
    """Single-latent TRM trunk — no separate z_H/answer state, just one
    latent z recursively refined and read out directly. Mirrors
    ``trm_singlez.py``. The ablation at the opposite end from
    ``TRMHier6Trunk``: collapsing the two-variable (z, y) design down to
    one variable, to isolate how much of TRM's benefit comes from the
    z_L/z_H split itself versus the recursion mechanism in general.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 128,
        expansion: float = 4.0,
        L_layers: int = 2,
        H_cycles: int = 3,
        L_cycles: int = 6,
        rms_norm_eps: float = 1e-5,
        **_,
    ):
        super().__init__()
        assert H_cycles >= 1 and L_cycles >= 1

        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.input_proj, self.embed_scale = _input_embedding(in_dim, hidden_size)
        self.reasoning = _ReasoningCell(hidden_size, expansion, L_layers, rms_norm_eps)

        self.register_buffer("L_init", trunc_normal_init_(torch.empty(hidden_size), std=1.0))

        self._out_dim = hidden_size

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        input_emb = self.embed_scale * self.input_proj(x)

        z = self.L_init.expand(batch, -1)

        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z = self.reasoning(z, injection=input_emb)
                z = self.reasoning(z)

        for _ in range(self.L_cycles):
            z = self.reasoning(z, injection=input_emb)
        z = self.reasoning(z)

        return z
