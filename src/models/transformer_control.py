"""Standard Transformer-encoder trunk, for a head-to-head comparison against
the plain MLP (``models.dnn.DNNTrunk``) and the recursive TRM trunks
(``models.trm_control.{TRMTrunk,TRMHier6Trunk,TRMSingleZTrunk}``).

A textbook multi-head self-attention encoder (Vaswani et al., 2017,
"Attention Is All You Need", https://arxiv.org/abs/1706.03762) applied to
continuous-control observations.

Unlike the TRM trunks — which drop attention entirely because a flat
observation vector is a "sequence" of length 1 with nothing to attend to
(see ``models.trm_control``'s module docstring, point 1) — this trunk gives
attention something real to work over: each *scalar feature* of the
observation (angle, angular velocity, cart position, ...) becomes its own
token, via a per-feature linear "tokenizer" in the same spirit as
FT-Transformer for tabular data (Gorishniy et al., 2021, "Revisiting Deep
Learning Models for Tabular Data", https://arxiv.org/abs/2106.11959).
Self-attention then lets the model learn interactions between individual
state dimensions directly (e.g. angle vs. angular velocity), rather than
only through the fixed linear combinations an MLP's first layer computes.

Note this is a fresh design, not a revival of the vendored TRM repo's
``models/recursive_reasoning/transformers_baseline.py`` (pruned — see
``src/models/trm/README.md`` / ``notes.md``): that baseline operated over
puzzle-grid tokens (one token per ARC/Sudoku/Maze cell), which has no
equivalent for a flat control vector. The per-feature tokenizer here is
built for this project's inputs from scratch.

Pipeline:
    1. Tokenize — x: (B, in_dim) -> tokens: (B, in_dim, hidden_size), one
                  learned affine map per feature index, plus a learned
                  per-feature embedding (feature *identity* matters here —
                  dim 0 of Pendulum is always cos(theta), not an
                  interchangeable sequence position the way NLP positions
                  are, but the mechanism — an added learned vector per
                  slot — is the same as a Transformer positional embedding).
    2. [CLS]    — prepend a learned readout token (only if pooling="cls").
    3. Encode   — ``num_layers`` pre-norm ``nn.TransformerEncoderLayer``s
                  (multi-head self-attention + position-wise FFN).
    4. Pool     — read out the [CLS] token, or mean-pool all feature tokens
                  (pooling="mean"), down to a single (B, hidden_size)
                  vector — the trunk's out_dim, fed into the existing
                  ``Actor``/``Critic`` head exactly like every other trunk.

Same contract as ``models.dnn.DNNTrunk`` / ``models.trm_control.*Trunk``:
``instantiate(model_cfg, in_dim=...)`` returns a module with an ``.out_dim``
property and a ``.forward(x) -> Tensor``.
"""

import math

import torch
import torch.nn as nn


class TransformerTrunk(nn.Module):
    """Multi-head self-attention trunk over per-feature observation tokens.

    Config (all optional, sensible defaults given):
        hidden_size:    token width / attention d_model. Also the trunk's out_dim.
        num_heads:      attention heads (must divide hidden_size).
        num_layers:     stacked encoder layers.
        ffn_expansion:  feed-forward hidden width, as a multiple of hidden_size.
        dropout:        dropout applied inside attention and the feed-forward block.
        activation:     feed-forward activation — "relu" | "gelu".
        pooling:        "cls" (learned readout token) | "mean" (mean over feature tokens).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_expansion: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: str = "cls",
        **_,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown pooling: {pooling!r}. Choose from: cls, mean")
        if activation not in ("relu", "gelu"):
            raise ValueError(f"Unknown activation: {activation!r}. Choose from: relu, gelu")

        self.pooling = pooling

        # Per-feature tokenizer: x (B, in_dim) -> tokens (B, in_dim, hidden_size).
        # One affine map per feature index (FT-Transformer style), equivalent to
        # in_dim independent nn.Linear(1, hidden_size) layers batched together.
        self.feature_weight = nn.Parameter(torch.empty(in_dim, hidden_size))
        self.feature_bias = nn.Parameter(torch.zeros(in_dim, hidden_size))
        bound = 1.0 / math.sqrt(hidden_size)
        nn.init.uniform_(self.feature_weight, -bound, bound)

        # Learned per-feature embedding, added like a positional embedding
        # (see docstring point 1) so the encoder can tell features apart.
        self.feature_embed = nn.Parameter(torch.empty(in_dim, hidden_size))
        nn.init.trunc_normal_(self.feature_embed, std=0.02)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_size))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * ffn_expansion),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # pre-norm: stabler gradients at the depths swept here
        )
        # enable_nested_tensor's fast path is inapplicable to norm_first=True
        # anyway (torch warns on every construction otherwise) — turn it off
        # explicitly rather than let it hit the log on every run.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self._out_dim = hidden_size

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias  # (B, in_dim, hidden_size)
        tokens = tokens + self.feature_embed

        if self.pooling == "cls":
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.encoder(tokens)

        if self.pooling == "cls":
            return tokens[:, 0]
        return tokens.mean(dim=1)
