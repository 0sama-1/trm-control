"""LSTM trunk, added as a mechanistic control for the TRM-vs-recursion
comparison (see project docs under
``projects/trm-delay-robust-scaling/phase0-literature/ideation/proposal-final.json``).

Follows the same contract as ``models.dnn.DNNTrunk`` /
``models.trm_control.*Trunk`` / ``models.transformer_control.TransformerTrunk``:
``instantiate(model_cfg, in_dim=...)`` returns a module with an ``.out_dim``
property and a ``.forward(x) -> Tensor`` that the caller feeds into its own
output head (``models.dnn.make_head``).

Why this trunk exists: the comparison this project runs is between TRM's
weight-tied *iterative refinement* over a single augmented observation
(no persistent state carried between environment steps — each ``forward()``
call is independent, per ``models.trm_control``'s docstring point 2) and a
recurrent mechanism that instead integrates information via a *persistent
hidden state carried across time* (the mechanism Bakker (2007) and POPGym's
recurrent baselines study). A textbook recurrent POLICY with hidden state
carried across an entire episode would require replay-buffer changes to
sample whole trajectories rather than single transitions — exactly the
"redesign the core training loop" this project's execution constraints rule
out.

Scoping decision (documented, not hidden): this trunk instead applies an
LSTM WITHIN a single ``forward()`` call, over the same per-slot sequence the
transformer trunk already tokenizes — the current observation followed by
the pending-action queue slots exposed by observation augmentation (see
``augmentation.md``, ``delays.py::DelayedEnv``). This keeps the trunk
stateless across calls, exactly like every other trunk here, so it drops
into the existing training loop unchanged, while still giving the delay
history to a genuinely recurrent (not attention, not weight-tied-recursion)
mechanism for a fair, same-information, same-training-loop comparison. It is
NOT a full cross-episode recurrent policy in Bakker's sense; the eventual
report states this plainly rather than overclaiming a like-for-like replication.

Because the sequence is built the same way as the transformer trunk's
per-feature tokens (one token per scalar input dimension, via a learned
per-feature affine map — see ``models.transformer_control``), the LSTM
receives an identical information budget per call to the transformer and
TRM trunks: same augmented observation, same slot count, differing only in
HOW each trunk processes it (attention vs. weight-tied recursion vs.
sequential recurrence).
"""

import torch
import torch.nn as nn


class LSTMTrunk(nn.Module):
    """Per-call LSTM over per-feature observation tokens.

    Config (all optional, sensible defaults given):
        hidden_size: LSTM hidden width. Also the trunk's out_dim.
        num_layers:  stacked LSTM layers.
        token_dim:   width of the per-feature tokenizer before the LSTM
                     (kept small relative to hidden_size; the LSTM does the
                     actual representational work, matching the role
                     hidden_size plays in the transformer trunk's tokenizer).
        dropout:     inter-layer dropout (only active when num_layers > 1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        token_dim: int = 32,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__()
        self._in_dim = in_dim
        self._hidden_size = hidden_size

        # One learned affine map per input slot (feature identity matters —
        # slot 0 is always the same physical quantity across calls, exactly
        # the same tokenizer design as models.transformer_control.TransformerTrunk).
        self.tokenizer = nn.Linear(1, token_dim)
        self.feature_embed = nn.Parameter(torch.zeros(in_dim, token_dim))
        nn.init.normal_(self.feature_embed, std=0.02)

        self.lstm = nn.LSTM(
            input_size=token_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._out_dim = hidden_size

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) -> tokens: (B, in_dim, token_dim)
        tokens = self.tokenizer(x.unsqueeze(-1)) + self.feature_embed.unsqueeze(0)
        # LSTM over the slot sequence; take the final layer's last hidden state
        # as the pooled representation, the recurrent analogue of the
        # transformer trunk's [CLS]/mean pool.
        _, (h_n, _) = self.lstm(tokens)
        return h_n[-1]


class LSTMCarryTrunk(nn.Module):
    """LSTM whose hidden state is genuinely carried across environment steps
    (Phase 3 Stage 2's memory baseline, plan §4.3) -- unlike ``LSTMTrunk``
    above, which runs over the slots of a single observation and is
    stateless between calls.

    Follows Ni et al. (2022)'s recurrent model-free recipe as far as a
    single-transition replay allows: separate recurrent trunks for actor and
    critic (each Actor/Critic instantiates its own), an embedding layer before
    the recurrence, and the previous action as an input (supplied by the
    ``prev_action_obs`` observation wrapper). The carry is ``cat(h, c)``,
    detached and fed back from replay as a stored state (R2D2-style), the
    same mechanism ``TRMCarryTrunk`` uses, so the two memory trunks are
    trained identically. ``forward(x, carry) -> (h, next_carry)``.
    """

    stateful = True

    def __init__(self, in_dim: int, hidden_size: int = 128, embed_dim: int = 128, **_):
        super().__init__()
        self._hidden = hidden_size
        self.embed = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.ReLU())
        self.cell = nn.LSTMCell(embed_dim, hidden_size)
        self._out_dim = hidden_size

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @property
    def carry_dim(self) -> int:
        return 2 * self._hidden

    def initial_carry(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, 2 * self._hidden, device=device)

    def forward(self, x: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = carry.detach().split(self._hidden, dim=-1)
        h, c = self.cell(self.embed(x), (h.contiguous(), c.contiguous()))
        return h, torch.cat([h, c], dim=-1).detach()
