# TRM-for-Control: Integration Notes

Context for the `trm-integration` branch: a Tiny Recursive Model (TRM) trunk,
adapted from the supervised puzzle-solving model in `src/models/trm/`, made
usable as a drop-in backbone for the off-policy actor/critic networks
(DDPG, TD3, SAC), for a head-to-head comparison against the plain MLP trunk
(`src/models/dnn.py`).

Reference: Alexia Jolicoeur-Martineau, *"Less is More: Recursive Reasoning
with Tiny Networks"*, 2025. https://arxiv.org/abs/2510.04871

## Files added / changed

| File | Status | Purpose |
|---|---|---|
| `src/models/trm_control.py` | new | `TRMTrunk`, `TRMHier6Trunk`, `TRMSingleZTrunk` — the three RL-adapted trunks |
| `conf/model/trm.yaml` | new | Hydra config for `TRMTrunk` |
| `conf/model/trm_hier6.yaml` | new | Hydra config for `TRMHier6Trunk` |
| `conf/model/trm_singlez.yaml` | new | Hydra config for `TRMSingleZTrunk` |
| `.gitignore` | modified | `models` → `/models` (see below) |

Run with, e.g.:
```bash
python src/main.py arch=td3 model=trm
python src/main.py arch=ddpg model=trm_hier6
python src/main.py arch=sac model=trm_singlez
```

### `.gitignore` fix

The old `models` rule (meant only for the root-level checkpoint output dir,
`cfg.model_folder`) was unanchored, so it also silently matched `src/models/`.
`src/models/trm/` (the vendored TRM repo) and the new `trm_control.py` were
invisible to git until this was anchored to `/models`. `src/models/dnn.py`
and `src/models/__init__.py` were already tracked from before the rule
existed, which is why they weren't affected.

## The trunk contract

Every algo's `Actor`/`Critic` builds its backbone the same way:

```python
self.trunk = instantiate(model_cfg, in_dim=obs_dim)      # or obs_dim + act_dim for critics
self.head  = make_head(self.trunk.out_dim, out_dim, model_cfg.output_init)
```

So any drop-in replacement for `DNNTrunk` just needs `__init__(in_dim, **kwargs)`,
an `.out_dim` property, and `.forward(x) -> Tensor`. All three TRM trunks
follow this exactly — no changes were needed to `ddpg.py`, `td3.py`, or
`sac.py`.

## What changed vs. the supervised (ARC/Sudoku/Maze) TRM

### Pruned entirely

| Removed | Why |
|---|---|
| Token embedding lookup, `vocab_size`, `seq_len` | Observations are continuous vectors, not discrete tokens |
| Self-attention + RoPE (`Attention`, `RotaryEmbedding`) | A length-1 "sequence" has nothing to attend to — degenerates to a no-op |
| `mlp_t` (token-mixing MLP) | Same reason — no sequence axis to mix over |
| Puzzle embeddings (`CastedSparseEmbedding`, `puzzle_emb_*`, `num_puzzle_identifiers`) | No "puzzle ID" concept in a control observation |
| `lm_head` | Trunk returns the raw latent; the existing `Actor`/`Critic` head consumes it, same as `DNNTrunk` |
| ACT halting (`q_head`, `q_halt_logits`/`q_continue_logits`, `halt_max_steps`, `halt_exploration_prob`, `no_ACT_continue`, `steps`/`halted`/`carry`) | Training-time mechanism: the same carry is revisited across many optimizer steps on one batch, with a learned head deciding when to stop. Off-policy RL calls the trunk once per transition with no reuse — nothing for a halting policy to learn from |
| Deep supervision outer loop (`pretrain.py::train_batch` re-feeding carry) | Same reason as above |
| bf16 casting (`CastedLinear`/`CastedEmbedding`, `forward_dtype`) | Plain float32 throughout; control nets are tiny enough not to need it |
| Distributed / EMA / loss-head infra (`CastedSparseEmbeddingSignSGD_Distributed`, `ACTLossHead`, `stablemax_cross_entropy`, EMA helper) | Training-pipeline code, not part of the model; each RL algo already supplies its own actor/critic losses |
| `H_layers` config field | Was already dead/unused in the original |

### Added / changed for the RL setting

- **Continuous input projection** — `nn.Linear(in_dim, hidden_size)` replacing the token-embedding lookup, keeping the original's `embed_scale` (÷√hidden_size) normalization trick.
- **Trunk contract glue** — `out_dim` property, `forward(x) -> Tensor`, `**_` to absorb stray kwargs (e.g. `output_init`, which `Actor`/`Critic` read directly off `model_cfg`, not off the trunk).
- **Stateless per-call latents** — every forward pass reinitializes `z_H`/`z_L` (or `z`) from the fixed init buffers and runs one full recursion. The original persists a carry *across* calls, reset only when `halted`; that doesn't apply here since each RL transition is independent, not a refinement of the previous one.
- **`n_latents` generalized in hier6** — original hardcodes exactly 6 latents with `L_cycles` implicitly required to equal 6. Parametrized as `n_latents` (default 6) with round-robin indexing (`step % n_latents`), decoupling it from `L_cycles`.
- **Local reimplementation of primitives** (`rms_norm`, `SwiGLU`, `trunc_normal_init_`) on plain `nn.Linear` instead of importing the vendored `CastedLinear`/`CastedEmbedding`. Functionally identical, but avoids depending on `src/models/trm`'s internal absolute imports, which assume that directory is its own project root (`from models.layers import ...` resolves differently depending on what's on `sys.path`) and would break if imported directly from the main codebase.

### Kept faithful (the actual point of the comparison)

- A single tiny shared cell (`_ReasoningCell`) reused across **every** latent-update role — TRM's central simplification over HRM's two separate networks.
- The exact `H_cycles` × `L_cycles` recursion nesting.
- The truncated-backprop / "1-step-gradient" trick: the first `H_cycles - 1` outer cycles run under `torch.no_grad()`; only the final cycle is tracked for backprop. This is what makes deep recursion affordable — preserving it is the whole reason to test this mechanism against a plain MLP trunk with matched depth.
- The three-variant split at the same conceptual point as the source: `trm` (2-latent z_H/z_L), `trm_hier6` (6 parallel low-level latents), `trm_singlez` (single latent, no z_H/answer split).

## Vendor subtree pruning

`src/models/trm/` is a vendored copy of the paper's official repo, kept only
as reference source for what `trm_control.py` was ported from — it is never
imported by this project (see "Local reimplementation of primitives" above).
Everything not needed for that reference purpose was removed:

| Removed | Why |
|---|---|
| `dataset/` (ARC/Sudoku/**maze** builders) | Puzzle-domain data pipeline, irrelevant to control observations |
| `kaggle/` (~11MB ARC-AGI competition JSON) | Same — puzzle task data |
| `evaluators/` | ARC-specific evaluator |
| `pretrain.py` (**the supervised-training "main file"**) | The entrypoint for the whole ACT/deep-supervision training loop this project doesn't use |
| `puzzle_dataset.py` | `pretrain.py`'s dataloader |
| `config/cfg_pretrain.yaml` | `pretrain.py`'s Hydra config |
| `models/ema.py`, `models/sparse_embedding.py`, `models/losses.py`, `utils/` | Only ever used by `pretrain.py` |
| `models/recursive_reasoning/hrm.py` + `config/arch/hrm.yaml` | HRM baseline — not one of the 3 requested variants |
| `models/recursive_reasoning/transformers_baseline.py` + `config/arch/transformers_baseline.yaml` | Transformer baseline — same reason |
| `requirements.txt`, `specific_requirements.txt` | Deps for the now-removed pretrain pipeline |

**Kept:** `models/recursive_reasoning/{trm,trm_hier6,trm_singlez}.py`,
`models/common.py`, `models/layers.py`, `config/arch/{trm,trm_hier6,trm_singlez}.yaml`,
plus `README.md`/`LICENSE`/`assets/` for attribution.

Caveat: the three kept `recursive_reasoning/*.py` files still have a source
line `from models.sparse_embedding import CastedSparseEmbedding` (puzzle
embeddings) that no longer resolves, since that file was removed. Left as-is
— these files were never executable standalone even before pruning (they
needed the full `pretrain.py` harness, which is also gone); they now serve
purely as read-only reference for the recursion logic, not as code that runs.
A short banner was added to the top of `src/models/trm/README.md` noting the
subtree is pruned and pointing back to this file.

## Defaults (worth revisiting on real hardware)

`hidden_size=128`, `H_cycles=3`, `L_cycles=6`, `expansion=4`, `L_layers=2` —
scaled down from the paper's ARC defaults (`hidden_size=512`, same cycle
counts) to a size more proportionate to small continuous-control
observations. These are arbitrary starting points, not tuned; worth
sweeping once running on the HPC.

## Verification performed

Couldn't run the full `python src/main.py` CLI in this environment (no GPU/
gym/wandb here). What *was* verified, in a disposable CPU-only venv:

- Standalone forward/backward on all three trunks at several configs — output shapes, no-NaN gradients, gradients reaching both the input projection and the reasoning cell.
- `hydra.utils.instantiate` against each new YAML exactly as `Actor`/`Critic` call it (both `obs_dim` and `obs_dim + act_dim` inputs).
- Full Hydra config composition (`model=trm|trm_hier6|trm_singlez` × `arch=ddpg|td3|sac`) resolving cleanly against `conf/config.yaml`.

Not yet run: an actual training loop against a real env (Pendulum/CartPole) — left for validation on the HPC.
