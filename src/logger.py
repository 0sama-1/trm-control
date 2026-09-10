from __future__ import annotations

import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from datetime import datetime


class WandbLogger:
    """Thin wrapper around wandb that centralises init/log/finish.

    All algo files stay free of wandb imports — they just call logger.log().
    When wandb.enabled is false the class is a no-op, so no guard clauses
    are needed at call sites.
    """

    def __init__(self, cfg: DictConfig, algo_name: str):
        cur_dir = Path(__file__).resolve().parent
        src_dir = cur_dir.parent
        env_path = src_dir / ".env"
        load_dotenv(dotenv_path=env_path)
        self._enabled = cfg.wandb.enabled
        self.run_path: str | None = None
        if not self._enabled:
            return

        # Login in any other case
        import wandb
        wandb.login(
            key=os.getenv("WANDB_API_KEY")
        )


        self._wandb = wandb
        env_id = cfg.env.id
        delay_name = cfg.delay.name
        model_name = cfg.model.name
        t = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=f"{t}_{env_id}_{algo_name}_{model_name}_{delay_name}_seed{cfg.seed}",
            group=env_id,
            job_type=algo_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[env_id, algo_name, model_name, delay_name],
        )
        # Cross-reference: the results_summary.json this run also writes
        # (trainer.py::_write_results_summary) stores this so the two are
        # linkable after the fact without grepping wandb by run name.
        self.run_path = run.path if run is not None else None

        # Episode-level and step-level metrics log at very different rates
        # within the same run (one call per episode vs. one per env step);
        # without this, W&B's default implicit step counter interleaves them
        # on one shared x-axis. Give each its own explicit x-axis instead.
        wandb.define_metric("episode")
        wandb.define_metric("*", step_metric="episode")
        wandb.define_metric("global_step")
        wandb.define_metric("step/*", step_metric="global_step")

    def log(self, metrics: dict) -> None:
        if self._enabled:
            self._wandb.log(metrics)

    def log_step(self, step_row: dict) -> None:
        """Per-env-step log, namespaced under 'step/' so it plots against
        'global_step' (see the define_metric calls in __init__) rather than
        colliding with the per-episode log's 'episode' axis. Only called
        when cfg.wandb.log_steps is true (see OffPolicyTrainer._training_loop)
        -- this can be a lot of API calls over a full run, so it is opt-in."""
        if not self._enabled:
            return
        row = {"global_step": step_row["global_step"]}
        for k, v in step_row.items():
            if k == "global_step":
                continue
            # Lists (action/applied_action) log fine as JSON but won't plot
            # as a line by default; unwrap 1-D actions to a scalar so the
            # common case (this project's envs are all 1-D action) still
            # gets a normal line chart.
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            row[f"step/{k}"] = v
        self._wandb.log(row)

    def finish(self) -> None:
        if self._enabled:
            self._wandb.finish()

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, *_) -> None:
        self.finish()
