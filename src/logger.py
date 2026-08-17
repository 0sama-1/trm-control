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

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=f"{t}_{env_id}_{algo_name}_{model_name}_{delay_name}_seed{cfg.seed}",
            group=env_id,
            job_type=algo_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[env_id, algo_name, model_name, delay_name],
        )

    def log(self, metrics: dict) -> None:
        if self._enabled:
            self._wandb.log(metrics)

    def finish(self) -> None:
        if self._enabled:
            self._wandb.finish()

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, *_) -> None:
        self.finish()
