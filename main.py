import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.model_folder = str(REPO_ROOT / cfg.model_folder)
    cfg.video_folder = str(REPO_ROOT / cfg.video_folder)
    cfg.results_folder = str(REPO_ROOT / cfg.results_folder)

    algo = cfg.arch.name

    if algo == "ddpg":
        from algos.ddpg import DDPGTrainer as Trainer
    elif algo == "td3":
        from algos.td3 import TD3Trainer as Trainer
    elif algo == "ppo":
        from algos.ppo import PPOTrainer as Trainer
    elif algo == "sac":
        from algos.sac import SACTrainer as Trainer
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}. Choose from: ddpg, td3, ppo, sac")

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
