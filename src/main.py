import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    algo = cfg.arch.name

    if algo == "ddpg":
        from algos.ddpg import DDPGTrainer as Trainer
    elif algo == "td3":
        from algos.td3 import TD3Trainer as Trainer
    elif algo == "ppo":
        from algos.ppo import PPOTrainer as Trainer
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}. Choose from: ddpg, td3, ppo")

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
