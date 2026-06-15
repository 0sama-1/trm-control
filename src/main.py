import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    algo = cfg.arch.name

    if algo == "ddpg":
        from algos.ddpg import train
    elif algo == "td3":
        from algos.td3 import train
    elif algo == "ppo":
        from algos.ppo import train
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}. Choose from: ddpg, td3, ppo")

    train(cfg)


if __name__ == "__main__":
    main()
