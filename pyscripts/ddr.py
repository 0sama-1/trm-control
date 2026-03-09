import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    rank, local_rank, world_size = setup()

    try:
        device = torch.device(f"cuda:{local_rank}")

        print(
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, "
            f"cuda_device_count={torch.cuda.device_count()}, device={device}"
        )

        model = nn.Linear(10, 10).to(device)
        model = DDP(model, device_ids=[local_rank])

        x = torch.randn(20, 10, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()

        print(f"Rank {rank} finished successfully on {device}")

    finally:
        cleanup()


if __name__ == "__main__":
    main()