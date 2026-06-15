from __future__ import annotations

import torch

from .env import DistributedEnv, get_distributed_env


def maybe_init_distributed(backend: str = "nccl") -> DistributedEnv:
    env = get_distributed_env()
    if env.is_distributed and not torch.distributed.is_initialized():
        if not torch.cuda.is_available():
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(env.local_rank)
    return env


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
