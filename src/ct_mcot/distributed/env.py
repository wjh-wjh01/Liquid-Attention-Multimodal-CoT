from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DistributedEnv:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def get_distributed_env() -> DistributedEnv:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedEnv(rank=rank, world_size=world_size, local_rank=local_rank, is_distributed=world_size > 1)
