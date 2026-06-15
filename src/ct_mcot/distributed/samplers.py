from __future__ import annotations

from torch.utils.data import Dataset, DistributedSampler, RandomSampler, Sampler, SequentialSampler

from .env import DistributedEnv


def build_sampler(dataset: Dataset, env: DistributedEnv, shuffle: bool) -> Sampler:
    if env.is_distributed:
        return DistributedSampler(dataset, num_replicas=env.world_size, rank=env.rank, shuffle=shuffle)
    return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
