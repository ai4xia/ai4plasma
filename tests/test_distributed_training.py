from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_masked_unet3d import DistributedEvalSampler  # noqa: E402


def test_distributed_eval_sampler_covers_each_sample_once():
    dataset = list(range(23))
    world_size = 4
    shards = [
        list(DistributedEvalSampler(dataset, rank=rank, world_size=world_size))
        for rank in range(world_size)
    ]

    combined = [index for shard in shards for index in shard]
    assert sorted(combined) == list(range(len(dataset)))
    assert len(combined) == len(set(combined))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
