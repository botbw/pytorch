from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch.distributed._tensor.placement_types import Partition, Shard, Replicate, TensorMeta
from torch.distributed._tensor.api import DTensorSpec
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

@dataclass
class FakeDeviceMesh:
    mesh: torch.Tensor

    def size(self, mesh_dim: Optional[int] = None) -> int:
        return self.mesh.numel() if mesh_dim is None else self.mesh.size(mesh_dim)

class TestPartition(TestCase):
    def test_partition(self):
        global_tensor = torch.arange(72).reshape(6, 12)
        device_mesh = FakeDeviceMesh(torch.arange(6).reshape(2, 3))

        before_spec = DTensorSpec(mesh=device_mesh, placements=[Shard(0), Shard(1)], tensor_meta=TensorMeta.from_global_tensor(global_tensor))
        after_spec = DTensorSpec(mesh=device_mesh, placements=[Shard(1), Shard(0)], tensor_meta=TensorMeta.from_global_tensor(global_tensor))
        
        before_partitions = Partition.from_tensor_spec(before_spec)
        after_partitions = Partition.from_tensor_spec(after_spec)

        # print(f"{len(before_partitions)=} {before_partitions=}")
        # print(f"{len(after_partitions)=} {after_partitions=}")
        print(before_partitions[0].intersect_with_multiple(after_partitions))
        print(after_partitions[0].intersect_with_multiple(before_partitions))

if __name__ == "__main__":
    run_tests()
