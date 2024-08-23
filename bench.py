from time import time
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor.placement_types import Shard, Replicate, Partial
from torch.distributed._tensor._redistribute import Partition



device_mesh = DeviceMesh('cuda', torch.arange(4).reshape(2, 2))
global_tensor = torch.randn(16, 8192, 8192).cuda()
rank = dist.get_rank()
warmup = 10
runs = 50

def bench(src_placement, tgt_placement):
    src_dtensor = distribute_tensor(global_tensor, device_mesh) # replicate from rank 0
    src_dtensor = src_dtensor.redistribute(device_mesh, src_placement)

    for _ in range(warmup):
        tgt_dtensor0 = src_dtensor.redistribute(device_mesh, tgt_placement)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_t = time()
    for _ in range(runs):
        tgt_dtensor0 = src_dtensor.redistribute(device_mesh, tgt_placement)
    dist.barrier()
    torch.cuda.synchronize()
    rule_time = time() - start_t
    rule_mem = torch.cuda.max_memory_allocated()

    for _ in range(warmup):
        tgt_dtensor1 = src_dtensor.redistribute(device_mesh, tgt_placement, p2p=True)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_t = time()
    for _ in range(runs):
        tgt_dtensor1 = src_dtensor.redistribute(device_mesh, tgt_placement, p2p=True)
    dist.barrier()
    torch.cuda.synchronize()
    p2p_time = time() - start_t
    p2p_mem = torch.cuda.max_memory_allocated()
    assert torch.allclose(tgt_dtensor0._local_tensor, tgt_dtensor1._local_tensor)
    if rank == 0:
        print(f"============================{src_dtensor._spec} -> {tgt_dtensor0._spec}================================")
        print(f"Rule based time: {rule_time}, max memory: {rule_mem / 1024**2} MB")
        print(f"P2P based time: {p2p_time} max memory: {p2p_mem / 1024**2} MB")
        print("Is P2P better? time: " + ("✅" if p2p_time < rule_time else "❌") + " mem: " + ("✅" if p2p_mem < rule_mem else "❌"))

if __name__ == "__main__":
    # choices = [Replicate(), Shard(0), Shard(1), Shard(2)]
    # for a in choices:
    #     for b in choices:
    #         for c in choices:
    #             for d in choices:
    #                 bench([a, b], [c, d])
    # exit()

    bench([Shard(0), Shard(0)], [Replicate(), Replicate()])
    bench([Replicate(), Replicate()], [Shard(0), Shard(0)])

    bench([Shard(0), Replicate()], [Shard(1), Replicate()])
    bench([Shard(1), Replicate()], [Shard(0), Replicate()])

    bench([Shard(1), Replicate()], [Shard(2), Replicate()])
    bench([Shard(2), Replicate()], [Shard(1), Replicate()])

    bench([Shard(0), Shard(1)], [Shard(2), Shard(1)])
    bench([Shard(2), Shard(1)], [Shard(0), Shard(1)])

    bench([Shard(0), Shard(1)], [Replicate(), Replicate()])
    bench([Replicate(), Replicate()], [Shard(0), Shard(1)])

    bench([Shard(0), Shard(1)], [Shard(1), Shard(0)])
    bench([Shard(1), Shard(0)], [Shard(0), Shard(1)])

    bench([Shard(2), Shard(1)], [Shard(1), Shard(2)])
    bench([Shard(1), Shard(2)], [Shard(2), Shard(1)])

    bench([Shard(0), Shard(0)], [Shard(0), Shard(1)])
    bench([Shard(0), Shard(1)], [Shard(0), Shard(0)])

    bench([Shard(0), Shard(0)], [Shard(1), Shard(1)])
    bench([Shard(1), Shard(1)], [Shard(0), Shard(0)])
