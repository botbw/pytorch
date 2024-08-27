from time import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor.placement_types import Shard, Replicate, Partial
from torch.distributed._tensor._redistribute import Partition

MARGIN=0

def compare(p2p, time):
    if abs(p2p - time) / time <= MARGIN:
        return '🟰'
    return '✅' if p2p < time else '❌'

device_mesh = DeviceMesh('cuda', torch.arange(4).reshape(2, 2))
rank = dist.get_rank()
WARMUP = 10
RUN = 20

profile_percent_time = defaultdict(list)
profile_percent_mem = defaultdict(list)

def all_reduce_scalar(scalar):
    tensor = torch.tensor([scalar]).cuda()
    dist.all_reduce(tensor)
    return tensor.item()

def bench(src_placement, tgt_placement, global_tensor):
    src_dtensor = distribute_tensor(global_tensor, device_mesh) # replicate from rank 0
    src_dtensor = src_dtensor.redistribute(device_mesh, src_placement)

    for _ in range(WARMUP):
        tgt_dtensor0 = src_dtensor.redistribute(device_mesh, tgt_placement)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_t = time()
    for _ in range(RUN):
        tgt_dtensor0 = src_dtensor.redistribute(device_mesh, tgt_placement)
    dist.barrier()
    torch.cuda.synchronize()
    rule_time = time() - start_t
    rule_mem = torch.cuda.max_memory_allocated()

    for _ in range(WARMUP):
        tgt_dtensor1 = src_dtensor.redistribute(device_mesh, tgt_placement, p2p=True)
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_t = time()
    for _ in range(RUN):
        tgt_dtensor1 = src_dtensor.redistribute(device_mesh, tgt_placement, p2p=True)
    dist.barrier()
    torch.cuda.synchronize()
    p2p_time = time() - start_t
    p2p_mem = torch.cuda.max_memory_allocated()

    rule_time = all_reduce_scalar(rule_time)
    rule_mem = all_reduce_scalar(rule_mem)
    p2p_time = all_reduce_scalar(p2p_time)
    p2p_mem = all_reduce_scalar(p2p_mem)
    assert torch.allclose(tgt_dtensor0._local_tensor, tgt_dtensor1._local_tensor)
    if rank == 0:
        print(f"============================{src_dtensor._spec} -> {tgt_dtensor0._spec}================================")
        print(f"Rule based time: {rule_time}, max memory: {rule_mem / 1024**2} MB")
        print(f"P2P based time: {p2p_time} max memory: {p2p_mem / 1024**2} MB")
        print("Is P2P better? time: " + compare(p2p_time, rule_time) + " mem: " + compare(p2p_mem, rule_mem))
        profile_percent_time[compare(p2p_time, rule_time)].append((p2p_time - rule_time) / rule_time)
        profile_percent_mem[compare(p2p_mem, rule_mem)].append((p2p_mem - rule_mem) / rule_mem)

    return p2p_time, rule_time

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    size_base = torch.tensor([256, 256], device='cuda')
    scale = range(4, 101, 5)

    p2p_time_list = []
    rule_time_list = []
    for multipler in scale:
        tensor = torch.randn((size_base * multipler).tolist())
        p2p_time, rule_time = bench([Shard(0), Shard(1)], [Shard(1), Shard(0)], tensor)
        p2p_time_list.append(p2p_time / RUN)
        rule_time_list.append(rule_time / RUN)

    if dist.get_rank() == 0:
        plt.figure(dpi=300)
        plt.plot(list(scale), p2p_time_list, label='p2p')
        plt.plot(list(scale), rule_time_list, label='rule')
        plt.legend()
        plt.xlabel(f'size {size_base.tolist()} * n')
        plt.ylabel('time (s)')
        plt.title(f"[Shard(0), Shard(1)] -> [Shard(1), Shard(0)] on {device_mesh.mesh.shape} mesh")
        plt.savefig('case1.png')

    p2p_time_list = []
    rule_time_list = []
    for multipler in scale:
        tensor = torch.randn((size_base * multipler).tolist())
        p2p_time, rule_time = bench([Shard(0), Shard(0)], [Shard(1), Shard(1)], tensor)
        p2p_time_list.append(p2p_time / RUN)
        rule_time_list.append(rule_time / RUN)

    if dist.get_rank() == 0:
        plt.figure(dpi=300)
        plt.plot(list(scale), p2p_time_list, label='p2p')
        plt.plot(list(scale), rule_time_list, label='rule')
        plt.legend()
        plt.xlabel(f'size {size_base.tolist()} * n')
        plt.ylabel('time (s)')
        plt.title(f"[Shard(0), Shard(0)] -> [Shard(1), Shard(1)] on {device_mesh.mesh.shape} mesh")
        plt.savefig('case2.png')

