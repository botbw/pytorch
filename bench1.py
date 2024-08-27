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

device_mesh = DeviceMesh('cuda', torch.arange(16).reshape(4, 4))
global_tensor = torch.randn(32, 8192, 4096).cuda()
rank = dist.get_rank()
WARMUP = 10
RUN = 20

profile_percent_time = defaultdict(list)
profile_percent_mem = defaultdict(list)

def all_reduce_scalar(scalar):
    tensor = torch.tensor([scalar]).cuda()
    dist.all_reduce(tensor)
    return tensor.item()

def bench(src_placement, tgt_placement):
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
    choices = [Replicate(), Shard(0), Shard(1), Shard(2)]
    for a in choices:
        for b in choices:
            for c in choices:
                for d in choices:
                    if (a, b) == (c, d):
                        continue
                    bench([a, b], [c, d])

    if rank == 0:
        import matplotlib.pyplot as plt
        import seaborn as sns
        def plot_box(di, name):
            plt.figure(dpi=300, figsize=(6, 12))
            color = {
                '🟰': 'grey',
                '✅': 'green',
                '❌': 'red'
            }
            for k, v in di.items():
                sns.swarmplot(x=[' ' for _ in v], y=v, color=color[k])
            plt.ylabel("Percent of optimizationi compared to main")
            plt.xlabel("All kind of placements")
            plt.title(f"benchmark on {device_mesh.mesh.shape} mesh")
            plt.gca().set_aspect(aspect='auto')
            plt.tight_layout()
            plt.savefig(f'{name}.png')

        plot_box(profile_percent_time, f'{device_mesh.mesh.shape}_optimization')
