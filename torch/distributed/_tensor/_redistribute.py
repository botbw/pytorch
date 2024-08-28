# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from copy import deepcopy
import logging
from functools import lru_cache
from itertools import permutations
from typing import Dict, cast, List, NamedTuple, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)


logger = logging.getLogger(__name__)


class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: Tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: List[int]


@lru_cache(maxsize=None)
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
) -> List[_TransformInfo]:
    """
    Generate the transform infos from the source placements to the target placements.

    To transform from source to target placement it might have multiple steps, i.e. it
    might decompose Si -> Sj into Si -> R -> Sj.
    This would detect if there're mis-aligned/nested shardings between src/dst placements.
    E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
    in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
    the former is a nested-sharding of a tensor already already sharded dimension 0, whereras
    the latter is the first sharding on tensor dimension 0.
    """
    transform_infos: List[_TransformInfo] = []

    device_mesh = src_spec.device_mesh
    my_coordinate = device_mesh.get_coordinate()
    assert my_coordinate is not None

    # logical shape records the logic tensor shape on the mesh dimension
    # this is useful to ensure uneven sharding gets correct output shape
    initial_logical_shape = list(src_spec.shape)
    mesh_dims_to_logical_shape = [initial_logical_shape]

    if device_mesh.ndim == 1:
        # if device_mesh is 1D, redistribute is a simple direct transformation
        transform_infos.append(
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(src_spec.placements[0], dst_spec.placements[0]),
                logical_shape=initial_logical_shape,
            )
        )
        return transform_infos

    # Handle multi-dim device mesh placement redistribution
    # First, we need to build the logical shape for each mesh dim
    # for correct allgathering uneven shards on each mesh dim (with dynamic padding)
    for i, (src, _) in enumerate(zip(src_spec.placements, dst_spec.placements)):
        current_logical_shape = mesh_dims_to_logical_shape[i]
        if isinstance(src, Shard):
            if i < device_mesh.ndim - 1:
                # calculate and save the logical shape for this sharding
                mesh_dim_size = device_mesh.size(mesh_dim=i)
                local_shard_size, _ = src._local_shard_size_on_dim(
                    current_logical_shape[src.dim],
                    mesh_dim_size,
                    my_coordinate[i],
                )
                new_logical_shape = list(current_logical_shape)
                new_logical_shape[src.dim] = local_shard_size
                mesh_dims_to_logical_shape.append(new_logical_shape)
        else:
            mesh_dims_to_logical_shape.append(current_logical_shape)

    # Next, we need to derive the transform infos from src to dst placements,
    # here we use a greedy search with step by step state transformations
    current_placements = list(src_spec.placements)
    target_placements = list(dst_spec.placements)

    if src_spec.num_shards > 1:
        # If src_spec have sharding, it could potentially have sharding that is misaligned with dst_spec
        # a common case of this is nested sharding (i.e. (S(0), S(0)) -> (R, S(0))).
        # In those cases, we first traverse from inner placement to outer placement
        # to detect misaligned shardings and properly replicate nested sharding first.
        for mesh_dim in reversed(range(len(current_placements))):
            current = current_placements[mesh_dim]
            target = target_placements[mesh_dim]
            # If target is not Shard, we can directly redistribute since we are traversing from innner
            # to outer placements here
            if isinstance(target, Shard):
                # If target is Shard, check for nested sharding on the tensor dim BEFORE the current mesh_dim
                shard_dim = target.dim
                current_mesh_sharding, target_mesh_sharding = [], []
                for i, (s, p) in enumerate(zip(current_placements, target_placements)):
                    if i >= mesh_dim:
                        break
                    if s.is_shard(shard_dim):
                        current_mesh_sharding.append(i)
                    if p.is_shard(shard_dim):
                        target_mesh_sharding.append(i)

                if current_mesh_sharding != target_mesh_sharding:
                    # if current/target_placements have misaligned sharding on the tensor dim BEFORE the current
                    # mesh_dim, we need to replicate the tensor on the mesh dim first to clear the nested sharding
                    target = Replicate()

            if current != target:
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(current, target),
                        logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                    )
                )
                current_placements[mesh_dim] = target

    # We always traverse from outer placement to inner placement to collect the remaining
    # needed transform infos (i.e. the replication from nested sharding might need to further
    # perform resharding to Shard again)
    for mesh_dim, (current, target) in enumerate(
        zip(current_placements, target_placements)
    ):
        if current != target:
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=mesh_dim,
                    src_dst_placements=(current, target),
                    logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                )
            )
            current_placements[mesh_dim] = target

    return transform_infos


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    transform_infos = _gen_transform_infos(current_spec, target_spec)

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)

        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                new_local_tensor = target_placement._replicate_to_shard(
                    local_tensor, device_mesh, i, my_coordinate[i]
                )
            else:
                assert (
                    current.is_shard()
                ), f"Current placement should be shard but found {current}"
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    new_local_tensor = shard_spec._to_new_shard_dim(
                        local_tensor,
                        device_mesh,
                        i,
                        transform_info.logical_shape,
                        target_placement.dim,
                    )
        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # skip the replicate to partial transformation when we are in backward pass
                # In this case we keep the grad as replicate, this is because we don't
                # want to convert the replicated gradients back to partial, although
                # that's logically conform with the same layout, converting the gradients
                # back to partial is actually useless as you would have to do reduce later
                # which would be more expensive than keeping it replicate! For this reason,
                # we keep the replicate grad here.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # for backward shard -> partial, we just need to convert the shard to replicate
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        assert new_local_tensor is not None
        local_tensor = new_local_tensor

    del local_tensor
    assert new_local_tensor is not None, "redistribute failed!"

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        async_op: bool = False,
        p2p: bool = False
    ):
        current_spec = input._spec
        ctx.current_spec = current_spec
        ctx.async_op = async_op
        ctx.p2p = p2p

        if current_spec.placements != placements:
            target_spec = DTensorSpec(
                device_mesh, placements, tensor_meta=input._spec.tensor_meta
            )

            local_tensor = input._local_tensor
            if p2p:
                output = redistribute_local_tensor_p2p(
                    local_tensor, current_spec, target_spec, async_op=async_op
                )
            else:
                output = redistribute_local_tensor(
                    local_tensor, current_spec, target_spec, async_op=async_op
                )
        else:
            # use the same local tensor if placements are the same.
            output = input._local_tensor
            target_spec = current_spec

        return dtensor.DTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        previous_spec = ctx.current_spec
        current_spec = grad_output._spec
        async_op = ctx.async_op
        p2p = ctx.p2p

        local_tensor = grad_output._local_tensor
        if p2p:
            output = redistribute_local_tensor_p2p(
                local_tensor,
                current_spec,
                previous_spec,
                async_op=async_op,
                is_backward=True,
            )
        else:
            output = redistribute_local_tensor(
                local_tensor,
                current_spec,
                previous_spec,
                async_op=async_op,
                is_backward=True,
            )
        # normalize the target placement to replicate if it is partial
        normalized_placements: List[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # keep target placement to replicate instead of partial in this case
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=grad_output.dtype,
            ),
        )
        output_dtensor = dtensor.DTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
            None
        )


@dataclass
class Partition:
    '''
        Tensor partition from a logical view, and the rank that holds it.
    '''
    start_coord: Tuple[int, ...]  # start coord of this partition
    end_coord: Tuple[int, ...]  # end coord of this partition
    rank: int  # rank that hold this partition
    partial_coord: Tuple[int, ...]  # partial coord of this partition

    def __post_init__(self):
        self._hash: Optional[int] = None

    def _hash_impl(self) -> int:
        return hash((self.start_coord, self.end_coord, self.rank, self.partial_coord))

    def __hash__(self) -> int:
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def __repr__(self) -> str:
        return f"{{{self.start_coord}->{self.end_coord} partial {self.partial_coord} on rank{self.rank}}}"

    def shard(self, tensor_dim: int, shard_num: int, shard_idx: int) -> "Partition":
        block_size = (self.end_coord[tensor_dim] - self.start_coord[tensor_dim] + shard_num - 1) // shard_num
        return Partition(
            self.start_coord[:tensor_dim] + (min(self.start_coord[tensor_dim] + block_size * shard_idx, self.end_coord[tensor_dim]),) + self.start_coord[tensor_dim+1:],
            self.end_coord[:tensor_dim] + (min(self.start_coord[tensor_dim] + block_size * (shard_idx + 1), self.end_coord[tensor_dim]),) + self.end_coord[tensor_dim+1:],
            self.rank,
            self.partial_coord
        )

    def partial(self, new_partial_idx: int) -> "Partition":
        return Partition(self.start_coord, self.end_coord, self.rank, self.partial_coord + (new_partial_idx,))

    @staticmethod
    def from_tensor_spec(spec: DTensorSpec) -> Tuple["Partition"]:
        if spec.tensor_meta is None:
            raise ValueError("tensor_meta is not set")

        tensor_mesh = spec.mesh.mesh
        global_shape, _, _ = spec.tensor_meta
        placements = spec.placements

        unravel = torch.unravel_index(torch.arange(tensor_mesh.numel()), tensor_mesh.shape)
        rank_to_coord = []
        for i in range(unravel[0].shape[0]):
            rank_to_coord.append(
                tuple(unravel[j][i].item() for j in range(tensor_mesh.ndim))
            )
        partitions = [
            Partition((0,) * len(global_shape), tuple(global_shape), rank, tuple()) for rank in tensor_mesh.flatten().tolist()
        ]

        for mesh_dim, placement in enumerate(placements):
            if placement.is_shard():
                shard = cast(Shard, placement)
                tensor_dim = shard.dim
                shard_num = tensor_mesh.size(mesh_dim)
                for partition in partitions:
                    partitions[partition.rank] = partition.shard(tensor_dim, shard_num, rank_to_coord[partition.rank][mesh_dim])
            elif placement.is_partial():
                partial = cast(Partial, placement)
                for partition in partitions:
                    partitions[partition.rank] = partition.partial(rank_to_coord[partition.rank][mesh_dim])
        return tuple(partitions)

    def from_src(self, src_partition: "Partition") -> Optional["Partition"]:

        start_coord = tuple(max(s1, s2) for s1, s2 in zip(self.start_coord, src_partition.start_coord))
        end_coord = tuple(min(e1, e2) for e1, e2 in zip(self.end_coord, src_partition.end_coord))
        if any(s >= e for s, e in zip(start_coord, end_coord)) or src_partition.partial_coord != self.partial_coord:
            return None

        return Partition(start_coord, end_coord, src_partition.rank, src_partition.partial_coord)

    @staticmethod
    @lru_cache(maxsize=None)
    def gen_recv_meta(src_partitions: Tuple["Partition"], tgt_partitions: Tuple["Partition"]) -> Dict[int, List["Partition"]]:
        recv_meta = defaultdict(list)
        for tgt in tgt_partitions:
            cache = defaultdict(int)
            # TODO better load balance strategy
            for src in sorted(src_partitions, key=lambda x: abs(x.rank - tgt.rank)):
                intersection = tgt.from_src(src)
                if intersection is None:
                    continue
                cache_str = f"{intersection.start_coord}{intersection.end_coord}"
                if cache_str not in cache:
                    recv_meta[tgt.rank].append(intersection)
                cache[cache_str] += 1

        return recv_meta

    @staticmethod
    def gen_p2p_utils(rank: int, src_tensor: torch.Tensor, local_src_partition: "Partition", local_dst_partition: "Partition", recv_meta: Dict[int, List["Partition"]]) -> Tuple[List[dist.P2POp], List[dist.P2POp], torch.Tensor, List[Tuple[slice]]]:
        send_ops = []
        recv_ops = []
        recv_slices = []
        buffer_shape = tuple(e - s for s, e in zip(local_dst_partition.start_coord, local_dst_partition.end_coord))
        recv_buffer = torch.empty(buffer_shape, dtype=src_tensor.dtype, device=src_tensor.device)
        for recv_rank, intersections in recv_meta.items():
            for intersection in intersections:
                send_rank = intersection.rank
                src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))

                if rank == send_rank == recv_rank:
                    # assign the static intersection to the buffer
                    recv_buffer[tgt_slice] = src_tensor[src_slice]
                    continue

                if rank == send_rank:
                    src_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_src_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    # TODO: possible to eliminate contiguous call to reduce mem usage?
                    send_ops.append(dist.P2POp(dist.isend, src_tensor[src_slice].contiguous(), recv_rank))

                if rank == recv_rank:
                    tgt_slice = tuple(slice(st - base, en - base) for base, st, en in zip(local_dst_partition.start_coord, intersection.start_coord, intersection.end_coord))
                    # TODO: possible to eliminate contiguous call to reduce mem usage?
                    recv_ops.append(dist.P2POp(dist.irecv, recv_buffer[tgt_slice].contiguous(), send_rank))
                    recv_slices.append(tgt_slice)

        # TODO: handle the case where there is no communication, currently sending a dummy tensor
        if len(send_ops + recv_ops) == 0:
            # NOTE: recv_buffer, recv_slices will still be empty
            dummy_tensor = torch.zeros(1, dtype=src_tensor.dtype, device=src_tensor.device)
            send_ops.append(dist.P2POp(dist.isend, dummy_tensor, rank))
            recv_ops.append(dist.P2POp(dist.irecv, dummy_tensor, rank))

        return send_ops, recv_ops, recv_buffer, recv_slices

    @staticmethod
    def fill_recv_buffer(recv_ops: List[dist.P2POp], recv_buffer: torch.Tensor, recv_slices: List[Tuple[slice]]) -> torch.Tensor:
        for op, slice in zip(recv_ops, recv_slices):
            if op.op is not dist.irecv:
                raise ValueError("recv_ops must be irecv")
            recv_buffer[slice] = op.tensor

        return recv_buffer


@lru_cache(maxsize=None)
def _gen_immediate_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
) -> Tuple[List[_TransformInfo], DTensorSpec]:
    """
    Similar to _gen_transform_infos, but only perform simple transformation by dim if it can be achieved by one single collective operation.
    Consider device dim k:
        1. R -> R: No-op
        2. R -> S(i): Safe when num of S(i) are the same in cur_spec[:k] and dst_spec[:k] and no S(i) in cur_spec[k+1:]
        3. R -> P: Always safe

        4. S(i) -> R: Safe when no S(i) in cur_spec[k+1:]
        5. S(i) -> S(j): Safe when S(i) meets 4 and S(j) meets 2
        6. S(i) -> P: Used as S(i) -> R during backward

        7. P -> R: Always safe
        8. P -> S(i): P -> R -> S(i), safe when 7 and 2 are satisfied
        9. P -> P: No-op

        10. if P -> S(i) is not executed, do P -> R since P2P cannot handle reduce_scatter
    """
    transform_infos: List[_TransformInfo] = []

    device_mesh = src_spec.device_mesh
    my_coordinate = device_mesh.get_coordinate()
    assert my_coordinate is not None

    # logical shape records the logic tensor shape on the mesh dimension
    # this is useful to ensure uneven sharding gets correct output shape
    initial_logical_shape = list(src_spec.shape)
    mesh_dims_to_logical_shape = [initial_logical_shape]

    if device_mesh.ndim == 1:
        # if device_mesh is 1D, redistribute is a simple direct transformation
        transform_infos.append(
            _TransformInfo(
                mesh_dim=0,
                src_dst_placements=(src_spec.placements[0], dst_spec.placements[0]),
                logical_shape=initial_logical_shape,
            )
        )
        return transform_infos, dst_spec

    # Handle multi-dim device mesh placement redistribution
    # First, we need to build the logical shape for each mesh dim
    # for correct allgathering uneven shards on each mesh dim (with dynamic padding)
    for k, (src, _) in enumerate(zip(src_spec.placements, dst_spec.placements)):
        current_logical_shape = mesh_dims_to_logical_shape[k]
        if isinstance(src, Shard):
            if k < device_mesh.ndim - 1:
                # calculate and save the logical shape for this sharding
                mesh_dim_size = device_mesh.size(mesh_dim=k)
                local_shard_size, _ = src._local_shard_size_on_dim(
                    current_logical_shape[src.dim],
                    mesh_dim_size,
                    my_coordinate[k],
                )
                new_logical_shape = list(current_logical_shape)
                new_logical_shape[src.dim] = local_shard_size
                mesh_dims_to_logical_shape.append(new_logical_shape)
        else:
            mesh_dims_to_logical_shape.append(current_logical_shape)

    tgt_placements = list(dst_spec.placements)

    def count_shards(placements: List[Placement], shard: Shard) -> int:
        return sum(1 for p in placements if p.is_shard(shard.dim))

    src_placements = src_spec.placements
    max_immediate_transform_cnt = 0
    best_transform_infos = []
    best_cur_placements = deepcopy(list(src_placements))
    for seq in permutations(range(len(src_placements)-1, -1, -1)):
        immediate_transform_cnt = 0
        cur_placements = deepcopy(list(src_placements))
        transform_infos = []
        def to_shard(k: int, tgt_shard: Shard):
            return count_shards(cur_placements[:k], tgt_shard) == count_shards(cur_placements[:k], tgt_shard) and count_shards(cur_placements[k + 1:], tgt_shard) == 0

        def from_shard(k: int, cur_shard: Shard):
            return count_shards(cur_placements[k + 1:], cur_shard) == 0

        for k in seq:
            cur = cur_placements[k]
            tgt = tgt_placements[k]

            if cur == tgt:
                continue

            if cur.is_replicate():
                cur = cast(Replicate, cur)
                if tgt.is_shard():
                    tgt = cast(Shard, tgt)
                    if to_shard(k, tgt):
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=k,
                                src_dst_placements=(cur, tgt),
                                logical_shape=mesh_dims_to_logical_shape[k],
                            )
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
                elif tgt.is_partial():
                    tgt = cast(Partial, tgt)
                    transform_infos.append(
                        _TransformInfo(
                            mesh_dim=k,
                            src_dst_placements=(cur, tgt),
                            logical_shape=mesh_dims_to_logical_shape[k],
                        )
                    )
                    cur_placements[k] = tgt
                    immediate_transform_cnt += 1
            elif cur.is_shard():
                cur = cast(Shard, cur)
                if tgt.is_replicate():
                    tgt = cast(Replicate, tgt)
                    if from_shard(k, cur):
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=k,
                                src_dst_placements=(cur, tgt),
                                logical_shape=mesh_dims_to_logical_shape[k],
                            )
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
                elif tgt.is_shard():
                    tgt = cast(Shard, tgt)
                    if cur.dim != tgt.dim and from_shard(k, cur) and to_shard(k, tgt):
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=k,
                                src_dst_placements=(cur, tgt),
                                logical_shape=mesh_dims_to_logical_shape[k],
                            )
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
                elif tgt.is_partial():
                    tgt = cast(Partial, tgt)
                    if from_shard(k, cur):
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=k,
                                src_dst_placements=(cur, tgt),
                                logical_shape=mesh_dims_to_logical_shape[k],
                            )
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1
            else:
                cur = cast(Partial, cur)
                if tgt.is_replicate():
                    tgt = cast(Replicate, tgt)
                    transform_infos.append(
                        _TransformInfo(
                            mesh_dim=k,
                            src_dst_placements=(cur, tgt),
                            logical_shape=mesh_dims_to_logical_shape[k],
                        )
                    )
                    cur_placements[k] = tgt
                    immediate_transform_cnt += 1
                elif tgt.is_shard():
                    tgt = cast(Shard, tgt)
                    if to_shard(k, tgt):
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=k,
                                src_dst_placements=(cur, tgt),
                                logical_shape=mesh_dims_to_logical_shape[k],
                            )
                        )
                        cur_placements[k] = tgt
                        immediate_transform_cnt += 1

        if immediate_transform_cnt > max_immediate_transform_cnt:
            max_immediate_transform_cnt = immediate_transform_cnt
            best_transform_infos = transform_infos
            best_cur_placements = cur_placements

    # transform rest P -> S to R -> S since P2P cannot handle reduce_scatter
    for k, (cur, tgt) in enumerate(zip(best_cur_placements, tgt_placements)):
        if cur.is_partial() and tgt.is_shard():
            cur = cast(Partial, cur)
            tgt = Replicate()
            best_transform_infos.append(
                _TransformInfo(
                    mesh_dim=k,
                    src_dst_placements=(cur, tgt),
                    logical_shape=mesh_dims_to_logical_shape[k],
                )
            )
            best_cur_placements[k] = tgt

    return best_transform_infos, DTensorSpec(src_spec.mesh, tuple(best_cur_placements), src_spec.tensor_meta)

def redistribute_local_tensor_p2p(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
) -> torch.Tensor:
    """
    Similar to redistribute_local_tensor, but P2P is used for optimizing complicated transformation
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = local_tensor
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    transform_infos, current_spec = _gen_immediate_transform_infos(current_spec, target_spec)

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)

        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                new_local_tensor = target_placement._replicate_to_shard(
                    local_tensor, device_mesh, i, my_coordinate[i]
                )
            else:
                assert (
                    current.is_shard()
                ), f"Current placement should be shard but found {current}"
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    new_local_tensor = shard_spec._to_new_shard_dim(
                        local_tensor,
                        device_mesh,
                        i,
                        transform_info.logical_shape,
                        target_placement.dim,
                    )
        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # skip the replicate to partial transformation when we are in backward pass
                # In this case we keep the grad as replicate, this is because we don't
                # want to convert the replicated gradients back to partial, although
                # that's logically conform with the same layout, converting the gradients
                # back to partial is actually useless as you would have to do reduce later
                # which would be more expensive than keeping it replicate! For this reason,
                # we keep the replicate grad here.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # for backward shard -> partial, we just need to convert the shard to replicate
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        assert new_local_tensor is not None
        local_tensor = new_local_tensor

    del local_tensor
    assert new_local_tensor is not None, "redistribute failed!"

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    if current_spec != target_spec:
        src_partitions = Partition.from_tensor_spec(current_spec)
        tgt_partitions = Partition.from_tensor_spec(target_spec)
        recv_info = Partition.gen_recv_meta(src_partitions, tgt_partitions)
        rank = current_spec.mesh.get_rank()
        send_ops, recv_ops, new_local_tensor, recv_slices = Partition.gen_p2p_utils(rank, new_local_tensor, src_partitions[rank], tgt_partitions[rank], recv_info)
        works = dist.batch_isend_irecv(send_ops + recv_ops)
        for work in works:
            work.wait()
        new_local_tensor = Partition.fill_recv_buffer(recv_ops, new_local_tensor, recv_slices)

    return new_local_tensor
