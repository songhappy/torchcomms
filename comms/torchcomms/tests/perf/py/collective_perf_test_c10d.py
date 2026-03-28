# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

"""
collective_perf_test_c10d.py — Collective perf tests using pure torch.distributed (c10d).

This uses the standard ProcessGroup path with NO torchcomms routing.
For the torchcomms-via-c10d variant, see collective_perf_test_c10d_comms.py.

Launch with:
  torchrun --nproc_per_node=<N> -m torchcomms.tests.perf.py.collective_perf_test_c10d [args]
"""

import os
import sys
from typing import Any

import torch
import torch.distributed as dist
from torchcomms.tests.perf.py.collective_perf_test import (
    dtype_to_string,
    parse_args,
    print_usage,
    run_collectives,
    validate_params,
)


class C10dComm:
    def __init__(self, backend: str):
        self._backend = backend

    def get_backend(self) -> str:
        return self._backend

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_size(self) -> int:
        return dist.get_world_size()

    def all_reduce(
        self, tensor: torch.Tensor, op: Any, async_op: bool = False
    ):
        del op
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)

    def all_gather(
        self, output_list: list[torch.Tensor], input_tensor: torch.Tensor, async_op: bool
    ):
        return dist.all_gather(output_list, input_tensor, async_op=async_op)

    def all_gather_single(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, async_op: bool
    ):
        return dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=async_op)

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_list: list[torch.Tensor],
        op: Any,
        async_op: bool,
    ):
        del op
        return dist.reduce_scatter(
            output_tensor, input_list, op=dist.ReduceOp.SUM, async_op=async_op
        )

    def reduce_scatter_single(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: Any,
        async_op: bool,
    ):
        del op
        return dist.reduce_scatter_tensor(
            output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=async_op
        )

    def all_to_all(
        self,
        output_list: list[torch.Tensor],
        input_list: list[torch.Tensor],
        async_op: bool,
    ):
        return dist.all_to_all(output_list, input_list, async_op=async_op)

    def all_to_all_single(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, async_op: bool
    ):
        return dist.all_to_all_single(output_tensor, input_tensor, async_op=async_op)

    def broadcast(self, tensor: torch.Tensor, root: int, async_op: bool):
        return dist.broadcast(tensor, src=root, async_op=async_op)

    def reduce(
        self, tensor: torch.Tensor, root: int, op: Any, async_op: bool
    ):
        del op
        return dist.reduce(tensor, dst=root, op=dist.ReduceOp.SUM, async_op=async_op)

    def scatter(
        self,
        output_tensor: torch.Tensor,
        input_list: list[torch.Tensor],
        root: int,
        async_op: bool,
    ):
        scatter_list = input_list if self.get_rank() == root else None
        return dist.scatter(output_tensor, scatter_list=scatter_list, src=root, async_op=async_op)

    def gather(
        self,
        output_list: list[torch.Tensor],
        input_tensor: torch.Tensor,
        root: int,
        async_op: bool,
    ):
        gather_list = output_list if self.get_rank() == root else None
        return dist.gather(input_tensor, gather_list=gather_list, dst=root, async_op=async_op)

    def send(self, tensor: torch.Tensor, peer: int, async_op: bool):
        if async_op:
            return dist.isend(tensor, dst=peer)
        return dist.send(tensor, dst=peer)

    def recv(self, tensor: torch.Tensor, peer: int, async_op: bool):
        if async_op:
            return dist.irecv(tensor, src=peer)
        return dist.recv(tensor, src=peer)

    def barrier(self, async_op: bool):
        return dist.barrier(async_op=async_op)

    def finalize(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def _resolve_backend(device: torch.device) -> str:
    backend = os.environ.get("TEST_BACKEND")
    if backend:
        return backend
    if device.type == "xpu":
        return "xccl"
    return "nccl"


def _setup_device() -> torch.device:
    requested = os.environ.get("TEST_DEVICE", "cuda")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if requested == "xpu":
        torch.xpu.set_device(local_rank)
        return torch.device("xpu", local_rank)

    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def main() -> int:
    collective, params, parse_error = parse_args(sys.argv[1:])

    if parse_error == "help":
        print_usage(sys.argv[0])
        return 0

    error = validate_params(collective, params)
    if error:
        print(f"Error: {error}\n", file=sys.stderr)
        print_usage(sys.argv[0])
        return 1

    device = _setup_device()
    backend = _resolve_backend(device)

    dist.init_process_group(backend=backend, init_method="env://")
    comm = C10dComm(backend)
    rank = comm.get_rank()
    num_ranks = comm.get_size()

    if rank == 0:
        print("C10d Collective Performance Test")
        print("===============================")
        print(f"Backend: {comm.get_backend()}")
        print(f"Ranks: {num_ranks}")
        print(f"Collective: {collective}")
        print(f"Mode: {'async' if params.async_op else 'sync'}")
        print(f"Dtype: {dtype_to_string(params.dtype)}")
        print(f"Warmup: {params.warmup_iterations}")
        print(f"Iterations: {params.measure_iterations}")
        print(f"Window: {params.iteration_window}")
        print(
            f"Size range: {params.min_size} - {params.max_size} "
            f"bytes (x{params.size_scaling_factor})"
        )
        print()

    run_collectives(collective, comm, params, device)

    if rank == 0:
        print("\nPerformance test completed.")

    comm.finalize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
