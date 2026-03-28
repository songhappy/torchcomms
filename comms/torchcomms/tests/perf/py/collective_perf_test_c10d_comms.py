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
import torchcomms  # noqa: F401  — must be imported before torch.distributed
import torch.distributed as dist
import torch.distributed.config as dist_config
dist_config.use_torchcomms = True


from torchcomms.tests.perf.py.collective_perf_test import (
    dtype_to_string,
    parse_args,
    print_usage,
    run_collectives,
    validate_params,
)
from torchcomms.tests.perf.py.collective_perf_test_c10d import (
    C10dComm, _resolve_backend, _setup_device
)


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
