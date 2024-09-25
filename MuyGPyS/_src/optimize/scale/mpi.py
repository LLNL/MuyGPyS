# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from mpi4py import MPI

import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._src.optimize.scale.numpy import (
    _analytic_scale_optim_unnormalized,
)

world = config.mpi_state.comm_world
# rank = config.mpi_state.comm_world.Get_rank()
# size = config.mpi_state.comm_world.Get_size()


def _analytic_scale_optim(
    Kin: np.ndarray, nn_targets: np.ndarray, batch_dim_count: int = 1, **kwargs
) -> np.ndarray:
    in_dim_count = (Kin.ndim - batch_dim_count) // 2

    batch_shape = Kin.shape[:batch_dim_count]
    in_shape = Kin.shape[batch_dim_count + in_dim_count :]

    local_batch_size = np.prod(batch_shape, dtype=int)
    in_size = np.prod(in_shape, dtype=int)

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    nn_targets_flat = nn_targets.reshape(batch_shape + (in_size, 1))
    local_sum = _analytic_scale_optim_unnormalized(
        Kin_flat, nn_targets_flat, **kwargs
    )
    global_sum = world.allreduce(local_sum, op=MPI.SUM)
    global_batch_count = world.allreduce(local_batch_size, op=MPI.SUM)
    return global_sum / (in_size * global_batch_count)
