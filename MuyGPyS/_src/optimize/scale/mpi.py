# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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
    K: np.ndarray,
    nn_targets: np.ndarray,
) -> np.ndarray:
    local_batch_count, nn_count, _ = nn_targets.shape
    local_sum = _analytic_scale_optim_unnormalized(K, nn_targets)
    global_sum = world.allreduce(local_sum, op=MPI.SUM)
    global_batch_count = world.allreduce(local_batch_count, op=MPI.SUM)
    return global_sum / (nn_count * global_batch_count)
