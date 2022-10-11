# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.optimize.sigma_sq.numpy import (
    _analytic_sigma_sq_optim_unnormalized,
)
from MuyGPyS import config

from mpi4py import MPI

import numpy as np

world = config.mpi_state.comm_world
# rank = config.mpi_state.comm_world.Get_rank()
# size = config.mpi_state.comm_world.Get_size()


def _analytic_sigma_sq_optim(
    K: np.ndarray,
    nn_targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    local_batch_count, nn_count, _ = nn_targets.shape
    local_sum = _analytic_sigma_sq_optim_unnormalized(K, nn_targets, eps)
    global_sum = world.allreduce(local_sum, op=MPI.SUM)
    global_batch_count = world.allreduce(local_batch_count, op=MPI.SUM)
    return global_sum / (nn_count * global_batch_count)
