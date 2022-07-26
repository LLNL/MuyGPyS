# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.gp.muygps.numpy import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
    _muygps_sigma_sq_optim_unnormalized,
)
from MuyGPyS import config

from mpi4py import MPI

import numpy as np

world = config.mpi_state.comm_world
# rank = config.mpi_state.comm_world.Get_rank()
# size = config.mpi_state.comm_world.Get_size()


def _muygps_sigma_sq_optim(
    K: np.ndarray,
    nn_targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    local_batch_count, _, _ = nn_targets.shape
    local_sum = _muygps_sigma_sq_optim_unnormalized(K, nn_targets, eps)
    global_batch_count, global_sum = world.allreduce(
        (local_batch_count, local_sum), op=MPI.SUM
    )
    return global_sum / global_batch_count
