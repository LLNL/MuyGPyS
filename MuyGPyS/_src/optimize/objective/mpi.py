# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.optimize.objective.numpy import _mse_fn_unnormalized
from MuyGPyS import config

from mpi4py import MPI

import numpy as np

world = config.mpi_state.comm_world


def _mse_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    local_batch_count, response_count = predictions.shape
    local_squared_errors = _mse_fn_unnormalized(predictions, targets)
    global_batch_count = world.allreduce(local_batch_count, op=MPI.SUM)
    global_squared_errors = world.allreduce(local_squared_errors, op=MPI.SUM)
    return global_squared_errors / (global_batch_count * response_count)


def _make_mpi_obj_fn(kwargs_opt_fn, comm):
    def caller_fn(**kwargs):
        local_obj = kwargs_opt_fn(**kwargs)
        global_obj = comm.allreduce(local_obj, op=MPI.SUM)
        return global_obj

    return caller_fn


def _cross_entropy_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
    ll_eps: float = 1e-15,
) -> float:
    pass
