# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from mpi4py import MPI

import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._src.optimize.loss.numpy import (
    _mse_fn_unnormalized,
    _cross_entropy_fn as _cross_entropy_fn_n,
    _lool_fn as _lool_fn_n,
    _pseudo_huber_fn as _pseudo_huber_fn_n,
    _looph_fn as _looph_fn_n,
)

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
    **kwargs,
) -> float:
    local_log_loss = _cross_entropy_fn_n(predictions, targets, **kwargs)
    global_log_loss = world.allreduce(local_log_loss, op=MPI.SUM)
    return global_log_loss


def _lool_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
    variances: np.ndarray,
    scale: np.ndarray,
) -> float:
    local_likelihoods = _lool_fn_n(predictions, targets, variances, scale)
    global_likelihood = world.allreduce(local_likelihoods, op=MPI.SUM)
    return global_likelihood


def _lool_fn_unscaled(
    predictions: np.ndarray, targets: np.ndarray, variances: np.ndarray
) -> float:
    local_likelihoods = _lool_fn_unscaled(predictions, targets, variances)
    global_likelihood = world.allreduce(local_likelihoods, op=MPI.SUM)
    return global_likelihood


def _pseudo_huber_fn(
    predictions: np.ndarray, targets: np.ndarray, boundary_scale: float = 1.5
) -> float:
    local_pseudo_huber = _pseudo_huber_fn_n(
        predictions, targets, boundary_scale=boundary_scale
    )
    global_pseudo_huber = world.allreduce(local_pseudo_huber, op=MPI.SUM)
    return global_pseudo_huber


def _looph_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
    variances: np.ndarray,
    scale: np.ndarray,
    boundary_scale: float = 3.0,
) -> float:
    local_looph = _looph_fn_n(
        predictions, targets, variances, scale, boundary_scale=boundary_scale
    )
    global_looph = world.allreduce(local_looph, op=MPI.SUM)
    return global_looph
