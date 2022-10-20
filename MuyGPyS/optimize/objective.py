# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Objective Handling

MuyGPyS includes predefined objective functions and convenience functions for
indicating them to optimization.
"""

import numpy as np

from typing import Callable, Tuple

from MuyGPyS import config

from MuyGPyS.optimize.utils import _switch_on_opt_method, _switch_on_loss_method
from MuyGPyS.optimize.loss import get_loss_func


def make_obj_fn(
    obj_method: str, opt_method: str, loss_method: str, *args
) -> Callable:
    """
    Prepare an objective function as a function purely of the hyperparameters
    to be optimized.

    This function is designed for use with
    :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()`, and the format
    depends on the `opt_method` argument.

    Args:
        obj_method:
            The name of the objective function to be minimized.
        opt_method:
            The name of the optimization method to be utilized.
        loss_method:
            Indicates the loss function to be used.

    Returns:
        A Callable `objective_fn`, whose format depends on `opt_method`.
    """
    if obj_method == "loo_crossval":
        return make_loo_crossval_fn(opt_method, loss_method, *args)
    else:
        raise ValueError(f"Unsupported objective method: {obj_method}")


def make_loo_crossval_fn(
    opt_method: str,
    loss_method: str,
    loss_fn: Callable,
    kernel_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    """
    Prepare a leave-one-out cross validation function as a function purely of
    the hyperparameters to be optimized.

    This function is designed for use with
    :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()`, and the format
    depends on the `opt_method` argument.

    Args:
        opt_method:
            The name of the optimization method to be utilized.
        loss_method:
            Indicates the loss function to be used.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        mean_fn:
            A function that realizes MuyGPs posterior mean prediction given an
            epsilon value. The given value is unused if epsilon is fixed.
        var_fn:
            A function that realizes MuyGPs posterior variance prediction given
            an epsilon value. The given value is unused if epsilon is fixed.
        sigma_sq_fn:
            A function that realizes `sigma_sq` optimization given an epsilon
            value. The given value is unused if epsilon is fixed.
        pairwise_dists:
            Distance tensor of floats of shape
            `(batch_count, nn_count, nn_count)` whose second two dimensions give
            the pairwise distances between the nearest neighbors of each batch
            element.
        crosswise_dists:
            Distance matrix of floats of shape `(batch_count, nn_count)` whose
            rows give the distances between each batch element and its nearest
            neighbors.
        batch_nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.
        batch_targets:
            Matrix of floats of shape `(batch_count, response_count)` whose rows
            give the expected response for each  batch element.

    Returns:
        A Callable `objective_fn`, whose format depends on `opt_method`.
    """
    kernels_fn = _switch_on_opt_method(
        opt_method,
        make_kwargs_kernels_fn,
        make_array_kernels_fn,
        kernel_fn,
        pairwise_dists,
        crosswise_dists,
    )
    predict_and_loss_fn = _switch_on_loss_method(
        loss_method,
        make_raw_predict_and_loss_fn,
        make_raw_predict_and_loss_fn,
        make_var_predict_and_loss_fn,
        opt_method,
        loss_fn,
        mean_fn,
        var_fn,
        sigma_sq_fn,
        batch_nn_targets,
        batch_targets,
    )
    return _switch_on_opt_method(
        opt_method,
        make_kwargs_obj_fn,
        make_array_obj_fn,
        kernels_fn,
        predict_and_loss_fn,
    )


def make_array_obj_fn(
    kernels_fn: Callable, predict_and_loss_fn: Callable
) -> Callable:
    def obj_fn(x0):
        K, Kcross = kernels_fn(x0)
        return predict_and_loss_fn(K, Kcross, x0)

    return obj_fn


def make_kwargs_obj_fn(
    kernels_fn: Callable, predict_and_loss_fn: Callable, **kwargs
) -> Callable:
    def obj_fn(**kwargs):
        K, Kcross = kernels_fn(**kwargs)
        return predict_and_loss_fn(K, Kcross, **kwargs)

    return obj_fn


def make_array_kernels_fn(
    kernel_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
) -> Callable:
    def kernels_fn(x0):
        K = kernel_fn(pairwise_dists, x0)
        Kcross = kernel_fn(crosswise_dists, x0)
        return K, Kcross

    return kernels_fn


def make_kwargs_kernels_fn(
    kernel_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
) -> Callable:
    def kernels_fn(**kwargs):
        K = kernel_fn(pairwise_dists, **kwargs)
        Kcross = kernel_fn(crosswise_dists, **kwargs)
        return K, Kcross

    return kernels_fn


def make_raw_predict_and_loss_fn(
    opt_method: str,
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    return _switch_on_opt_method(
        opt_method,
        make_raw_kwargs_predict_and_loss_fn,
        make_raw_array_predict_and_loss_fn,
        loss_fn,
        mean_fn,
        var_fn,
        sigma_sq_fn,
        batch_nn_targets,
        batch_targets,
    )


def make_raw_array_predict_and_loss_fn(
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    def predict_and_loss_fn(K, Kcross, x0):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            x0,
        )

        return loss_fn(predictions, batch_targets)

    return predict_and_loss_fn


def make_raw_kwargs_predict_and_loss_fn(
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    def predict_and_loss_fn(K, Kcross, **kwargs):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            **kwargs,
        )

        return -loss_fn(predictions, batch_targets)

    return predict_and_loss_fn

def make_var_predict_and_loss_fn(
    opt_method: str,
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    return _switch_on_opt_method(
        opt_method,
        make_var_kwargs_predict_and_loss_fn,
        make_var_array_predict_and_loss_fn,
        loss_fn,
        mean_fn,
        var_fn,
        sigma_sq_fn,
        batch_nn_targets,
        batch_targets,
    )


def make_var_array_predict_and_loss_fn(
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    def predict_and_loss_fn(K, Kcross, x0):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            x0,
        )

        sigma_sq = sigma_sq_fn(K, batch_nn_targets, x0)

        variances = var_fn(K, Kcross, x0)
        return loss_fn(predictions, batch_targets, variances,sigma_sq)

    return predict_and_loss_fn


def make_var_kwargs_predict_and_loss_fn(
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    def predict_and_loss_fn(K, Kcross, **kwargs):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            **kwargs,
        )
        sigma_sq = sigma_sq_fn(K, batch_nn_targets, **kwargs)

        variances = var_fn(K, Kcross, **kwargs)

        return -loss_fn(predictions, batch_targets, variances, sigma_sq)

    return predict_and_loss_fn
