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

from typing import Callable

from MuyGPyS import config

from MuyGPyS.optimize.utils import _switch_on_opt_method


def make_obj_fn(obj_method: str, opt_method: str, *args) -> Callable:
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

    Returns:
        A Callable `objective_fn`, whose format depends on `opt_method`.
    """
    if obj_method == "loo_crossval":
        return make_loo_crossval_fn(opt_method, *args)
    else:
        raise ValueError(f"Unsupported objective method: {obj_method}")


def make_loo_crossval_fn(
    opt_method: str,
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
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
        loss_fn:
            The loss function to be minimized. Can be any function that accepts
            two `numpy.ndarray` objects indicating the prediction and target
            values, in that order.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        predict_fn:
            A function that realizes MuyGPs prediction given an epsilon value.
            The given value is unused if epsilon is fixed.
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
    return _switch_on_opt_method(
        opt_method,
        make_loo_crossval_kwargs_fn,
        make_loo_crossval_array_fn,
        loss_fn,
        kernel_fn,
        predict_fn,
        pairwise_dists,
        crosswise_dists,
        batch_nn_targets,
        batch_targets,
    )


def loo_crossval_array(
    x0: np.ndarray,
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> float:
    """
    Leave-one-out cross validation.

    Returns leave-one-out cross validation performance for a set `MuyGPS`
    object.
    Predicts on all of the training data at once.
    This function is designed for use with
    :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
    `opt_method="scipy"`, and embeds the optimization parameters into the `x0`
    vector.

    Args:
        x0:
            Current guess for hyperparameter values of shape `(opt_count,)`.
        loss_fn:
            The loss function to be minimized. Can be any function that accepts
            two `numpy.ndarray` objects indicating the prediction and target
            values, in that order.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        predict_fn:
            A function that realizes MuyGPs prediction given an epsilon value.
            The given value is unused if epsilon is fixed.
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
        The evaluation of `objective_fn` on the predicted versus expected
        response.
    """
    K = kernel_fn(pairwise_dists, x0)
    Kcross = kernel_fn(crosswise_dists, x0)

    predictions = predict_fn(
        K,
        Kcross,
        batch_nn_targets,
        x0,
    )

    return loss_fn(predictions, batch_targets)


def make_loo_crossval_array_fn(
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    """
    Prepare a leave-one-out cross validation function as a function purely of
    the hyperparameters to be optimized.

    This function is designed for use with
    :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
    `opt_method="scipy"`, and assumes that the optimization parameters will be
    passed in an `(optim_count,)` vector.

    Args:
        loss_fn:
            The loss function to be minimized. Can be any function that accepts
            two `numpy.ndarray` objects indicating the prediction and target
            values, in that order.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        predict_fn:
            A function that realizes MuyGPs prediction given an epsilon value.
            The given value is unused if epsilon is fixed.
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
        A Callable `objective_fn` as a function of only an `(optim_count,)`
        vector.
    """

    def caller_fn(x0):
        return loo_crossval_array(
            x0,
            loss_fn,
            kernel_fn,
            predict_fn,
            pairwise_dists,
            crosswise_dists,
            batch_nn_targets,
            batch_targets,
        )

    return caller_fn


def loo_crossval_kwargs(
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
    **kwargs,
) -> float:
    """
    Leave-one-out cross validation.

    Returns leave-one-out cross validation performance for a set `MuyGPS`
    object.
    Predicts on all of the training data at once.
    This function is designed for use with
    :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
    `opt_method="bayesian"`, and the optimization parameters as additional
    kwargs.

    Args:
        loss_fn:
            The loss function to be minimized. Can be any function that accepts
            two `numpy.ndarray` objects indicating the prediction and target
            values, in that order.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        predict_fn:
            A function that realizes MuyGPs prediction given an epsilon value.
            The given value is unused if epsilon is fixed.
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
        kwargs:
            Hyperparameter values to be optimized, e.g. `nu=0.32`.

    Returns:
        The evaluation of `objective_fn` on the predicted versus expected
        response.
    """
    K = kernel_fn(pairwise_dists, **kwargs)
    Kcross = kernel_fn(crosswise_dists, **kwargs)

    predictions = predict_fn(
        K,
        Kcross,
        batch_nn_targets,
        **kwargs,
    )

    return -loss_fn(predictions, batch_targets)


def make_loo_crossval_kwargs_fn(
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> Callable:
    """
    Prepare a leave-one-out cross validation function as a function purely of
    the hyperparameters to be optimized.

    This function is designed for use with
    :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
    `opt_method="bayesian"`, and assumes that the optimization parameters will
    be passed as keyword arguments.

    Args:
        loss_fn:
            The loss function to be minimized. Can be any function that accepts
            two `numpy.ndarray` objects indicating the prediction and target
            values, in that order.
        kernel_fn:
            A function that realizes kernel tensors given a list of the free
            parameters.
        predict_fn:
            A function that realizes MuyGPs prediction given an epsilon value.
            The given value is unused if epsilon is fixed.
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
        A Callable `objective_fn` as a function of only an `(optim_count,)`
        vector.
    """

    def caller_fn(**kwargs):
        return loo_crossval_kwargs(
            loss_fn,
            kernel_fn,
            predict_fn,
            pairwise_dists,
            crosswise_dists,
            batch_nn_targets,
            batch_targets,
            **kwargs,
        )

    return caller_fn
