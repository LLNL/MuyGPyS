# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Objective and Loss Function Handling

MuyGPyS includes predefined loss functions and convenience functions for 
indicating them to optimization.
"""

import numpy as np

from scipy.special import softmax
from sklearn.metrics import log_loss
from typing import Callable, Dict

from MuyGPyS.gp.muygps import MuyGPS


def get_loss_func(loss_method: str) -> Callable:
    """
    Select a loss function based upon string key.

    Currently supports strings `"log"` or `"cross-entropy"` for
    :func:`MuyGPyS.optimize.objective.cross_entropy_fn` and `"mse"` for
    :func:`MuyGPyS.optimize.objective.mse_fn`.

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.

    Returns:
        The loss function Callable.

    Raises:
        NotImplementedError:
            Unrecognized strings will result in an error.
    """
    loss_method = loss_method.lower()
    if loss_method == "cross-entropy" or loss_method == "log":
        return cross_entropy_fn
    elif loss_method == "mse":
        return mse_fn
    else:
        raise NotImplementedError(
            f"Loss function {loss_method} is not implemented."
        )


def cross_entropy_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Cross entropy function.

    Computes the cross entropy loss the predicted versus known response.
    Transforms `predictions` to be row-stochastic, and ensures that `targets`
    contains no negative elements.

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.

    Returns:
        The cross-entropy loss of the prediction.
    """
    one_hot_targets = np.zeros(targets.shape)
    one_hot_targets[targets > 0.0] = 1.0

    return log_loss(
        one_hot_targets, softmax(predictions, axis=1), eps=1e-6, normalize=False
    )


def mse_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Mean squared error function.

    Computes mean squared error loss of the predicted versus known response.
    Treats multivariate outputs as interchangeable in terms of loss penalty.

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.

    Returns:
        The mse loss of the prediction.
    """
    batch_count = predictions.shape[0]
    response_count = predictions.shape[1]
    squared_errors = np.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)


def loo_crossval(
    x0: np.ndarray,
    objective_fn: Callable,
    muygps: MuyGPS,
    optim_params: Dict,
    pairwise_dists: np.ndarray,
    crosswise_dists: np.ndarray,
    batch_nn_targets: np.ndarray,
    batch_targets: np.ndarray,
) -> float:
    """
    Leave-one-out cross validation.

    Returns leave-one-out cross validation performance for a set `MuyGPS`
    object. Predicts on all of the training data at once.

    Args:
        x0:
            Current guess for hyperparameter values of shape `(opt_count,)`.
        objective_fn:
            The function to be optimized. Can be any function that accepts two
            `numpy.ndarray` objects indicating the prediction and target values,
            in that order.
        muygps:
            The MuyGPS object.
        optim_params:
            Dictionary of references of unfixed hyperparameters belonging to the
            MuyGPS object. Keys should be strings, and values should be objects
            of type :class:`MuyGPyS.gp.kernels.Hyperparameter`. Must have an
            `opt_count` number of key-value pairs, matching the shape of `x0`.
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
    for i, key in enumerate(optim_params):
        lb, ub = optim_params[key].get_bounds()
        if x0[i] < lb:
            optim_params[key]._set_val(lb)
        elif x0[i] > ub:
            optim_params[key]._set_val(ub)
        else:
            optim_params[key]._set_val(x0[i])

    K = muygps.kernel(pairwise_dists)
    Kcross = muygps.kernel(crosswise_dists)

    predictions = muygps.regress(
        K, Kcross, batch_nn_targets, apply_sigma_sq=False
    )

    return objective_fn(predictions, batch_targets)
