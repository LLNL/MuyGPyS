# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Objective and Loss Function Handling

MuyGPyS includes predefined loss functions and convenience functions for 
indicating them to optimization.
"""

import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import Callable

from jax import jit, grad
from scipy.special import softmax
from sklearn.metrics import log_loss


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
    one_hot_targets = jnp.zeros(targets.shape)
    one_hot_targets[targets > 0.0] = 1.0

    return log_loss(
        one_hot_targets, softmax(predictions, axis=1), eps=1e-6, normalize=False
    )


@jit
def mse_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
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


# @partial(jit, static_argnums=(1, 2, 3))
def loo_crossval(
    x0: np.ndarray,
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
    pairwise_dists: jnp.ndarray,
    crosswise_dists: jnp.ndarray,
    batch_nn_targets: jnp.ndarray,
    batch_targets: jnp.ndarray,
) -> float:
    """
    Leave-one-out cross validation.

    Returns leave-one-out cross validation performance for a set `MuyGPS`
    object. Predicts on all of the training data at once.

    Args:
        x0:
            Current guess for hyperparameter values of shape `(opt_count,)`.
        loss_fn:
            The function to be optimized. Can be any function that accepts two
            `numpy.ndarray` objects indicating the prediction and target values,
            in that order.
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
        x0[-1],
    )

    return loss_fn(predictions, batch_targets)


def make_loo_crossval_fn(
    loss_fn: Callable,
    kernel_fn: Callable,
    predict_fn: Callable,
    pairwise_dists: jnp.ndarray,
    crosswise_dists: jnp.ndarray,
    batch_nn_targets: jnp.ndarray,
    batch_targets: jnp.ndarray,
) -> Callable:
    def caller_fn(x0):
        return loo_crossval(
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