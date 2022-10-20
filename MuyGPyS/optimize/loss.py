# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Loss Function Handling

MuyGPyS includes predefined loss functions and convenience functions for
indicating them to optimization.
"""

import numpy as np

from typing import Callable

from MuyGPyS import config

from MuyGPyS._src.optimize.loss import (
    _mse_fn,
    _cross_entropy_fn,
    _lool_fn,
)
from MuyGPyS.optimize.utils import _switch_on_loss_method


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
    return _switch_on_loss_method(
        loss_method, lambda: cross_entropy_fn, lambda: mse_fn, lambda: lool_fn
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

    @NOTE[bwp] I don't remember why we hard-coded eps=1e-6. Might need to
    revisit.

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.

    Returns:
        The cross-entropy loss of the prediction.
    """
    return _cross_entropy_fn(predictions, targets, ll_eps=1e-6)


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
    return _mse_fn(predictions, targets)


def lool_fn(
    predictions: np.ndarray,
    targets: np.ndarray,
    variances: np.ndarray,
    sigma_sq: np.ndarray,
) -> float:
    """
    Leave-one-out likelihood function.

    Computes leave-one-out likelihood (LOOL) loss of the predicted versus known
    response. Treats multivariate outputs as interchangeable in terms of loss
    penalty.

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.
        variances:
            The unscaled variance of the predicted responses of shape
            `(batch_count, response_count)`.
        sigma_sq:
            The sigma_sq variance scaling parameter of shape
            `(response_count,)`.

    Returns:
        The LOOL loss of the prediction.
    """
    return _lool_fn(predictions, targets, variances, sigma_sq)
