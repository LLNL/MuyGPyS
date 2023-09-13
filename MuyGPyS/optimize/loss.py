# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Loss Function Handling

MuyGPyS includes predefined loss functions and convenience functions for
indicating them to optimization.
"""

from typing import Callable

import MuyGPyS._src.math as mm
from MuyGPyS._src.optimize.loss import (
    _mse_fn,
    _cross_entropy_fn,
    _lool_fn,
    _lool_fn_unscaled,
    _pseudo_huber_fn,
    _looph_fn,
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
        loss_method,
        lambda: cross_entropy_fn,
        lambda: mse_fn,
        lambda: lool_fn,
        lambda: pseudo_huber_fn,
        lambda: looph_fn,
    )


def cross_entropy_fn(
    predictions: mm.ndarray,
    targets: mm.ndarray,
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
    predictions: mm.ndarray,
    targets: mm.ndarray,
) -> float:
    """
    Mean squared error function.

    Computes mean squared error loss of the predicted versus known response.
    Treats multivariate outputs as interchangeable in terms of loss penalty. The
    function computes

    .. math::
        \\ell_\\textrm{MSE}(f(x), y) = \\frac{1}{b} \\sum_{i=1}^b (f(x_i) - y)^2

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
    predictions: mm.ndarray,
    targets: mm.ndarray,
    variances: mm.ndarray,
    sigma_sq: mm.ndarray,
) -> float:
    """
    Leave-one-out likelihood function.

    Computes leave-one-out likelihood (LOOL) loss of the predicted versus known
    response. Treats multivariate outputs as interchangeable in terms of loss
    penalty. The function computes

    .. math::
        \\ell_\\textrm{lool}(f(x), y \\mid \\sigma^2) =
        \\sum_{i=1}^b \\sum_{j=1}^s
        \\left ( \\frac{(f(x_i) - y)}{\\sigma_j} \\right )^2 + \\log \\sigma_j^2

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


def lool_fn_unscaled(
    predictions: mm.ndarray, targets: mm.ndarray, variances: mm.ndarray
) -> float:
    """
    Leave-one-out likelihood function.

    Computes leave-one-out likelihood (LOOL) loss of the predicted versus known
    response. Treats multivariate outputs as interchangeable in terms of loss
    penalty. Unlike lool_fn, does not require sigma_sq as an argument. The
    function computes

    .. math::
        \\ell_\\textrm{lool}(f(x), y \\mid \\sigma^2) = \\sum_{i=1}^b
        \\left ( \\frac{(f(x_i) - y)}{\\sigma_i} \\right )^2 + \\log \\sigma_i^2

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.
        variances:
            The unscaled variance of the predicted responses of shape
            `(batch_count, response_count)`.

    Returns:
        The LOOL loss of the prediction.
    """
    return _lool_fn_unscaled(predictions, targets, variances)


def pseudo_huber_fn(
    predictions: mm.ndarray, targets: mm.ndarray, boundary_scale: float = 1.5
) -> float:
    """
    Pseudo-Huber loss function.

    Computes a smooth approximation to the Huber loss function, which balances
    sensitive squared-error loss for relatively small errors and
    robust-to-outliers absolute loss for larger errors, so that the loss is not
    overly sensitive to outliers. Used the form from
    [wikipedia](https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function).
    The function computes

    .. math::
        \\ell_\\textrm{Pseudo-Huber}(f(x), y \\mid \\delta) =
            \\sum_{i=1}^b \\delta^2 \\left ( \\sqrt{
            1 + \\left ( \\frac{y_i - f(x_i)}{\\delta} \\right )^2
            } - 1 \\right )

    Args:
        predictions:
            The predicted response of shape `(batch_count, response_count)`.
        targets:
            The expected response of shape `(batch_count, response_count)`.
        boundary_scale:
            The boundary value for the residual beyond which the loss becomes
            approximately linear. Useful values depend on the scale of the
            response.

    Returns:
        The sum of pseudo-Huber losses of the predictions.
    """
    return _pseudo_huber_fn(predictions, targets, boundary_scale=boundary_scale)


def looph_fn(
    predictions: mm.ndarray,
    targets: mm.ndarray,
    variances: mm.ndarray,
    sigma_sq: mm.ndarray,
    boundary_scale: float = 1.5,
) -> float:
    """
    Variance-regularized pseudo-Huber loss function.

    Computes a smooth approximation to the Huber loss function, similar to
    :func:`pseudo_huber_fn`, with the addition of both a variance scaling and a
    additive logarithmic variance regularization term to avoid exploding the
    variance. The function computes

    .. math::
        \\ell_\\textrm{lool}(f(x), y \\mid \\delta, \\sigma^2) =
            \\sum_{i=1}^b \\delta^2 \\left ( \\sqrt{
            1 + \\left ( \\frac{y_i - f(x_i)}{\\delta \\sigma_i^2} \\right )^2
            } - 1 \\right ) + \\log \\sigma_i^2

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
        boundary_scale:
            The boundary value for the residual beyond which the loss becomes
            approximately linear. Useful values depend on the scale of the
            response.

    Returns:
        The sum of pseudo-Huber losses of the predictions.
    """
    return _looph_fn(
        predictions, targets, variances, sigma_sq, boundary_scale=boundary_scale
    )
