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


class LossFn:
    """
    Loss function base class.

    MuyGPyS-compatible loss functions should inherit from this class, and
    possess the following static methods:

    * A `__new__` method taking predictions and target vectors, optionally
        a variance vector and sigma_sq vector, and optionally additional keyword
        arguments that produces a float.
    * A `make_predict_and_loss_fn method`, for example
        :func:`_make_raw_predict_and_loss_fn` or
        :func:`_make_raw_predict_and_loss_fn`, or a new function with the same
        signature.

    These LossFn classes act like global functions with a member function, in
    that it is impossible to instantiate objects of their type and
    "initializing" the class produces a float instead of an object of that type.
    This depends on an abuse of the `__new__` semantics, and might stop working
    in future versions of Python.
    """

    @staticmethod
    def __new__(*args, **kwargs):
        raise ValueError("Base loss functor cannot be called!")

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        raise ValueError(
            "Base loss functor cannot produce predict_and_loss_fn!"
        )


def _make_raw_predict_and_loss_fn(
    loss_fn: LossFn,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: mm.ndarray,
    batch_targets: mm.ndarray,
    **loss_kwargs,
) -> Callable:
    def predict_and_loss_fn(K, Kcross, *args, **kwargs):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            **kwargs,
        )

        return -loss_fn(predictions, batch_targets, **loss_kwargs)

    return predict_and_loss_fn


def _make_var_predict_and_loss_fn(
    loss_fn: LossFn,
    mean_fn: Callable,
    var_fn: Callable,
    sigma_sq_fn: Callable,
    batch_nn_targets: mm.ndarray,
    batch_targets: mm.ndarray,
    **loss_kwargs,
) -> Callable:
    def predict_and_loss_fn(K, Kcross, *args, **kwargs):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            **kwargs,
        )
        sigma_sq = sigma_sq_fn(K, batch_nn_targets, **kwargs)

        variances = var_fn(K, Kcross, **kwargs)

        return -loss_fn(
            predictions, batch_targets, variances, sigma_sq, **loss_kwargs
        )

    return predict_and_loss_fn


class cross_entropy_fn(LossFn):
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

    @staticmethod
    def __new__(  # type: ignore
        cls,
        predictions: mm.ndarray,
        targets: mm.ndarray,
    ) -> float:
        return _cross_entropy_fn(predictions, targets, ll_eps=1e-6)

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        return _make_raw_predict_and_loss_fn(cross_entropy_fn, *args, **kwargs)


class mse_fn(LossFn):
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

    @staticmethod
    def __new__(  # type: ignore
        cls,
        predictions: mm.ndarray,
        targets: mm.ndarray,
    ) -> float:
        return _mse_fn(predictions, targets)

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        return _make_raw_predict_and_loss_fn(mse_fn, *args, **kwargs)


class lool_fn(LossFn):
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

    @staticmethod
    def __new__(  # type: ignore
        cls,
        predictions: mm.ndarray,
        targets: mm.ndarray,
        variances: mm.ndarray,
        sigma_sq: mm.ndarray,
    ) -> float:
        return _lool_fn(predictions, targets, variances, sigma_sq)

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        return _make_var_predict_and_loss_fn(lool_fn, *args, **kwargs)


class lool_fn_unscaled(LossFn):
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

    @staticmethod
    def __new__(  # type: ignore
        cls, predictions: mm.ndarray, targets: mm.ndarray, variances: mm.ndarray
    ) -> float:
        return _lool_fn_unscaled(predictions, targets, variances)

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        return _make_var_predict_and_loss_fn(lool_fn_unscaled, *args, **kwargs)


class pseudo_huber_fn(LossFn):
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

    @staticmethod
    def __new__(  # type: ignore
        cls,
        predictions: mm.ndarray,
        targets: mm.ndarray,
        boundary_scale: float = 1.5,
    ) -> float:
        return _pseudo_huber_fn(
            predictions, targets, boundary_scale=boundary_scale
        )

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        return _make_raw_predict_and_loss_fn(pseudo_huber_fn, *args, **kwargs)


class looph_fn(LossFn):
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

    @staticmethod
    def __new__(  # type: ignore
        cls,
        predictions: mm.ndarray,
        targets: mm.ndarray,
        variances: mm.ndarray,
        sigma_sq: mm.ndarray,
        boundary_scale: float = 1.5,
    ) -> float:
        return _looph_fn(
            predictions,
            targets,
            variances,
            sigma_sq,
            boundary_scale=boundary_scale,
        )

    @staticmethod
    def make_predict_and_loss_fn(*args, **kwargs):
        return _make_var_predict_and_loss_fn(looph_fn, *args, **kwargs)
