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


def make_raw_predict_and_loss_fn(
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    scale_fn: Callable,
    batch_nn_targets: mm.ndarray,
    batch_targets: mm.ndarray,
    **loss_kwargs,
) -> Callable:
    """
    Make a predict_and_loss function that depends only on the posterior mean.

    Assembles a new function with signature `(K, Kcross, *args, **kwargs)` that
    computes the posterior mean and uses the passed `loss_fn` to score it
    against the batch targets.

    Args:
        loss_fn:
            A loss function Callable with signature
            `(predictions, responses, **kwargs)`, where `predictions` and
            `targets` are matrices of shape `(batch_count, response_count)`.
        mean_fn:
            A MuyGPS posterior mean function Callable with signature
            `(K, Kcross, batch_nn_targets)`, which are tensors of shape
            `(batch_count, nn_count, nn_count)`, `(batch_count, nn_count)`, and
            `(batch_count, nn_count, response_count)`, respectively.
        var_fn:
            A MuyGPS posterior variance function Callable with signature
            `(K, Kcross)`, which are tensors of shape
            `(batch_count, nn_count, nn_count)` and `(batch_count, nn_count)`,
            respectively. Unused by this function, but still required by the
            signature.
        scale_fn:
            A MuyGPS `scale` optimization function Callable with signature
            `(K, batch_nn_targets)`, which are tensors of shape
            `(batch_count, nn_count, nn_count)` and
            `(batch_count, nn_count, response_count)`, respectively. Unused by
            this function, but still required by the signature.
        batch_nn_targets:
            A tensor of shape `(batch_count, nn_count, response_count)`
            containing the expected response of the nearest neighbors of each
            batch element.
        batch_targets:
            A matrix of shape `(batch_count, response_count)` containing the
            expected response of each batch element.
        loss_kwargs:
            Additionall keyword arguments used by the loss function.

    Returns:
        A Callable with signature `(K, Kcross, *args, **kwargs) -> float` that
        computes the posterior mean and applies the loss function to it and the
        `batch_targets`.
    """

    def predict_and_loss_fn(K, Kcross, *args, **kwargs):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            **kwargs,
        )

        return -loss_fn(predictions, batch_targets, **loss_kwargs)

    return predict_and_loss_fn


def make_var_predict_and_loss_fn(
    loss_fn: Callable,
    mean_fn: Callable,
    var_fn: Callable,
    scale_fn: Callable,
    batch_nn_targets: mm.ndarray,
    batch_targets: mm.ndarray,
    **loss_kwargs,
) -> Callable:
    """
    Make a predict_and_loss function that depends on the posterior mean and
    variance.

    Assembles a new function with signature `(K, Kcross, *args, **kwargs)` that
    computes the posterior mean and variance and uses the passed `loss_fn` to
    score them against the batch targets.

    Args:
        loss_fn:
            A loss function Callable with signature
            `(predictions, responses, **kwargs)`, where `predictions` and
            `targets` are matrices of shape `(batch_count, response_count)`.
        mean_fn:
            A MuyGPS posterior mean function Callable with signature
            `(K, Kcross, batch_nn_targets)`, which are tensors of shape
            `(batch_count, nn_count, nn_count)`, `(batch_count, nn_count)`, and
            `(batch_count, nn_count, response_count)`, respectively.
        var_fn:
            A MuyGPS posterior variance function Callable with signature
            `(K, Kcross)`, which are tensors of shape
            `(batch_count, nn_count, nn_count)` and `(batch_count, nn_count)`,
            respectively.
        scale_fn:
            A MuyGPS `scale` optimization function Callable with signature
            `(K, batch_nn_targets)`, which are tensors of shape
            `(batch_count, nn_count, nn_count)` and
            `(batch_count, nn_count, response_count)`, respectively.
        batch_nn_targets:
            A tensor of shape `(batch_count, nn_count, response_count)`
            containing the expected response of the nearest neighbors of each
            batch element.
        batch_targets:
            A matrix of shape `(batch_count, response_count)` containing the
            expected response of each batch element.
        loss_kwargs:
            Additionall keyword arguments used by the loss function.

    Returns:
        A Callable with signature `(K, Kcross, *args, **kwargs) -> float` that
        computes the posterior mean and applies the loss function to it and the
        `batch_targets`.
    """

    def predict_and_loss_fn(K, Kcross, *args, **kwargs):
        predictions = mean_fn(
            K,
            Kcross,
            batch_nn_targets,
            **kwargs,
        )
        scale = scale_fn(K, batch_nn_targets, **kwargs)

        variances = var_fn(K, Kcross, **kwargs)

        return -loss_fn(
            predictions, batch_targets, variances, scale, **loss_kwargs
        )

    return predict_and_loss_fn


class LossFn:
    """
    Loss functor class.

    MuyGPyS-compatible loss functions are objects of this class. Creating a new
    loss function is as simple as instantiation a new `LossFn` object.

    Args:
        loss_fn:
            A Callable with signature `(predictions, targets, **kwargs)` or
            `(predictions, targets, variances, scale, **kwargs)` tha computes
            a floating-point loss score of a set of predictions given posterior
            means and possibly posterior variances. Individual loss functions
            can implement different `kwargs` as needed.
        make_precit_and_loss_fn:
            A Callable with signature
            `(loss_fn, mean_fn, var_fn, scale_fn, batch_nn_targets, batch_targets, **loss_kwargs)`
            that produces a function that computes posterior predictions and
            scores them using the loss function.
            :func:~MuyGPyS.optimize.loss._make_raw_predict_and_loss_fn` and
            :func:~MuyGPyS.optimize.loss._make_var_predict_and_loss_fn` are two
            candidates.
    """

    def __init__(self, loss_fn: Callable, make_predict_and_loss_fn: Callable):
        self._fn = loss_fn
        self._make_predict_and_loss_fn = make_predict_and_loss_fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def make_predict_and_loss_fn(self, *args, **kwargs):
        return self._make_predict_and_loss_fn(self._fn, *args, **kwargs)


cross_entropy_fn = LossFn(_cross_entropy_fn, make_raw_predict_and_loss_fn)
"""
Cross entropy function.

Computes the cross entropy loss the predicted versus known response. Transforms
`predictions` to be row-stochastic, and ensures that `targets` contains no
negative elements. Only defined for two or more labels.
For a sample with true labels :math:`y_i \\in \\{0, 1\\}` and estimates
:math:`\\bar{\\mu(x_i)} = \\textrm{Pr}(y = 1)`, the function computes

.. math::
    \\ell_\\textrm{cross-entropy}(\\bar{\\mu}, y) =
        \\sum_{i=1}^{b} y_i \\log(\\bar{\\mu}_i)
        - (1 - y_i) \\log(1 - \\bar{\\mu}_i).

The numpy backend uses
`sklearn's implementation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html>`_.

Args:
    predictions:
        The predicted response of shape `(batch_count, response_count)`.
    targets:
        The expected response of shape `(batch_count, response_count)`.
    eps:
        Probabilities are clipped to the range `[eps, 1 - eps]`.

Returns:
    The cross-entropy loss of the prediction.
"""

mse_fn = LossFn(_mse_fn, make_raw_predict_and_loss_fn)
"""
Mean squared error function.

Computes mean squared error loss of the predicted versus known response. Treats
multivariate outputs as interchangeable in terms of loss penalty. The function
computes

.. math::
    \\ell_\\textrm{MSE}(\\bar{\\mu}, y) =
        \\frac{1}{b} \\sum_{i=1}^b (\\bar{\\mu}_i - y)^2.

Args:
    predictions:
        The predicted response of shape `(batch_count, response_count)`.
    targets:
        The expected response of shape `(batch_count, response_count)`.

Returns:
    The mse loss of the prediction.
"""

lool_fn = LossFn(_lool_fn, make_var_predict_and_loss_fn)
"""
Leave-one-out likelihood function.

Computes leave-one-out likelihood (LOOL) loss of the predicted versus known
response. Treats multivariate outputs as interchangeable in terms of loss
penalty. The function computes

.. math::
    \\ell_\\textrm{lool}(\\bar{\\mu}, y, \\bar{\\Sigma}) =
    \\sum_{i=1}^b \\sum_{j=1}^s
    \\left ( \\frac{(\\bar{\\mu}_i - y)^2}{\\bar{\\Sigma}_{ii}} \\right )_j
    +  \\left ( \\log \\bar{\\Sigma}_{ii} \\right )_j

Args:
    predictions:
        The predicted response of shape `(batch_count, response_count)`.
    targets:
        The expected response of shape `(batch_count, response_count)`.
    variances:
        The unscaled variance of the predicted responses of shape
        `(batch_count, response_count)`.
    scale:
        The scale variance scaling parameter of shape `(response_count,)`.

Returns:
    The LOOL loss of the prediction.
"""

lool_fn_unscaled = LossFn(_lool_fn_unscaled, make_var_predict_and_loss_fn)
"""
Leave-one-out likelihood function.

Computes leave-one-out likelihood (LOOL) loss of the predicted versus known
response. Treats multivariate outputs as interchangeable in terms of loss
penalty. Unlike lool_fn, does not require scale as an argument. The function
computes

.. math::
    \\ell_\\textrm{lool}(\\bar{\\mu}, y, \\bar{\\Sigma}) = \\sum_{i=1}^b
    \\frac{(\\bar{\\mu}_i - y)^2}{\\bar{\\Sigma}_{ii}} + \\log \\bar{\\Sigma}_{ii}.

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

pseudo_huber_fn = LossFn(_pseudo_huber_fn, make_raw_predict_and_loss_fn)
"""
Pseudo-Huber loss function.

Computes a smooth approximation to the Huber loss function, which balances
sensitive squared-error loss for relatively small errors and robust-to-outliers
absolute loss for larger errors, so that the loss is not overly sensitive to
outliers. Uses the form from
`wikipedia <https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function>`_.
The function computes

.. math::
    \\ell_\\textrm{Pseudo-Huber}(\\bar{\\mu}, y \\mid \\delta) =
        \\sum_{i=1}^b \\delta^2 \\left ( \\sqrt{
        1 + \\left ( \\frac{y_i - \\bar{\\mu}_i}{\\delta} \\right )^2
        } - 1 \\right ).

Args:
    predictions:
        The predicted response of shape `(batch_count, response_count)`.
    targets:
        The expected response of shape `(batch_count, response_count)`.
    boundary_scale:
        The boundary value for the residual beyond which the loss becomes
        approximately linear. Useful values depend on the scale of the response.

Returns:
    The sum of pseudo-Huber losses of the predictions.
"""

looph_fn = LossFn(_looph_fn, make_var_predict_and_loss_fn)
"""
Variance-regularized pseudo-Huber loss function.

Computes a smooth approximation to the Huber loss function, similar to
:func:`pseudo_huber_fn`, with the addition of both a variance scaling and a
additive logarithmic variance regularization term to avoid exploding the
variance. This is intended to be an outlier-robust replacement to the lool
function. Similar to pseudo-Huber, the `boundary_scale` parameter is sensitive
to the units of the responses, and must be set accordingly. The function
computes

.. math::
    \\ell_\\textrm{lool}(\\bar{\\mu}, y, \\bar{\\Sigma} \\mid \\delta) =
        \\sum_{i=1}^b \\sum_{j=1}^s 2\\delta^2
        \\left ( \\sqrt{
            1 + \\left (
                \\frac{(y_i - \\bar{\\mu}_i)^2}{\\delta^2 \\bar{\\Sigma}_{ii}}
            \\right )_j
        } - 1 \\right ) + \\left ( \\log \\bar{\\Sigma}_{ii} \\right )_j.

Args:
    predictions:
        The predicted response of shape `(batch_count, response_count)`.
    targets:
        The expected response of shape `(batch_count, response_count)`.
    variances:
        The unscaled variance of the predicted responses of shape
        `(batch_count, response_count)`.
    scale:
        The scale variance scaling parameter of shape `(response_count,)`.
    boundary_scale:
        The boundary value for the residual beyond which the loss becomes
        approximately linear. Corresponds to the number of standard deviations
        beyond which to linearize the loss. The default value is 3.0, which is
        sufficiently tight for most realistic problems.

Returns:
    The sum of leave-one-out pseudo-Huber losses of the predictions.
"""
