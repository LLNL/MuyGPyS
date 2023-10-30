# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit
from jax.nn import softmax

import MuyGPyS._src.math.jax as jnp


@jit
def _cross_entropy_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    **kwargs,
) -> float:
    one_hot_targets = jnp.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = softmax(predictions, axis=1)

    return _log_loss(one_hot_targets, softmax_predictions, **kwargs)


@jit
def _log_loss(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, eps: float = 1e-15
) -> jnp.ndarray:
    """
    Lifted whole-cloth from [0].

    References
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/metrics/_classification.py#L2309
    """
    # Clipping
    y_pred = jnp.clip(y_pred, eps, 1 - eps)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, jnp.newaxis]
    loss = -(y_true * jnp.log(y_pred)).sum(axis=1)

    return jnp.sum(loss)


@jit
def _mse_fn_unnormalized(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum((predictions - targets) ** 2)


@jit
def _mse_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    return _mse_fn_unnormalized(predictions, targets) / (
        batch_count * response_count
    )


@jit
def _lool_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    return _lool_fn_unscaled(predictions, targets, jnp.outer(variances, scale))


@jit
def _lool_fn_unscaled(
    predictions: jnp.ndarray, targets: jnp.ndarray, variances: jnp.ndarray
) -> float:
    return jnp.sum(
        jnp.divide((predictions - targets) ** 2, variances) + jnp.log(variances)
    )


@jit
def _pseudo_huber_fn(
    predictions: jnp.ndarray, targets: jnp.ndarray, boundary_scale: float = 1.5
) -> float:
    return boundary_scale**2 * jnp.sum(
        jnp.sqrt(1 + jnp.divide(targets - predictions, boundary_scale) ** 2) - 1
    )


@jit
def _looph_fn_unscaled(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
    boundary_scale: float = 3.0,
) -> float:
    boundary_scale_sq = boundary_scale**2
    return jnp.sum(
        2
        * boundary_scale_sq
        * (
            jnp.sqrt(
                1
                + jnp.divide(
                    (targets - predictions) ** 2, boundary_scale_sq * variances
                )
            )
            - 1
        )
        + jnp.log(variances)
    )


@jit
def _looph_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
    scale: jnp.ndarray,
    boundary_scale: float = 3.0,
) -> float:
    return _looph_fn_unscaled(
        predictions,
        targets,
        jnp.outer(variances, scale),
        boundary_scale=boundary_scale,
    )
