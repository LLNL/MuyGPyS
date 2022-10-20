# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from jax import jit

from jax.nn import softmax


@jit
def _cross_entropy_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    ll_eps: float = 1e-15,
) -> float:
    one_hot_targets = jnp.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = softmax(predictions, axis=1)

    return _log_loss(one_hot_targets, softmax_predictions, eps=ll_eps)


@jit
def _log_loss(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, eps: float = 1e-15
) -> float:
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
) -> float:
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
    sigma_sq: jnp.ndarray,
) -> float:
    scaled_variances = jnp.outer(variances, sigma_sq)
    return jnp.sum(
        jnp.divide((predictions - targets) ** 2, scaled_variances)
        + jnp.log(scaled_variances)
    )
