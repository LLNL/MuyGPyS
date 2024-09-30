# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
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
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    eps: float = 1e-15,
    **kwargs,
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
    predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
    return jnp.sum((predictions - targets) ** 2)


@jit
def _mse_fn(predictions: jnp.ndarray, targets: jnp.ndarray, **kwargs) -> float:
    return _mse_fn_unnormalized(predictions, targets) / (
        jnp.prod(jnp.array(predictions.shape))
    )


@jit
def _lool_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
    scale: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    return _lool_fn_unscaled(predictions, targets, scale * variances, **kwargs)


@jit
def _lool_fn_unscaled(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
    **kwargs,
) -> float:
    if variances.ndim == 3:
        residual = jnp.atleast_3d(predictions - targets)
        quad_form = jnp.squeeze(
            residual.swapaxes(-2, -1) @ jnp.linalg.solve(variances, residual)
        )
        logdet = jnp.linalg.slogdet(variances)
        return jnp.sum(quad_form + logdet)
    else:
        return jnp.sum(
            jnp.divide((predictions - targets) ** 2, variances)
            + jnp.log(variances)
        )


@jit
def _pseudo_huber_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    boundary_scale: float = 1.5,
    **kwargs,
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
    **kwargs,
) -> float:
    boundary_scale_sq = boundary_scale**2
    if variances.ndim == 1:
        return jnp.sum(
            2
            * boundary_scale_sq
            * (
                jnp.sqrt(
                    1
                    + jnp.divide(
                        (targets - predictions) ** 2,
                        boundary_scale_sq * variances,
                    )
                )
                - 1
            )
            + jnp.log(variances)
        )
    else:
        raise ValueError("looph does not yet support multivariate inference")


@jit
def _looph_fn(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    variances: jnp.ndarray,
    scale: float,
    boundary_scale: float = 3.0,
    **kwargs,
) -> float:
    return _looph_fn_unscaled(
        predictions, targets, scale * variances, boundary_scale=boundary_scale
    )
