# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit

import MuyGPyS._src.math.jax as jnp


@jit
def _analytic_scale_optim_unnormalized(
    Kin: jnp.ndarray, nn_targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
    nn_targets = jnp.atleast_3d(nn_targets)
    return jnp.sum(
        jnp.einsum("ijk,ijk->ik", nn_targets, jnp.linalg.solve(Kin, nn_targets))
    )


@jit
def _analytic_scale_optim(
    Kin: jnp.ndarray, nn_targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
    if Kin.ndim == 3:
        assert nn_targets.ndim == 2
        return _analytic_scale_optim_univariate(Kin, nn_targets, **kwargs)
    elif Kin.ndim == 5:
        assert nn_targets.ndim == 3
        return _analytic_scale_optim_multivariate(Kin, nn_targets, **kwargs)
    raise ValueError("should not be possible to get here (jax scale)")


@jit
def _analytic_scale_optim_univariate(
    Kin: jnp.ndarray, nn_targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
    batch_count, nn_count = Kin.shape[:2]
    return _analytic_scale_optim_unnormalized(Kin, nn_targets, **kwargs) / (
        batch_count * nn_count
    )


@jit
def _analytic_scale_optim_multivariate(
    Kin: jnp.ndarray, nn_targets: jnp.ndarray, **kwargs
) -> jnp.ndarray:
    batch_count, nn_count, response_count = Kin.shape[:3]
    Kin_flat = Kin.reshape(
        batch_count, nn_count * response_count, nn_count * response_count
    )
    nn_targets_flat = nn_targets.reshape(batch_count, nn_count * response_count)
    return _analytic_scale_optim_unnormalized(
        Kin_flat, nn_targets_flat, **kwargs
    ) / (batch_count * nn_count)
