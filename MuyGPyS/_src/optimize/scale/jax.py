# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit

import MuyGPyS._src.math.jax as jnp


@jit
def _analytic_scale_optim_unnormalized(
    Kin: jnp.ndarray,
    nn_targets: jnp.ndarray,
) -> jnp.ndarray:
    nn_targets = jnp.atleast_3d(nn_targets)
    return jnp.sum(
        jnp.einsum("ijk,ijk->ik", nn_targets, jnp.linalg.solve(Kin, nn_targets))
    )


@jit
def _analytic_scale_optim(
    Kin: jnp.ndarray,
    nn_targets: jnp.ndarray,
    batch_dim_count: int = 1,
) -> jnp.ndarray:
    in_dim_count = (Kin.ndim - batch_dim_count) // 2

    batch_shape = Kin.shape[:batch_dim_count]
    in_shape = Kin.shape[batch_dim_count + in_dim_count :]

    batch_size = jnp.prod(jnp.array(batch_shape), dtype=int)
    in_size = jnp.prod(jnp.array(in_shape), dtype=int)

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    nn_targets_flat = nn_targets.reshape(batch_shape + (in_size, 1))

    return _analytic_scale_optim_unnormalized(Kin_flat, nn_targets_flat) / (
        batch_size * in_size
    )
