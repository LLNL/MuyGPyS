# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from jax import jit


@jit
def _analytic_sigma_sq_optim_unnormalized(
    K: jnp.ndarray,
    nn_targets: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum(
        jnp.einsum("ijk,ijk->ik", nn_targets, jnp.linalg.solve(K, nn_targets)),
        axis=0,
    )


@jit
def _analytic_sigma_sq_optim(
    K: jnp.ndarray,
    nn_targets: jnp.ndarray,
) -> jnp.ndarray:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_sigma_sq_optim_unnormalized(K, nn_targets) / (
        batch_count * nn_count
    )
