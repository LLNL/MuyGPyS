# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from jax import jit


@jit
def _analytic_sigma_sq_optim_unnormalized(
    K: jnp.ndarray,
    nn_targets: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    _, nn_count, _ = nn_targets.shape
    return jnp.sum(
        jnp.einsum(
            "ijk,ijk->ik",
            nn_targets,
            jnp.linalg.solve(K + eps * jnp.eye(nn_count), nn_targets),
        ),
        axis=0,
    )


@jit
def _analytic_sigma_sq_optim(
    K: jnp.ndarray,
    nn_targets: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_sigma_sq_optim_unnormalized(K, nn_targets, eps) / (
        batch_count * nn_count
    )
