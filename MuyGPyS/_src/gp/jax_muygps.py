# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from jax import jit


@jit
def _muygps_compute_solve(
    K: jnp.ndarray,
    Kcross: jnp.ndarray,
    batch_nn_targets: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    batch_count, nn_count, response_count = batch_nn_targets.shape
    responses = Kcross.reshape(batch_count, 1, nn_count) @ jnp.linalg.solve(
        K + eps * jnp.eye(nn_count), batch_nn_targets
    )
    return responses.reshape(batch_count, response_count)


@jit
def _muygps_compute_diagonal_variance(
    K: jnp.ndarray,
    Kcross: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    batch_count, nn_count = Kcross.shape
    return 1 - jnp.sum(
        Kcross
        * jnp.linalg.solve(
            K + eps * jnp.eye(nn_count),
            Kcross.reshape(batch_count, nn_count, 1),
        ).reshape(batch_count, nn_count),
        axis=1,
    )


@jit
def _muygps_sigma_sq_optim(
    K: jnp.ndarray,
    nn_indices: jnp.ndarray,
    targets: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    batch_count, nn_count = nn_indices.shape

    nn_targets = targets[nn_indices, :]
    return jnp.sum(
        jnp.einsum(
            "ijk,ijk->ik",
            nn_targets,
            jnp.linalg.solve(K + eps * jnp.eye(nn_count), nn_targets),
        ),
        axis=0,
    ) / (nn_count * batch_count)
