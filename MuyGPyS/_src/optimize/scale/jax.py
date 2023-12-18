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
    return jnp.sum(
        jnp.einsum("ijk,ijk->ik", nn_targets, jnp.linalg.solve(Kin, nn_targets))
    )


@jit
def _analytic_scale_optim(
    Kin: jnp.ndarray,
    nn_targets: jnp.ndarray,
) -> jnp.ndarray:
    batch_count, nn_count = nn_targets.shape[:2]
    return _analytic_scale_optim_unnormalized(Kin, nn_targets) / (
        batch_count * nn_count
    )
