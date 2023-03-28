# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit

import MuyGPyS._src.math.jax as jnp


@jit
def _homoscedastic_perturb(K: jnp.ndarray, eps: float) -> jnp.ndarray:
    _, nn_count, _ = K.shape
    return K + eps * jnp.eye(nn_count)


@jit
def _heteroscedastic_perturb(K: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    batch_count, nn_count, _ = K.shape
    ret = K.copy()
    indices = (
        jnp.repeat(jnp.arange(batch_count), nn_count),
        jnp.tile(jnp.arange(nn_count), batch_count),
        jnp.tile(jnp.arange(nn_count), batch_count),
    )
    eps_tens = jnp.zeros((batch_count, nn_count, nn_count))

    eps_tens.at[indices].set(eps.flatten())
    ret += eps_tens
    return ret
