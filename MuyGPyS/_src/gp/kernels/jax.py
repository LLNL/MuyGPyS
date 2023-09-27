# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit
from jax.scipy.special import gammaln
from tensorflow_probability.substrates import jax as tfp

import MuyGPyS._src.math.jax as jnp


@jit
def _rbf_fn(squared_dists: jnp.ndarray, **kwargs) -> jnp.ndarray:
    return jnp.exp(-squared_dists / 2)


@jit
def _matern_05_fn(dists: jnp.ndarray, **kwargs) -> jnp.ndarray:
    return jnp.exp(-dists)


@jit
def _matern_15_fn(dists: jnp.ndarray, **kwargs) -> jnp.ndarray:
    K = dists * jnp.sqrt(3)
    return (1.0 + K) * jnp.exp(-K)


@jit
def _matern_25_fn(dists: jnp.ndarray, **kwargs) -> jnp.ndarray:
    K = dists * jnp.sqrt(5)
    return (1.0 + K + K**2 / 3.0) * jnp.exp(-K)


@jit
def _matern_inf_fn(dists: jnp.ndarray, **kwargs) -> jnp.ndarray:
    return jnp.exp(-(dists**2) / 2.0)


@jit
def _matern_gen_fn(
    dists: jnp.ndarray, smoothness: float, **kwargs
) -> jnp.ndarray:
    K = dists
    diag_indices = jnp.arange(K.shape[1])
    if len(K.shape) == 3:
        K = K.at[:, diag_indices, diag_indices].set(1.0)
    tmp = jnp.sqrt(2 * smoothness) * K
    const_val = (2 ** (1.0 - smoothness)) / jnp.exp(gammaln(smoothness))
    if len(K.shape) == 2:
        K = K.at[:, :].set(const_val)
    elif len(K.shape) == 3:
        K = K.at[:, :, :].set(const_val)
    K *= tmp**smoothness
    K *= tfp.math.bessel_kve(smoothness, tmp) / jnp.exp(jnp.abs(tmp))
    if len(K.shape) == 3:
        K = K.at[:, jnp.arange(K.shape[1]), jnp.arange(K.shape[1])].set(1.0)
    return K
