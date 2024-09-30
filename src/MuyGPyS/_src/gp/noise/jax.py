# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit

import MuyGPyS._src.math.jax as jnp


@jit
def _homoscedastic_perturb(
    Kin: jnp.ndarray, noise_variance: float
) -> jnp.ndarray:
    if Kin.ndim == 3:
        _, nn_count, _ = Kin.shape
        return Kin + noise_variance * jnp.eye(nn_count)
    elif Kin.ndim == 5:
        b, in_count, nn_count, in_count2, nn_count2 = Kin.shape
        assert nn_count == nn_count2
        assert in_count == in_count2
        all_count = in_count * nn_count
        Kin_flat = Kin.reshape(b, all_count, all_count)
        Kin_flat = Kin_flat + noise_variance * jnp.eye(all_count)
        return Kin_flat.reshape(b, in_count, nn_count, in_count, nn_count)
    else:
        raise ValueError(
            "homoscedastic perturbation is not implemented for tensors of "
            f"shape {Kin.shape}"
        )


@jit
def _shear_perturb33(Kin: jnp.ndarray, noise_variance: float) -> jnp.ndarray:
    convergence_variance = noise_variance * 2
    if Kin.ndim == 5:
        b, in_count, nn_count, in_count2, nn_count2 = Kin.shape
        assert nn_count == nn_count2
        assert in_count == in_count2
        assert in_count == 3
        all_count = in_count * nn_count
        Kin_flat = Kin.reshape(b, all_count, all_count)
        nugget = jnp.diag(
            jnp.hstack(
                (
                    convergence_variance * jnp.ones(nn_count),
                    noise_variance * jnp.ones(2 * nn_count),
                )
            )
        )
        Kin_flat = Kin_flat + nugget
        return Kin_flat.reshape(b, in_count, nn_count, in_count, nn_count)
    else:
        raise ValueError(
            "homoscedastic perturbation is not implemented for tensors of "
            f"shape {Kin.shape}"
        )


@jit
def _heteroscedastic_perturb(
    Kin: jnp.ndarray, noise_variances: jnp.ndarray
) -> jnp.ndarray:
    batch_count, nn_count, _ = Kin.shape
    ret = Kin.copy()
    indices = (
        jnp.repeat(jnp.arange(batch_count), nn_count),
        jnp.tile(jnp.arange(nn_count), batch_count),
        jnp.tile(jnp.arange(nn_count), batch_count),
    )
    ret = ret.at[indices].add(noise_variances.flatten())

    return ret
