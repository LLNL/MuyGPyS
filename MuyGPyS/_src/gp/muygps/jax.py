# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit

import MuyGPyS._src.math.jax as jnp


@jit
def _find_matching_ndim(nn_targets: jnp.ndarray, Kin: jnp.ndarray):
    # assumes that nntargets.ndim <= Kin.ndim
    nshape = jnp.array(nn_targets.shape)
    Kshape = jnp.array(Kin.shape)
    shared_shape = jnp.equal(nshape, Kshape[: nn_targets.ndim])
    return jnp.count_nonzero(shared_shape)


@jit
def _muygps_posterior_mean(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    batch_nn_targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    batch_in_ndim = _find_matching_ndim(batch_nn_targets, Kin)
    in_shape = Kin.shape[batch_in_ndim:]  # (i_1, ..., i_n)
    out_shape = Kcross.shape[batch_in_ndim:]  # (j_1, ..., j_m)
    batch_shape = Kin.shape[: -2 * len(in_shape)]  # (b_1, ..., b_b)
    extra_shape = batch_nn_targets.shape[len(batch_shape) + len(in_shape) :]

    in_size = jnp.prod(in_shape, dtype=int)
    out_size = jnp.prod(out_shape, dtype=int)
    extra_size = jnp.prod(extra_shape, dtype=int)

    batch_nn_targets_flat = batch_nn_targets.reshape(
        batch_shape + (in_size, extra_size)
    )
    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = jnp.linalg.solve(Kin_flat, Kcross_flat)

    ret = F_flat.swapaxes(-2, -1) @ batch_nn_targets_flat
    ret = ret.reshape(batch_shape + out_shape + extra_shape)
    return ret


@jit
def _muygps_diagonal_variance(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    batch_size: int = 1,
    **kwargs,
) -> jnp.ndarray:
    in_dim_count = (Kin.ndim - batch_size) // 2

    batch_shape = Kin.shape[:batch_size]
    in_shape = Kin.shape[batch_size + in_dim_count :]
    out_shape = Kcross.shape[batch_size + in_dim_count :]

    in_size = jnp.prod(in_shape, dtype=int)
    out_size = jnp.prod(out_shape, dtype=int)

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = jnp.linalg.solve(Kin_flat, Kcross_flat)

    Kpost = F_flat.swapaxes(-2, -1) @ Kcross_flat

    return 1 - Kpost.reshape(batch_shape + out_shape + out_shape)


@jit
def _muygps_fast_posterior_mean(
    Kcross: jnp.ndarray,
    coeffs_tensor: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.squeeze(
        jnp.einsum("ij,ijk->ik", Kcross, jnp.atleast_3d(coeffs_tensor))
    )


@jit
def _mmuygps_fast_posterior_mean(
    Kcross: jnp.ndarray,
    coeffs_tensor: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.einsum("ijk,ijk->ik", Kcross, coeffs_tensor)


@jit
def _muygps_fast_posterior_mean_precompute(
    Kin: jnp.ndarray,
    train_nn_targets_fast: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.linalg.solve(Kin, train_nn_targets_fast)
