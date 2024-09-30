# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
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


# NOTE this cannot work while batch_in_ndim is variable. The shapes all must be
# fixed and insensitive to inputs


@jit
def _muygps_posterior_mean(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    nn_targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    if Kin.ndim == 3:
        if nn_targets.ndim == 2:
            return _muygps_posterior_mean_univariate(
                Kin, Kcross, nn_targets, **kwargs
            )
        elif nn_targets.ndim == 3:
            return _muygps_posterior_mean_diagonal_multivariate(
                Kin, Kcross, nn_targets, **kwargs
            )
    else:
        return _muygps_posterior_mean_multivariate(
            Kin, Kcross, nn_targets, **kwargs
        )
    raise ValueError("should not be possible to get here (jax mean)")


@jit
def _muygps_posterior_mean_univariate(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    nn_targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    batch_count = Kin.shape[0]
    F_flat = jnp.linalg.solve(Kin, Kcross[:, :, None])
    ret = F_flat.swapaxes(-2, -1) @ nn_targets[:, :, None]
    ret = ret.reshape(batch_count)
    return ret


@jit
def _muygps_posterior_mean_diagonal_multivariate(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    nn_targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    batch_count = Kin.shape[0]
    response_count = nn_targets.shape[-1]
    F_flat = jnp.linalg.solve(Kin, Kcross[:, :, None])
    ret = F_flat.swapaxes(-2, -1) @ nn_targets
    ret = ret.reshape(batch_count, response_count)
    return ret


@jit
def _muygps_posterior_mean_multivariate(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    nn_targets: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    batch_count, nn_count, response_count = Kin.shape[:3]
    out_count = Kcross.shape[-1]
    Kin_flat = Kin.reshape(
        batch_count, nn_count * response_count, nn_count * response_count
    )
    Kcross_flat = Kcross.reshape(
        batch_count, nn_count * response_count, out_count
    )
    nn_targets_flat = nn_targets.reshape(batch_count, nn_count * response_count)
    F_flat = jnp.linalg.solve(Kin_flat, Kcross_flat)
    ret = F_flat.swapaxes(-2, -1) @ nn_targets_flat
    ret = ret.reshape(batch_count, out_count)
    return ret


@jit
def _muygps_diagonal_variance(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    Kout: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    if Kin.ndim == 3:
        return _muygps_posterior_variance_univariate(
            Kin, Kcross, Kout, **kwargs
        )
    elif Kin.ndim == 5:
        return _muygps_posterior_variance_multivariate(
            Kin, Kcross, Kout, **kwargs
        )
    raise ValueError("should not be possible to get here (jax variance)")


@jit
def _muygps_posterior_variance_univariate(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    Kout: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    batch_count = Kin.shape[0]
    Kcross_flat = Kcross[:, :, None]
    F_flat = jnp.linalg.solve(Kin, Kcross_flat)
    Kpost = F_flat.swapaxes(-2, -1) @ Kcross_flat

    return Kout - Kpost.reshape(batch_count)


@jit
def _muygps_posterior_variance_multivariate(
    Kin: jnp.ndarray,
    Kcross: jnp.ndarray,
    Kout: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    batch_count, nn_count, response_count = Kin.shape[:3]
    out_count = Kcross.shape[-1]
    Kin_flat = Kin.reshape(
        batch_count, nn_count * response_count, nn_count * response_count
    )
    Kcross_flat = Kcross.reshape(
        batch_count, nn_count * response_count, out_count
    )
    F_flat = jnp.linalg.solve(Kin_flat, Kcross_flat)
    Kpost = F_flat.swapaxes(-2, -1) @ Kcross_flat

    return Kout - Kpost.reshape(batch_count, out_count, out_count)


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
