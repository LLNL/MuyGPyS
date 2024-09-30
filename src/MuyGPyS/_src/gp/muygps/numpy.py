# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _find_matching_ndim(nn_targets: np.ndarray, Kin: np.ndarray):
    # assumes that nntargets.ndim <= Kin.ndim
    nshape = np.array(nn_targets.shape)
    Kshape = np.array(Kin.shape)
    shared_shape = np.equal(nshape, Kshape[: nn_targets.ndim])
    return np.count_nonzero(shared_shape)


def _muygps_posterior_mean(
    Kin: np.ndarray,
    Kcross: np.ndarray,
    nn_targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    batch_in_ndim = _find_matching_ndim(nn_targets, Kin)
    in_shape = Kin.shape[batch_in_ndim:]  # (i_1, ..., i_n)
    out_shape = Kcross.shape[batch_in_ndim:]  # (j_1, ..., j_m)
    batch_shape = Kin.shape[: -2 * len(in_shape)]  # (b_1, ..., b_b)
    extra_shape = nn_targets.shape[len(batch_shape) + len(in_shape) :]

    in_size = np.prod(in_shape, dtype=int)
    out_size = np.prod(out_shape, dtype=int)
    extra_size = np.prod(extra_shape, dtype=int)

    nn_targets_flat = nn_targets.reshape(batch_shape + (in_size, extra_size))
    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = np.linalg.solve(Kin_flat, Kcross_flat)

    ret = F_flat.swapaxes(-2, -1) @ nn_targets_flat
    ret = ret.reshape(batch_shape + out_shape + extra_shape)
    return ret


def _muygps_diagonal_variance(
    Kin: np.ndarray,
    Kcross: np.ndarray,
    Kout: np.ndarray,
    batch_size: int = 1,
    **kwargs,
) -> np.ndarray:
    in_dim_count = (Kin.ndim - batch_size) // 2

    batch_shape = Kin.shape[:batch_size]
    in_shape = Kin.shape[batch_size + in_dim_count :]
    out_shape = Kcross.shape[batch_size + in_dim_count :]

    in_size = np.prod(in_shape, dtype=int)
    out_size = np.prod(out_shape, dtype=int)

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = np.linalg.solve(Kin_flat, Kcross_flat)

    Kpost = F_flat.swapaxes(-2, -1) @ Kcross_flat

    return Kout - Kpost.reshape(batch_shape + out_shape + out_shape)


def _muygps_fast_posterior_mean(
    Kcross: np.ndarray,
    coeffs_tensor: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.squeeze(
        np.einsum("ij,ijk->ik", Kcross, np.atleast_3d(coeffs_tensor))
    )


def _mmuygps_fast_posterior_mean(
    Kcross: np.ndarray,
    coeffs_tensor: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.einsum("ijk,ijk->ik", Kcross, coeffs_tensor)


def _muygps_fast_posterior_mean_precompute(
    Kin: np.ndarray,
    train_nn_targets_fast: np.ndarray,
    **kwargs,
) -> np.ndarray:
    if train_nn_targets_fast.ndim == 2:
        train_nn_targets_fast = train_nn_targets_fast[:, :, None]
    return np.squeeze(np.linalg.solve(Kin, train_nn_targets_fast))
