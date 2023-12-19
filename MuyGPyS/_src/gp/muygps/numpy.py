# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _old_muygps_posterior_mean(
    Kin: np.ndarray,
    Kcross: np.ndarray,
    batch_nn_targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.squeeze(Kcross @ np.linalg.solve(Kin, batch_nn_targets))


def _find_matching_ndim(nn_targets: np.ndarray, Kin: np.ndarray):
    # assumes that nntargets.ndim <= Kin.ndim
    nshape = np.array(nn_targets.shape)
    Kshape = np.array(Kin.shape)
    shared_shape = np.equal(nshape, Kshape[: nn_targets.ndim])
    return np.count_nonzero(shared_shape)


def _muygps_posterior_mean(
    Kin: np.ndarray,
    Kcross: np.ndarray,
    batch_nn_targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    #### TODO: Optionally ignore the last dimension of batch_nn_targets

    # print(f"Kin.shape = {Kin.shape}")
    # print(f"Kcross.shape = {Kcross.shape}")
    # print(f"batch_nn_targets.shape = {batch_nn_targets.shape}")

    batch_in_ndim = _find_matching_ndim(batch_nn_targets, Kin)
    in_shape = Kin.shape[batch_in_ndim:]  # (i_1, ..., i_n)
    out_shape = Kcross.shape[batch_in_ndim:]  # (j_1, ..., j_m)
    batch_shape = Kin.shape[: -2 * len(in_shape)]  # (b_1, ..., b_b)
    extra_shape = batch_nn_targets.shape[len(batch_shape) + len(in_shape) :]

    # print(f"batch_shape = {batch_shape}")
    # print(f"in_shape = {in_shape}")
    # print(f"out_shape = {out_shape}")
    # print(f"extra_shape = {extra_shape}")

    in_size = np.prod(in_shape, dtype=int)
    out_size = np.prod(out_shape, dtype=int)
    extra_size = np.prod(extra_shape, dtype=int)

    # print(f"in_size = {in_size}")
    # print(f"out_size = {out_size}")

    batch_nn_targets_flat = batch_nn_targets.reshape(
        batch_shape + (in_size, extra_size)
    )
    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    # print(f"Kin_flat.shape = {Kin_flat.shape}")
    # print(f"Kcross_flat.shape = {Kcross_flat.shape}")

    F_flat = np.linalg.solve(Kin_flat, Kcross_flat)

    # print(f"F_flat.shape = {F_flat.shape}")
    # print(f"F_flat.swapaxes(-2, -1).shape = {F_flat.swapaxes(-2, -1).shape}")
    # print(f"batch_nn_targets_flat.shape = {batch_nn_targets_flat.shape}")

    ret = F_flat.swapaxes(-2, -1) @ batch_nn_targets_flat

    # print(f"ret.shape = {ret.shape}")

    ret = ret.reshape(batch_shape + out_shape + extra_shape)

    return ret


def _old_muygps_diagonal_variance(
    Kin: np.ndarray,
    Kcross: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.squeeze(
        1 - Kcross @ np.linalg.solve(Kin, Kcross.transpose(0, 2, 1))
    )


def _muygps_diagonal_variance(
    Kin: np.ndarray,
    Kcross: np.ndarray,
    batch_dim_count: int = 1,
    **kwargs,
) -> np.ndarray:
    in_dim_count = (Kin.ndim - batch_dim_count) // 2

    batch_shape = Kin.shape[:batch_dim_count]
    in_shape = Kin.shape[batch_dim_count + in_dim_count :]
    out_shape = Kcross.shape[batch_dim_count + in_dim_count :]

    in_size = np.prod(in_shape, dtype=int)
    out_size = np.prod(out_shape, dtype=int)

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = np.linalg.solve(Kin_flat, Kcross_flat)

    Kpost = F_flat.swapaxes(-2, -1) @ Kcross_flat

    return 1 - Kpost.reshape(batch_shape + out_shape + out_shape)


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
    return np.linalg.solve(Kin, train_nn_targets_fast)
