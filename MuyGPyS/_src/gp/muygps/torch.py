# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _find_matching_ndim(nn_targets: torch.ndarray, Kin: torch.ndarray):
    # assumes that nntargets.ndim <= Kin.ndim
    nshape = torch.array(nn_targets.shape)
    Kshape = torch.array(Kin.shape)
    shared_shape = torch.eq(nshape, Kshape[: nn_targets.ndim])
    return torch.count_nonzero(shared_shape)


def _muygps_posterior_mean(
    Kin: torch.ndarray,
    Kcross: torch.ndarray,
    nn_targets: torch.ndarray,
    **kwargs,
) -> torch.ndarray:
    batch_in_ndim = _find_matching_ndim(nn_targets, Kin)
    in_shape = Kin.shape[batch_in_ndim:]  # (i_1, ..., i_n)
    out_shape = Kcross.shape[batch_in_ndim:]  # (j_1, ..., j_m)
    batch_shape = Kin.shape[: -2 * len(in_shape)]  # (b_1, ..., b_b)
    extra_shape = nn_targets.shape[len(batch_shape) + len(in_shape) :]

    in_size = in_shape.numel()
    out_size = out_shape.numel()
    extra_size = extra_shape.numel()

    nn_targets_flat = nn_targets.reshape(batch_shape + (in_size, extra_size))
    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = torch.linalg.solve(Kin_flat, Kcross_flat)

    ret = F_flat.swapaxes(-2, -1) @ nn_targets_flat

    ret = ret.reshape(batch_shape + out_shape + extra_shape)
    return ret


def _muygps_diagonal_variance(
    Kin: torch.ndarray,
    Kcross: torch.ndarray,
    Kout: torch.ndarray,
    batch_size: int = 1,
    **kwargs,
) -> torch.ndarray:
    in_dim_count = (Kin.ndim - batch_size) // 2

    batch_shape = Kin.shape[:batch_size]
    in_shape = Kin.shape[batch_size + in_dim_count :]
    out_shape = Kcross.shape[batch_size + in_dim_count :]

    in_size = in_shape.numel()
    out_size = out_shape.numel()

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    Kcross_flat = Kcross.reshape(batch_shape + (in_size, out_size))

    F_flat = torch.linalg.solve(Kin_flat, Kcross_flat)

    Kpost = F_flat.swapaxes(-2, -1) @ Kcross_flat

    return Kout - Kpost.reshape(batch_shape + out_shape + out_shape)


def _muygps_fast_posterior_mean(
    Kcross: torch.ndarray,
    coeffs_tensor: torch.ndarray,
) -> torch.ndarray:
    return torch.squeeze(
        torch.einsum("ij,ijk->ik", Kcross, torch.atleast_3d(coeffs_tensor))
    )


def _mmuygps_fast_posterior_mean(
    Kcross: torch.ndarray,
    coeffs_ndarray: torch.ndarray,
) -> torch.ndarray:
    return torch.einsum("ijk,ijk->ik", Kcross, coeffs_ndarray)


def _muygps_fast_posterior_mean_precompute(
    Kin: torch.ndarray,
    train_nn_targets_fast: torch.ndarray,
) -> torch.ndarray:
    return torch.linalg.solve(Kin, train_nn_targets_fast)
