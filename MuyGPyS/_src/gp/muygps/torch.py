# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _muygps_posterior_mean(
    K: torch.ndarray,
    Kcross: torch.ndarray,
    batch_nn_targets: torch.ndarray,
    **kwargs,
) -> torch.ndarray:
    return torch.squeeze(Kcross @ torch.linalg.solve(K, batch_nn_targets))


def _muygps_diagonal_variance(
    K: torch.ndarray,
    Kcross: torch.ndarray,
    **kwargs,
) -> torch.ndarray:
    return torch.squeeze(
        1 - Kcross @ torch.linalg.solve(K, Kcross.transpose(1, -1))
    )


def _muygps_fast_posterior_mean(
    Kcross: torch.ndarray,
    coeffs_ndarray: torch.ndarray,
) -> torch.ndarray:
    return torch.einsum("ij,ijk->ik", Kcross, coeffs_ndarray)


def _mmuygps_fast_posterior_mean(
    Kcross: torch.ndarray,
    coeffs_ndarray: torch.ndarray,
) -> torch.ndarray:
    return torch.einsum("ijk,ijk->ik", Kcross, coeffs_ndarray)


def _muygps_fast_posterior_mean_precompute(
    K: torch.ndarray,
    train_nn_targets_fast: torch.ndarray,
) -> torch.ndarray:
    return torch.linalg.solve(K, train_nn_targets_fast)
