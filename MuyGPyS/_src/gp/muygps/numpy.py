# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _muygps_posterior_mean(
    Kcov: np.ndarray,
    Kcross: np.ndarray,
    batch_nn_targets: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.squeeze(Kcross @ np.linalg.solve(Kcov, batch_nn_targets))


def _muygps_diagonal_variance(
    Kcov: np.ndarray,
    Kcross: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.squeeze(
        1 - Kcross @ np.linalg.solve(Kcov, Kcross.transpose(0, 2, 1))
    )


def _muygps_fast_posterior_mean(
    Kcross: np.ndarray,
    coeffs_tensor: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.einsum("ij,ijk->ik", Kcross, coeffs_tensor)


def _mmuygps_fast_posterior_mean(
    Kcross: np.ndarray,
    coeffs_tensor: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.einsum("ijk,ijk->ik", Kcross, coeffs_tensor)


def _muygps_fast_posterior_mean_precompute(
    Kcov: np.ndarray,
    train_nn_targets_fast: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.linalg.solve(Kcov, train_nn_targets_fast)
