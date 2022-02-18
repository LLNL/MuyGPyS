# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from typing import Generator


def _muygps_compute_solve(
    K: np.ndarray,
    Kcross: np.ndarray,
    batch_nn_targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    batch_count, nn_count, response_count = batch_nn_targets.shape
    responses = Kcross.reshape(batch_count, 1, nn_count) @ np.linalg.solve(
        K + eps * np.eye(nn_count), batch_nn_targets
    )
    return responses.reshape(batch_count, response_count)


def _muygps_compute_diagonal_variance(
    K: np.ndarray,
    Kcross: np.ndarray,
    eps: float,
) -> np.ndarray:
    batch_count, nn_count = Kcross.shape
    return 1 - np.sum(
        Kcross
        * np.linalg.solve(
            K + eps * np.eye(nn_count), Kcross.reshape(batch_count, nn_count, 1)
        ).reshape(batch_count, nn_count),
        axis=1,
    )


def _muygps_sigma_sq_optim(
    K: np.ndarray,
    nn_indices: np.ndarray,
    targets: np.ndarray,
    eps: float,
):
    batch_count, nn_count = nn_indices.shape
    _, response_count = targets.shape

    sigma_sq = np.zeros((response_count,))
    for i in range(response_count):
        sigma_sq[i] = sum(_get_sigma_sq(K, targets[:, i], nn_indices, eps)) / (
            nn_count * batch_count
        )
    return sigma_sq


def _get_sigma_sq(
    K: np.ndarray,
    target_col: np.ndarray,
    nn_indices: np.ndarray,
    eps: float,
) -> Generator[float, None, None]:
    batch_count, nn_count = nn_indices.shape
    for j in range(batch_count):
        Y_0 = target_col[nn_indices[j, :]]
        yield Y_0 @ np.linalg.solve(K[j, :, :] + eps * np.eye(nn_count), Y_0)
