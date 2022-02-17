# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


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
    return np.array(
        [
            1.0
            - Kcross[i, :]
            @ np.linalg.solve(K[i, :, :] + eps * np.eye(nn_count), Kcross[i, :])
            for i in range(batch_count)
        ]
    )
