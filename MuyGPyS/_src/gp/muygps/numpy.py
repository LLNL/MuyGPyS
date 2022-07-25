# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
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
) -> np.ndarray:
    batch_count, nn_count = nn_indices.shape

    nn_targets = targets[nn_indices, :]
    return np.sum(
        np.einsum(
            "ijk,ijk->ik",
            nn_targets,
            np.linalg.solve(K + eps * np.eye(nn_count), nn_targets),
        ),
        axis=0,
    ) / (nn_count * batch_count)
