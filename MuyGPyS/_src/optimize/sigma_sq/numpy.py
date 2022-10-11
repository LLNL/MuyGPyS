# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


def _analytic_sigma_sq_optim_unnormalized(
    K: np.ndarray,
    nn_targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    _, nn_count, _ = nn_targets.shape
    return np.sum(
        np.einsum(
            "ijk,ijk->ik",
            nn_targets,
            np.linalg.solve(K + eps * np.eye(nn_count), nn_targets),
        ),
        axis=0,
    )


def _analytic_sigma_sq_optim(
    K: np.ndarray,
    nn_targets: np.ndarray,
    eps: float,
) -> np.ndarray:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_sigma_sq_optim_unnormalized(K, nn_targets, eps) / (
        nn_count * batch_count
    )
