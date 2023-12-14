# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _analytic_scale_optim_unnormalized(
    Kin: np.ndarray,
    nn_targets: np.ndarray,
) -> np.ndarray:
    return np.sum(
        np.einsum("ijk,ijk->ik", nn_targets, np.linalg.solve(Kin, nn_targets)),
        axis=0,
    )


def _analytic_scale_optim(
    Kin: np.ndarray,
    nn_targets: np.ndarray,
) -> np.ndarray:
    batch_count, nn_count, _ = nn_targets.shape
    return _analytic_scale_optim_unnormalized(Kin, nn_targets) / (
        nn_count * batch_count
    )
