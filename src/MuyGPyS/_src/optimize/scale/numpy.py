# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _analytic_scale_optim_unnormalized(
    Kin: np.ndarray, nn_targets: np.ndarray, **kwargs
) -> np.ndarray:
    nn_targets = np.atleast_3d(nn_targets)
    return np.sum(
        np.einsum("ijk,ijk->ik", nn_targets, np.linalg.solve(Kin, nn_targets))
    )


def _analytic_scale_optim(
    Kin: np.ndarray, nn_targets: np.ndarray, batch_dim_count: int = 1, **kwargs
) -> np.ndarray:
    in_dim_count = (Kin.ndim - batch_dim_count) // 2

    batch_shape = Kin.shape[:batch_dim_count]
    in_shape = Kin.shape[batch_dim_count + in_dim_count :]

    batch_size = np.prod(batch_shape, dtype=int)
    in_size = np.prod(in_shape, dtype=int)

    Kin_flat = Kin.reshape(batch_shape + (in_size, in_size))
    nn_targets_flat = nn_targets.reshape(batch_shape + (in_size, 1))

    return _analytic_scale_optim_unnormalized(
        Kin_flat, nn_targets_flat, **kwargs
    ) / (batch_size * in_size)
