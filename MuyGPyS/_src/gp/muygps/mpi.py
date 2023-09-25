# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.gp.muygps.numpy import (
    _muygps_posterior_mean,
    _muygps_diagonal_variance,
)


def _muygps_fast_posterior_mean(
    Kcross: np.ndarray,
    coeffs_tensor: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError(
        'Function "muygps_fast_posterior_mean" does not support mpi!'
    )


def _mmuygps_fast_posterior_mean(
    Kcross: np.ndarray,
    coeffs_tensor: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError(
        'Function "mmuygps_fast_posterior_mean" does not support mpi!'
    )


def _muygps_fast_posterior_mean_precompute(
    K: np.ndarray,
    train_nn_targets_fast: np.ndarray,
    **kwargs,
) -> np.ndarray:
    raise NotImplementedError(
        'Function "muygps_fast_posterior_mean_precompute" does not support mpi!'
    )
