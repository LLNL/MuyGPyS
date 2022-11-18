# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
import numpy as np
from MuyGPyS._src.gp.muygps.numpy import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
)


def _muygps_fast_regress_solve(
    Kcross: np.ndarray,
    coeffs_mat: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError(
        f'Function "muygps_fast_regress_solve" does not support mpi!'
    )


def _muygps_fast_regress_precompute(
    K: np.ndarray,
    eps: float,
    train_nn_targets_fast: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError(
        f'Function "muygps_fast_regress_precompute" does not support mpi!'
    )
