# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.gp.noise.numpy import _homoscedastic_perturb, np


def _heteroscedastic_perturb(
    K: np.ndarray, noise_variances: np.ndarray
) -> np.ndarray:
    raise NotImplementedError("heteroscedastic noise does not support mpi!")
