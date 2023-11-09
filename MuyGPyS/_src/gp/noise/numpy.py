# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _homoscedastic_perturb(Kcov: np.ndarray, noise_variance: float) -> np.ndarray:
    _, nn_count, _ = Kcov.shape
    return Kcov + noise_variance * np.eye(nn_count)


def _heteroscedastic_perturb(
    Kcov: np.ndarray, noise_variances: np.ndarray
) -> np.ndarray:
    ret = Kcov.copy()
    batch_count, nn_count, _ = Kcov.shape
    indices = (
        np.repeat(range(batch_count), nn_count),
        np.tile(np.arange(nn_count), batch_count),
        np.tile(np.arange(nn_count), batch_count),
    )
    ret[indices] += noise_variances.flatten()
    return ret
