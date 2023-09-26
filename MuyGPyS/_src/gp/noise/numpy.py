# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _homoscedastic_perturb(K: np.ndarray, noise_variance: float) -> np.ndarray:
    _, nn_count, _ = K.shape
    return K + noise_variance * np.eye(nn_count)


def _heteroscedastic_perturb(
    K: np.ndarray, noise_variances: np.ndarray
) -> np.ndarray:
    ret = K.copy()
    batch_count, nn_count, _ = K.shape
    indices = (
        np.repeat(range(batch_count), nn_count),
        np.tile(np.arange(nn_count), batch_count),
        np.tile(np.arange(nn_count), batch_count),
    )
    ret[indices] += noise_variances.flatten()
    return ret
