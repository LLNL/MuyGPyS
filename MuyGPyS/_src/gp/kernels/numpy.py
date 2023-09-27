# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from scipy.special import gamma, kv

import MuyGPyS._src.math.numpy as np


def _rbf_fn(squared_dists: np.ndarray, **kwargs) -> np.ndarray:
    return np.exp(-squared_dists / 2.0)


def _matern_05_fn(dists: np.ndarray, **kwargs) -> np.ndarray:
    return np.exp(-dists)


def _matern_15_fn(dists: np.ndarray, **kwargs) -> np.ndarray:
    K = dists * np.sqrt(3)
    return (1.0 + K) * np.exp(-K)


def _matern_25_fn(dists: np.ndarray, **kwargs) -> np.ndarray:
    K = dists * np.sqrt(5)
    return (1.0 + K + K**2 / 3.0) * np.exp(-K)


def _matern_inf_fn(dists: np.ndarray, **kwargs) -> np.ndarray:
    return np.exp(-(dists**2) / 2.0)


def _matern_gen_fn(
    dists: np.ndarray, smoothness: float, **kwargs
) -> np.ndarray:
    K = dists
    K[K == 0.0] += np.finfo(float).eps
    tmp = np.sqrt(2 * smoothness) * K
    K.fill((2 ** (1.0 - smoothness)) / gamma(smoothness))
    K *= tmp**smoothness
    K *= kv(smoothness, tmp)
    return K
