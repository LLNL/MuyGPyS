# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from scipy.special import gamma, kv


def _rbf_fn(squared_dists: np.ndarray, length_scale: float) -> np.ndarray:
    return np.exp(-squared_dists / (2 * length_scale**2))


def _matern_05_fn(dists: np.ndarray, length_scale: float) -> np.ndarray:
    dists = dists / length_scale
    return np.exp(-dists)


def _matern_15_fn(dists: np.ndarray, length_scale: float) -> np.ndarray:
    dists = dists / length_scale
    K = dists * np.sqrt(3)
    return (1.0 + K) * np.exp(-K)


def _matern_25_fn(dists: np.ndarray, length_scale: float) -> np.ndarray:
    dists = dists / length_scale
    K = dists * np.sqrt(5)
    return (1.0 + K + K**2 / 3.0) * np.exp(-K)


def _matern_inf_fn(dists: np.ndarray, length_scale: float) -> np.ndarray:
    dists = dists / length_scale
    return np.exp(-(dists**2) / 2.0)


def _matern_gen_fn(
    dists: np.ndarray, nu: float, length_scale: float
) -> np.ndarray:
    dists = dists / length_scale
    K = dists
    K[K == 0.0] += np.finfo(float).eps
    tmp = np.sqrt(2 * nu) * K
    K.fill((2 ** (1.0 - nu)) / gamma(nu))
    K *= tmp**nu
    K *= kv(nu, tmp)
    return K
