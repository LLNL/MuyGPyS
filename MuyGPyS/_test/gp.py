# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Union

import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.gp.tensors import _pairwise_differences
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HeteroscedasticNoise, HomoscedasticNoise, NullNoise
from MuyGPyS.gp.sigma_sq import SigmaSq


class BenchmarkGP:
    """
    A basic Gaussian Process.

    Performs GP inference and simulation by way of analytic computations.

    Args:
        kernel:
            The kernel to be used. Only supports Matern.
        eps:
            The noise model.
    """

    def __init__(
        self,
        kernel: Matern,
        eps: Union[
            HeteroscedasticNoise, HomoscedasticNoise, NullNoise
        ] = HomoscedasticNoise(0.0),
    ):
        """
        Initialize.
        """
        self.kernel = kernel
        self.eps = eps
        self.sigma_sq = SigmaSq()

    def fixed(self) -> bool:
        """
        Checks whether all kernel and model parameters are fixed.

        This is a convenience utility to determine whether optimization is
        required.

        Returns:
            Returns `True` if all parameters are fixed, and `False` otherwise.
        """
        for p in self.kernel._hyperparameters:
            if not self.kernel._hyperparameters[p].fixed():
                return False
        if not self.eps.fixed():
            return False
        return True


def benchmark_sample_full(
    gp: BenchmarkGP,
    test: np.ndarray,
    train: np.ndarray,
) -> np.ndarray:
    """
    Sample from a GP prior for a dataset separated into train and test.

    Args:
        gp:
            The gp object
        test:
            The full testing data matrix of shape
            `(test_count, feature_count)`.
        train:
            The full training data matrix of shape
            `(train_count, feature_count)`.

    Returns:
        A sample from the GP prior for a train/test split.
    """
    return benchmark_sample(gp, np.vstack((test, train)))


def benchmark_prepare_cholK(
    gp: BenchmarkGP,
    data: np.ndarray,
) -> np.ndarray:
    """
    Sample from a GP prior for a dataset.

    Args:
        gp:
            The gp object
        train:
            The full training data matrix of shape
            `(train_count, feature_count)`.

    Returns:
        The Cholesky decomposition of a dense covariance matrix.
    """
    pairwise_diffs = _pairwise_differences(data)
    data_count, _ = data.shape
    Kfull = gp.sigma_sq()[0] * (
        gp.kernel(pairwise_diffs) + gp.eps() * np.eye(data_count)
    )
    return np.linalg.cholesky(Kfull)


def benchmark_sample(
    gp: BenchmarkGP,
    data: np.ndarray,
) -> np.ndarray:
    """
    Sample from a GP prior for a dataset.

    Args:
        gp:
            The gp object
        train:
            The full training data matrix of shape
            `(train_count, feature_count)`.
    """
    cholK = benchmark_prepare_cholK(gp, data)
    return benchmark_sample_from_cholK(cholK)


def benchmark_sample_from_cholK(cholK: np.ndarray) -> np.ndarray:
    data_count, _ = cholK.shape
    return (
        cholK @ np.array(np.random.normal(0, 1, size=(data_count,)))
    ).reshape(data_count, 1)


def get_analytic_sigma_sq(K, y):
    assert y.shape[0] == K.shape[0]
    return (1 / y.shape[0]) * y.T @ np.linalg.solve(K, y)
