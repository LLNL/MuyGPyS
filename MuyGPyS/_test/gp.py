# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np
from MuyGPyS.gp.hyperparameter import FixedScale
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise, NoiseFn


class BenchmarkGP:
    """
    A basic Gaussian Process.

    Performs GP inference and simulation by way of analytic computations.

    Args:
        kernel:
            The kernel to be used. Only supports Matern.
        noise:
            The noise model.
    """

    def __init__(
        self,
        kernel: Matern,
        noise: NoiseFn = HomoscedasticNoise(0.0),
    ):
        """
        Initialize.
        """
        self.kernel = kernel
        self.noise = noise
        self.scale = FixedScale()

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
        if not self.noise.fixed():
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
    if test.ndim == train.ndim == 1:
        thing = np.hstack((test, train))
    else:
        thing = np.vstack((test, train))
    return benchmark_sample(gp, thing)


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
    data_count = data.shape[0]
    pairwise_dists = gp.kernel.deformation.pairwise_tensor(
        data, np.arange(data_count)
    )
    Kfull = gp.scale() * (
        gp.kernel(pairwise_dists) + gp.noise() * np.eye(data_count)
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
    return np.squeeze(benchmark_sample_from_cholK(cholK))


def benchmark_sample_from_cholK(cholK: np.ndarray) -> np.ndarray:
    data_count, _ = cholK.shape
    return (
        cholK @ np.array(np.random.normal(0, 1, size=(data_count,)))
    ).reshape(data_count, 1)


def get_analytic_scale(Kin, y):
    assert y.shape[0] == Kin.shape[0]
    return (1 / y.shape[0]) * np.squeeze(y.T @ np.linalg.solve(Kin, y))
