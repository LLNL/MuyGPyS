# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
import MuyGPyS._src.math.numpy as np

from absl.testing import parameterized

from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import DifferenceIsotropy, dot
from MuyGPyS.gp.hyperparameter import Parameter
from MuyGPyS.gp.kernels.experimental import SOAPKernel
from MuyGPyS.gp.noise import HomoscedasticNoise


def pad_atom_count(train_features, test_features):

    a_train = train_features.shape[3]
    a_test = test_features.shape[3]
    a_max = max(a_train, a_test)

    train_pad = a_max - a_train
    test_pad = a_max - a_test

    train_pad_width = ((0, 0), (0, 0), (0, 0), (0, train_pad), (0, 0))

    test_pad_width = ((0, 0), (0, 0), (0, 0), (0, test_pad), (0, 0))

    train_features_padded = np.pad(
        array=train_features,
        pad_width=train_pad_width,
        mode="constant",
        constant_values=0,
    )

    test_features_padded = np.pad(
        array=test_features,
        pad_width=test_pad_width,
        mode="constant",
        constant_values=0,
    )

    return train_features_padded, test_features_padded


def explicit_crosswise(locations, points):
    crosswise_similarity = np.zeros(shape=(2, 3, 2, 3, 2, 2, 10, 10))

    for i in range(2):
        for c1 in range(3):
            for k in range(2):
                for c2 in range(3):
                    for q1 in range(2):
                        for q2 in range(2):
                            for a1 in range(10):
                                for a2 in range(10):
                                    crosswise_similarity[
                                        i, c1, k, c2, q1, q2, a1, a2
                                    ] = (
                                        locations[i, c1, q1, a1]
                                        * points[i, c2, k, q2, a2]
                                    ).sum()
    return crosswise_similarity.reshape(2, 3, 2, 3, 4, 10, 10)


def explicit_pairwise(points):
    pairwise_similarity = np.zeros(shape=(2, 3, 2, 3, 2, 2, 2, 10, 10))

    for i in range(2):
        for c1 in range(3):
            for k1 in range(2):
                for c2 in range(3):
                    for k2 in range(2):
                        for q1 in range(2):
                            for q2 in range(2):
                                for a1 in range(10):
                                    for a2 in range(10):
                                        pairwise_similarity[
                                            i, c1, k1, c2, k2, q1, q2, a1, a2
                                        ] = (
                                            points[i, c1, k1, q1, a1]
                                            * points[i, c2, k2, q2, a2]
                                        ).sum()

    return pairwise_similarity.reshape(2, 3, 2, 3, 2, 4, 10, 10)


class BenchmarkTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTestCase, cls).setUpClass()
        cls.nn_count = 2
        cls.zeta = 2.0
        cls.noise_prior = 1e-15
        cls.nn_envs = [[3, 6], [3, 6]]

        # features shape (env_count, 3, 2, atom_count, desc_count)
        cls.raw_train_features = np.random.rand(10, 3, 2, 10, 116)
        cls.raw_test_features = np.random.rand(2, 3, 2, 2, 116)
        cls.train_features = pad_atom_count(
            cls.raw_train_features, cls.raw_test_features
        )[0]
        cls.test_features = pad_atom_count(
            cls.raw_train_features, cls.raw_test_features
        )[1]
        cls.train_forces = np.random.rand(10, 3)

        cls.test_count = cls.test_features.shape[0]

        cls.sim_fn = DifferenceIsotropy(metric=dot, length_scale=Parameter(1.0))
        cls.model = MuyGPS(
            kernel=SOAPKernel(deformation=cls.sim_fn),
            noise=HomoscedasticNoise(cls.noise_prior),
        )
