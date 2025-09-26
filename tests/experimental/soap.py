# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest

from MuyGPyS._test.soap import (
    BenchmarkTestCase,
    explicit_pairwise,
    explicit_crosswise,
)


class SimTestCase(BenchmarkTestCase):
    @classmethod
    def SetUpClass(cls):
        super(SimTestCase, cls).setUpClass()

    def _crosswise_sim_chassis(self):
        explicit_similarity = explicit_crosswise(
            data=self.test_features,
            nn_data=self.train_features,
            indices=np.arange(self.test_count),
            nn_indices=self.nn_envs.astype(int),
        )
        library_similarity = self.sim_fn.crosswise_tensor(
            data=self.test_features,
            nn_data=self.train_features,
            data_indices=np.arange(self.test_count),
            nn_indices=self.nn_envs.astype(int),
        )
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))

    def _pairwise_sim_chassis(self):
        explicit_similarity = explicit_pairwise(
            data=self.train_features, nn_indices=self.nn_envs.astype(int)
        )
        library_similarity = self.sim_fn.pairwise_tensor(
            data=self.train_features, nn_indices=self.nn_envs.astype(int)
        )
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))


class SimTest(SimTestCase):
    def test_crosswise(self):
        self._crosswise_sim_chassis()

    def test_pairwise(self):
        self._pairwise_sim_chassis()


class KernelTestCase(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()

        cls.crosswise_similarity = cls.sim_fn.crosswise_tensor(
            data=cls.test_features,
            nn_data=cls.train_features,
            data_indices=np.arange(cls.test_count),
            nn_indices=cls.nn_envs.astype(int),
        )
        cls.pairwise_similarity = cls.sim_fn.pairwise_tensor(
            data=cls.train_features, nn_indices=cls.nn_envs.astype(int)
        )

    def _Kin_chassis(self, Kernel_fn):
        Kin = Kernel_fn(self.pairwise_similarity)
        expected_shape = (self.test_count, 3, self.nn_count, 3, self.nn_count)

        self.assertEqual(Kin.shape, expected_shape)

    def _Kcross_chassis(self, Kernel_fn):
        Kcross = Kernel_fn(self.crosswise_similarity)
        expected_shape = (self.test_count, 3, self.nn_count, 3)

        self.assertEqual(Kcross.shape, expected_shape)


class KernelTest(KernelTestCase):
    def test_Kcross(self):
        self._Kcross_chassis(Kernel_fn=self.model.kernel)

    def test_Kin(self):
        self._Kin_chassis(Kernel_fn=self.model.kernel)


class PosteriorTestCase(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(PosteriorTestCase, cls).setUpClass()

        cls.crosswise_similarity = cls.sim_fn.crosswise_tensor(
            data=cls.test_features,
            nn_data=cls.train_features,
            data_indices=np.arange(cls.test_count),
            nn_indices=cls.nn_envs.astype(int),
        )
        cls.pairwise_similarity = cls.sim_fn.pairwise_tensor(
            data=cls.train_features, nn_indices=cls.nn_envs.astype(int)
        )
        cls.out_similarity = cls.sim_fn.out_tensor(
            data=cls.test_features, data_indices=np.arange(cls.test_count)
        )
        cls.Kin = cls.model.kernel(cls.pairwise_similarity)

        cls.Kcross = cls.model.kernel(cls.crosswise_similarity)

        cls.Kout = cls.model.kernel(cls.out_similarity)

    def _mean_chassis(self):
        nn_targets = self.train_forces[self.nn_envs.astype(int)].swapaxes(-2, -1)
        posterior_mean = self.model.posterior_mean(
            Kin=self.Kin,
            Kcross=self.Kcross,
            batch_nn_targets=nn_targets
        )
        self.assertEqual(posterior_mean.shape, (self.test_count, 3))

    def _variance_chassis(self):
        posterior_var = self.model.posterior_variance(
            Kin=self.Kin,
            Kcross=self.Kcross,
            Kout=self.Kout
        )
        self.assertEqual(posterior_var.shape, (self.test_count, 3, 3))


class PosteriorTest(PosteriorTestCase):
    def test_mean(self):
        self._mean_chassis()

    def test_variance(self):
        self._variance_chassis()


if __name__ == "__main__":
    absltest.main()
