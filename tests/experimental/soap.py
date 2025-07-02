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
            nn_indices=self.nn_envs,
        )
        library_similarity = self.sim_fn.crosswise_tensor(
            data=self.test_features,
            nn_data=self.train_features,
            data_indices=np.arange(self.test_count),
            nn_indices=self.nn_envs,
        )
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))

    def _pairwise_sim_chassis(self):
        explicit_similarity = explicit_pairwise(
            data=self.train_features, nn_indices=self.nn_envs
        )
        library_similarity = self.sim_fn.pairwise_tensor(
            data=self.train_features, nn_indices=self.nn_envs
        )
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))


class SimTest(SimTestCase):
    def test_crosswise(self):
        self._crosswise_sim_chassis()

    def test_pairwise(self):
        self._pairwise_sim_chassis()


if __name__ == "__main__":
    absltest.main()
