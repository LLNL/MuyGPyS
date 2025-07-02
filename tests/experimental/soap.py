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

    def _crosswise_sim_chassis(
        self,
        locations,
        points,
    ):
        explicit_similarity = explicit_crosswise(
            locations=locations,
            points=points
        )
        library_similarity = self.sim_fn.crosswise_tensor(
            data=self.test_features,
            nn_data=self.train_features,
            data_indices=np.arange(self.test_count),
            nn_indices=self.nn_envs
        )
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))

    def _pairwise_sim_chassis(
        self,
        points,
    ):
        explicit_similarity = explicit_pairwise(
            points=points,
        )
        library_similarity = self.sim_fn.pairwise_tensor(
            data=self.train_features,
            nn_indices=self.nn_envs
        )
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))


class SimTest(SimTestCase):
    def test_crosswise(self):
        self._crosswise_sim_chassis(self.test_features, self.train_features)

    def test_pairwise(self):
        self._pairwise_sim_chassis(self.train_features)


if __name__ == "__main__":
    absltest.main()
