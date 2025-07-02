# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest

from MuyGPyS._test.soap import (
    BenchmarkTestCase,
    explicit_pairwise,
    explicit_crosswise
)


class DiffsTestCase(BenchmarkTestCase):
    @classmethod
    def SetUpClass(cls):
        super(DiffsTestCase, cls).setUpClass()

        cls.train_features = cls.train_features
        cls.test_features = cls.test_features
        cls.pairwise_similarity = cls.sim_fn.pairwise_tensor(
            data=cls.train_features,
            nn_indices=cls.nn_envs
        )
        cls.crosswise_similarity = cls.sim_fn.crosswise_tensor(
            data=cls.test_features,
            nn_data=cls.train_features,
            data_indices=np.arange(cls.test_features.shape[0]),
            nn_indices=cls.nn_envs
        )

    def _crosswise_sim_chassis(
        self,
        locations,
        points,
    ):
        explicit_similarity = explicit_crosswise(
            locations=locations,
            points=points
        )
        library_similarity = self.crosswise_similarity
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))

    def _pairwise_sim_chassis(
        self,
        points,
    ):
        explicit_similarity = explicit_pairwise(
            points=points,
        )
        library_similarity = self.pairwise_similarity
        self.assertEqual(explicit_similarity.shape, library_similarity.shape)
        self.assertTrue(np.allclose(explicit_similarity, library_similarity))


if __name__ == "__main__":
    absltest.main()
