# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.testing.test_utils import (
    _make_gaussian_matrix,
    _basic_nn_kwarg_options,
)


class NeighborsTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 100, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_neighbors_batch_shape(
        self, data_count, feature_count, nn_count, batch_count, nn_kwargs
    ):
        data = _make_gaussian_matrix(data_count, feature_count)
        nbrs_lookup = NN_Wrapper(data, nn_count, **nn_kwargs)
        indices = np.random.choice(data_count, batch_count, replace=False)
        nn_indices, nn_dists = nbrs_lookup.get_batch_nns(indices)
        self.assertEqual(nn_indices.shape, (batch_count, nn_count))
        self.assertEqual(nn_dists.shape, (batch_count, nn_count))

    @parameterized.parameters(
        (
            (1000, f, nn, 100, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_neighbors_query_shape(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        self.assertEqual(nn_indices.shape, (test_count, nn_count))
        self.assertEqual(nn_dists.shape, (test_count, nn_count))

    ## NOTE[bwp] Should we validate actual KNN behavior, or just trust that we
    # are using the APIs correctly and that the libraries work internally? I
    # don't want to try to develop tests for third-party software...


if __name__ == "__main__":
    absltest.main()
