# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _check_ndarray,
    _make_gaussian_matrix,
)
from MuyGPyS.neighbors import NN_Wrapper


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
        indices = mm.iarray(
            np.random.choice(data_count, batch_count, replace=False)
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        nn_indices, nn_dists = nbrs_lookup.get_batch_nns(indices)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_dists, mm.ftype)
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
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_dists, mm.ftype)
        self.assertEqual(nn_indices.shape, (test_count, nn_count))
        self.assertEqual(nn_dists.shape, (test_count, nn_count))

    # NOTE[bwp] Should we validate actual KNN behavior, or just trust that we
    # are using the APIs correctly and that the libraries work internally? I
    # don't want to try to develop tests for third-party software...


if __name__ == "__main__":
    absltest.main()
