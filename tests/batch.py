# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _check_ndarray,
    _make_gaussian_matrix,
    _make_gaussian_dict,
)
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import (
    full_filtered_batch,
    sample_batch,
    sample_balanced_batch,
)

if config.state.backend == "torch":
    raise ValueError("batch.py does not support torch backend at this time")


class BatchTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, b, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for b in [10000, 1000, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_batch(
        self, data_count, feature_count, nn_count, batch_count, nn_kwargs
    ):
        data = _make_gaussian_matrix(data_count, feature_count)
        _check_ndarray(self.assertEqual, data, mm.ftype)
        nbrs_lookup = NN_Wrapper(data, nn_count, **nn_kwargs)
        indices, nn_indices = sample_batch(nbrs_lookup, batch_count, data_count)
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, indices, mm.itype)
        target_count = np.min((data_count, batch_count))
        self.assertEqual(indices.shape, (target_count,))
        self.assertEqual(nn_indices.shape, (target_count, nn_count))

    @parameterized.parameters(
        (
            (1000, f, r, nn, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_full_filtered_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["labels"])
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, _ in enumerate(indices):
            self.assertNotEqual(
                len(mm.unique(data["labels"][nn_indices[i, :]])), 1
            )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [10000, 1000, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["labels"], mm.itype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, _ in enumerate(indices):
            self.assertNotEqual(
                len(mm.unique(data["labels"][nn_indices[i, :]])), 1
            )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch_lo_dist(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["labels"], mm.itype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)

        hist, _ = np.array(
            np.histogram(data["labels"][indices], bins=response_count),
            dtype=object,
        )
        self.assertSequenceAlmostEqual(
            hist, (batch_count / response_count) * np.ones((response_count))
        )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [1000, 10000]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch_hi_dist(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["labels"], mm.itype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)

        target_count = np.min((data_count, batch_count))
        hist, _ = np.array(
            np.histogram(data["labels"][indices], bins=response_count),
            dtype=object,
        )
        self.assertGreaterEqual(
            np.mean(hist) + 0.1 * (target_count / response_count),
            target_count / response_count,
        )
        self.assertGreaterEqual(
            np.min(hist) + 0.45 * (target_count / response_count),
            target_count / response_count,
        )


if __name__ == "__main__":
    absltest.main()
