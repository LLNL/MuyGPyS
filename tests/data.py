# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._test.utils import (
    _balanced_subsample,
    _check_ndarray,
    _make_gaussian_dict,
    _normalize,
    _precision_assert,
    _subsample,
)


class NormalizeTest(parameterized.TestCase):
    @parameterized.parameters(((1000, f) for f in [1000, 100, 2]))
    def test_normalize(self, data_count, feature_count):
        data = mm.array(np.random.randn(data_count, feature_count))
        normalized_data = _normalize(data)
        _check_ndarray(self.assertEqual, normalized_data, mm.ftype)
        _precision_assert(
            self.assertSequenceAlmostEqual,
            np.array(mm.linalg.norm(normalized_data, axis=1)),
            np.array(mm.sqrt(mm.array(feature_count)) * mm.ones((data_count,))),
        )


class SampleTest(parameterized.TestCase):
    @parameterized.parameters((1000, 100, 10, 20))
    def test_subsample_shape(
        self, data_count, feature_count, response_count, sample_count
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        sub_data = _subsample(data, sample_count)
        self.assertEqual(sub_data["input"].shape, (sample_count, feature_count))

    @parameterized.parameters(
        ((1000, 100, r, s) for r in [10, 2] for s in [500, 200, 20])
    )
    def test_balanced_subsample_shape(
        self, data_count, feature_count, response_count, sample_count
    ):
        data = _make_gaussian_dict(
            data_count, feature_count, response_count, categorical=True
        )
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["output"], mm.ftype)
        sub_data = _balanced_subsample(data, sample_count)
        _check_ndarray(self.assertEqual, sub_data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, sub_data["output"], mm.ftype)

    @parameterized.parameters(((1000, 100, r, 500) for r in [2, 10]))
    def test_balanced_subsample_dist(
        self, data_count, feature_count, response_count, sample_count
    ):
        data = _make_gaussian_dict(
            data_count, feature_count, response_count, categorical=True
        )
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["output"], mm.ftype)
        sub_data = _balanced_subsample(data, sample_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["output"], mm.ftype)
        labels = mm.argmax(sub_data["output"], axis=1)

        hist, _ = np.histogram(labels, bins=response_count)
        self.assertSequenceAlmostEqual(
            hist, (sample_count / response_count) * mm.ones((response_count))
        )


if __name__ == "__main__":
    absltest.main()
