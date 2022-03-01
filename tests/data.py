# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS._test.utils import (
    _normalize,
    _subsample,
    _balanced_subsample,
    _make_gaussian_dict,
)


class NormalizeTest(parameterized.TestCase):
    @parameterized.parameters(((1000, f) for f in [1000, 100, 2]))
    def test_normalize(self, data_count, feature_count):
        data = np.random.randn(data_count, feature_count)
        normalized_data = _normalize(data)
        self.assertSequenceAlmostEqual(
            np.linalg.norm(normalized_data, axis=1),
            np.sqrt(feature_count) * np.ones((data_count,)),
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
        sub_data = _balanced_subsample(data, sample_count)
        self.assertEqual(sub_data["input"].shape, (sample_count, feature_count))

    @parameterized.parameters(((1000, 100, r, 500) for r in [2, 10]))
    def test_balanced_subsample_dist(
        self, data_count, feature_count, response_count, sample_count
    ):
        data = _make_gaussian_dict(
            data_count, feature_count, response_count, categorical=True
        )
        sub_data = _balanced_subsample(data, sample_count)
        labels = np.argmax(sub_data["output"], axis=1)
        hist, _ = np.array(np.histogram(labels, bins=response_count))
        self.assertSequenceAlmostEqual(
            hist, (sample_count / response_count) * np.ones((response_count))
        )


if __name__ == "__main__":
    absltest.main()
