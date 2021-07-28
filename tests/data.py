# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.embed import embed_all
from MuyGPyS.data.utils import normalize, subsample, balanced_subsample
from MuyGPyS.testing.test_utils import _make_gaussian_dict, _make_gaussian_data


class EmbedTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 900, f, r, e, m)
            for f in [100, 1000, 40]
            for r in [1, 10]
            for e in [10, 20, 40]
            for m in ["pca"]
        )
    )
    def test_embed_copy(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        embed_dim,
        method,
    ):
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )
        embedded_train, embedded_test = embed_all(
            train,
            test,
            embed_dim,
            embed_method=method,
            do_normalize=False,
            in_place=False,
        )
        self.assertEqual(
            embedded_train["input"].shape, (train_count, embed_dim)
        )
        self.assertEqual(embedded_test["input"].shape, (test_count, embed_dim))

    @parameterized.parameters(
        (
            (1000, 900, f, r, e, m)
            for f in [100, 1000, 40]
            for r in [1, 10]
            for e in [10, 20, 40]
            for m in ["pca"]
        )
    )
    def test_embed_inplace(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        embed_dim,
        method,
    ):
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )
        embed_all(
            train,
            test,
            embed_dim,
            embed_method=method,
            do_normalize=False,
            in_place=True,
        )
        self.assertEqual(train["input"].shape, (train_count, embed_dim))
        self.assertEqual(test["input"].shape, (test_count, embed_dim))

    @parameterized.parameters(((1000, f) for f in [1000, 100, 2]))
    def test_normalize(self, data_count, feature_count):
        data = np.random.randn(data_count, feature_count)
        normalized_data = normalize(data)
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
        sub_data = subsample(data, sample_count)
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
        sub_data = balanced_subsample(data, sample_count)
        self.assertEqual(sub_data["input"].shape, (sample_count, feature_count))

    @parameterized.parameters(((1000, 100, r, 500) for r in [2, 10]))
    def test_balanced_subsample_dist(
        self, data_count, feature_count, response_count, sample_count
    ):
        data = _make_gaussian_dict(
            data_count, feature_count, response_count, categorical=True
        )
        sub_data = balanced_subsample(data, sample_count)
        labels = np.argmax(sub_data["output"], axis=1)
        hist, _ = np.array(np.histogram(labels, bins=response_count))
        self.assertSequenceAlmostEqual(
            hist, (sample_count / response_count) * np.ones((response_count))
        )


if __name__ == "__main__":
    absltest.main()
