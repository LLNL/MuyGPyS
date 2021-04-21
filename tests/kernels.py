# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import RBF as sk_RBF

from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.testing.test_utils import (
    _make_gaussian_matrix,
    _basic_nn_kwarg_options,
    _exact_nn_kwarg_options,
    _fast_nn_kwarg_options,
)
from MuyGPyS.gp.distance import pairwise_distances
from MuyGPyS.gp.kernels import Hyperparameter, RBF, Matern


class DistancesTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, m, 10, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for m in ["l2", "F2", "ip", "cosine"]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [100]
            # for nn in [3]
            # for nn_kwargs in _exact_nn_kwarg_options
        )
    )
    def test_distances_shapes(
        self,
        train_count,
        feature_count,
        nn_count,
        metric,
        test_count,
        nn_kwargs,
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices = nbrs_lookup.get_nns(test)
        self.assertEqual(nn_indices.shape, (test_count, nn_count))
        dists = pairwise_distances(train, nn_indices, metric=metric)
        self.assertEqual(dists.shape, (test_count, nn_count, nn_count))

    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_l2(
        self, train_count, feature_count, nn_count, test_count, nn_kwargs
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices = nbrs_lookup.get_nns(test)
        points = train[nn_indices]
        self.assertEqual(points.shape, (test_count, nn_count, feature_count))
        dists = np.array(
            [
                np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1) ** 2
                for mat in points
            ]
        )
        self.assertEqual(dists.shape, (test_count, nn_count, nn_count))
        ll_dists = points[:, :, None, :] - points[:, None, :, :]
        ll_dists = ll_dists ** 2
        ll_dists = np.sum(ll_dists, axis=-1)
        self.assertEqual(ll_dists.shape, (test_count, nn_count, nn_count))
        l2_dists = pairwise_distances(train, nn_indices, metric="l2")
        self.assertEqual(l2_dists.shape, (test_count, nn_count, nn_count))
        F2_dists = pairwise_distances(train, nn_indices, metric="F2")
        self.assertEqual(l2_dists.shape, (test_count, nn_count, nn_count))
        self.assertTrue(np.allclose(dists, F2_dists))
        self.assertTrue(np.allclose(dists, l2_dists ** 2))
        self.assertTrue(np.allclose(dists, ll_dists))

    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [100]
            # for nn in [5]
            # for nn_kwargs in _fast_nn_kwarg_options
        )
    )
    def test_cosine(
        self, train_count, feature_count, nn_count, test_count, nn_kwargs
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        train = train / np.linalg.norm(train, axis=1)[:, None]
        test = test / np.linalg.norm(test, axis=1)[:, None]
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices = nbrs_lookup.get_nns(test)
        points = train[nn_indices]
        ip_dists = pairwise_distances(train, nn_indices, metric="ip")
        co_dists = pairwise_distances(train, nn_indices, metric="cosine")
        self.assertTrue(np.allclose(ip_dists, co_dists))


class RBFTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            # for f in [100]
            # for nn in [5]
            # for nn_kwargs in _fast_nn_kwarg_options
            # for k_kwargs in [
            #     {"length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)}}
            # ]
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {"length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)}},
                {"length_scale": {"val": 2.0, "bounds": (1e-4, 1e3)}},
            ]
        )
    )
    def test_rbf(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices = nbrs_lookup.get_nns(test)
        F2_dists = pairwise_distances(train, nn_indices, metric="F2")
        rbf = RBF(**k_kwargs)
        kern = rbf(F2_dists)
        self.assertEqual(kern.shape, (test_count, nn_count, nn_count))
        self.assertEqual(k_kwargs["length_scale"]["val"], rbf.length_scale())
        self.assertSequenceAlmostEqual(
            k_kwargs["length_scale"]["bounds"], rbf.length_scale.get_bounds()
        )
        points = train[nn_indices]
        sk_rbf = sk_RBF(length_scale=rbf.length_scale())
        sk_kern = np.array([sk_rbf(mat) for mat in points])
        self.assertEqual(sk_kern.shape, (test_count, nn_count, nn_count))
        self.assertTrue(np.allclose(kern, sk_kern))


class MaternTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [100]
            # for nn in [5]
            # for nn_kwargs in _fast_nn_kwarg_options
            for k_kwargs in [
                {
                    "nu": {"val": 0.42, "bounds": (1e-4, 5e1)},
                    "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
                },
                {
                    "nu": {"val": 0.5, "bounds": (1e-4, 5e1)},
                    "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
                },
                {
                    "nu": {"val": 1.5, "bounds": (1e-4, 5e1)},
                    "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
                },
                {
                    "nu": {"val": 2.5, "bounds": (1e-4, 5e1)},
                    "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
                },
                {
                    "nu": {"val": np.inf, "bounds": (1e-4, 5e1)},
                    "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
                },
            ]
        )
    )
    def test_matern(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices = nbrs_lookup.get_nns(test)
        l2_dists = pairwise_distances(train, nn_indices, metric="l2")
        mtn = Matern(**k_kwargs)
        kern = mtn(l2_dists)
        self.assertEqual(kern.shape, (test_count, nn_count, nn_count))
        self.assertEqual(k_kwargs["length_scale"]["val"], mtn.length_scale())
        self.assertSequenceAlmostEqual(
            k_kwargs["length_scale"]["bounds"], mtn.length_scale.get_bounds()
        )
        self.assertEqual(k_kwargs["nu"]["val"], mtn.nu())
        self.assertSequenceAlmostEqual(
            k_kwargs["nu"]["bounds"], mtn.nu.get_bounds()
        )
        points = train[nn_indices]
        sk_mtn = sk_Matern(nu=mtn.nu(), length_scale=mtn.length_scale())
        sk_kern = np.array([sk_mtn(mat) for mat in points])
        self.assertEqual(sk_kern.shape, (test_count, nn_count, nn_count))
        self.assertTrue(np.allclose(kern, sk_kern))

    # @parameterized.parameters(
    #     (
    #         (1000, f, nn, 10, nn_kwargs, k_kwargs)
    #         # for f in [100, 10, 2, 1]
    #         # for nn in [5, 10, 100]
    #         # for nn_kwargs in _basic_nn_kwarg_options
    #         for f in [100]
    #         for nn in [5]
    #         for nn_kwargs in _fast_nn_kwarg_options
    #         for k_kwargs in [
    #             {
    #                 "nu": {"val": 1.5, "bounds": (1e-5, 1e2)},
    #                 "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
    #             },
    #         ]
    #     )
    # )
    # def test_matern(
    #     self,
    #     train_count,
    #     feature_count,
    #     nn_count,
    #     test_count,
    #     nn_kwargs,
    #     k_kwargs,
    # ):
    #     train = _make_gaussian_matrix(train_count, feature_count)
    #     test = _make_gaussian_matrix(test_count, feature_count)
    #     nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
    #     nn_indices = nbrs_lookup.get_nns(test)
    #     F2_dists = pairwise_distances(train, nn_indices, metric="F2")
    #     rbf = RBF(**k_kwargs)
    #     kern = rbf(F2_dists)
    #     self.assertEqual(kern.shape, (test_count, nn_count, nn_count))
    #     self.assertEqual(k_kwargs["length_scale"]["val"], rbf.length_scale())
    #     self.assertSequenceAlmostEqual(
    #         k_kwargs["length_scale"]["bounds"], rbf.length_scale.get_bounds()
    #     )


if __name__ == "__main__":
    absltest.main()
