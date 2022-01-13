# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

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
from MuyGPyS.gp.kernels import Hyperparameter, SigmaSq, RBF, Matern


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
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        self.assertEqual(nn_indices.shape, (test_count, nn_count))
        self.assertEqual(nn_dists.shape, (test_count, nn_count))
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
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
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
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        points = train[nn_indices]
        ip_dists = pairwise_distances(train, nn_indices, metric="ip")
        co_dists = pairwise_distances(train, nn_indices, metric="cosine")
        self.assertTrue(np.allclose(ip_dists, co_dists))


class SigmaSqTest(parameterized.TestCase):
    def test_unlearned(self):
        param = SigmaSq()
        self.assertEqual("unlearned", param())
        val = np.array([1.0])
        param._set(val)
        self.assertEqual(val, param())


class HyperparameterTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1.0, "bounds": (1e-5, 1e1)},
                {"val": 1e-5, "bounds": (1e-5, 1e1)},
                {"val": 1e1, "bounds": (1e-5, 1e1)},
            )
        )
    )
    def test_full_init(self, val, bounds):
        param = Hyperparameter(val, bounds)
        self.assertEqual(val, param())
        self._check_in_bounds(bounds, param)

    def _check_in_bounds(self, given_bounds, param):
        bounds = param.get_bounds()
        self.assertEqual(given_bounds, bounds)
        self.assertGreaterEqual(param(), bounds[0])
        self.assertLessEqual(param(), bounds[1])

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": "fixed"},))
    )
    def test_fixed_init(self, val, bounds):
        param = Hyperparameter(val, bounds)
        self.assertEqual(val, param())
        self.assertTrue(param.fixed())

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": "sample", "bounds": (1e-4, 1e2), "reps": 100},
                {"val": "log_sample", "bounds": (1e-4, 1e2), "reps": 100},
            )
        )
    )
    def test_sample(self, val, bounds, reps):
        for _ in range(reps):
            param = Hyperparameter(val, bounds)
            self._check_in_bounds(bounds, param)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": "sample", "bounds": "fixed"},
                {"val": "log_sample", "bounds": "fixed"},
            )
        )
    )
    def test_fixed_sample(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "Fixed bounds do not support "):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1e-2, "bounds": (1e-1, 1e2)},
                {"val": 1e3, "bounds": (1e-1, 1e2)},
            )
        )
    )
    def test_oob(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "bound"):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": np.array([1e-2, 1e1]), "bounds": "fixed"},
                {"val": np.array([1e3]), "bounds": (1e-1, 1e2)},
            )
        )
    )
    def test_nonscalar(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "Nonscalar hyperparameter"):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": "wut", "bounds": (1e-1, 1e2)},))
    )
    def test_bad_val_string(self, val, bounds):
        with self.assertRaisesRegex(
            ValueError, "Unsupported string hyperparameter"
        ):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": "sample", "bounds": "fixed"},))
    )
    def test_string_on_fixed(self, val, bounds):
        with self.assertRaisesRegex(
            ValueError, "Fixed bounds do not support string"
        ):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": "badstring"},))
    )
    def test_bad_val_bounds(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": (1e2, 1e-1)},))
    )
    def test_bad_bounds(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "not lesser than upper bound"):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1.0, "bounds": (1e2, 1e-1, 1e2)},
                {"val": 1.0, "bounds": [1e2]},
            )
        )
    )
    def test_bad_bound_length(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "unsupported length"):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1.0, "bounds": ("a", 1e-2)},
                {"val": 1.0, "bounds": (1e-1, "b")},
            )
        )
    )
    def test_bad_bound_vals(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "supported hyperparameter"):
            Hyperparameter(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": 1e-2},))
    )
    def test_noniterable_bound(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "non-iterable type"):
            Hyperparameter(val, bounds)


class KernelTest(parameterized.TestCase):
    def _check_params_chassis(self, kern_fn, **kwargs):
        for p in kern_fn.hyperparameters:
            self._check_params(kern_fn, p, **kwargs.get(p, dict()))

    def _check_params(self, kern_fn, param, val=None, bounds=None):
        if val is not None:
            self.assertEqual(val, kern_fn.hyperparameters[param]())
        if bounds is not None:
            if bounds == "fixed":
                self.assertTrue(kern_fn.hyperparameters[param].fixed())
            else:
                self.assertFalse(kern_fn.hyperparameters[param].fixed())
                self.assertEqual(
                    bounds, kern_fn.hyperparameters[param].get_bounds()
                )


class RBFTest(KernelTest):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            # for f in [100]
            # for nn in [5]
            # # for nn_kwargs in _fast_nn_kwarg_options
            # for nn_kwargs in _exact_nn_kwarg_options
            # for k_kwargs in [
            #     {"length_scale": {"val": 10.0, "bounds": (1e-5, 1e1)}}
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
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        F2_dists = pairwise_distances(train, nn_indices, metric="F2")
        rbf = RBF(**k_kwargs)
        self._check_params_chassis(rbf, **k_kwargs)
        kern = rbf(F2_dists)
        self.assertEqual(kern.shape, (test_count, nn_count, nn_count))
        points = train[nn_indices]
        sk_rbf = sk_RBF(length_scale=rbf.length_scale())
        sk_kern = np.array([sk_rbf(mat) for mat in points])
        self.assertEqual(sk_kern.shape, (test_count, nn_count, nn_count))
        self.assertTrue(np.allclose(kern, sk_kern))
        Kcross = rbf(nn_dists)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        sk_Kcross = np.array(
            [sk_rbf(vec, mat) for vec, mat in zip(test, points)]
        ).reshape(test_count, nn_count)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertTrue(np.allclose(Kcross, sk_Kcross))


class ParamTest(KernelTest):
    @parameterized.parameters(
        (
            (k_kwargs, alt_kwargs)
            for k_kwargs in [
                {"length_scale": {"val": 10.0, "bounds": (1e-5, 1e1)}}
            ]
            for alt_kwargs in [
                {"length_scale": {"val": 1.0, "bounds": (1e-2, 1e4)}},
                {"length_scale": {"bounds": (1e-3, 1e2)}},
                {"length_scale": {"val": 2.0}},
            ]
        )
    )
    def test_rbf(self, k_kwargs, alt_kwargs):
        self._test_chassis(RBF, k_kwargs, alt_kwargs)

    @parameterized.parameters(
        (
            (k_kwargs, alt_kwargs)
            for k_kwargs in [
                {
                    "nu": {"val": 0.42, "bounds": (1e-4, 5e1)},
                    "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
                }
            ]
            for alt_kwargs in [
                {
                    "nu": {"val": 1.0, "bounds": (1e-2, 5e4)},
                    "length_scale": {"val": 7.2, "bounds": (2e-5, 2e1)},
                },
                {
                    "nu": {"val": 1.0},
                    "length_scale": {"bounds": (2e-5, 2e1)},
                },
                {
                    "nu": {"bounds": (1e-2, 5e4)},
                },
                {
                    "length_scale": {"val": 7.2},
                },
            ]
        )
    )
    def test_matern(self, k_kwargs, alt_kwargs):
        self._test_chassis(Matern, k_kwargs, alt_kwargs)

    def _test_chassis(self, kern, k_kwargs, alt_kwargs):
        kern_fn = kern(**k_kwargs)
        self._check_params_chassis(kern_fn, **k_kwargs)
        kern_fn.set_params(**alt_kwargs)
        self._check_params_chassis(kern_fn, **alt_kwargs)


class MaternTest(KernelTest):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "nu": {"val": 0.42, "bounds": "fixed"},
                    "length_scale": {"val": 1.0, "bounds": "fixed"},
                },
                {
                    "nu": {"val": 0.5, "bounds": "fixed"},
                    "length_scale": {"val": 1.0, "bounds": "fixed"},
                },
                {
                    "nu": {"val": 1.5, "bounds": "fixed"},
                    "length_scale": {"val": 1.0, "bounds": "fixed"},
                },
                {
                    "nu": {"val": 2.5, "bounds": "fixed"},
                    "length_scale": {"val": 1.0, "bounds": "fixed"},
                },
                {
                    "nu": {"val": np.inf, "bounds": "fixed"},
                    "length_scale": {"val": 1.0, "bounds": "fixed"},
                },
            ]
            # for f in [100]
            # for nn in [5]
            # for nn_kwargs in _fast_nn_kwarg_options
            # for k_kwargs in [
            #     {
            #         "nu": {"val": 0.42, "bounds": (1e-4, 5e1)},
            #         "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
            #     }
            # ]
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
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = np.sqrt(nn_dists)
        l2_dists = pairwise_distances(train, nn_indices, metric="l2")
        mtn = Matern(**k_kwargs)
        self._check_params_chassis(mtn, **k_kwargs)
        kern = mtn(l2_dists)
        self.assertEqual(kern.shape, (test_count, nn_count, nn_count))
        points = train[nn_indices]
        sk_mtn = sk_Matern(nu=mtn.nu(), length_scale=mtn.length_scale())
        sk_kern = np.array([sk_mtn(mat) for mat in points])
        self.assertEqual(sk_kern.shape, (test_count, nn_count, nn_count))
        self.assertTrue(np.allclose(kern, sk_kern))
        Kcross = mtn(nn_dists)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        sk_Kcross = np.array(
            [sk_mtn(vec, mat) for vec, mat in zip(test, points)]
        ).reshape(test_count, nn_count)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertTrue(np.allclose(Kcross, sk_Kcross))


if __name__ == "__main__":
    absltest.main()
