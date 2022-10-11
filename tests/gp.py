# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

from MuyGPyS.examples.regress import make_regressor
from MuyGPyS.examples.classify import make_classifier

from MuyGPyS.gp.distance import (
    make_train_tensors,
    make_regress_tensors,
    pairwise_distances,
    crosswise_distances,
)
from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS._test.gp import BenchmarkGP
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.sigma_sq import muygps_sigma_sq_optim
from MuyGPyS._test.utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
    _exact_nn_kwarg_options,
    _get_sigma_sq_series,
    _consistent_assert,
)
from MuyGPyS._src.mpi_utils import (
    _consistent_unchunk_tensor,
)


class GPInitTest(parameterized.TestCase):
    @parameterized.parameters(
        (k_kwargs, e, gp)
        for k_kwargs in (
            (
                "matern",
                {
                    "nu": {"val": 1.0},
                    "length_scale": {"val": 7.2},
                },
            ),
            (
                "rbf",
                {
                    "length_scale": {"val": 1.5},
                },
            ),
        )
        for e in (({"val": 1e-5},))
        for gp in (MuyGPS, BenchmarkGP)
    )
    def test_bounds_defaults_init(self, k_kwargs, eps, gp_type):
        kern, kwargs = k_kwargs
        muygps = gp_type(kern=kern, eps=eps, **kwargs)
        for param in kwargs:
            self.assertEqual(
                kwargs[param]["val"],
                muygps.kernel.hyperparameters[param](),
            )
            self.assertTrue(
                muygps.kernel.hyperparameters[param].fixed(),
            )
        self.assertEqual(eps["val"], muygps.eps())
        self.assertTrue(muygps.eps.fixed())
        if gp_type == MuyGPS:
            self.assertFalse(muygps.sigma_sq.trained())
            self.assertEqual(np.array([1.0]), muygps.sigma_sq())
        elif gp_type == BenchmarkGP:
            self.assertFalse(muygps.sigma_sq.trained())
            self.assertEqual(np.array([1.0]), muygps.sigma_sq())

    @parameterized.parameters(
        (k_kwargs, e, gp)
        for k_kwargs in (
            (
                "matern",
                {
                    "nu": {"val": 1.0, "bounds": (1e-2, 5e4)},
                    "length_scale": {"val": 7.2, "bounds": (2e-5, 2e1)},
                },
            ),
            (
                "rbf",
                {
                    "length_scale": {"val": 1.5, "bounds": (1e-1, 1e2)},
                },
            ),
            (
                "matern",
                {
                    "nu": {"val": 1.0, "bounds": "fixed"},
                    "length_scale": {"val": 7.2, "bounds": "fixed"},
                },
            ),
            (
                "rbf",
                {
                    "length_scale": {"val": 1.5, "bounds": "fixed"},
                },
            ),
        )
        for e in (
            (
                {"val": 1e-5, "bounds": (1e-8, 1e-2)},
                {"val": 1e-5, "bounds": "fixed"},
            )
        )
        for gp in (MuyGPS, BenchmarkGP)
    )
    def test_full_init(self, k_kwargs, eps, gp_type):
        kern, kwargs = k_kwargs
        muygps = gp_type(kern=kern, eps=eps, **kwargs)
        for param in kwargs:
            self.assertEqual(
                kwargs[param]["val"],
                muygps.kernel.hyperparameters[param](),
            )
            if kwargs[param]["bounds"] == "fixed":
                self.assertTrue(muygps.kernel.hyperparameters[param].fixed())
            else:
                self.assertFalse(muygps.kernel.hyperparameters[param].fixed())
                self.assertEqual(
                    kwargs[param]["bounds"],
                    muygps.kernel.hyperparameters[param].get_bounds(),
                )
        self.assertEqual(eps["val"], muygps.eps())
        if eps["bounds"] == "fixed":
            self.assertTrue(muygps.eps.fixed())
        else:
            self.assertFalse(muygps.eps.fixed())
            self.assertEqual(eps["bounds"], muygps.eps.get_bounds())
        if gp_type == MuyGPS:
            self.assertFalse(muygps.sigma_sq.trained())
            self.assertEqual(1.0, muygps.sigma_sq())
        elif gp_type == BenchmarkGP:
            self.assertFalse(muygps.sigma_sq.trained())
            self.assertEqual(np.array([1.0]), muygps.sigma_sq())

    @parameterized.parameters(
        (k_kwargs, e, gp)
        for k_kwargs in (
            (
                "matern",
                {
                    "nu": {"val": -1.0, "bounds": (1e-2, 5e4)},
                    "length_scale": {"val": 7000.2, "bounds": (2e-5, 2e1)},
                },
            ),
            (
                "rbf",
                {
                    "length_scale": {"val": 1e-2, "bounds": (1e-1, 1e2)},
                },
            ),
        )
        for e in (
            (
                {"val": 1e-1, "bounds": (1e-8, 1e-2)},
                {"val": 1e-9, "bounds": (1e-8, 1e-2)},
            )
        )
        for gp in (MuyGPS, BenchmarkGP)
    )
    def test_oob_init(self, k_kwargs, eps, gp_init):
        kern, kwargs = k_kwargs
        with self.assertRaises(ValueError):
            _ = gp_init(kern=kern, eps=eps, **kwargs)

    @parameterized.parameters(
        (k_kwargs, e, gp, 100)
        for k_kwargs in (
            (
                "matern",
                {
                    "nu": {"val": "sample", "bounds": (1e-2, 5e4)},
                    "length_scale": {"val": "sample", "bounds": (2e-5, 1e1)},
                },
            ),
            (
                "rbf",
                {
                    "length_scale": {"val": "sample", "bounds": (1e-1, 1e2)},
                },
            ),
            (
                "matern",
                {
                    "nu": {"val": "log_sample", "bounds": (1e-2, 5e4)},
                    "length_scale": {
                        "val": "log_sample",
                        "bounds": (2e-5, 1e1),
                    },
                },
            ),
            (
                "rbf",
                {
                    "length_scale": {
                        "val": "log_sample",
                        "bounds": (1e-1, 1e2),
                    },
                },
            ),
        )
        for e in (
            (
                {"val": "sample", "bounds": (1e-8, 1e-2)},
                {"val": "log_sample", "bounds": (1e-8, 1e-2)},
            )
        )
        for gp in (MuyGPS, BenchmarkGP)
    )
    def test_sample_init(self, k_kwargs, eps, gp_type, reps):
        kern, kwargs = k_kwargs
        for _ in range(reps):
            muygps = gp_type(kern=kern, eps=eps, **kwargs)
            for param in kwargs:
                self._check_in_bounds(
                    kwargs[param]["bounds"],
                    muygps.kernel.hyperparameters[param],
                )
            self._check_in_bounds(eps["bounds"], muygps.eps)

    def _check_in_bounds(self, given_bounds, param):
        bounds = param.get_bounds()
        self.assertEqual(given_bounds, bounds)
        self.assertGreaterEqual(param(), bounds[0])
        self.assertLessEqual(param(), bounds[1])


class GPMathTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 100, f, 10, nn_kwargs, k_kwargs)
            for f in [100, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "eps": {"val": 1e-5},
                    "nu": {"val": 1.0},
                    "length_scale": {"val": 7.2},
                },
                {
                    "kern": "rbf",
                    "metric": "F2",
                    "eps": {"val": 1e-5},
                    "length_scale": {"val": 1.5},
                },
            )
        )
    )
    def test_tensor_shapes(
        self,
        train_count,
        test_count,
        feature_count,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(**k_kwargs)

        # prepare data
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)

        # neighbors and distances
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, _ = nbrs_lookup.get_nns(test)
        indices = np.arange(test_count)
        nn_dists = crosswise_distances(
            test, train, indices, nn_indices, metric=muygps.kernel.metric
        )
        F2_dists = pairwise_distances(
            train, nn_indices, metric=muygps.kernel.metric
        )

        # make kernels
        K = _consistent_unchunk_tensor(muygps.kernel(F2_dists))
        Kcross = _consistent_unchunk_tensor(muygps.kernel(nn_dists))
        # do validation
        self.assertEqual(K.shape, (test_count, nn_count, nn_count))
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0))
        self.assertTrue(np.all(Kcross >= 0.0))
        self.assertTrue(np.all(Kcross <= 1.0))
        # # Check that kernels are positive semidefinite
        for i in range(K.shape[0]):
            eigvals = np.linalg.eigvals(K[i, :, :])
            self.assertTrue(
                np.all(np.logical_or(eigvals >= 0.0, np.isclose(eigvals, 0.0)))
            )

    @parameterized.parameters(
        (
            (1000, 100, f, r, 10, nn_kwargs, k_kwargs)
            for f in [100, 1]
            for r in [5, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "eps": {"val": 1e-5},
                    "nu": {"val": 1.0},
                    "length_scale": {"val": 7.2},
                },
                {
                    "kern": "rbf",
                    "metric": "F2",
                    "eps": {"val": 1e-5},
                    "length_scale": {"val": 1.5},
                },
            )
            # for f in [100]
            # for r in [5]
            # for nn_kwargs in _exact_nn_kwarg_options
        )
    )
    def test_tensor_solve(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(**k_kwargs)

        # prepare data
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )

        # neighbors and distances
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)
        nn_indices, _ = nbrs_lookup.get_nns(test["input"])
        indices = np.arange(test_count)
        (nn_dists, F2_dists, train_targets) = make_regress_tensors(
            muygps.kernel.metric,
            indices,
            nn_indices,
            test["input"],
            train["input"],
            train["output"],
        )

        # make kernels
        K, Kcross = muygps.kernel(F2_dists), muygps.kernel(nn_dists)
        # solve GP regression
        responses = _consistent_unchunk_tensor(
            muygps._compute_solve(K, Kcross, train_targets, muygps.eps())
        )

        K = _consistent_unchunk_tensor(K)
        Kcross = _consistent_unchunk_tensor(Kcross)

        # validate
        self.assertEqual(responses.shape, (test_count, response_count))
        for i in range(test_count):
            _consistent_assert(
                self.assertSequenceAlmostEqual,
                responses[i, :],
                Kcross[i, :]
                @ np.linalg.solve(
                    K[i, :, :] + muygps.eps() * np.eye(nn_count),
                    train["output"][nn_indices[i], :],
                ),
            )

    @parameterized.parameters(
        (
            (1000, 100, f, r, 10, nn_kwargs, k_kwargs)
            for f in [100, 1]
            for r in [10, 2, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [1]
            # for r in [10]
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "eps": {"val": 1e-5},
                    "nu": {"val": 1.0},
                    "length_scale": {"val": 7.2},
                },
                {
                    "kern": "rbf",
                    "metric": "F2",
                    "eps": {"val": 1e-5},
                    "length_scale": {"val": 1.5},
                },
            )
        )
    )
    def test_diagonal_variance(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(**k_kwargs)

        # prepare data
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )

        # neighbors and distances
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)
        nn_indices, _ = nbrs_lookup.get_nns(test["input"])
        indices = np.arange(test_count)
        (nn_dists, F2_dists, _) = make_regress_tensors(
            muygps.kernel.metric,
            indices,
            nn_indices,
            test["input"],
            train["input"],
            train["output"],
        )

        # make kernels and variance
        K, Kcross = muygps.kernel(F2_dists), muygps.kernel(nn_dists)
        diagonal_variance = _consistent_unchunk_tensor(
            muygps._compute_diagonal_variance(K, Kcross, muygps.eps())
        )

        K = _consistent_unchunk_tensor(K)
        Kcross = _consistent_unchunk_tensor(Kcross)

        # validate
        self.assertEqual(diagonal_variance.shape, (test_count,))
        for i in range(test_count):
            _consistent_assert(
                self.assertAlmostEqual,
                diagonal_variance[i],
                1.0
                - Kcross[i, :]
                @ np.linalg.solve(
                    K[i, :, :] + muygps.eps() * np.eye(nn_count),
                    Kcross[i, :],
                ),
            )
            self.assertGreater(diagonal_variance[i], 0.0)


class MakeClassifierTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 1000, 10, b, n, nn_kwargs, lm, rt, k_kwargs)
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lm in ["mse"]
            for rt in [True, False]
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                    "length_scale": {"val": 1.5},
                    "eps": {"val": 1e-5},
                },
            )
        )
    )
    def test_make_multivariate_classifier(
        self,
        train_count,
        test_count,
        feature_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_method,
        return_distances,
        k_kwargs,
    ):
        response_count = 2
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )

        classifier_args = make_classifier(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            return_distances=return_distances,
            verbose=False,
        )

        if len(classifier_args) == 2:
            muygps, _ = classifier_args
        elif len(classifier_args) == 4:
            muygps, _, crosswise_dists, pairwise_dists = classifier_args
            crosswise_dists = _consistent_unchunk_tensor(crosswise_dists)
            pairwise_dists = _consistent_unchunk_tensor(pairwise_dists)
            self.assertEqual(crosswise_dists.shape, (batch_count, nn_count))
            self.assertEqual(
                pairwise_dists.shape, (batch_count, nn_count, nn_count)
            )
        print(k_kwargs)
        for key in k_kwargs:
            if key == "eps":
                self.assertEqual(k_kwargs[key]["val"], muygps.eps())
            elif key == "kern":
                self.assertEqual(k_kwargs[key], muygps.kern)
            elif key == "metric":
                self.assertEqual(k_kwargs[key], muygps.kernel.metric)
            elif isinstance(k_kwargs[key]["val"], str):
                print(
                    f"optimized to find value "
                    f"{muygps.kernel.hyperparameters[key]()}"
                )
            else:
                self.assertEqual(
                    k_kwargs[key]["val"],
                    muygps.kernel.hyperparameters[key](),
                )


class MakeRegressorTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 1000, 10, b, n, nn_kwargs, lm, ssm, rt, k_kwargs)
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lm in ["mse"]
            # for ssm in ["analytic"]
            # for rt in [True]
            for ssm in ["analytic", None]
            for rt in [True, False]
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                    # "nu": {"val": 0.38},
                    "length_scale": {"val": 1.5},
                    "eps": {"val": 1e-5},
                },
            )
        )
    )
    def test_make_multivariate_regressor(
        self,
        train_count,
        test_count,
        feature_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_method,
        sigma_method,
        return_distances,
        k_kwargs,
    ):
        response_count = 1
        # construct the observation locations
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=False,
        )

        regressor_args = make_regressor(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            sigma_method=sigma_method,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            return_distances=return_distances,
        )

        if len(regressor_args) == 2:
            muygps, _ = regressor_args
        elif len(regressor_args) == 4:
            muygps, _, crosswise_dists, pairwise_dists = regressor_args
            crosswise_dists = _consistent_unchunk_tensor(crosswise_dists)
            pairwise_dists = _consistent_unchunk_tensor(pairwise_dists)
            self.assertEqual(crosswise_dists.shape, (batch_count, nn_count))
            self.assertEqual(
                pairwise_dists.shape, (batch_count, nn_count, nn_count)
            )

        for key in k_kwargs:
            if key == "eps":
                self.assertEqual(k_kwargs[key]["val"], muygps.eps())
            elif key == "kern":
                self.assertEqual(k_kwargs[key], muygps.kern)
            elif key == "metric":
                self.assertEqual(k_kwargs[key], muygps.kernel.metric)
            elif k_kwargs[key]["val"] == "sample":
                print(
                    f"\toptimized {key} to find value "
                    f"{muygps.kernel.hyperparameters[key]()}"
                )
            else:
                self.assertEqual(
                    k_kwargs[key]["val"],
                    muygps.kernel.hyperparameters[key](),
                )
        if sigma_method is None:
            self.assertFalse(muygps.sigma_sq.trained())
            self.assertEqual(np.array([1.0]), muygps.sigma_sq())
        else:
            self.assertTrue(muygps.sigma_sq.trained())
            print(f"\toptimized sigma_sq to find value " f"{muygps.sigma_sq()}")


class GPSigmaSqTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, r, sm, 10, nn_kwargs, k_kwargs)
            for f in [100, 1]
            for r in [10, 2, 1]
            for sm in ["analytic"]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "eps": {"val": 1e-5},
                    "nu": {"val": 1.0},
                    "length_scale": {"val": 7.2},
                },
                {
                    "kern": "rbf",
                    "metric": "F2",
                    "eps": {"val": 1e-5},
                    "length_scale": {"val": 1.5},
                },
            )
            # for f in [100]
            # for r in [10]
            # for nn_kwargs in _exact_nn_kwarg_options
        )
    )
    def test_batch_sigma_sq_shapes(
        self,
        data_count,
        feature_count,
        response_count,
        sigma_method,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(**k_kwargs)

        # prepare data
        data = _make_gaussian_dict(data_count, feature_count, response_count)

        # neighbors and distances
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices = np.arange(data_count)
        nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
        (_, F2_dists, _, nn_targets) = make_train_tensors(
            muygps.kernel.metric,
            indices,
            nn_indices,
            data["input"],
            data["output"],
        )

        K = muygps.kernel(F2_dists)
        muygps = muygps_sigma_sq_optim(
            muygps, F2_dists, nn_targets, sigma_method=sigma_method
        )

        K = _consistent_unchunk_tensor(K)
        nn_targets = _consistent_unchunk_tensor(nn_targets)

        if response_count > 1:
            self.assertEqual(len(muygps.sigma_sq()), response_count)
            for i in range(response_count):
                sigmas = _get_sigma_sq_series(
                    K,
                    nn_targets[:, :, i].reshape(data_count, nn_count, 1),
                    muygps.eps(),
                )
                self.assertEqual(sigmas.shape, (data_count,))
                self.assertAlmostEqual(muygps.sigma_sq()[i], np.mean(sigmas), 5)
        else:
            sigmas = _get_sigma_sq_series(
                K,
                nn_targets[:, :, 0].reshape(data_count, nn_count, 1),
                muygps.eps(),
            )
            self.assertEqual(sigmas.shape, (data_count,))
            self.assertAlmostEqual(muygps.sigma_sq()[0], np.mean(sigmas), 5)


if __name__ == "__main__":
    absltest.main()
