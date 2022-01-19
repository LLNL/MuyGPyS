# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.gp.distance import pairwise_distances, crosswise_distances
from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import (
    sample_batch,
    sample_balanced_batch,
    full_filtered_batch,
)
from MuyGPyS.optimize.chassis import (
    scipy_optimize_from_tensors,
    scipy_optimize_from_indices,
)
from MuyGPyS.testing.gp import (
    benchmark_pairwise_distances,
    benchmark_sample,
    benchmark_sample_full,
    BenchmarkGP,
    get_analytic_sigma_sq,
)

from MuyGPyS.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
    _sq_rel_err,
)


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
        nbrs_lookup = NN_Wrapper(data, nn_count, **nn_kwargs)
        indices, nn_indices = sample_batch(nbrs_lookup, batch_count, data_count)
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
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["labels"])
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["labels"][nn_indices[i, :]])), 1
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
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        target_count = np.min((data_count, batch_count))
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["labels"][nn_indices[i, :]])), 1
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
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        target_count = np.min((data_count, batch_count))
        hist, _ = np.array(
            np.histogram(data["labels"][indices], bins=response_count)
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
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        target_count = np.min((data_count, batch_count))
        hist, _ = np.array(
            np.histogram(data["labels"][indices], bins=response_count)
        )
        self.assertGreaterEqual(
            np.mean(hist) + 0.1 * (target_count / response_count),
            target_count / response_count,
        )
        self.assertGreaterEqual(
            np.min(hist) + 0.45 * (target_count / response_count),
            target_count / response_count,
        )


class ObjectiveTest(parameterized.TestCase):
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
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["labels"])
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["labels"][nn_indices[i, :]])), 1
            )


class BalancedBatchTest(parameterized.TestCase):
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
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["labels"])
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, ind in enumerate(indices):
            self.assertNotEqual(
                len(np.unique(data["labels"][nn_indices[i, :]])), 1
            )


class GPSigmaSqBaselineTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1001,
                5,
                ss,
                k_kwargs,
            )
            for ss in [(1.0, 5e-2), (0.02453, 5e-2), (19.32, 5e-2)]
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": 0.38},
                    "length_scale": {"val": 1.5},
                    "eps": {"val": 1e-5},
                },
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": 2.5},
                    "length_scale": {"val": 1.5},
                    "eps": {"val": 1e-5},
                },
                {
                    "kern": "rbf",
                    "metric": "F2",
                    "length_scale": {"val": 1.5},
                    "eps": {"val": 1e-5},
                },
            )
        )
    )
    def test_baseline_sigma_sq_optim(
        self,
        data_count,
        its,
        sigma_sq,
        k_kwargs,
    ):
        sigma_sq, tol = sigma_sq
        x = np.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
        mrse = 0.0
        gp = BenchmarkGP(**k_kwargs)
        gp._set_sigma_sq(sigma_sq)
        pairwise_dists = benchmark_pairwise_distances(
            x, metric=gp.kernel.metric
        )
        K = gp.kernel(pairwise_dists) + gp.eps() * np.eye(data_count)
        for _ in range(its):
            y = benchmark_sample(gp, x)[:, 0]
            ss = get_analytic_sigma_sq(K, y)
            mrse += _sq_rel_err(sigma_sq, ss)
        mrse /= its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, tol)


class GPSigmaSqOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, 5, b, n, nn_kwargs, ss, k_kwargs)
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for ss in ((1.0, 5e-2), (0.002453, 5e-2), (19.32, 5e-2))
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": 0.3},
                    "length_scale": {"val": 1e-2},
                    "eps": {"val": 1e-5},
                },
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": 2.5},
                    "length_scale": {"val": 1e-2},
                    "eps": {"val": 1e-5},
                },
                {
                    "kern": "rbf",
                    "metric": "F2",
                    "length_scale": {"val": 1e-2},
                    "eps": {"val": 1e-5},
                },
            )
        )
    )
    def test_sigma_sq_optim(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        sigma_sq,
        k_kwargs,
    ):
        sigma_sq, tol = sigma_sq
        muygps = MuyGPS(**k_kwargs)

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count, _ = sim_train["input"].shape
        test_count, _ = sim_test["input"].shape

        mrse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
        pairwise_dists = pairwise_distances(
            sim_train["input"], batch_nn_indices, metric=muygps.kernel.metric
        )
        K = muygps.kernel(pairwise_dists)

        for _ in range(its):
            # Make GP benchmark.
            gp = BenchmarkGP(**k_kwargs)
            gp._set_sigma_sq(sigma_sq)

            # Simulate the response
            y = benchmark_sample_full(gp, sim_test["input"], sim_train["input"])
            sim_test["output"] = y[:test_count].reshape(test_count, 1)
            sim_train["output"] = y[test_count:].reshape(train_count, 1)

            # Find MuyGPyS optim
            muygps.sigma_sq_optim(K, batch_nn_indices, sim_train["output"])
            estimate = muygps.sigma_sq()[0]

            mrse += _sq_rel_err(sigma_sq, estimate)
        mrse /= its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, tol)


class GPOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, 10, b, n, nn_kwargs, lm, k_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for k_kwargs in (
                (
                    0.38,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-5},
                    },
                ),
            )
        )
    )
    def test_hyper_optim(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_method,
        k_kwargs,
    ):
        target, kwargs = k_kwargs

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        mrse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
        # nn_indices, _ = nbrs_lookup.get_nns(sim_test["input"])
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
        crosswise_dists = crosswise_distances(
            sim_train["input"],
            sim_train["input"],
            batch_indices,
            batch_nn_indices,
            metric=kwargs["metric"],
        )
        pairwise_dists = pairwise_distances(
            sim_train["input"], batch_nn_indices, metric=kwargs["metric"]
        )

        for _ in range(its):
            # Make GP benchmark.
            gp_kwargs = kwargs.copy()
            gp_kwargs["nu"]["val"] = target
            gp = BenchmarkGP(**gp_kwargs)

            # Simulate the response
            y = benchmark_sample_full(gp, sim_test["input"], sim_train["input"])
            sim_test["output"] = y[:test_count].reshape(test_count, 1)
            sim_train["output"] = y[test_count:].reshape(train_count, 1)

            # set up MuyGPS object
            muygps = MuyGPS(**kwargs)

            batch_targets = sim_train["output"][batch_indices, :]
            batch_nn_targets = sim_train["output"][batch_nn_indices, :]

            muygps = scipy_optimize_from_tensors(
                muygps,
                batch_targets,
                batch_nn_targets,
                crosswise_dists,
                pairwise_dists,
                loss_method=loss_method,
            )

            # mse += (estimate - target) ** 2
            estimate = muygps.kernel.hyperparameters["nu"]()
            mrse += _sq_rel_err(target, estimate)
        mrse /= its
        print(f"optimizes with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mrse, 0.0, 0)

    @parameterized.parameters(
        (
            (1001, b, n, nn_kwargs, lm, k_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for k_kwargs in (
                (
                    0.38,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-5},
                    },
                ),
            )
        )
    )
    def test_hyper_optim_from_indices(
        self,
        data_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_method,
        k_kwargs,
    ):
        target, kwargs = k_kwargs

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
        # nn_indices, _ = nbrs_lookup.get_nns(sim_test["input"])
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
        # Make GP benchmark.
        gp_kwargs = kwargs.copy()
        gp_kwargs["nu"]["val"] = target
        gp = BenchmarkGP(**gp_kwargs)

        # Sample a response curve
        y = benchmark_sample_full(gp, sim_test["input"], sim_train["input"])
        sim_test["output"] = y[:test_count].reshape(test_count, 1)
        sim_train["output"] = y[test_count:].reshape(train_count, 1)

        # set up MuyGPS object
        muygps = MuyGPS(**kwargs)

        muygps = scipy_optimize_from_indices(
            muygps,
            batch_indices,
            batch_nn_indices,
            sim_train["input"],
            sim_train["output"],
            loss_method=loss_method,
        )

        estimate = muygps.kernel.hyperparameters["nu"]()

        rse = _sq_rel_err(target, estimate)
        print(f"optimizes with relative squared error {rse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(rse, 0.0, 0)


if __name__ == "__main__":
    absltest.main()
