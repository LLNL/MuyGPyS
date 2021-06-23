# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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
from MuyGPyS.testing.gp import BenchmarkGP
from MuyGPyS.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
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
                10,
                k_dict[0],
                k_dict[1],
            )
            for k_dict in (
                (
                    "matern",
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                    },
                ),
                (
                    "rbf",
                    {"length_scale": 1.5, "eps": 0.00001},
                ),
                (
                    "nngp",
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.00001,
                    },
                ),
            )
        )
    )
    def test_baseline_sigma_sq_optim(
        self,
        data_count,
        its,
        kern,
        hyper_dict,
    ):
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2]
        sim_test["input"] = x[1::2]
        mean_sigma_sq = 0.0
        for i in range(its):
            gp = BenchmarkGP(kern=kern, **hyper_dict)
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            mean_sigma_sq += gp.get_sigma_sq(y)
        mean_sigma_sq /= its
        self.assertAlmostEqual(mean_sigma_sq, 1.0, 1)


class GPSigmaSqOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, 10, b, n, nn_kwargs, e, k_kwargs)
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for e in (({"val": 1e-5},))
            for k_kwargs in (
                (
                    "matern",
                    "l2",
                    {
                        "nu": {"val": 0.38},
                        "length_scale": {"val": 1.5},
                    },
                ),
                (
                    "rbf",
                    "F2",
                    {"length_scale": {"val": 1.5}},
                ),
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
        eps,
        k_kwargs,
    ):
        kern, metric, kwargs = k_kwargs
        muygps = MuyGPS(kern=kern, eps=eps, metric=metric, **kwargs)

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        mse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
        # nn_indices, _ = nbrs_lookup.get_nns(sim_test["input"])
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
        pairwise_dists = pairwise_distances(
            sim_train["input"], batch_nn_indices, metric=metric
        )
        K = muygps.kernel(pairwise_dists)

        for _ in range(its):
            # Make GP benchmark. Still uses old API.
            hyper_dict = {key: kwargs[key]["val"] for key in kwargs}
            hyper_dict["eps"] = eps["val"]
            hyper_dict["sigma_sq"] = np.array([1.0])
            # print(hyper_dict)
            gp = BenchmarkGP(kern=kern, **hyper_dict)

            # Simulate the response
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            sim_train["output"] = y[:train_count].reshape(train_count, 1)
            sim_test["output"] = y[train_count:].reshape(test_count, 1)

            # Ground truth sigma sq
            target = gp.get_sigma_sq(y)

            # Find MuyGPyS optim
            muygps.sigma_sq_optim(K, batch_nn_indices, sim_train["output"])
            estimate = muygps.sigma_sq()

            mse += (estimate - target) ** 2
        mse /= its
        print(f"optimizes with mse {mse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mse, 0.0, 1)


class GPOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, 10, b, n, nn_kwargs, lm, e, k_kwargs)
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for e in (({"val": 1e-5},))
            for k_kwargs in (
                (
                    "matern",
                    "l2",
                    0.38,
                    {
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {"val": 1.5},
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
        eps,
        k_kwargs,
    ):
        kern, metric, target, kwargs = k_kwargs

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        mse = 0.0

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
            metric=metric,
        )
        pairwise_dists = pairwise_distances(
            sim_train["input"], batch_nn_indices, metric=metric
        )

        for i in range(its):
            # Make GP benchmark. Still uses old API.
            hyper_dict = {
                key: kwargs[key]["val"]
                if not isinstance(kwargs[key]["val"], str)
                else target
                for key in kwargs
            }
            hyper_dict["eps"] = eps["val"]
            hyper_dict["sigma_sq"] = np.array([1.0])
            gp = BenchmarkGP(kern=kern, **hyper_dict)

            # Simulate the response
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            sim_train["output"] = y[:train_count].reshape(train_count, 1)
            sim_test["output"] = y[train_count:].reshape(test_count, 1)

            # set up MuyGPS object
            muygps = MuyGPS(kern=kern, eps=eps, metric=metric, **kwargs)

            estimate = scipy_optimize_from_tensors(
                muygps,
                batch_indices,
                batch_nn_indices,
                crosswise_dists,
                pairwise_dists,
                sim_train["output"],
                loss_method=loss_method,
            )[0]

            mse += (estimate - target) ** 2
        mse /= its
        print(f"optimizes with mse {mse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mse, 0.0, 1)

    @parameterized.parameters(
        (
            (1001, b, n, nn_kwargs, lm, e, k_kwargs)
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for e in (({"val": 1e-5},))
            for k_kwargs in (
                (
                    "matern",
                    "l2",
                    0.38,
                    {
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {"val": 1.5},
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
        eps,
        k_kwargs,
    ):
        kern, metric, target, kwargs = k_kwargs

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        mse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
        # nn_indices, _ = nbrs_lookup.get_nns(sim_test["input"])
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
        # Make GP benchmark. Still uses old API.
        hyper_dict = {
            key: kwargs[key]["val"]
            if not isinstance(kwargs[key]["val"], str)
            else target
            for key in kwargs
        }
        hyper_dict["eps"] = eps["val"]
        hyper_dict["sigma_sq"] = np.array([1.0])
        gp = BenchmarkGP(kern=kern, **hyper_dict)

        # Simulate the response
        gp.fit(sim_train["input"], sim_test["input"])
        y = gp.simulate()
        sim_train["output"] = y[:train_count].reshape(train_count, 1)
        sim_test["output"] = y[train_count:].reshape(test_count, 1)

        # set up MuyGPS object
        muygps = MuyGPS(kern=kern, eps=eps, metric=metric, **kwargs)

        estimate = scipy_optimize_from_indices(
            muygps,
            batch_indices,
            batch_nn_indices,
            sim_train["input"],
            sim_train["input"],
            sim_train["output"],
            loss_method=loss_method,
        )[0]

        mse += (estimate - target) ** 2
        print(f"optimizes with mse {mse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mse, 0.0, 1)


if __name__ == "__main__":
    absltest.main()
