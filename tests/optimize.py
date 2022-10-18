# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np


from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

from MuyGPyS.gp.distance import (
    pairwise_distances,
    crosswise_distances,
    make_train_tensors,
)
from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import (
    sample_batch,
    sample_balanced_batch,
    full_filtered_batch,
)
from MuyGPyS.optimize.chassis import (
    optimize_from_tensors,
    optimize_from_indices,
)
from MuyGPyS.optimize.loss import get_loss_func
from MuyGPyS.optimize.objective import (
    make_loo_crossval_fn,
)
from MuyGPyS.optimize.sigma_sq import muygps_sigma_sq_optim
from MuyGPyS._test.gp import (
    benchmark_pairwise_distances,
    benchmark_sample,
    benchmark_sample_full,
    BenchmarkGP,
    get_analytic_sigma_sq,
)
from MuyGPyS._test.utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _basic_nn_kwarg_options,
    _exact_nn_kwarg_options,
    _advanced_opt_method_and_kwarg_options,
    _sq_rel_err,
)
from MuyGPyS._src.mpi_utils import _consistent_chunk_tensor


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
        # target_count = np.min((data_count, batch_count))
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, _ in enumerate(indices):
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
        indices, _ = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        # target_count = np.min((data_count, batch_count))
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
            (1001, 5, b, n, nn_kwargs, ss, sm, k_kwargs)
            for b in [250]
            for n in [34]
            # for nn_kwargs in _basic_nn_kwarg_options
            # for ss in ((1.0, 5e-2), (0.002453, 5e-2), (19.32, 5e-2))
            for sm in ["analytic"]
            for k_kwargs in (
                {
                    "kern": "matern",
                    "metric": "l2",
                    "nu": {"val": 0.3},
                    "length_scale": {"val": 1e-2},
                    "eps": {"val": 1e-5},
                },
                # {
                #     "kern": "matern",
                #     "metric": "l2",
                #     "nu": {"val": 2.5},
                #     "length_scale": {"val": 1e-2},
                #     "eps": {"val": 1e-5},
                # },
                # {
                #     "kern": "rbf",
                #     "metric": "F2",
                #     "length_scale": {"val": 1e-2},
                #     "eps": {"val": 1e-5},
                # },
            )
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for ss in [(1.0, 5e-2)]
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
        sigma_method,
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

            batch_nn_targets = _consistent_chunk_tensor(
                sim_train["output"][batch_nn_indices, :]
            )

            # Find MuyGPyS optim
            muygps = muygps_sigma_sq_optim(
                muygps,
                pairwise_dists,
                batch_nn_targets,
                sigma_method=sigma_method,
            )
            estimate = muygps.sigma_sq()[0]

            mrse += _sq_rel_err(sigma_sq, estimate)
        mrse /= its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, tol)


class GPTensorsOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, 10, b, n, nn_kwargs, lm, om, opt_method_and_kwargs, k_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _advanced_opt_method_and_kwarg_options
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
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _advanced_opt_method_and_kwarg_options[0]
            # ]
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
        obj_method,
        opt_method_and_kwargs,
        k_kwargs,
    ):
        target, kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs

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

            batch_targets = _consistent_chunk_tensor(
                sim_train["output"][batch_indices, :]
            )
            batch_nn_targets = _consistent_chunk_tensor(
                sim_train["output"][batch_nn_indices, :]
            )

            muygps = optimize_from_tensors(
                muygps,
                batch_targets,
                batch_nn_targets,
                crosswise_dists,
                pairwise_dists,
                loss_method=loss_method,
                obj_method=obj_method,
                opt_method=opt_method,
                **opt_kwargs,
            )

            # mse += (estimate - target) ** 2
            estimate = muygps.kernel.hyperparameters["nu"]()
            mrse += _sq_rel_err(target, estimate)
        mrse /= its
        print(f"optimizes with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mrse, 0.0, 0)


class GPIndicesOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, b, n, nn_kwargs, lm, om, opt_method_and_kwargs, k_kwargs)
            for b in [250]
            for n in [20]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _advanced_opt_method_and_kwarg_options[0]
            # ]
            for nn_kwargs in _basic_nn_kwarg_options
            for opt_method_and_kwargs in _advanced_opt_method_and_kwarg_options
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
        obj_method,
        opt_method_and_kwargs,
        k_kwargs,
    ):
        target, kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs

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

        muygps = optimize_from_indices(
            muygps,
            batch_indices,
            batch_nn_indices,
            sim_train["input"],
            sim_train["output"],
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            **opt_kwargs,
        )

        estimate = muygps.kernel.hyperparameters["nu"]()

        rse = _sq_rel_err(target, estimate)
        print(f"optimizes with relative squared error {rse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(rse, 0.0, 0)


class MethodsAgreementTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        data_count = 1001
        feature_count = 10
        response_count = 2
        batch_count = 250
        nn_count = 20

        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(
            data["input"], nn_count, **_exact_nn_kwarg_options[0]
        )
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, data_count
        )

        (
            cls.crosswise_dists,
            cls.pairwise_dists,
            cls.batch_targets,
            cls.batch_nn_targets,
        ) = make_train_tensors(
            "l2", batch_indices, batch_nn_indices, data["input"], data["output"]
        )

    def _make_x0(self, params):
        x0 = list()
        if "nu" in params:
            x0.append(params["nu"])
        if "length_scale" in params:
            x0.append(params["length_scale"])
        if "eps" in params:
            x0.append(params["eps"])
        return x0

    @parameterized.parameters(
        (
            (lm, k_kwargs)
            for lm in ["mse"]
            for k_kwargs in (
                (
                    {"nu": 0.38},
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-5},
                    },
                ),
                (
                    {"nu": 0.38, "length_scale": 1.5, "eps": 1e-5},
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {
                            "val": "sample",
                            "bounds": (1e-2, 1e0),
                        },
                        "eps": {"val": "sample", "bounds": (1e-6, 1e-3)},
                    },
                ),
            )
        )
    )
    def test_kernel_fn(self, loss_method, k_kwargs):
        loss_fn = get_loss_func(loss_method)
        params, k_kwargs = k_kwargs
        muygps = MuyGPS(**k_kwargs)

        x0 = self._make_x0(params)

        array_kernel_fn = muygps.kernel.get_array_opt_fn()
        kwargs_kernel_fn = muygps.kernel.get_kwargs_opt_fn()

        K_array = array_kernel_fn(self.pairwise_dists, x0)
        Kcross_array = array_kernel_fn(self.crosswise_dists, x0)

        K_kwargs = kwargs_kernel_fn(self.pairwise_dists, **params)
        Kcross_kwargs = kwargs_kernel_fn(self.crosswise_dists, **params)

        self.assertTrue(np.allclose(K_array, K_kwargs))
        self.assertTrue(np.allclose(Kcross_array, Kcross_kwargs))

        array_predict_fn = muygps.get_array_opt_fn()
        kwargs_predict_fn = muygps.get_kwargs_opt_fn()

        predictions_array = array_predict_fn(
            K_array, Kcross_array, self.batch_nn_targets, x0
        )
        predictions_kwargs = kwargs_predict_fn(
            K_kwargs, Kcross_kwargs, self.batch_nn_targets, **params
        )

        self.assertTrue(np.allclose(predictions_array, predictions_kwargs))

        array_obj_fn = make_loo_crossval_fn(
            "scipy",
            loss_method,
            loss_fn,
            array_kernel_fn,
            array_predict_fn,
            self.pairwise_dists,
            self.crosswise_dists,
            self.batch_nn_targets,
            self.batch_targets,
        )
        kwargs_obj_fn = make_loo_crossval_fn(
            "bayes",
            loss_method,
            loss_fn,
            kwargs_kernel_fn,
            kwargs_predict_fn,
            self.pairwise_dists,
            self.crosswise_dists,
            self.batch_nn_targets,
            self.batch_targets,
        )

        array_val = array_obj_fn(x0)
        kwargs_val = kwargs_obj_fn(**params)

        self.assertAlmostEqual(array_val, -kwargs_val)


if __name__ == "__main__":
    absltest.main()
