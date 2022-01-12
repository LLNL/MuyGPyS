# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.examples.classify import make_multivariate_classifier, classify_any
from MuyGPyS.examples.regress import make_multivariate_regressor, regress_any
from MuyGPyS.gp.distance import pairwise_distances, crosswise_distances
from MuyGPyS.gp.muygps import MultivariateMuyGPS as MMuyGPS
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.chassis import (
    scipy_optimize_from_indices,
    scipy_optimize_from_tensors,
)
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.testing.gp import (
    benchmark_prepare_cholK,
    benchmark_sample_from_cholK,
    benchmark_sample_full,
    BenchmarkGP,
)
from MuyGPyS.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
    _fast_nn_kwarg_options,
)


class InitTest(parameterized.TestCase):
    @parameterized.parameters(
        (model_args)
        for model_args in (
            (
                "matern",
                [
                    {
                        "nu": {"val": 1.0},
                        "length_scale": {"val": 7.2},
                        "eps": {"val": 1e-5},
                    },
                    {
                        "nu": {"val": 1.2},
                        "length_scale": {"val": 2.2},
                        "eps": {"val": 1e-6},
                    },
                ],
            ),
            (
                "matern",
                [
                    {
                        "nu": {"val": 1.0},
                        "length_scale": {"val": 7.2},
                        "eps": {"val": 1e-5},
                    },
                ],
            ),
        )
    )
    def test_bounds_defaults_init(self, kern, model_args):
        # kern, kwargs = k_kwargs
        mmuygps = MMuyGPS(kern, *model_args)
        self.assertEqual(len(mmuygps.models), len(model_args))
        self.assertEqual(mmuygps.kern, kern)
        for i, muygps in enumerate(mmuygps.models):
            this_kwargs = model_args[i]
            for param in this_kwargs:
                if param == "eps":
                    continue
                self.assertEqual(
                    this_kwargs[param]["val"],
                    muygps.kernel.hyperparameters[param](),
                )
                self.assertEqual(
                    "fixed",
                    muygps.kernel.hyperparameters[param].get_bounds(),
                )
            self.assertEqual(this_kwargs["eps"]["val"], muygps.eps())
            self.assertEqual("fixed", muygps.eps.get_bounds())
            self.assertEqual("unlearned", muygps.sigma_sq())


class SigmaSqTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, 10, nn_kwargs, model_args)
            for nn_kwargs in _basic_nn_kwarg_options
            for f in [100]
            for model_args in (
                (
                    "matern",
                    [
                        {
                            "nu": {"val": 1.0},
                            "length_scale": {"val": 7.2},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 1.2},
                            "length_scale": {"val": 2.2},
                            "eps": {"val": 1e-6},
                        },
                        {
                            "nu": {"val": 0.38},
                            "length_scale": {"val": 12.4},
                            "eps": {"val": 1e-6},
                        },
                    ],
                ),
            )
        )
    )
    def test_batch_sigma_sq_shapes(
        self,
        data_count,
        feature_count,
        nn_count,
        nn_kwargs,
        model_args,
    ):
        kern, args = model_args
        response_count = len(args)
        mmuygps = MMuyGPS(kern, *args)

        # prepare data
        data = _make_gaussian_dict(data_count, feature_count, response_count)

        # neighbors and distances
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices = np.arange(data_count)
        nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
        pairwise_dists = pairwise_distances(
            data["input"], nn_indices, metric=mmuygps.metric
        )

        # fit sigmas
        mmuygps.sigma_sq_optim(pairwise_dists, nn_indices, data["output"])

        K = np.zeros((data_count, nn_count, nn_count))
        for i, muygps in enumerate(mmuygps.models):
            K = muygps.kernel(pairwise_dists)
            sigmas = muygps._get_sigma_sq_series(
                K, nn_indices, data["output"][:, i]
            )
            self.assertEqual(sigmas.shape, (data_count,))
            self.assertAlmostEqual(muygps.sigma_sq()[0], np.mean(sigmas), 5)


class OptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1001, 10, b, n, nn_kwargs, lm, k_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for k_kwargs in (
                (
                    "matern",
                    "l2",
                    [0.38, 0.78],
                    [
                        {
                            "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                            "length_scale": {"val": 0.7},
                            "eps": {"val": 1e-5},
                        },
                    ],
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
        kern, metric, target, args = k_kwargs
        response_count = len(args)

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        mse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
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

        gp_args = args.copy()
        for i, m in enumerate(gp_args):
            m["nu"]["val"] = target[i]
        gps = [BenchmarkGP(kern=kern, **a) for a in gp_args]
        cholKs = [
            benchmark_prepare_cholK(
                gp, np.vstack((sim_test["input"], sim_train["input"]))
            )
            for gp in gps
        ]
        for _ in range(its):
            # Simulate the response
            sim_test["output"] = np.zeros((test_count, response_count))
            sim_train["output"] = np.zeros((train_count, response_count))
            for i, cholK in enumerate(cholKs):
                y = benchmark_sample_from_cholK(cholK)
                sim_test["output"][:, i] = y[:test_count, 0]
                sim_train["output"][:, i] = y[test_count:, 0]

            mmuygps = MMuyGPS(kern, *args)

            batch_targets = sim_train["output"][batch_indices, :]
            batch_nn_targets = sim_train["output"][batch_nn_indices, :]

            for i, muygps in enumerate(mmuygps.models):
                mmuygps.models[i] = scipy_optimize_from_tensors(
                    muygps,
                    batch_targets[:, i].reshape(batch_count, 1),
                    batch_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
                    crosswise_dists,
                    pairwise_dists,
                    loss_method=loss_method,
                )
                estimate = mmuygps.models[i].kernel.hyperparameters["nu"]()
                mse += np.sum(estimate - target[i]) ** 2
        mse /= its * response_count
        print(f"optimizes with mse {mse}")
        self.assertAlmostEqual(mse, 0.0, 1)

    @parameterized.parameters(
        (
            (1001, 10, b, n, nn_kwargs, lm, k_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for k_kwargs in (
                (
                    "matern",
                    "l2",
                    [0.38, 0.78],
                    [
                        {
                            "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                            "length_scale": {"val": 0.7},
                            "eps": {"val": 1e-5},
                        },
                    ],
                ),
            )
        )
    )
    def test_hyper_optim_from_indices(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_method,
        k_kwargs,
    ):
        kern, metric, target, args = k_kwargs
        response_count = len(args)

        # construct the observation locations
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]

        mse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )

        gp_args = args.copy()
        for i, m in enumerate(gp_args):
            m["nu"]["val"] = target[i]
        gps = [BenchmarkGP(kern=kern, **a) for a in gp_args]
        cholKs = [
            benchmark_prepare_cholK(
                gp, np.vstack((sim_test["input"], sim_train["input"]))
            )
            for gp in gps
        ]
        for _ in range(its):
            # Simulate the response
            sim_test["output"] = np.zeros((test_count, response_count))
            sim_train["output"] = np.zeros((train_count, response_count))
            for i, cholK in enumerate(cholKs):
                y = benchmark_sample_from_cholK(cholK)
                sim_test["output"][:, i] = y[:test_count, 0]
                sim_train["output"][:, i] = y[test_count:, 0]

            mmuygps = MMuyGPS(kern, *args)

            for i, muygps in enumerate(mmuygps.models):
                mmuygps.models[i] = scipy_optimize_from_indices(
                    muygps,
                    batch_indices,
                    batch_nn_indices,
                    sim_train["input"],
                    sim_train["output"][:, i].reshape(train_count, 1),
                    loss_method=loss_method,
                )
                estimate = mmuygps.models[i].kernel.hyperparameters["nu"]()
                mse += np.sum(estimate - target[i]) ** 2
        mse /= its * response_count
        print(f"optimizes with mse {mse}")
        self.assertAlmostEqual(mse, 0.0, 1)


class ClassifyTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, f, nn, nn_kwargs, k_kwargs)
            for f in [100, 10, 2]
            for nn in [5, 10, 100]
            # for f in [10]
            # for nn in [5]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                (
                    "matern",
                    # [0.38, 0.78],
                    [
                        {
                            "nu": {"val": 0.38},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 0.79},
                            "length_scale": {"val": 0.7},
                            "eps": {"val": 1e-5},
                        },
                    ],
                ),
            )
        )
    )
    def test_classify_any(
        self,
        train_count,
        test_count,
        feature_count,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        kern, args = k_kwargs
        response_count = len(args)

        mmuygps = MMuyGPS(kern, *args)

        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)

        predictions, _ = classify_any(
            mmuygps,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
        )
        self.assertEqual(predictions.shape, (test_count, response_count))


class RegressTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, f, nn, vm, nn_kwargs, k_kwargs)
            for f in [100, 2]
            for nn in [5, 10]
            for vm in [None, "diagonal"]
            # for f in [2]
            # for nn in [5]
            # for vm in ["diagonal"]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for k_kwargs in (
                (
                    "matern",
                    [
                        {
                            "nu": {"val": 0.38},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 0.79},
                            "length_scale": {"val": 0.7},
                            "eps": {"val": 1e-5},
                        },
                    ],
                ),
            )
        )
    )
    def test_regress_any(
        self,
        train_count,
        test_count,
        feature_count,
        nn_count,
        variance_mode,
        nn_kwargs,
        k_kwargs,
    ):
        kern, args = k_kwargs
        response_count = len(args)

        mmuygps = MMuyGPS(kern, *args)

        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)

        self.assertEqual(mmuygps.sigma_sq(), "unlearned")

        predictions, _ = regress_any(
            mmuygps,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
            variance_mode=variance_mode,
            apply_sigma_sq=False,
        )
        if variance_mode is not None:
            predictions, diagonal_variance = predictions
            self.assertEqual(
                diagonal_variance.shape, (test_count, response_count)
            )
        self.assertEqual(predictions.shape, (test_count, response_count))


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
                (
                    "matern",
                    [
                        {
                            "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 0.8},
                            "length_scale": {"val": 0.7},
                            "eps": {"val": 1e-5},
                        },
                    ],
                ),
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
        kern, args = k_kwargs
        response_count = len(args)

        # construct the observation locations
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )

        classifier_args = make_multivariate_classifier(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            nn_kwargs=nn_kwargs,
            kern=kern,
            k_args=args,
            return_distances=return_distances,
        )

        if len(classifier_args) == 2:
            mmuygps, _ = classifier_args
        elif len(classifier_args) == 4:
            mmuygps, _, crosswise_dists, pairwise_dists = classifier_args
            self.assertEqual(crosswise_dists.shape, (batch_count, nn_count))
            self.assertEqual(
                pairwise_dists.shape, (batch_count, nn_count, nn_count)
            )

        for i, muygps in enumerate(mmuygps.models):
            for key in args[i]:
                if key == "eps":
                    self.assertEqual(args[i][key]["val"], muygps.eps())
                elif isinstance(args[i][key]["val"], str):
                    print(
                        f"optimized to find value "
                        f"{muygps.kernel.hyperparameters[key]()}"
                    )
                else:
                    self.assertEqual(
                        args[i][key]["val"],
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
            for ssm in ["analytic", None]
            for rt in [True, False]
            for k_kwargs in (
                (
                    "matern",
                    [
                        {
                            "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 0.8},
                            "length_scale": {"val": 0.7},
                            "eps": {"val": 1e-5},
                        },
                    ],
                ),
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
        kern, args = k_kwargs
        response_count = len(args)

        # construct the observation locations
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=False,
        )

        regressor_args = make_multivariate_regressor(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            sigma_method=sigma_method,
            nn_kwargs=nn_kwargs,
            kern=kern,
            k_args=args,
            return_distances=return_distances,
        )

        if len(regressor_args) == 2:
            mmuygps, _ = regressor_args
        elif len(regressor_args) == 4:
            mmuygps, _, crosswise_dists, pairwise_dists = regressor_args
            self.assertEqual(crosswise_dists.shape, (batch_count, nn_count))
            self.assertEqual(
                pairwise_dists.shape, (batch_count, nn_count, nn_count)
            )

        for i, muygps in enumerate(mmuygps.models):
            print(f"For model {i}:")
            for key in args[i]:
                if key == "eps":
                    self.assertEqual(args[i][key]["val"], muygps.eps())
                elif args[i][key]["val"] == "sample":
                    print(
                        f"\toptimized {key} to find value "
                        f"{muygps.kernel.hyperparameters[key]()}"
                    )
                else:
                    self.assertEqual(
                        args[i][key]["val"],
                        muygps.kernel.hyperparameters[key](),
                    )
            if sigma_method == None:
                self.assertEqual("unlearned", muygps.sigma_sq())
            else:
                print(
                    f"\toptimized sigma_sq to find value "
                    f"{muygps.sigma_sq()}"
                )


if __name__ == "__main__":
    absltest.main()
