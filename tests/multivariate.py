# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.mpi_utils import (
    _consistent_unchunk_tensor,
    _consistent_chunk_tensor,
    _warn0,
)
from MuyGPyS._test.gp import (
    benchmark_prepare_cholK,
    benchmark_sample_from_cholK,
    BenchmarkGP,
)
from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _basic_opt_method_and_kwarg_options,
    _check_ndarray,
    _get_sigma_sq_series,
    _make_gaussian_dict,
    _make_gaussian_data,
)
from MuyGPyS.examples.classify import make_multivariate_classifier, classify_any
from MuyGPyS.examples.regress import make_multivariate_regressor, regress_any
from MuyGPyS.gp import MultivariateMuyGPS as MMuyGPS
from MuyGPyS.gp.distance import pairwise_distances, crosswise_distances
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import optimize_from_tensors
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.sigma_sq import mmuygps_sigma_sq_optim


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
                self.assertTrue(muygps.kernel.hyperparameters[param].fixed())
            self.assertEqual(this_kwargs["eps"]["val"], muygps.eps())
            self.assertTrue(muygps.eps.fixed())
            self.assertFalse(muygps.sigma_sq.trained)


class SigmaSqTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, sm, 10, nn_kwargs, model_args)
            for nn_kwargs in _basic_nn_kwarg_options
            for f in [100]
            for sm in ["analytic"]
            for model_args in (
                (
                    "matern",
                    [
                        {
                            "nu": {"val": 1.5},
                            "length_scale": {"val": 7.2},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 0.5},
                            "length_scale": {"val": 2.2},
                            "eps": {"val": 1e-6},
                        },
                        {
                            "nu": {"val": mm.inf},
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
        sigma_method,
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
        indices = mm.arange(data_count)
        nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
        nn_targets = _consistent_chunk_tensor(data["output"][nn_indices, :])
        pairwise_dists = pairwise_distances(
            data["input"], nn_indices, metric=mmuygps.metric
        )

        # fit sigmas
        mmuygps = mmuygps_sigma_sq_optim(
            mmuygps, pairwise_dists, nn_targets, sigma_method=sigma_method
        )

        K = mm.zeros((data_count, nn_count, nn_count))
        nn_targets = _consistent_unchunk_tensor(nn_targets)
        for i, muygps in enumerate(mmuygps.models):
            K = _consistent_unchunk_tensor(muygps.kernel(pairwise_dists))
            sigmas = _get_sigma_sq_series(
                K,
                nn_targets[:, :, i].reshape(data_count, nn_count, 1),
                muygps.eps(),
            )
            _check_ndarray(self.assertEqual, sigmas, mm.ftype)
            _check_ndarray(self.assertEqual, muygps.sigma_sq(), mm.ftype)
            self.assertEqual(sigmas.shape, (data_count,))
            self.assertAlmostEqual(
                np.array(muygps.sigma_sq()[0]),
                np.mean(np.array(sigmas)),
                5,
            )


class OptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1001,
                10,
                b,
                n,
                nn_kwargs,
                loss_and_sigma_methods,
                om,
                opt_method_and_kwargs,
                k_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_and_sigma_methods in [["mse", None]]
            for om in ["loo_crossval"]
            # for nn_kwargs in _basic_nn_kwarg_options
            # for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for opt_method_and_kwargs in [
                _basic_opt_method_and_kwarg_options[1]
            ]
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
        loss_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
        k_kwargs,
    ):
        if config.state.backend != "numpy":
            _warn0(
                f"{self.__class__.__name__} relies on "
                f"{BenchmarkGP.__class__.__name__}, which only supports numpy. "
                f"Skipping."
            )
            return
        kern, metric, target, args = k_kwargs
        loss_method, sigma_method = loss_and_sigma_methods
        opt_method, opt_kwargs = opt_method_and_kwargs
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
        nbrs_lookup = NN_Wrapper(
            mm.array(sim_train["input"]), nn_count, **nn_kwargs
        )
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_count
        )
        crosswise_dists = crosswise_distances(
            mm.array(sim_train["input"]),
            mm.array(sim_train["input"]),
            batch_indices,
            batch_nn_indices,
            metric=metric,
        )
        pairwise_dists = pairwise_distances(
            mm.array(sim_train["input"]), batch_nn_indices, metric=metric
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

            print(sim_train["output"])
            batch_targets = sim_train["output"][batch_indices, :]
            batch_nn_targets = sim_train["output"][
                np.iarray(batch_nn_indices), :
            ]

            for i, muygps in enumerate(mmuygps.models):
                b_t = _consistent_chunk_tensor(
                    batch_targets[:, i].reshape(batch_count, 1)
                )
                b_nn_t = _consistent_chunk_tensor(
                    batch_nn_targets[:, :, i].reshape(batch_count, nn_count, 1)
                )
                mmuygps.models[i] = optimize_from_tensors(
                    muygps,
                    b_t,
                    b_nn_t,
                    crosswise_dists,
                    pairwise_dists,
                    loss_method=loss_method,
                    obj_method=obj_method,
                    opt_method=opt_method,
                    sigma_method=sigma_method,
                    **opt_kwargs,
                )
                estimate = mmuygps.models[i].kernel.hyperparameters["nu"]()
                mse += mm.sum(estimate - target[i]) ** 2
        mse /= its * response_count
        print(f"optimizes with mse {mse}")
        self.assertAlmostEqual(mse, 0.0, 1)


class ClassifyTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, f, nn, nn_kwargs, k_kwargs)
            for f in [100, 10, 2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [10]
            # for nn in [5]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
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
        if config.state.backend != "numpy":
            _warn0(
                f"classify_any() does not support {config.state.backend} "
                f"backend. Skipping."
            )
            return

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
        predictions = _consistent_unchunk_tensor(predictions)
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
                            "nu": {"val": 1.5},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-5},
                        },
                        {
                            "nu": {"val": 0.5},
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
        if config.state.backend != "numpy":
            _warn0(
                f"regress_any() does not support {config.state.backend} "
                f"backend. Skipping."
            )
            return
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

        self.assertFalse(mmuygps.sigma_sq.trained)

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
            diagonal_variance = _consistent_unchunk_tensor(diagonal_variance)
            self.assertEqual(
                diagonal_variance.shape, (test_count, response_count)
            )
        predictions = _consistent_unchunk_tensor(predictions)
        self.assertEqual(predictions.shape, (test_count, response_count))


class MakeClassifierTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1000,
                1000,
                10,
                b,
                n,
                nn_kwargs,
                lm,
                opt_method_and_kwargs,
                rt,
                k_kwargs,
            )
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lm in ["mse"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
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
        opt_method_and_kwargs,
        return_distances,
        k_kwargs,
    ):
        # skip if we are using the MPI implementation
        if config.state.backend == "torch":
            _warn0(f"optimization does not support MPI. skipping.")
            return
        if config.state.backend == "torch":
            _warn0(f"optimization does not support torch. skipping.")
            return

        kern, args = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
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
            opt_method=opt_method,
            nn_kwargs=nn_kwargs,
            kern=kern,
            k_args=args,
            opt_kwargs=opt_kwargs,
            return_distances=return_distances,
        )

        if len(classifier_args) == 2:
            mmuygps, _ = classifier_args
        elif len(classifier_args) == 4:
            mmuygps, _, crosswise_dists, pairwise_dists = classifier_args
            crosswise_dists = _consistent_unchunk_tensor(crosswise_dists)
            pairwise_dists = _consistent_unchunk_tensor(pairwise_dists)
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
            (
                1000,
                1000,
                10,
                b,
                n,
                nn_kwargs,
                lm,
                opt_method_and_kwargs,
                ssm,
                rt,
                k_kwargs,
            )
            for b in [250]
            for n in [10]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
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
        opt_method_and_kwargs,
        sigma_method,
        return_distances,
        k_kwargs,
    ):
        if config.state.backend == "mpi":
            _warn0(f"optimization does not support mpi. skipping.")
            return
        if config.state.backend == "torch":
            _warn0(f"optimization does not support torch. skipping.")
            return
        # skip if we are using the MPI implementation
        kern, args = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
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
            opt_method=opt_method,
            sigma_method=sigma_method,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            kern=kern,
            k_args=args,
            return_distances=return_distances,
        )

        if len(regressor_args) == 2:
            mmuygps, _ = regressor_args
        elif len(regressor_args) == 4:
            mmuygps, _, crosswise_dists, pairwise_dists = regressor_args
            crosswise_dists = _consistent_unchunk_tensor(crosswise_dists)
            pairwise_dists = _consistent_unchunk_tensor(pairwise_dists)
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
            if sigma_method is None:
                self.assertFalse(muygps.sigma_sq.trained)
            else:
                self.assertTrue(muygps.sigma_sq.trained)
                print(
                    f"\toptimized sigma_sq to find value "
                    f"{muygps.sigma_sq()}"
                )


if __name__ == "__main__":
    absltest.main()
