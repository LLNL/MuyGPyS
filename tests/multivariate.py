# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
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
from MuyGPyS.gp.distortion import IsotropicDistortion, AnisotropicDistortion, l2
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.sigma_sq import AnalyticSigmaSq, SigmaSq
from MuyGPyS.gp.tensors import pairwise_tensor, crosswise_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import optimize_from_tensors
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import mse_fn


class InitTest(parameterized.TestCase):
    @parameterized.parameters(
        (model_args,)
        for model_args in (
            [
                {
                    "kernel": Matern(
                        nu=ScalarHyperparameter(1.0),
                        metric=IsotropicDistortion(
                            metric=l2, length_scale=ScalarHyperparameter(7.2)
                        ),
                    ),
                    "eps": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": Matern(
                        nu=ScalarHyperparameter(1.2),
                        metric=IsotropicDistortion(
                            metric=l2, length_scale=ScalarHyperparameter(2.2)
                        ),
                    ),
                    "eps": HomoscedasticNoise(1e-6),
                },
            ],
            [
                {
                    "kernel": Matern(
                        nu=ScalarHyperparameter(1.0),
                        metric=IsotropicDistortion(
                            metric=l2, length_scale=ScalarHyperparameter(7.2)
                        ),
                    ),
                    "eps": HomoscedasticNoise(1e-5),
                },
            ],
        )
    )
    def test_bounds_defaults_init(self, model_args):
        mmuygps = MMuyGPS(*model_args)
        self.assertEqual(len(mmuygps.models), len(model_args))
        for i, muygps in enumerate(mmuygps.models):
            this_kwargs = model_args[i]
            for name, param in this_kwargs["kernel"]._hyperparameters.items():
                self.assertEqual(
                    param(),
                    muygps.kernel._hyperparameters[name](),
                )
                self.assertTrue(muygps.kernel._hyperparameters[name].fixed())
            self.assertEqual(this_kwargs["eps"](), muygps.eps())
            self.assertTrue(muygps.eps.fixed())
            self.assertFalse(muygps.sigma_sq.trained)


class SigmaSqTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 2, 10, nn_kwargs, model_args)
            for nn_kwargs in _basic_nn_kwarg_options
            for model_args in (
                [
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(1.5),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(7.2),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.5),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(2.2),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-6),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(mm.inf),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(12.4),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-6),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(1.5),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(7.2),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.5),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(2.2),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-6),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(mm.inf),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(12.4),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-6),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                ],
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
        response_count = len(model_args)
        mmuygps = MMuyGPS(*model_args)

        # prepare data
        data = _make_gaussian_dict(data_count, feature_count, response_count)

        # neighbors and differences
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices = mm.arange(data_count)
        nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
        nn_targets = _consistent_chunk_tensor(data["output"][nn_indices, :])
        pairwise_diffs = pairwise_tensor(data["input"], nn_indices)

        # fit sigmas
        mmuygps = mmuygps.optimize_sigma_sq(pairwise_diffs, nn_targets)

        K = mm.zeros((data_count, nn_count, nn_count))
        nn_targets = _consistent_unchunk_tensor(nn_targets)
        for i, model in enumerate(mmuygps.models):
            K = _consistent_unchunk_tensor(model.kernel(pairwise_diffs))
            sigmas = _get_sigma_sq_series(
                K,
                nn_targets[:, :, i].reshape(data_count, nn_count, 1),
                model.eps(),
            )
            _check_ndarray(self.assertEqual, sigmas, mm.ftype)
            _check_ndarray(self.assertEqual, model.sigma_sq(), mm.ftype)
            self.assertEqual(sigmas.shape, (data_count,))
            self.assertAlmostEqual(
                np.array(model.sigma_sq()[0]),
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
                loss_fn,
                om,
                opt_method_and_kwargs,
                k_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_fn in [mse_fn]
            for om in ["loo_crossval"]
            # for nn_kwargs in _basic_nn_kwarg_options
            # for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for opt_method_and_kwargs in [
                _basic_opt_method_and_kwarg_options[1]
            ]
            for k_kwargs in (
                (
                    [0.63, 0.78],
                    [
                        {
                            "kernel": Matern(
                                nu=ScalarHyperparameter("sample", (1e-2, 1e0)),
                                metric=IsotropicDistortion(
                                    metric=l2,
                                    length_scale=ScalarHyperparameter(1.5),
                                ),
                            ),
                            "eps": HomoscedasticNoise(1e-5),
                        },
                        {
                            "kernel": Matern(
                                nu=ScalarHyperparameter("sample", (1e-2, 1e0)),
                                metric=IsotropicDistortion(
                                    metric=l2,
                                    length_scale=ScalarHyperparameter(0.7),
                                ),
                            ),
                            "eps": HomoscedasticNoise(1e-5),
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
        loss_fn,
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
        target, args = k_kwargs
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
        crosswise_diffs = crosswise_tensor(
            mm.array(sim_train["input"]),
            mm.array(sim_train["input"]),
            batch_indices,
            batch_nn_indices,
        )
        pairwise_diffs = pairwise_tensor(
            mm.array(sim_train["input"]), batch_nn_indices
        )

        gps = [BenchmarkGP(**a) for a in args]
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

            mmuygps = MMuyGPS(*args)

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
                    crosswise_diffs,
                    pairwise_diffs,
                    loss_fn=loss_fn,
                    obj_method=obj_method,
                    opt_method=opt_method,
                    **opt_kwargs,
                )
                estimate = mmuygps.models[i].kernel._hyperparameters["nu"]()
                mse += mm.sum(estimate - target[i]) ** 2
        mse /= its * response_count
        print(f"optimizes with mse {mse}")
        self.assertLessEqual(mse, 0.5)


class ClassifyTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 200, 2, nn, nn_kwargs, args)
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [10]
            # for nn in [5]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for args in (
                (
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.63),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.79),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(0.7),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.63),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(1.5),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.79),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(0.7),
                                length_scale1=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
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
        args,
    ):
        if config.state.backend != "numpy":
            _warn0(
                f"classify_any() does not support {config.state.backend} "
                f"backend. Skipping."
            )
            return

        response_count = len(args)

        mmuygps = MMuyGPS(*args)

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
            (1000, 200, 2, nn, nn_kwargs, args)
            for nn in [5, 10]
            # for f in [2]
            # for nn in [5]
            # for vm in ["diagonal"]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for args in (
                (
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(1.5),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            ScalarHyperparameter(0.5),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(0.7),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(1.5),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(1.5),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            ScalarHyperparameter(0.5),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(0.7),
                                length_scale1=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
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
        nn_kwargs,
        args,
    ):
        if config.state.backend != "numpy":
            _warn0(
                f"regress_any() does not support {config.state.backend} "
                f"backend. Skipping."
            )
            return
        response_count = len(args)

        mmuygps = MMuyGPS(*args)

        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )
        nbrs_lookup = NN_Wrapper(train["input"], nn_count, **nn_kwargs)

        self.assertFalse(mmuygps.sigma_sq.trained)

        predictions, diagonal_variance, _ = regress_any(
            mmuygps,
            test["input"],
            train["input"],
            nbrs_lookup,
            train["output"],
        )
        diagonal_variance = _consistent_unchunk_tensor(diagonal_variance)
        self.assertEqual(diagonal_variance.shape, (test_count, response_count))
        predictions = _consistent_unchunk_tensor(predictions)
        self.assertEqual(predictions.shape, (test_count, response_count))


class MakeClassifierTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1000,
                1000,
                2,
                b,
                n,
                nn_kwargs,
                lf,
                opt_method_and_kwargs,
                args,
            )
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lf in [mse_fn]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for args in (
                (
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.8),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(0.7),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(1.5),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.8),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(0.7),
                                length_scale1=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
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
        loss_fn,
        opt_method_and_kwargs,
        args,
    ):
        if config.state.backend == "torch":
            _warn0("optimization does not support MPI. skipping.")
            return

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

        mmuygps, _ = make_multivariate_classifier(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_method=opt_method,
            nn_kwargs=nn_kwargs,
            k_args=args,
            opt_kwargs=opt_kwargs,
        )

        for i, muygps in enumerate(mmuygps.models):
            self.assertEqual(args[i]["eps"](), muygps.eps())
            for name, param in args[i]["kernel"]._hyperparameters.items():
                if param.fixed() is False:
                    print(
                        f"optimized to find value "
                        f"{muygps.kernel._hyperparameters[name]()}"
                    )
                else:
                    self.assertEqual(
                        param(),
                        muygps.kernel._hyperparameters[name](),
                    )


class MakeRegressorTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1000,
                1000,
                2,
                b,
                n,
                nn_kwargs,
                lf,
                opt_method_and_kwargs,
                args,
            )
            for b in [250]
            for n in [10]
            for nn_kwargs in _basic_nn_kwarg_options
            for lf in [mse_fn]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for args in (
                (
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.8),
                            metric=IsotropicDistortion(
                                metric=l2,
                                length_scale=ScalarHyperparameter(0.7),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(1.5),
                                length_scale1=ScalarHyperparameter(0.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                        "sigma_sq": AnalyticSigmaSq(),
                    },
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter(0.8),
                            metric=AnisotropicDistortion(
                                metric=l2,
                                length_scale0=ScalarHyperparameter(0.7),
                                length_scale1=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                        "sigma_sq": SigmaSq(),
                    },
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
        loss_fn,
        opt_method_and_kwargs,
        args,
    ):
        if config.state.backend == "mpi":
            _warn0("optimization does not support mpi. skipping.")
            return
        if config.state.backend == "torch":
            _warn0("optimization does not support torch. skipping.")
            return
        # skip if we are using the MPI implementation
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

        mmuygps, _ = make_multivariate_regressor(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_method=opt_method,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            k_args=args,
        )

        for i, muygps in enumerate(mmuygps.models):
            print(f"For model{i}:")
            self.assertEqual(args[i]["eps"](), muygps.eps())
            for name, param in args[i]["kernel"]._hyperparameters.items():
                if param.fixed() is False:
                    print(
                        f"optimized to find value "
                        f"{muygps.kernel._hyperparameters[name]()}"
                    )
                else:
                    self.assertEqual(
                        param(),
                        muygps.kernel._hyperparameters[name](),
                    )
            self.assertTrue(muygps.sigma_sq.trained)
            if isinstance(muygps.sigma_sq, AnalyticSigmaSq):
                print(
                    f"\toptimized sigma_sq to find value "
                    f"{muygps.sigma_sq()}"
                )
            else:
                self.assertEqual(mm.array([1.0]), muygps.sigma_sq())


if __name__ == "__main__":
    absltest.main()
