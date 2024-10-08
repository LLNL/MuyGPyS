# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm

from MuyGPyS import config

from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
    _make_gaussian_data,
)
from MuyGPyS.examples.fast_posterior_mean import do_fast_posterior_mean
from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import AnalyticScale, ScalarParam, FixedScale
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn

if config.state.backend in ["mpi", "torch"]:
    raise ValueError("This test only supports numpy and jax!")


class MakeFastRegressorTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 1000, 10, b, n, nn_kwargs, lf, opt_fn_and_kwargs, k_kwargs)
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lf in [mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            # for ssm in ["analytic"]
            # for rt in [True]
            for k_kwargs in (
                {
                    "kernel": Matern(
                        smoothness=ScalarParam("sample", (1e-1, 1e0)),
                        deformation=Isotropy(l2, length_scale=ScalarParam(1.5)),
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_make_fast_regressor(
        self,
        train_count,
        test_count,
        feature_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_fn,
        opt_fn_and_kwargs,
        k_kwargs,
    ):
        # skip if we are using the MPI implementation

        # construct the observation locations
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        response_count = 2
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count=response_count,
            categorical=False,
        )

        (
            _,
            _,
            predictions,
            precomputed_coefficient_matrix,
            _,
        ) = do_fast_posterior_mean(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            opt_kwargs=opt_kwargs,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
        )
        self.assertEqual(
            precomputed_coefficient_matrix.shape,
            (train_count, nn_count, response_count),
        )
        self.assertEqual(predictions.shape, (test_count, response_count))


class MakeFastMultivariateRegressorTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1000,
                1000,
                10,
                b,
                n,
                nn_kwargs,
                lf,
                opt_fn_and_kwargs,
                k_kwargs,
            )
            for b in [250]
            for n in [10]
            for nn_kwargs in _basic_nn_kwarg_options
            for lf in [mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for k_kwargs in (
                (
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam(0.5),
                            deformation=Isotropy(
                                l2, length_scale=ScalarParam(1.5)
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-5),
                        "scale": AnalyticScale(),
                    },
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam(0.8),
                            deformation=Isotropy(
                                l2, length_scale=ScalarParam(0.7)
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-5),
                        "scale": FixedScale(),
                    },
                ),
            )
        )
    )
    def test_make_fast_multivariate_regressor(
        self,
        train_count,
        test_count,
        feature_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_fn,
        opt_fn_and_kwargs,
        k_kwargs,
    ):
        # skip if we are using the MPI implementation
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        response_count = len(k_kwargs)

        # construct the observation locations
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=False,
        )

        (
            mmuygps,
            _,
            predictions,
            precomputed_coefficient_matrix,
            _,
        ) = do_fast_posterior_mean(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            k_kwargs=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
        )

        self.assertEqual(
            precomputed_coefficient_matrix.shape,
            (train_count, nn_count, response_count),
        )
        self.assertEqual(predictions.shape, (test_count, response_count))

        for i, muygps in enumerate(mmuygps.models):
            print(f"For model{i}:")
            self.assertEqual(k_kwargs[i]["noise"](), muygps.noise())
            for name, param in k_kwargs[i]["kernel"]._hyperparameters.items():
                if param.fixed() is False:
                    print(
                        f"optimized to find value "
                        f"{muygps.kernel._hyperparameters[name]()}"
                    )
                    self.assertTrue(muygps.scale.trained)
                else:
                    self.assertEqual(
                        param(),
                        muygps.kernel._hyperparameters[name](),
                    )
                    self.assertFalse(muygps.scale.trained)
            self.assertFalse(muygps.scale.trained)
            self.assertEqual(mm.array([1.0]), muygps.scale())


if __name__ == "__main__":
    absltest.main()
