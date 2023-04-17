# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS import config
from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _basic_opt_method_and_kwarg_options,
    _make_gaussian_data,
)
from MuyGPyS.examples.fast_posterior_mean import do_fast_posterior_mean
from MuyGPyS.gp.distortion import IsotropicDistortion
from MuyGPyS.gp.kernels import Hyperparameter, Matern
from MuyGPyS.gp.noise import HomoscedasticNoise

if config.state.backend in ["mpi", "torch"]:
    raise ValueError("This test only supports numpy and jax!")


class MakeFastRegressorTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 1000, 10, b, n, nn_kwargs, lm, k_kwargs)
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lm in ["mse"]
            # for ssm in ["analytic"]
            # for rt in [True]
            for k_kwargs in (
                {
                    "kernel": Matern(
                        nu=Hyperparameter("sample", (1e-1, 1e0)),
                        metric=IsotropicDistortion(
                            "l2", length_scale=Hyperparameter(1.5)
                        ),
                    ),
                    "eps": HomoscedasticNoise(1e-5),
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
        loss_method,
        k_kwargs,
    ):
        # skip if we are using the MPI implementation

        # construct the observation locations
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
            loss_method=loss_method,
            opt_method="bayes",
            opt_kwargs={
                "allow_duplicate_points": True,
                "init_points": 2,
                "n_iter": 2,
            },
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
                lm,
                opt_method_and_kwargs,
                ssm,
                k_kwargs,
            )
            for b in [250]
            for n in [10]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for ssm in ["analytic", None]
            for k_kwargs in (
                (
                    {
                        "kernel": Matern(
                            nu=Hyperparameter(0.5),
                            metric=IsotropicDistortion(
                                "l2", length_scale=Hyperparameter(1.5)
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
                    },
                    {
                        "kernel": Matern(
                            nu=Hyperparameter(0.8),
                            metric=IsotropicDistortion(
                                "l2", length_scale=Hyperparameter(0.7)
                            ),
                        ),
                        "eps": HomoscedasticNoise(1e-5),
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
        loss_method,
        opt_method_and_kwargs,
        sigma_method,
        k_kwargs,
    ):
        # skip if we are using the MPI implementation
        opt_method, opt_kwargs = opt_method_and_kwargs
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
            nbrs_lookup,
            predictions,
            precomputed_coefficient_matrix,
            timings,
        ) = do_fast_posterior_mean(
            test["input"],
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
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
            self.assertEqual(k_kwargs[i]["eps"](), muygps.eps())
            for name, param in k_kwargs[i]["kernel"].hyperparameters.items():
                if param.fixed() is False:
                    print(
                        f"optimized to find value "
                        f"{muygps.kernel.hyperparameters[name]()}"
                    )
                else:
                    self.assertEqual(
                        param(),
                        muygps.kernel.hyperparameters[name](),
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
