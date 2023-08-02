# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm

from MuyGPyS import config
from MuyGPyS._test.utils import (
    _advanced_opt_method_and_kwarg_options,
    _basic_opt_method_and_kwarg_options,
    _basic_nn_kwarg_options,
    _check_ndarray,
    _sq_rel_err,
)
from MuyGPyS.gp import MuyGPS

from MuyGPyS.gp.distortion import IsotropicDistortion, l2
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.experimental.chassis import (
    optimize_from_tensors_mini_batch,
)

from optimize import BenchmarkTestCase

if config.state.backend != "numpy":
    raise ValueError("optimize.py only supports the numpy backend at this time")


class MiniBatchBenchmarkTestCase(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(MiniBatchBenchmarkTestCase, cls).setUpClass()

    def _optim_chassis_mini_batch(
        self,
        muygps,
        name,
        itr,
        nbrs_lookup,
        batch_count,
        loss_method,
        obj_method,
        sigma_method,
        loss_kwargs,
        opt_kwargs,
    ) -> float:
        muygps = optimize_from_tensors_mini_batch(
            muygps,
            self.train_features,
            self.train_responses[itr, :, :],
            nbrs_lookup,
            batch_count,
            self.train_count,
            num_epochs=1,  # Optimizing over one epoch (for now)
            batch_features=None,
            loss_method=loss_method,
            obj_method=obj_method,
            sigma_method=sigma_method,
            loss_kwargs=loss_kwargs,
            verbose=False,  # TODO True,
            **opt_kwargs,
        )

        estimate = muygps.kernel._hyperparameters[name]()
        return _sq_rel_err(self.params[name](), estimate)

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)


class NuTest(MiniBatchBenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(NuTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                b,
                n,
                nn_kwargs,
                loss_kwargs_and_sigma_methods,
                om,
                opt_method_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for loss_kwargs_and_sigma_methods in [
                ["lool", dict(), "analytic"],
                ["mse", dict(), None],
                ["huber", {"boundary_scale": 1.5}, None],
            ]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in [
                _basic_opt_method_and_kwarg_options[1]
            ]
        )
    )
    def test_nu_mini_batch(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_kwargs_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
    ):
        loss_method, loss_kwargs, sigma_method = loss_kwargs_and_sigma_methods
        _, opt_kwargs = opt_method_and_kwargs

        mrse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(self.train_features, nn_count, **nn_kwargs)

        for i in range(self.its):
            # set up MuyGPS object
            muygps = MuyGPS(
                kernel=Matern(
                    nu=ScalarHyperparameter(
                        "sample", self.params["nu"].get_bounds()
                    ),
                    metric=IsotropicDistortion(
                        metric=l2,
                        length_scale=ScalarHyperparameter(
                            self.params["length_scale"]()
                        ),
                    ),
                ),
                eps=HomoscedasticNoise(self.params["eps"]()),
            )

            mrse += self._optim_chassis_mini_batch(
                muygps,
                "nu",
                i,
                nbrs_lookup,
                batch_count,
                loss_method,
                obj_method,
                sigma_method,
                loss_kwargs,
                opt_kwargs,
            )
        mrse /= self.its
        print(f"optimizes nu with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertLessEqual(mrse, self.nu_tol[loss_method])


class LengthScaleTest(MiniBatchBenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(LengthScaleTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                b,
                n,
                nn_kwargs,
                loss_kwargs_and_sigma_methods,
                om,
                opt_method_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for loss_kwargs_and_sigma_methods in [["lool", dict(), "analytic"]]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in [
                _advanced_opt_method_and_kwarg_options[1]
            ]
        )
    )
    def test_length_scale_mini_batch(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_kwargs_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
    ):
        loss_method, loss_kwargs, sigma_method = loss_kwargs_and_sigma_methods
        _, opt_kwargs = opt_method_and_kwargs

        error_vector = mm.zeros((self.its,))

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(self.train_features, nn_count, **nn_kwargs)

        for i in range(self.its):
            # set up MuyGPS object
            muygps = MuyGPS(
                kernel=Matern(
                    nu=ScalarHyperparameter(self.params["nu"]()),
                    metric=IsotropicDistortion(
                        metric=l2,
                        length_scale=ScalarHyperparameter(
                            "sample", self.params["length_scale"].get_bounds()
                        ),
                    ),
                ),
                eps=HomoscedasticNoise(self.params["eps"]()),
            )

            error_vector[i] = self._optim_chassis_mini_batch(
                muygps,
                "length_scale",
                i,
                nbrs_lookup,
                batch_count,
                loss_method,
                obj_method,
                sigma_method,
                loss_kwargs,
                opt_kwargs,
            )
        median_error = mm.median(error_vector)
        print(
            "optimizes length_scale with "
            f"median relative squared error {median_error}"
        )
        # Is this a strong enough guarantee?
        self.assertLessEqual(median_error, self.length_scale_tol[loss_method])


if __name__ == "__main__":
    absltest.main()
