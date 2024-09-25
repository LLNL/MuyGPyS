# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm

from MuyGPyS import config
from MuyGPyS._test.optimize import BenchmarkTestCase
from MuyGPyS._test.utils import (
    _check_ndarray,
    _sq_rel_err,
)
from MuyGPyS.gp import MuyGPS

from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import AnalyticScale, FixedScale, ScalarParam
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.experimental.chassis import (
    optimize_from_tensors_mini_batch,
)
from MuyGPyS.optimize.loss import (
    lool_fn,
    mse_fn,
    pseudo_huber_fn,
    looph_fn,
)

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
        loss_fn,
        obj_method,
        opt_kwargs,
        loss_kwargs=dict(),
    ) -> float:
        (
            muygps,
            _,
            _,
            _,
            _,
        ) = optimize_from_tensors_mini_batch(
            muygps,
            self.train_features,
            self.train_responses_list[itr],
            self.nn_count,
            self.batch_count,
            self.train_count,
            num_epochs=1,  # Optimizing over one epoch (for now)
            keep_state=False,
            probe_previous=False,
            batch_features=None,
            loss_fn=loss_fn,
            obj_method=obj_method,
            loss_kwargs=loss_kwargs,
            verbose=False,
            **opt_kwargs,
        )
        estimate = muygps.kernel._hyperparameters[name]()
        return _sq_rel_err(self.params[name](), estimate)

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)


class SmoothnessTest(MiniBatchBenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(SmoothnessTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                loss_kwargs_and_scale,
                om,
                opt_method_and_kwargs,
            )
            for loss_kwargs_and_scale in [
                ["lool", lool_fn, dict(), AnalyticScale()],
                ["mse", mse_fn, dict(), FixedScale()],
                [
                    "huber",
                    pseudo_huber_fn,
                    {"boundary_scale": 1.5},
                    FixedScale(),
                ],
                ["looph", looph_fn, {"boundary_scale": 1.5}, AnalyticScale()],
            ]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in [
                [
                    "bayesian",
                    {
                        "random_state": 1,
                        "init_points": 3,
                        "n_iter": 10,
                        "allow_duplicate_points": True,
                    },
                ],
            ]
        )
    )
    def test_smoothness_mini_batch(
        self,
        loss_kwargs_and_scale,
        obj_method,
        opt_method_and_kwargs,
    ):
        (
            loss_name,
            loss_fn,
            loss_kwargs,
            scale,
        ) = loss_kwargs_and_scale
        _, opt_kwargs = opt_method_and_kwargs

        mrse = 0.0

        for i in range(self.its):
            # set up MuyGPS object
            muygps = MuyGPS(
                kernel=Matern(
                    smoothness=ScalarParam(
                        "sample", self.params["smoothness"].get_bounds()
                    ),
                    deformation=Isotropy(
                        metric=l2,
                        length_scale=ScalarParam(self.params["length_scale"]()),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                scale=scale,
            )

            mrse += self._optim_chassis_mini_batch(
                muygps,
                "smoothness",
                i,
                loss_fn,
                obj_method,
                opt_kwargs,
                loss_kwargs=loss_kwargs,
            )
        mrse /= self.its
        print(f"optimizes smoothness with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertLessEqual(mrse, self.smoothness_tol[loss_name])


class LengthScaleTest(MiniBatchBenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(LengthScaleTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                loss_kwargs_and_scale,
                om,
                opt_method_and_kwargs,
            )
            for loss_kwargs_and_scale in [
                ["lool", lool_fn, dict(), AnalyticScale()],
            ]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in [
                [
                    "bayesian",
                    {
                        "random_state": 1,
                        "init_points": 5,
                        "n_iter": 20,
                        "allow_duplicate_points": True,
                    },
                ],
            ]
        )
    )
    def test_length_scale_mini_batch(
        self,
        loss_kwargs_and_scale,
        obj_method,
        opt_method_and_kwargs,
    ):
        (
            loss_name,
            loss_fn,
            loss_kwargs,
            scale,
        ) = loss_kwargs_and_scale
        _, opt_kwargs = opt_method_and_kwargs

        error_vector = mm.zeros((self.its,))

        for i in range(self.its):
            # set up MuyGPS object
            muygps = MuyGPS(
                kernel=Matern(
                    smoothness=ScalarParam(self.params["smoothness"]()),
                    deformation=Isotropy(
                        metric=l2,
                        length_scale=ScalarParam(
                            "sample", self.params["length_scale"].get_bounds()
                        ),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                scale=scale,
            )

            error_vector[i] = self._optim_chassis_mini_batch(
                muygps,
                "length_scale",
                i,
                loss_fn,
                obj_method,
                opt_kwargs,
                loss_kwargs=loss_kwargs,
            )
        median_error = mm.median(error_vector)
        print(
            "optimizes length_scale with "
            f"median relative squared error {median_error}"
        )
        # Is this a strong enough guarantee?
        self.assertLessEqual(median_error, self.length_scale_tol[loss_name])


if __name__ == "__main__":
    absltest.main()
