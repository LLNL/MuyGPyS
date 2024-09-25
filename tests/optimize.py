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
    # _advanced_opt_fn_and_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
)
from MuyGPyS.gp import MuyGPS

from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import AnalyticScale, FixedScale, ScalarParam
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import lool_fn, mse_fn, pseudo_huber_fn, looph_fn

if config.state.backend != "numpy":
    raise ValueError("optimize.py only supports the numpy backend at this time")


class BenchmarkTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTest, cls).setUpClass()

    def test_types(self):
        self._check_ndarray(
            self.train_features,
            mm.ftype,
            ctype=mm.ndarray,
            shape=(self.train_count, self.feature_count),
        )
        self._check_ndarray(
            self.test_features,
            mm.ftype,
            ctype=mm.ndarray,
            shape=(self.test_count, self.feature_count),
        )
        for i in range(self.its):
            self._check_ndarray(
                self.train_responses_list[i],
                mm.ftype,
                ctype=mm.ndarray,
                shape=(self.train_count, self.response_count),
            )
            self._check_ndarray(
                self.test_responses_list[i],
                mm.ftype,
                ctype=mm.ndarray,
                shape=(self.test_count, self.response_count),
            )


class SmoothnessTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(SmoothnessTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                loss_kwargs_and_scale,
                opt_fn_and_kwargs,
            )
            for loss_kwargs_and_scale in [
                ["mse", mse_fn, dict(), FixedScale()],
                [
                    "huber",
                    pseudo_huber_fn,
                    {"boundary_scale": 1.5},
                    FixedScale(),
                ],
                ["lool", lool_fn, dict(), AnalyticScale()],
                ["looph", looph_fn, {"boundary_scale": 3.0}, AnalyticScale()],
            ]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            # for opt_fn_and_kwargs in [_basic_opt_fn_and_kwarg_options[0]]
        )
    )
    def test_smoothness(
        self,
        loss_kwargs_and_scale,
        opt_fn_and_kwargs,
    ):
        (
            loss_name,
            loss_fn,
            loss_kwargs,
            scale,
        ) = loss_kwargs_and_scale
        opt_fn, opt_kwargs = opt_fn_and_kwargs

        error_vector = list()

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

            error_vector.append(
                self._optim_chassis(
                    muygps,
                    "smoothness",
                    i,
                    loss_fn,
                    opt_fn,
                    opt_kwargs,
                    loss_kwargs=loss_kwargs,
                )
            )
        median_rse = mm.median(error_vector)
        print(
            f"optimizes smoothness with mean relative squared error {median_rse}"
        )
        # Is this a strong enough guarantee?
        self.assertLessEqual(median_rse, self.smoothness_tol[loss_name])


class LengthScaleTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(LengthScaleTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                loss_and_scale,
                opt_fn_and_kwargs,
            )
            for loss_and_scale in [
                ["lool", lool_fn, dict(), AnalyticScale()],
                ["looph", looph_fn, {"boundary_scale": 3.0}, AnalyticScale()],
            ]
            # for opt_fn_and_kwargs in _advanced_opt_fn_and_kwarg_options
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            # for opt_fn_and_kwargs in [_basic_opt_fn_and_kwarg_options[0]]
            # for opt_fn_and_kwargs in [_basic_opt_fn_and_kwarg_options[1]]
        )
    )
    def test_length_scale(
        self,
        loss_and_scale,
        opt_fn_and_kwargs,
    ):
        loss_name, loss_fn, loss_kwargs, scale = loss_and_scale
        opt_fn, opt_kwargs = opt_fn_and_kwargs

        error_vector = list()

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

            error_vector.append(
                self._optim_chassis(
                    muygps,
                    "length_scale",
                    i,
                    loss_fn,
                    opt_fn,
                    opt_kwargs,
                    loss_kwargs=loss_kwargs,
                )
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
