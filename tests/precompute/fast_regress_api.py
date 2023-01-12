# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys

import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

from MuyGPyS.examples.two_class_classify_uq import example_lambdas
from MuyGPyS._test.api import FastRegressionAPITest
from MuyGPyS._test.utils import (
    _balanced_subsample,
    _basic_nn_kwarg_options,
    _basic_opt_method_and_kwarg_options,
)


hardpath = "../data/"

stargal_dir = "star-gal/"

stargal_files = {
    "full": "galstar.pkl",
    "30": "embedded_30_galstar.pkl",
    "40": "embedded_40_galstar.pkl",
    "50": "embedded_50_galstar.pkl",
}

heaton_file = "heaton/sub_heaton.pkl"


class HeatonFastTest(FastRegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(HeatonFastTest, cls).setUpClass()
        with open(os.path.join(hardpath, heaton_file), "rb") as f:
            cls.train, cls.test = pkl.load(f)

    @parameterized.parameters(
        (
            (nn, bs, vm, lm, om, opt_method_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for vm in ["diagonal", None]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                (
                    11.0,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                    },
                ),
            )
        )
    )
    def test_fast_regress(
        self,
        nn_count,
        batch_count,
        variance_mode,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, k_kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs

        if variance_mode is None:
            sigma_method = None
            apply_sigma_sq = False
        else:
            sigma_method = "analytic"
            apply_sigma_sq = True

        self._do_fast_regress_test_chassis(
            train=self.train,
            test=self.test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
            variance_mode=variance_mode,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            apply_sigma_sq=apply_sigma_sq,
            verbose=False,
        )


class MultivariateStargalFastRegressTest(FastRegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(MultivariateStargalFastRegressTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
        ) as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
        # with open(os.path.join(hardpath, heaton_file), "rb") as f:
        #     cls.train, cls.test = pkl.load(f)

    @parameterized.parameters(
        (
            (nn, bs, vm, lm, om, opt_method_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for vm in [None]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for opt_method_and_kwargs in [
                _basic_opt_method_and_kwarg_options[0]
            ]
            for k_kwargs in (
                (
                    1.0,
                    "matern",
                    [
                        {
                            "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                            # "nu": {"val": 0.38},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-3},
                        },
                        {
                            "nu": {"val": 0.5},
                            # "nu": {"val": 0.38},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-3},
                        },
                    ],
                ),
                (
                    1.0,
                    "rbf",
                    [
                        {
                            "metric": "l2",
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-3},
                        },
                        {
                            "metric": "l2",
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-3},
                        },
                    ],
                ),
            )
        )
    )
    def test_fast_regress(
        self,
        nn_count,
        batch_count,
        variance_mode,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, kern, k_args = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)

        if variance_mode is None:
            sigma_method = None
            apply_sigma_sq = False
        else:
            sigma_method = "analytic"
            apply_sigma_sq = True

        self._do_fast_regress_test_chassis(
            train=train,
            test=test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
            variance_mode=variance_mode,
            nn_kwargs=nn_kwargs,
            kern=kern,
            k_kwargs=k_args,
            opt_kwargs=opt_kwargs,
            apply_sigma_sq=apply_sigma_sq,
            verbose=False,
        )


class StargalFastRegressTest(FastRegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(StargalFastRegressTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
        ) as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
        # with open(os.path.join(hardpath, heaton_file), "rb") as f:
        #     cls.train, cls.test = pkl.load(f)

    @parameterized.parameters(
        (
            (nn, bs, vm, lm, om, opt_method_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for vm in [None]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for nn_kwargs in _basic_nn_kwarg_options
            for opt_method_and_kwargs in [
                _basic_opt_method_and_kwarg_options[0]
            ]
            for k_kwargs in (
                (
                    1.0,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": 0.5},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                    },
                ),
                (
                    1.0,
                    {
                        "kern": "rbf",
                        "metric": "l2",
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                    },
                ),
            )
        )
    )
    def test_fast_regress(
        self,
        nn_count,
        batch_count,
        variance_mode,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, k_kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)

        if variance_mode is None:
            sigma_method = None
            apply_sigma_sq = False
        else:
            sigma_method = "analytic"
            apply_sigma_sq = True

        self._do_fast_regress_test_chassis(
            train=train,
            test=test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
            variance_mode=variance_mode,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            apply_sigma_sq=apply_sigma_sq,
            verbose=False,
        )


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
