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
from MuyGPyS._test.api import (
    ClassifyAPITest,
    RegressionAPITest,
    FastRegressionAPITest,
)
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

mnist_dir = "mnist/"

mnist_files = {
    "full": "mnist.pkl",
    "30": "embedded_30_mnist.pkl",
    "40": "embedded_40_mnist.pkl",
    "50": "embedded_50_mnist.pkl",
}

heaton_file = "heaton/sub_heaton.pkl"


class MNISTTest(ClassifyAPITest):
    @classmethod
    def setUpClass(cls):
        super(MNISTTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + mnist_dir, mnist_files["40"]), "rb"
        ) as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)

    @parameterized.parameters(
        (
            (nn, bs, lm, om, opt_method_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            # for lm in ["mse"]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lm in ["log", "mse"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                (
                    0.85,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": 0.5, "bounds": (1e-1, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                    },
                ),
                (
                    0.85,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                    },
                ),
            )
        )
    )
    def test_classify(
        self,
        nn_count,
        batch_count,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, k_kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 5000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=False,
        )


class StargalTest(ClassifyAPITest):
    @classmethod
    def setUpClass(cls):
        super(StargalTest, cls).setUpClass()
        # with open(os.path.join(hardpath, hardfiles["full"]), "rb") as f:
        #     cls.train, cls.test = pkl.load(f)
        # with open(os.path.join(hardpath, hardfiles["30"]), "rb") as f:
        #     cls.embedded_30_train, cls.embedded_30_test = pkl.load(f)
        with open(
            os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
        ) as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
        # with open(os.path.join(hardpath, hardfiles["50"]), "rb") as f:
        #     cls.embedded_50_train, cls.embedded_50_test = pkl.load(f)


class StargalClassifyTest(StargalTest):
    @parameterized.parameters(
        (
            (nn, bs, lm, om, opt_method_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            # for lm in ["log"]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lm in ["log", "mse"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                (
                    0.92,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        # "nu": {"val": 0.75},
                        "nu": {"val": 0.5, "bounds": (1e-1, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                    },
                ),
                (
                    0.9,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                    },
                ),
            )
        )
    )
    def test_classify(
        self,
        nn_count,
        batch_count,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, k_kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 5000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=False,
        )


class StargalUQTest(StargalTest):
    @parameterized.parameters(
        (
            (
                nn,
                obs,
                ubs,
                lm,
                om,
                opt_method_and_kwargs,
                uq,
                nn_kwargs,
                k_kwargs,
            )
            for nn in [30]
            for obs in [500]
            for ubs in [500]
            for lm in ["log", "mse"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for uq in [example_lambdas]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                (
                    0.92,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": 0.5, "bounds": (1e-1, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                    },
                ),
                (
                    0.9,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                    },
                ),
            )
        )
    )
    def test_classify_uq(
        self,
        nn_count,
        opt_batch_count,
        uq_batch_count,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        uq_objectives,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, k_kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_uq_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            opt_batch_count=opt_batch_count,
            uq_batch_count=uq_batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            k_kwargs=k_kwargs,
            verbose=False,
        )


class MultivariateStargalClassifyTest(StargalTest):
    @parameterized.parameters(
        (
            (nn, bs, lm, om, opt_method_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for k_kwargs in (
                (
                    0.92,
                    "matern",
                    [
                        {
                            "nu": {"val": 0.5, "bounds": (1e-1, 1e0)},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-3},
                        },
                        {
                            "nu": {"val": 0.5, "bounds": (1e-1, 1e0)},
                            "length_scale": {"val": 1.5},
                            "eps": {"val": 1e-3},
                        },
                    ],
                ),
                (
                    0.9,
                    "rbf",
                    [
                        {
                            "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                            "eps": {"val": 1e-3},
                        },
                        {
                            "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                            "eps": {"val": 1e-3},
                        },
                    ],
                ),
            )
        )
    )
    def test_classify(
        self,
        nn_count,
        batch_count,
        loss_method,
        obj_method,
        opt_method_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, kern, k_kwargs = k_kwargs
        opt_method, opt_kwargs = opt_method_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 5000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            nn_kwargs=nn_kwargs,
            kern=kern,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=False,
        )


class MultivariateStargalRegressTest(RegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(MultivariateStargalRegressTest, cls).setUpClass()
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
            for vm in [None, "diagonal"]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            for nn_kwargs in _basic_nn_kwarg_options
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            # for vm in ["diagonal"]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
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
                        {"length_scale": {"val": 1.5}, "eps": {"val": 1e-3}},
                        {"length_scale": {"val": 1.5}, "eps": {"val": 1e-3}},
                    ],
                ),
            )
        )
    )
    def test_regress(
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

        self._do_regress_test_chassis(
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


class HeatonTest(RegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(HeatonTest, cls).setUpClass()
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
                # (
                #     11.0,
                #     {
                #         "kern": "rbf",
                #         "metric": "F2",
                #         "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                #         "eps": {"val": 1e-3},
                #     },
                # ),
            )
            # for vm in ["diagonal"]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
        )
    )
    def test_regress(
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

        self._do_regress_test_chassis(
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
                # (
                #     11.0,
                #     {
                #         "kern": "rbf",
                #         "metric": "F2",
                #         "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                #         "eps": {"val": 1e-3},
                #     },
                # ),
            )
            # for vm in ["diagonal"]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
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
            for vm in [None, "diagonal"]
            for lm in ["mse"]
            for om in ["loo_crossval"]
            for nn_kwargs in _basic_nn_kwarg_options
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            # for vm in ["diagonal"]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
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
                        {"length_scale": {"val": 1.5}, "eps": {"val": 1e-3}},
                        {"length_scale": {"val": 1.5}, "eps": {"val": 1e-3}},
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


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
