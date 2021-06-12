# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import sys

import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.data.utils import balanced_subsample, subsample
from MuyGPyS.examples.classify import example_lambdas
from MuyGPyS.testing.api_tests import ClassifyAPITest, RegressionAPITest
from MuyGPyS.testing.test_utils import (
    _basic_nn_kwarg_options,
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
            (nn, bs, lm, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lm in ["log", "mse"]
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
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
                (
                    0.85,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
            )
        )
    )
    def test_classify(
        self, nn_count, batch_size, loss_method, nn_kwargs, k_kwargs
    ):
        target_accuracy, k_kwargs = k_kwargs
        train = balanced_subsample(self.embedded_40_train, 5000)
        test = balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_size=batch_size,
            loss_method=loss_method,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
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

    @parameterized.parameters(
        (
            (nn, bs, lm, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lm in ["log", "mse"]
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
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
                (
                    0.9,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
            )
        )
    )
    def test_classify(
        self, nn_count, batch_size, loss_method, nn_kwargs, k_kwargs
    ):
        target_accuracy, k_kwargs = k_kwargs
        train = balanced_subsample(self.embedded_40_train, 5000)
        test = balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_size=batch_size,
            loss_method=loss_method,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            verbose=False,
        )

    @parameterized.parameters(
        (
            (nn, obs, ubs, lm, uq, nn_kwargs, k_kwargs)
            for nn in [30]
            for obs in [500]
            for ubs in [500]
            for lm in ["log", "mse"]
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
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
                (
                    0.9,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
            )
        )
    )
    def test_classify_uq(
        self,
        nn_count,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        uq_objectives,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, k_kwargs = k_kwargs
        train = balanced_subsample(self.embedded_40_train, 10000)
        test = balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_uq_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            loss_method=loss_method,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
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
            (nn, bs, vm, lm, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            # for vm in [None]
            for vm in [None, "diagonal"]
            for nn_kwargs in _basic_nn_kwarg_options
            for lm in ["mse"]
            for k_kwargs in (
                (
                    11.0,
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-1, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-3},
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
                (
                    11.0,
                    {
                        "kern": "rbf",
                        "metric": "F2",
                        "length_scale": {"val": 1.5, "bounds": (0.5, 1e1)},
                        "eps": {"val": 1e-3},
                        "sigma_sq": [{"val": 1.0, "bounds": "fixed"}],
                    },
                ),
            )
        )
    )
    def test_regress(
        self,
        nn_count,
        batch_size,
        variance_mode,
        loss_method,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, k_kwargs = k_kwargs

        self._do_regress_test_chassis(
            train=self.train,
            test=self.test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_size=batch_size,
            loss_method=loss_method,
            variance_mode=variance_mode,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            verbose=False,
        )


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
