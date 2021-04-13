# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import sys

import numpy as np
import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.testing.api_tests import ClassifyAPITest, RegressionAPITest
from MuyGPyS.examples.classify import example_lambdas
from MuyGPyS.testing.test_utils import (
    _basic_nn_kwarg_options,
    _fast_nn_kwarg_options,
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

heaton_file = "heaton/heaton.pkl"


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
            (
                nn,
                ed,
                ob,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ed in [40]
            for ob in [500]
            # for nn_kwargs in _fast_nn_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    0.96,
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "rbf",
                    0.96,
                    {
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                # (
                #     "nngp",
                #     0.935,
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.015,
                #         "sigma_sq": np.array([1.0]),
                #     },
                # ),
            )
        )
    )
    def test_classify_notrain_noembed(
        self,
        nn_count,
        embed_dim,
        opt_batch_size,
        nn_kwargs,
        kern,
        target_accuracy,
        hyper_dict,
    ):

        self._do_classify_test_chassis(
            train=self.embedded_40_train,
            test=self.embedded_40_test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            embed_dim=embed_dim,
            opt_batch_size=opt_batch_size,
            uq_batch_size=None,
            loss_method=None,
            embed_method=None,
            uq_objectives=None,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
        )

    @parameterized.parameters(
        (
            (
                nn,
                ed,
                ob,
                lm,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ed in [40]
            for ob in [500]
            for lm in ["log", "mse"]
            for nn_kwargs in _basic_nn_kwarg_options
            # for lm in ["log"]
            # for nn_kwargs in _fast_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    0.96,
                    {
                        # "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "rbf",
                    0.96,
                    {
                        # "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                # (
                #     "nngp",
                #     0.935,
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.015,
                #         "sigma_sq": np.array([1.0]),
                #     },
                # ),
            )
        )
    )
    def test_classify_noembed(
        self,
        nn_count,
        embed_dim,
        opt_batch_size,
        loss_method,
        nn_kwargs,
        kern,
        target_accuracy,
        hyper_dict,
    ):

        self._do_classify_test_chassis(
            train=self.embedded_40_train,
            test=self.embedded_40_test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            embed_dim=embed_dim,
            opt_batch_size=opt_batch_size,
            uq_batch_size=None,
            loss_method=loss_method,
            embed_method=None,
            uq_objectives=None,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
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
            (
                nn,
                ed,
                ob,
                ub,
                uq,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ed in [40]
            for ob in [500]
            for ub in [500]
            # for uq in [example_lambdas]
            # for nn_kwargs in _fast_nn_kwarg_options
            for uq in [None, example_lambdas]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    0.96,
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "rbf",
                    0.945,
                    {
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                # (
                #     "nngp",
                #     0.935,
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.015,
                #         "sigma_sq": np.array([1.0]),
                #     },
                # ),
            )
        )
    )
    def test_classify_notrain_noembed(
        self,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        uq_objectives,
        nn_kwargs,
        kern,
        target_accuracy,
        hyper_dict,
    ):

        self._do_classify_test_chassis(
            train=self.embedded_40_train,
            test=self.embedded_40_test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            embed_dim=embed_dim,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            loss_method=None,
            embed_method=None,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
        )

    @parameterized.parameters(
        (
            (
                nn,
                ed,
                ob,
                ub,
                lm,
                uq,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ed in [40]
            for ob in [500]
            for ub in [500]
            # for lm in ["log"]
            # for uq in [example_lambdas]
            # for e in [True]
            for lm in ["log", "mse"]
            for uq in [None, example_lambdas]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    0.96,
                    {
                        # "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "rbf",
                    0.945,
                    {
                        # "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                # (
                #     "nngp",
                #     0.935,
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.015,
                #         "sigma_sq": np.array([1.0]),
                #     },
                # ),
            )
        )
    )
    def test_classify_noembed(
        self,
        nn_count,
        embed_dim,
        opt_batch_size,
        uq_batch_size,
        loss_method,
        uq_objectives,
        nn_kwargs,
        kern,
        target_accuracy,
        hyper_dict,
    ):

        self._do_classify_test_chassis(
            train=self.embedded_40_train,
            test=self.embedded_40_test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            embed_dim=embed_dim,
            opt_batch_size=opt_batch_size,
            uq_batch_size=uq_batch_size,
            loss_method=loss_method,
            embed_method=None,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
        )


class HeatonTest(RegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(HeatonTest, cls).setUpClass()
        with open(os.path.join(hardpath, heaton_file), "rb") as f:
            cls.train, cls.test = pkl.load(f)

    @parameterized.parameters(
        (
            (
                nn,
                ob,
                vm,
                nn_kwargs,
                k_ta_dict[0],
                k_ta_dict[1],
                k_ta_dict[2],
            )
            for nn in [30]
            for ob in [500]
            for vm in [None, "diagonal"]
            for nn_kwargs in _basic_nn_kwarg_options
            # for vm in [None]
            # for nn_kwargs in _fast_nn_kwarg_options
            for k_ta_dict in (
                (
                    "matern",
                    3.5,
                    {
                        "nu": 0.42,
                        "length_scale": 1.0,
                        "eps": 0.001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "rbf",
                    6.0,
                    {
                        "length_scale": 1.5,
                        "eps": 0.001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "nngp",
                    6.0,
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.015,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
            )
        )
    )
    def test_regress_notrain(
        self,
        nn_count,
        batch_size,
        variance_mode,
        nn_kwargs,
        kern,
        target_mse,
        hyper_dict,
    ):
        self._do_regress_test_chassis(
            train=self.train,
            test=self.test,
            target_mse=target_mse,
            nn_count=nn_count,
            embed_dim=None,
            batch_size=batch_size,
            loss_method=None,
            variance_mode=variance_mode,
            embed_method=None,
            nn_kwargs=nn_kwargs,
            kern=kern,
            hyper_dict=hyper_dict,
        )


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
