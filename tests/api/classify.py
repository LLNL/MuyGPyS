# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import os
import sys

import pickle as pkl

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._test.api import ClassifyAPITest
from MuyGPyS._test.utils import (
    _balanced_subsample,
    _basic_nn_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
)
from MuyGPyS.gp.deformation import Isotropy, F2, l2
from MuyGPyS.examples.two_class_classify_uq import example_lambdas
from MuyGPyS.gp.hyperparameter import ScalarParam
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import cross_entropy_fn, mse_fn

if config.state.backend == "torch":
    ValueError("MuyGPyS.examples.classify does not support torch!")
if config.state.backend == "mpi":
    ValueError("MuyGPyS.examples.classify does not support mpi!")


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


class MNISTTest(ClassifyAPITest):
    @classmethod
    def setUpClass(cls):
        super(MNISTTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + mnist_dir, mnist_files["40"]), "rb"
        ) as f:
            train, test = pkl.load(f)
            cls.embedded_40_train = {
                "input": np.array(train["input"]),
                "output": np.array(train["output"]),
            }
            cls.embedded_40_test = {
                "input": np.array(test["input"]),
                "output": np.array(test["output"]),
            }

    @parameterized.parameters(
        (
            (nn, bs, lf, opt_fn_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lf in [cross_entropy_fn, mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[1]
            # ]
            # for nn_kwargs in [_basic_nn_kwarg_options[1]]
            for k_kwargs in (
                (
                    0.85,
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam(0.5, (1e-1, 1e0)),
                            deformation=Isotropy(
                                l2,
                                length_scale=ScalarParam(1.5),
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
            )
        )
    )
    def test_mnist_classify(
        self,
        nn_count,
        batch_count,
        loss_fn,
        opt_fn_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "jax" and config.state.ftype == "32":
            import warnings

            warnings.warn(
                "classify api chassis does not currently support jax in 32 "
                "bit mode."
            )
        target_accuracy, k_kwargs = k_kwargs
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 5000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
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
            (nn, bs, lf, opt_fn_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lf in [cross_entropy_fn, mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[0]
            # ]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for k_kwargs in (
                (
                    0.92,
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam(0.5, (1e-1, 1e0)),
                            deformation=Isotropy(
                                l2,
                                length_scale=ScalarParam(1.5),
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
                (
                    0.9,
                    {
                        "kernel": RBF(
                            deformation=Isotropy(
                                F2,
                                length_scale=ScalarParam(1.5, (0.5, 1e1)),
                            )
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
            )
        )
    )
    def test_classify(
        self,
        nn_count,
        batch_count,
        loss_fn,
        opt_fn_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, k_kwargs = k_kwargs
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 5000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
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
                lf,
                opt_fn_and_kwargs,
                uq,
                nn_kwargs,
                k_kwargs,
            )
            for nn in [30]
            for obs in [500]
            for ubs in [500]
            for uq in [example_lambdas]
            for lf in [cross_entropy_fn, mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[1]
            # ]
            # for nn_kwargs in [_basic_nn_kwarg_options[1]]
            for k_kwargs in (
                (
                    0.92,
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam(0.5, (1e-1, 1e0)),
                            deformation=Isotropy(
                                l2,
                                length_scale=ScalarParam(1.5),
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
                (
                    0.9,
                    {
                        "kernel": RBF(
                            deformation=Isotropy(
                                F2,
                                length_scale=ScalarParam(1.5),
                            )
                        ),
                        "noise": HomoscedasticNoise(1e-3),
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
        loss_fn,
        opt_fn_and_kwargs,
        uq_objectives,
        nn_kwargs,
        k_kwargs,
    ):
        target_accuracy, k_kwargs = k_kwargs
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)
        self._do_classify_uq_test_chassis(
            train=train,
            test=test,
            target_acc=target_accuracy,
            nn_count=nn_count,
            opt_batch_count=opt_batch_count,
            uq_batch_count=uq_batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            uq_objectives=uq_objectives,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            k_kwargs=k_kwargs,
            verbose=False,
        )


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
