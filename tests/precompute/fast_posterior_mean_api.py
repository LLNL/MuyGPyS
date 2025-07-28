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

import MuyGPyS._src.math.numpy as np

from MuyGPyS._test.api import FastPosteriorMeanAPITest
from MuyGPyS._test.utils import (
    _balanced_subsample,
    _basic_nn_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
)

from MuyGPyS.gp.deformation import (
    Isotropy,
    Anisotropy,
    l2,
)
from MuyGPyS.gp.hyperparameter import ScalarParam, VectorParam
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn

if config.state.backend in ["mpi", "torch"]:
    raise ValueError("This test only supports numpy and jax backends!")


hardpath = "../data/"

stargal_dir = "star-gal/"

stargal_files = {
    "full": "galstar.pkl",
    "30": "embedded_30_galstar.pkl",
    "40": "embedded_40_galstar.pkl",
    "50": "embedded_50_galstar.pkl",
}

heaton_file = "heaton/sub_heaton.pkl"


class HeatonFastTest(FastPosteriorMeanAPITest):
    @classmethod
    def setUpClass(cls):
        super(HeatonFastTest, cls).setUpClass()
        with open(os.path.join(hardpath, heaton_file), "rb") as f:
            cls.train, cls.test = pkl.load(f)
            cls.train["output"] = np.squeeze(cls.train["output"])
            cls.test["output"] = np.squeeze(cls.test["output"])

    @parameterized.parameters(
        (
            (nn, bs, lf, opt_fn_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lf in [mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in (
                (
                    11.0,
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam("sample", (1e-1, 1e0)),
                            deformation=Isotropy(
                                l2, length_scale=ScalarParam(1.5)
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
                (
                    11.0,
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam("sample", (1e-1, 1e0)),
                            deformation=Anisotropy(
                                l2,
                                length_scale=VectorParam(
                                    ScalarParam(1.5), ScalarParam(1.5)
                                ),
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
            )
        )
    )
    def test_fast_posterior_mean(
        self,
        nn_count,
        batch_count,
        loss_fn,
        opt_fn_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, k_kwargs = k_kwargs
        opt_fn, opt_kwargs = opt_fn_and_kwargs

        self._do_fast_posterior_mean_test_chassis(
            train=self.train,
            test=self.test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=False,
        )


class StargalFastTest(FastPosteriorMeanAPITest):
    @classmethod
    def setUpClass(cls):
        super(StargalFastTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
        ) as f:
            cls.embedded_40_train, cls.embedded_40_test = pkl.load(f)
        # with open(os.path.join(hardpath, heaton_file), "rb") as f:
        #     cls.train, cls.test = pkl.load(f)

    @parameterized.parameters(
        (
            (nn, bs, lf, opt_fn_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lf in [mse_fn]
            for nn_kwargs in _basic_nn_kwarg_options
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[0]
            # ]
            for k_kwargs in (
                (
                    1.0,
                    {
                        "kernel": Matern(
                            smoothness=ScalarParam(0.5),
                            deformation=Isotropy(
                                l2, length_scale=ScalarParam(1.5)
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                    },
                ),
                # (
                #     1.0,
                #     {
                #         "kernel": RBF(
                #             deformation=Isotropy(
                #                 F2, length_scale=ScalarParam(1.5)
                #             )
                #         ),
                #         "noise": HomoscedasticNoise(1e-3),
                #     },
                # ),
            )
        )
    )
    def test_fast_posterior_mean(
        self,
        nn_count,
        batch_count,
        loss_fn,
        opt_fn_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, k_kwargs = k_kwargs
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)

        self._do_fast_posterior_mean_test_chassis(
            train=train,
            test=test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=False,
        )


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
