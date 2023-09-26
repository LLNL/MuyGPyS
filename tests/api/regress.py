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
from MuyGPyS._test.api import RegressionAPITest
from MuyGPyS._test.utils import (
    _balanced_subsample,
    _basic_nn_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
)

from MuyGPyS.gp.deformation import (
    Isotropy,
    Anisotropy,
    F2,
    l2,
)
from MuyGPyS.gp.hyperparameter import AnalyticScale, ScalarHyperparameter, Scale
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize.loss import mse_fn

if config.state.backend == "torch":
    ValueError("MuyGPyS.examples.regress does not support torch!")
if config.state.backend == "mpi":
    ValueError("MuyGPyS.examples.regress does not support mpi!")

hardpath = "../data/"

stargal_dir = "star-gal/"

stargal_files = {
    "full": "galstar.pkl",
    "30": "embedded_30_galstar.pkl",
    "40": "embedded_40_galstar.pkl",
    "50": "embedded_50_galstar.pkl",
}

heaton_file = "heaton/sub_heaton.pkl"


class MultivariateStargalRegressTest(RegressionAPITest):
    @classmethod
    def setUpClass(cls):
        super(MultivariateStargalRegressTest, cls).setUpClass()
        with open(
            os.path.join(hardpath + stargal_dir, stargal_files["40"]), "rb"
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
            for lf in [mse_fn]
            # for nn_kwargs in _basic_nn_kwarg_options
            # for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for opt_fn_and_kwargs in [_basic_opt_fn_and_kwarg_options[0]]
            for k_kwargs in (
                (
                    1.0,
                    [
                        {
                            "kernel": Matern(
                                nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                                deformation=Isotropy(
                                    l2,
                                    length_scale=ScalarHyperparameter(1.5),
                                ),
                            ),
                            "noise": HomoscedasticNoise(1e-3),
                            "scale": AnalyticScale(),
                        },
                        {
                            "kernel": Matern(
                                nu=ScalarHyperparameter(0.5, (1e-1, 1e0)),
                                deformation=Isotropy(
                                    l2,
                                    length_scale=ScalarHyperparameter(1.5),
                                ),
                            ),
                            "noise": HomoscedasticNoise(1e-3),
                            "scale": AnalyticScale(),
                        },
                    ],
                ),
                (
                    1.0,
                    [
                        {
                            "kernel": RBF(
                                deformation=Isotropy(
                                    F2,
                                    length_scale=ScalarHyperparameter(1.5),
                                )
                            ),
                            "noise": HomoscedasticNoise(1e-3),
                            "scale": Scale(),
                        },
                        {
                            "kernel": RBF(
                                deformation=Isotropy(
                                    F2,
                                    length_scale=ScalarHyperparameter(1.5),
                                )
                            ),
                            "noise": HomoscedasticNoise(1e-3),
                            "scale": Scale(),
                        },
                    ],
                ),
            )
        )
    )
    def test_stargal_regress(
        self,
        nn_count,
        batch_count,
        loss_fn,
        opt_fn_and_kwargs,
        nn_kwargs,
        k_kwargs,
    ):
        target_mse, k_args = k_kwargs
        opt_fn, opt_kwargs = opt_fn_and_kwargs
        train = _balanced_subsample(self.embedded_40_train, 10000)
        test = _balanced_subsample(self.embedded_40_test, 1000)

        self._do_regress_test_chassis(
            train=train,
            test=test,
            target_mse=target_mse,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_args,
            opt_kwargs=opt_kwargs,
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
            (nn, bs, lf, opt_fn_and_kwargs, nn_kwargs, k_kwargs)
            for nn in [30]
            for bs in [500]
            for lf in [mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in _basic_nn_kwarg_options
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[0]
            # ]
            for k_kwargs in (
                (
                    11.0,
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                            deformation=Isotropy(
                                l2,
                                length_scale=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                        "scale": AnalyticScale(),
                    },
                ),
                (
                    11.0,
                    {
                        "kernel": Matern(
                            nu=ScalarHyperparameter("sample", (1e-1, 1e0)),
                            deformation=Anisotropy(
                                l2,
                                length_scale0=ScalarHyperparameter(1.5),
                                length_scale1=ScalarHyperparameter(1.5),
                            ),
                        ),
                        "noise": HomoscedasticNoise(1e-3),
                        "scale": Scale(),
                    },
                ),
                # (
                #     11.0,
                #     {
                #         "kern": "rbf",
                #         "metric": "F2",
                #         "length_scale": ScalarHyperparameter(1.5, "bounds": (0.5, 1e1)},
                #         "noise": HomoscedasticNoise(1e-3),
                #     },
                # ),
            )
        )
    )
    def test_heaton_regress(
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

        self._do_regress_test_chassis(
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


if __name__ == "__main__":
    if os.path.isdir(sys.argv[-1]):
        hardpath = sys.argv[-1]
        sys.argv = sys.argv[:-1]
    absltest.main()
