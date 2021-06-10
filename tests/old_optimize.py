# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np


from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS.testing.gp import BenchmarkGP
from MuyGPyS.testing.opt import _old_optim_chassis
from MuyGPyS.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
    _basic_nn_kwarg_options,
)


class GPSigmaSqOptimTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (
                1001,
                10,
                b,
                n,
                nn_kwargs,
                k_dict[0],
                k_dict[1],
            )
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_dict in (
                (
                    "matern",
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                    },
                ),
                (
                    "rbf",
                    {"length_scale": 1.5, "eps": 0.00001},
                ),
                (
                    "nngp",
                    {
                        "sigma_w_sq": 1.5,
                        "sigma_b_sq": 1.0,
                        "eps": 0.00001,
                    },
                ),
            )
        )
    )
    def test_sigma_sq_optim(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        kern,
        hyper_dict,
    ):
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]
        mse = 0.0
        for i in range(its):
            gp = BenchmarkGP(kern=kern, **hyper_dict)
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            sim_train["output"] = y[:train_count].reshape(train_count, 1)
            sim_test["output"] = y[train_count:].reshape(test_count, 1)
            true_sigma_sq = gp.get_sigma_sq(y)
            global_sigmas, indiv_sigmas = _old_optim_chassis(
                sim_train,
                sim_train,
                nn_count,
                batch_count,
                kern=kern,
                hyper_dict=hyper_dict,
                nn_kwargs=nn_kwargs,
            )
            mse += (global_sigmas[0] - true_sigma_sq) ** 2
            # print(
            #     global_sigmas[0],
            #     true_sigma_sq,
            #     np.abs(global_sigmas[0] - true_sigma_sq),
            # )
        mse /= its
        print(f"optimized with mse {mse}")
        self.assertAlmostEqual(mse, 0.0, 0)


class GPOptimTest(parameterized.TestCase):
    """
    @NOTE[bwp] We are seeing bad performance on recovering true hyperparameters
    for all settings aside from the `matern`/`nu` pairing using exact nearest
    neighbors, where we are able to find low mse. Is this due to the example
    data (a 1d curve), or a poor choice of true hyperparameters? I am not sure.
    Likely the 1d problem is the source of trouble concerning HNSW, as HNSW is
    designed for high dimensional large data problems.
    """

    @parameterized.parameters(
        (
            (
                1001,
                10,
                b,
                n,
                nn_kwargs,
                k_p_b_dict[0],
                k_p_b_dict[1],
                k_p_b_dict[2],
                k_p_b_dict[3],
            )
            for b in [250]
            for n in [34]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_p_b_dict in (
                (
                    "matern",
                    "nu",
                    (1e-5, 1.0),
                    {
                        "nu": 0.38,
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                (
                    "rbf",
                    "length_scale",
                    (1.0, 2.0),
                    {
                        "length_scale": 1.5,
                        "eps": 0.00001,
                        "sigma_sq": np.array([1.0]),
                    },
                ),
                # (
                #     "nngp",
                #     "sigma_w_sq",
                #     (0.5, 3.0),
                #     {
                #         "sigma_w_sq": 1.5,
                #         "sigma_b_sq": 1.0,
                #         "eps": 0.00001,
                #         "sigma_sq": np.array([1.0]),
                #     },
                # ),
            )
        )
    )
    def test_hyper_optim(
        self,
        data_count,
        its,
        batch_count,
        nn_count,
        nn_kwargs,
        kern,
        param,
        optim_bounds,
        hyper_dict,
    ):
        sim_train = dict()
        sim_test = dict()
        x = np.linspace(-10.0, 10.0, data_count).reshape(1001, 1)
        sim_train["input"] = x[::2, :]
        sim_test["input"] = x[1::2, :]
        train_count = sim_train["input"].shape[0]
        test_count = sim_test["input"].shape[0]
        mse = 0.0
        for i in range(its):
            gp = BenchmarkGP(kern=kern, **hyper_dict)
            gp.fit(sim_train["input"], sim_test["input"])
            y = gp.simulate()
            sim_train["output"] = y[:train_count].reshape(train_count, 1)
            sim_test["output"] = y[train_count:].reshape(test_count, 1)
            subtracted_dict = {
                k: hyper_dict[k] for k in hyper_dict if k != param
            }
            tru_param = hyper_dict[param]
            est_param = _old_optim_chassis(
                sim_train,
                sim_train,
                nn_count,
                batch_count,
                kern=kern,
                hyper_dict=subtracted_dict,
                optim_bounds={param: optim_bounds},
                nn_kwargs=nn_kwargs,
            )[0]
            mse += (est_param - tru_param) ** 2
        mse /= its
        # Is this a strong enough guarantee?
        print(f"optimizes with mse {mse}")
        self.assertAlmostEqual(mse, 0.0, 1)


if __name__ == "__main__":
    absltest.main()
