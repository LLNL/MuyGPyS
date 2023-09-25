# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _pairwise_differences

from MuyGPyS import config
from MuyGPyS._src.mpi_utils import _consistent_chunk_tensor
from MuyGPyS._test.gp import get_analytic_sigma_sq
from MuyGPyS._test.optimize import BenchmarkTestCase
from MuyGPyS._test.utils import (
    _advanced_opt_fn_and_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
    _basic_nn_kwarg_options,
    _sq_rel_err,
)
from MuyGPyS.gp import MuyGPS

from MuyGPyS.gp.distortion import IsotropicDistortion, l2
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.sigma_sq import AnalyticSigmaSq, SigmaSq
from MuyGPyS.gp.tensors import pairwise_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import (
    lool_fn,
    mse_fn,
    pseudo_huber_fn,
    looph_fn,
)

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
        self._check_ndarray(
            self.train_responses,
            mm.ftype,
            ctype=mm.ndarray,
            shape=(self.its, self.train_count, self.response_count),
        )
        self._check_ndarray(
            self.test_responses,
            mm.ftype,
            ctype=mm.ndarray,
            shape=(self.its, self.test_count, self.response_count),
        )


class SigmaSqTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(SigmaSqTest, cls).setUpClass()

    def test_sigma_sq(self):
        mrse = 0.0
        pairwise_diffs = _pairwise_differences(self.train_features)
        K = self.gp.kernel(pairwise_diffs) + self.gp.noise() * mm.eye(
            self.feature_count
        )
        for i in range(self.its):
            ss = get_analytic_sigma_sq(K, self.train_features)
            mrse += _sq_rel_err(self.gp.sigma_sq(), ss)

        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.sigma_tol)


class SigmaSqOptimTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(SigmaSqOptimTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (b, n, nn_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
        )
    )
    def test_sigma_sq_optim(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
    ):
        mrse = 0.0

        nbrs_lookup = NN_Wrapper(self.train_features, nn_count, **nn_kwargs)

        for i in range(self.its):
            muygps = MuyGPS(
                kernel=Matern(
                    nu=ScalarHyperparameter(self.params["nu"]()),
                    metric=IsotropicDistortion(
                        metric=l2,
                        length_scale=ScalarHyperparameter(
                            self.params["length_scale"]()
                        ),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                sigma_sq=AnalyticSigmaSq(),
            )
            _, batch_nn_indices = sample_batch(
                nbrs_lookup, batch_count, self.train_count
            )
            batch_pairwise_diffs = pairwise_tensor(
                self.train_features, batch_nn_indices
            )
            batch_nn_targets = _consistent_chunk_tensor(
                self.train_responses[i, batch_nn_indices, :]
            )

            muygps = muygps.optimize_sigma_sq(
                batch_pairwise_diffs, batch_nn_targets
            )
            estimate = muygps.sigma_sq()[0]

            mrse += _sq_rel_err(self.gp.sigma_sq(), estimate)
        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.sigma_tol)


class NuTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(NuTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                b,
                n,
                nn_kwargs,
                loss_kwargs_and_sigma_sq,
                opt_fn_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_kwargs_and_sigma_sq in [
                ["lool", lool_fn, dict(), AnalyticSigmaSq()],
                ["mse", mse_fn, dict(), SigmaSq()],
                ["huber", pseudo_huber_fn, {"boundary_scale": 1.5}, SigmaSq()],
                ["looph", looph_fn, {"boundary_scale": 1.5}, AnalyticSigmaSq()],
            ]
            # for nn_kwargs in _basic_nn_kwarg_options
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[0]
            # ]
        )
    )
    def test_nu(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_kwargs_and_sigma_sq,
        opt_fn_and_kwargs,
    ):
        (
            loss_name,
            loss_fn,
            loss_kwargs,
            sigma_sq,
        ) = loss_kwargs_and_sigma_sq
        opt_fn, opt_kwargs = opt_fn_and_kwargs

        mrse = 0.0

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(self.train_features, nn_count, **nn_kwargs)

        for i in range(self.its):
            # set up MuyGPS object
            muygps = MuyGPS(
                kernel=Matern(
                    nu=ScalarHyperparameter(
                        "sample", self.params["nu"].get_bounds()
                    ),
                    metric=IsotropicDistortion(
                        metric=l2,
                        length_scale=ScalarHyperparameter(
                            self.params["length_scale"]()
                        ),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                sigma_sq=sigma_sq,
            )

            mrse += self._optim_chassis(
                muygps,
                "nu",
                i,
                nbrs_lookup,
                batch_count,
                loss_fn,
                opt_fn,
                opt_kwargs,
                loss_kwargs=loss_kwargs,
            )
        mrse /= self.its
        print(f"optimizes nu with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertLessEqual(mrse, self.nu_tol[loss_name])


class LengthScaleTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(LengthScaleTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                b,
                n,
                nn_kwargs,
                loss_and_sigma_sq,
                opt_fn_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_and_sigma_sq in [["lool", lool_fn, AnalyticSigmaSq()]]
            # for nn_kwargs in _basic_nn_kwarg_options
            for opt_fn_and_kwargs in _advanced_opt_fn_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_fn_and_kwargs in [
            #     _basic_opt_fn_and_kwarg_options[0]
            # ]
        )
    )
    def test_length_scale(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_and_sigma_sq,
        opt_fn_and_kwargs,
    ):
        loss_name, loss_fn, sigma_sq = loss_and_sigma_sq
        opt_fn, opt_kwargs = opt_fn_and_kwargs

        error_vector = mm.zeros((self.its,))

        # compute nearest neighbor structure
        nbrs_lookup = NN_Wrapper(self.train_features, nn_count, **nn_kwargs)

        for i in range(self.its):
            # set up MuyGPS object
            muygps = MuyGPS(
                kernel=Matern(
                    nu=ScalarHyperparameter(self.params["nu"]()),
                    metric=IsotropicDistortion(
                        metric=l2,
                        length_scale=ScalarHyperparameter(
                            "sample", self.params["length_scale"].get_bounds()
                        ),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                sigma_sq=sigma_sq,
            )

            error_vector[i] = self._optim_chassis(
                muygps,
                "length_scale",
                i,
                nbrs_lookup,
                batch_count,
                loss_fn,
                opt_fn,
                opt_kwargs,
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
