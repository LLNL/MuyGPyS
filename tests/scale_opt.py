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
from MuyGPyS._test.gp import get_analytic_scale
from MuyGPyS._test.optimize import BenchmarkTestCase
from MuyGPyS._test.utils import _basic_nn_kwarg_options, _sq_rel_err
from MuyGPyS.gp import MuyGPS

from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.hyperparameter import (
    AnalyticScale,
    DownSampleScale,
    ScalarParam,
)
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.tensors import pairwise_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch

if config.state.backend != "numpy":
    raise ValueError("optimize.py only supports the numpy backend at this time")


class ScaleTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(ScaleTest, cls).setUpClass()

    def test_scale(self):
        mrse = 0.0
        pairwise_diffs = _pairwise_differences(self.train_features)
        Kin = self.sampler.gp.kernel(
            pairwise_diffs
        ) + self.sampler.gp.noise() * mm.eye(self.train_count)

        for i in range(self.its):
            ss = get_analytic_scale(Kin, self.train_responses_list[i])
            mrse += _sq_rel_err(self.params["scale"](), ss)

        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.scale_tol)


class AnalyticOptimTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(AnalyticOptimTest, cls).setUpClass()

    def test_scale_optim(self):
        mrse = 0.0

        for i in range(self.its):
            muygps = MuyGPS(
                kernel=Matern(
                    smoothness=ScalarParam(self.params["smoothness"]()),
                    deformation=Isotropy(
                        metric=l2,
                        length_scale=ScalarParam(self.params["length_scale"]()),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                scale=AnalyticScale(),
            )

            muygps = muygps.optimize_scale(
                self.batch_pairwise_diffs_list[i], self.batch_nn_targets_list[i]
            )
            estimate = muygps.scale()

            mrse += _sq_rel_err(self.params["scale"](), estimate)
        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.scale_tol)


class DownSampleOptimTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(DownSampleOptimTest, cls).setUpClass()

    @parameterized.parameters(((d, i) for d in [8] for i in [10]))
    def test_scale_optim(self, down_count, iteration_count):
        mrse = 0.0

        for i in range(self.its):
            muygps = MuyGPS(
                kernel=Matern(
                    smoothness=ScalarParam(self.params["smoothness"]()),
                    deformation=Isotropy(
                        metric=l2,
                        length_scale=ScalarParam(self.params["length_scale"]()),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                scale=DownSampleScale(
                    down_count=down_count, iteration_count=iteration_count
                ),
            )

            muygps = muygps.optimize_scale(
                self.batch_pairwise_diffs_list[i], self.batch_nn_targets_list[i]
            )
            estimate = muygps.scale()

            mrse += _sq_rel_err(self.params["scale"](), estimate)
        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.scale_tol)


if __name__ == "__main__":
    absltest.main()
