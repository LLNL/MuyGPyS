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
        Kin = self.gp.kernel(pairwise_diffs) + self.gp.noise() * mm.eye(
            self.feature_count
        )
        for i in range(self.its):
            ss = get_analytic_scale(Kin, self.train_features)
            mrse += _sq_rel_err(self.gp.scale(), ss)

        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.scale_tol)


class AnalyticOptimTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(AnalyticOptimTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (b, n, nn_kwargs)
            for b in [250]
            for n in [20]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
        )
    )
    def test_scale_optim(
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
                    smoothness=ScalarParam(self.params["smoothness"]()),
                    deformation=Isotropy(
                        metric=l2,
                        length_scale=ScalarParam(self.params["length_scale"]()),
                    ),
                ),
                noise=HomoscedasticNoise(self.params["noise"]()),
                scale=AnalyticScale(),
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

            muygps = muygps.optimize_scale(
                batch_pairwise_diffs, batch_nn_targets
            )
            estimate = muygps.scale()[0]

            mrse += _sq_rel_err(self.gp.scale(), estimate)
        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.scale_tol)


class DownSampleOptimTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(DownSampleOptimTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (b, n, d, i, nn_kwargs)
            for b in [250]
            for n in [20]
            for d in [15]
            for i in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
        )
    )
    def test_scale_optim(
        self,
        batch_count,
        nn_count,
        down_count,
        iteration_count,
        nn_kwargs,
    ):
        mrse = 0.0

        nbrs_lookup = NN_Wrapper(self.train_features, nn_count, **nn_kwargs)

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
            _, batch_nn_indices = sample_batch(
                nbrs_lookup, batch_count, self.train_count
            )
            batch_pairwise_diffs = pairwise_tensor(
                self.train_features, batch_nn_indices
            )
            batch_nn_targets = _consistent_chunk_tensor(
                self.train_responses[i, batch_nn_indices, :]
            )

            muygps = muygps.optimize_scale(
                batch_pairwise_diffs, batch_nn_targets
            )
            estimate = muygps.scale()[0]

            mrse += _sq_rel_err(self.gp.scale(), estimate)
        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.scale_tol)


if __name__ == "__main__":
    absltest.main()
