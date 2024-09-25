# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import parameterized

from MuyGPyS._src.mpi_utils import _consistent_chunk_tensor
from MuyGPyS._test.utils import (
    _check_ndarray,
    _sq_rel_err,
    _basic_nn_kwarg_options,
)
from MuyGPyS._test.sampler import UnivariateSampler
from MuyGPyS.gp.deformation import Isotropy, l2
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.hyperparameter import ScalarParam, FixedScale
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch


class BenchmarkTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTestCase, cls).setUpClass()
        cls.data_count = 501
        cls.its = 10
        cls.train_ratio = 0.50
        cls.batch_count = 150
        cls.nn_count = 10

        cls.feature_count = 1
        cls.response_count = 1
        cls.length_scale = 1e-2

        cls.scale_tol = 5e-1
        cls.smoothness_tol = {
            "mse": 2.5e-1,
            "lool": 2.5e-1,
            "huber": 5e-1,
            "looph": 9e-1,
        }
        cls.length_scale_tol = {
            "mse": 9e-1,
            "lool": 9e-1,
            "huber": 9e-1,
            "looph": 9e-1,
        }

        cls.params = {
            "length_scale": ScalarParam(0.05, (1e-2, 1e0)),
            "smoothness": ScalarParam(2.0, (1e-1, 5)),
            "noise": HomoscedasticNoise(1e-5, (1e-8, 1e-2)),
            "scale": FixedScale(5.0),
        }

        cls.sampler = UnivariateSampler(
            data_count=cls.data_count,
            train_ratio=cls.train_ratio,
            kernel=Matern(
                smoothness=ScalarParam(cls.params["smoothness"]()),
                deformation=Isotropy(
                    metric=l2,
                    length_scale=ScalarParam(cls.params["length_scale"]()),
                ),
            ),
            noise=HomoscedasticNoise(cls.params["noise"]()),
        )
        cls.sampler.gp.scale._set(cls.params["scale"]())

        cls.train_features, cls.test_features = cls.sampler.features()
        cls.train_count = cls.train_features.shape[0]
        cls.test_count = cls.test_features.shape[0]

        cls.train_responses_list = list()
        cls.test_responses_list = list()
        for _ in range(cls.its):
            train_responses, test_responses = cls.sampler.sample()
            cls.train_responses_list.append(train_responses)
            cls.test_responses_list.append(test_responses)

        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features, cls.nn_count, **_basic_nn_kwarg_options[0]
        )
        cls.batch_indices_list = list()
        cls.batch_nn_indices_list = list()
        for _ in range(cls.its):
            batch_indices, batch_nn_indices = sample_batch(
                cls.nbrs_lookup, cls.batch_count, cls.train_count
            )
            cls.batch_indices_list.append(batch_indices)
            cls.batch_nn_indices_list.append(batch_nn_indices)

        cls.batch_crosswise_dists_list = list()
        cls.batch_pairwise_dists_list = list()
        cls.batch_targets_list = list()
        cls.batch_nn_targets_list = list()
        for i in range(cls.its):
            batch_crosswise_dists = (
                cls.sampler.gp.kernel.deformation.crosswise_tensor(
                    cls.train_features,
                    cls.train_features,
                    cls.batch_indices_list[i],
                    cls.batch_nn_indices_list[i],
                )
            )
            batch_pairwise_dists = (
                cls.sampler.gp.kernel.deformation.pairwise_tensor(
                    cls.train_features,
                    cls.batch_nn_indices_list[i],
                )
            )
            batch_targets = _consistent_chunk_tensor(
                cls.train_responses_list[i][cls.batch_indices_list[i]]
            )
            batch_nn_targets = _consistent_chunk_tensor(
                cls.train_responses_list[i][cls.batch_nn_indices_list[i]]
            )

            cls.batch_crosswise_dists_list.append(batch_crosswise_dists)
            cls.batch_pairwise_dists_list.append(batch_pairwise_dists)
            cls.batch_targets_list.append(batch_targets)
            cls.batch_nn_targets_list.append(batch_nn_targets)

    def _optim_chassis(
        self,
        muygps,
        name,
        itr,
        loss_fn,
        opt_fn,
        opt_kwargs,
        loss_kwargs=dict(),
    ) -> float:
        muygps = opt_fn(
            muygps,
            self.batch_targets_list[itr],
            self.batch_nn_targets_list[itr],
            self.batch_crosswise_dists_list[itr],
            self.batch_pairwise_dists_list[itr],
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
            **opt_kwargs,
            verbose=False,
        )
        estimate = muygps.kernel._hyperparameters[name]()
        print(
            f"\toptimizes {name} with {estimate} (target {self.params[name]()})"
            f" -- sq rel err = {_sq_rel_err(self.params[name](), estimate)}"
        )
        return _sq_rel_err(self.params[name](), estimate)

    def _check_ndarray(self, *args, shape=None, **kwargs):
        shape = tuple(x for x in shape if x > 1)
        return _check_ndarray(self.assertEqual, *args, shape=shape, **kwargs)
