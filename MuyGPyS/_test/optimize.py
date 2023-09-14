# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import parameterized

import MuyGPyS._src.math as mm

from MuyGPyS._src.mpi_utils import _consistent_chunk_tensor
from MuyGPyS._test.utils import _check_ndarray, _sq_rel_err
from MuyGPyS._test.gp import benchmark_sample_full, BenchmarkGP
from MuyGPyS.gp.distortion import IsotropicDistortion, l2
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.tensors import pairwise_tensor, crosswise_tensor
from MuyGPyS.optimize import optimize_from_tensors
from MuyGPyS.optimize.batch import sample_batch


class BenchmarkTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTestCase, cls).setUpClass()
        cls.data_count = 1001
        cls.its = 13
        cls.sim_train = dict()
        cls.xs = mm.linspace(-10.0, 10.0, cls.data_count).reshape(
            cls.data_count, 1
        )
        cls.train_features = cls.xs[::2, :]
        cls.test_features = cls.xs[1::2, :]
        cls.train_count, _ = cls.train_features.shape
        cls.test_count, _ = cls.test_features.shape
        cls.feature_count = 1
        cls.response_count = 1
        cls.length_scale = 1e-2

        cls.sigma_tol = 5e-1
        cls.nu_tol = {
            "mse": 2.5e-1,
            "lool": 2.5e-1,
            "huber": 5e-1,
            "looph": 9e-1,
        }
        cls.length_scale_tol = {
            "mse": 3e-1,
            "lool": 3e-1,
            "huber": 5e-1,
            "looph": 9e-1,
        }

        cls.params = {
            "length_scale": ScalarHyperparameter(1e-1, (1e-2, 1e0)),
            "nu": ScalarHyperparameter(0.78, (1e-1, 2e0)),
            "eps": HomoscedasticNoise(1e-5, (1e-8, 1e-2)),
        }

        cls.gp = BenchmarkGP(
            kernel=Matern(
                nu=ScalarHyperparameter(cls.params["nu"]()),
                metric=IsotropicDistortion(
                    metric=l2,
                    length_scale=ScalarHyperparameter(
                        cls.params["length_scale"]()
                    ),
                ),
            ),
            eps=HomoscedasticNoise(cls.params["eps"]()),
        )
        cls.gp.sigma_sq._set(mm.array([5.0]))
        cls.ys = mm.zeros((cls.its, cls.data_count, cls.response_count))
        cls.train_responses = mm.zeros(
            (cls.its, cls.train_count, cls.response_count)
        )
        cls.test_responses = mm.zeros(
            (cls.its, cls.test_count, cls.response_count)
        )
        for i in range(cls.its):
            ys = benchmark_sample_full(
                cls.gp, cls.test_features, cls.train_features
            )
            cls.train_responses = mm.assign(
                cls.train_responses,
                ys[cls.test_count :, :],
                i,
                slice(None),
                slice(None),
            )
            cls.test_responses = mm.assign(
                cls.test_responses,
                ys[: cls.test_count, :],
                i,
                slice(None),
                slice(None),
            )

    def _optim_chassis(
        self,
        muygps,
        name,
        itr,
        nbrs_lookup,
        batch_count,
        loss_fn,
        obj_method,
        opt_method,
        sigma_method,
        opt_kwargs,
        loss_kwargs=dict(),
    ) -> float:
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, self.train_count
        )
        batch_crosswise_diffs = crosswise_tensor(
            self.train_features,
            self.train_features,
            batch_indices,
            batch_nn_indices,
        )
        batch_pairwise_diffs = pairwise_tensor(
            self.train_features,
            batch_nn_indices,
        )
        batch_targets = _consistent_chunk_tensor(
            self.train_responses[itr, batch_indices, :]
        )
        batch_nn_targets = _consistent_chunk_tensor(
            self.train_responses[itr, batch_nn_indices, :]
        )
        muygps = optimize_from_tensors(
            muygps,
            batch_targets,
            batch_nn_targets,
            batch_crosswise_diffs,
            batch_pairwise_diffs,
            loss_fn=loss_fn,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
            loss_kwargs=loss_kwargs,
            **opt_kwargs,
            verbose=False,  # TODO True,
        )
        estimate = muygps.kernel._hyperparameters[name]()
        return _sq_rel_err(self.params[name](), estimate)

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)
