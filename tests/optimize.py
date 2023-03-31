# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

if config.state.backend == "torch":
    raise ValueError(f"optimize.py does not support torch backend at this time")

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.mpi_utils import _consistent_chunk_tensor, _warn0
from MuyGPyS._test.gp import (
    benchmark_pairwise_distances,
    benchmark_sample_full,
    BenchmarkGP,
    get_analytic_sigma_sq,
)
from MuyGPyS._test.utils import (
    _basic_opt_method_and_kwarg_options,
    _basic_nn_kwarg_options,
    _check_ndarray,
    _sq_rel_err,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.kernels import Hyperparameter, Matern, RBF
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.tensors import pairwise_tensor, crosswise_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import optimize_from_tensors
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.sigma_sq import muygps_sigma_sq_optim


class BenchmarkTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTestCase, cls).setUpClass()
        cls.data_count = 1001
        cls.its = 10
        cls.sigma_sqs = [1.0, 0.002353, 19.32]
        cls.sigma_tol = 5e-2
        cls.sim_train = dict()
        cls.x = np.linspace(-10.0, 10.0, cls.data_count).reshape(
            cls.data_count, 1
        )
        cls.train_features = cls.x[::2, :]
        cls.test_features = cls.x[1::2, :]
        cls.train_count, _ = cls.train_features.shape
        cls.test_count, _ = cls.test_features.shape
        cls.feature_count = 1
        cls.response_count = 1

        cls.k_kwargs = (
            {
                "kernel": Matern(
                    nu=Hyperparameter(0.5), length_scale=Hyperparameter(1e-2)
                ),
                "eps": HomoscedasticNoise(1e-5),
            },
            {
                "kernel": Matern(
                    nu=Hyperparameter(1.5), length_scale=Hyperparameter(1e-2)
                ),
                "eps": HomoscedasticNoise(1e-5),
            },
        )
        cls.sim_kwargs = (
            {
                "kernel": Matern(
                    nu=Hyperparameter(0.5),
                    length_scale=Hyperparameter(1e-2),
                    metric=None,
                ),
                "eps": HomoscedasticNoise(1e-5),
            },
            {
                "kernel": Matern(
                    nu=Hyperparameter(1.5),
                    length_scale=Hyperparameter(1e-2),
                    metric=None,
                ),
                "eps": HomoscedasticNoise(1e-5),
            },
        )
        cls.k_kwargs_opt = {
            "kernel": Matern(
                nu=Hyperparameter("sample", (0.1, 5.0)),
                length_scale=Hyperparameter(1e-2),
            ),
            "eps": HomoscedasticNoise(1e-5),
        }
        cls.model_count = len(cls.k_kwargs)
        cls.ss_count = len(cls.sigma_sqs)
        cls.gps = list()
        cls.ys = list()
        cls.train_targets_list = list()
        cls.test_targets_list = list()
        for i, kwargs in enumerate(cls.sim_kwargs):
            cls.gps.append(list())
            cls.ys.append(list())
            cls.test_targets_list.append(list())
            cls.train_targets_list.append(list())
            for j, ss in enumerate(cls.sigma_sqs):
                cls.gps[i].append(BenchmarkGP(**kwargs))
                cls.gps[i][j]._set_sigma_sq(mm.array([ss]))
                cls.ys[i].append(list())
                cls.test_targets_list[i].append(list())
                cls.train_targets_list[i].append(list())
                for k in range(cls.its):
                    cls.ys[i][j].append(list())
                    cls.test_targets_list[i][j].append(list())
                    cls.train_targets_list[i][j].append(list())
                    cls.ys[i][j][k] = benchmark_sample_full(
                        cls.gps[i][j], cls.train_features, cls.test_features
                    )
                    cls.test_targets_list[i][j][k] = cls.ys[i][j][k][
                        : cls.test_count
                    ].reshape(cls.test_count, 1)
                    cls.train_targets_list[i][j][k] = cls.ys[i][j][k][
                        cls.test_count :
                    ].reshape(cls.train_count, 1)

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)


class BenchmarkTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTest, cls).setUpClass()

    def test_types(self):
        self._check_ndarray(self.train_features, np.ftype, ctype=np.ndarray)
        self._check_ndarray(self.test_features, np.ftype, ctype=np.ndarray)
        self._check_ndarray(self.train_features, np.ftype, ctype=np.ndarray)
        for i, _ in enumerate(self.k_kwargs):
            for j, _ in enumerate(self.sigma_sqs):
                for k in range(self.its):
                    self._check_ndarray(
                        self.ys[i][j][k],
                        np.ftype,
                        ctype=np.ndarray,
                        shape=(self.data_count, 1),
                    )
                    self._check_ndarray(
                        self.test_targets_list[i][j][k],
                        np.ftype,
                        ctype=np.ndarray,
                        shape=(self.test_count, 1),
                    )
                    self._check_ndarray(
                        self.train_targets_list[i][j][k],
                        np.ftype,
                        ctype=np.ndarray,
                        shape=(self.train_count, 1),
                    )


class BenchmarkSigmaSqTest(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkSigmaSqTest, cls).setUpClass()

    def test_sigma_sq(self):
        mrse = 0.0
        for i, _ in enumerate(self.k_kwargs):
            for j, sigma_sq in enumerate(self.sigma_sqs):
                model = self.gps[i][j]
                pairwise_dists = benchmark_pairwise_distances(
                    self.x, metric=model.metric
                )
                K = model.kernel(pairwise_dists) + model.eps() * np.eye(
                    self.data_count
                )
                for k in range(self.its):
                    ss = get_analytic_sigma_sq(K, self.ys[i][j][k])
                    mrse += _sq_rel_err(sigma_sq, ss)
        mrse /= self.its * self.ss_count * self.model_count
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.sigma_tol)


class BenchmarkOptimTestCase(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkOptimTestCase, cls).setUpClass()
        cls.batch_count = cls.train_count
        cls.nn_count = 34
        cls.sm = ["analytic"]
        cls.nn_kwargs = _basic_nn_kwarg_options[0]

        nbrs_lookup = NN_Wrapper(
            cls.train_features, cls.nn_count, **cls.nn_kwargs
        )
        cls.nu_target_list = list()
        cls.batch_indices_list = list()
        cls.batch_nn_indices_list = list()
        cls.crosswise_diffs_list = list()
        cls.pairwise_diffs_list = list()
        cls.batch_nn_targets_list = list()
        for i, kwargs in enumerate(cls.k_kwargs):
            cls.nu_target_list.append(kwargs["kernel"].nu())
            cls.batch_indices_list.append(list())
            cls.batch_nn_indices_list.append(list())
            cls.crosswise_diffs_list.append(list())
            cls.pairwise_diffs_list.append(list())
            cls.batch_nn_targets_list.append(list())
            for j, sigma_sq in enumerate(cls.sigma_sqs):
                batch_indices, batch_nn_indices = sample_batch(
                    nbrs_lookup, cls.batch_count, cls.train_count
                )
                cls.batch_indices_list[i].append(batch_indices)
                cls.batch_nn_indices_list[i].append(batch_nn_indices)
                cls.crosswise_diffs_list[i].append(
                    crosswise_tensor(
                        cls.train_features,
                        cls.train_features,
                        cls.batch_indices_list[i][j],
                        cls.batch_nn_indices_list[i][j],
                    )
                )
                cls.pairwise_diffs_list[i].append(
                    pairwise_tensor(
                        cls.train_features,
                        cls.batch_nn_indices_list[i][j],
                    )
                )
                cls.batch_nn_targets_list[i].append(list())
                for k in range(cls.its):
                    cls.batch_nn_targets_list[i][j].append(
                        _consistent_chunk_tensor(
                            mm.array(
                                cls.train_targets_list[i][j][k][
                                    cls.batch_nn_indices_list[i][j], :
                                ]
                            )
                        )
                    )


class BenchmarkOptimTypesTest(BenchmarkOptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkOptimTypesTest, cls).setUpClass()

    def test_types(self):
        for i, kwargs in enumerate(self.k_kwargs):
            for j, sigma_sq in enumerate(self.sigma_sqs):
                self._check_ndarray(
                    self.batch_indices_list[i][j],
                    mm.itype,
                    shape=(self.batch_count,),
                )
                self._check_ndarray(
                    self.batch_nn_indices_list[i][j],
                    mm.itype,
                    shape=(self.batch_count, self.nn_count),
                )
                self._check_ndarray(
                    self.crosswise_diffs_list[i][j],
                    mm.ftype,
                    shape=(self.batch_count, self.nn_count, self.feature_count),
                )
                self._check_ndarray(
                    self.pairwise_diffs_list[i][j],
                    mm.ftype,
                    shape=(
                        self.batch_count,
                        self.nn_count,
                        self.nn_count,
                        self.feature_count,
                    ),
                )
                for k in range(self.its):
                    self._check_ndarray(
                        self.batch_nn_targets_list[i][j][k], mm.ftype
                    )
                    self.assertEqual(
                        self.batch_nn_targets_list[i][j][k].shape,
                        (self.batch_count, self.nn_count, self.response_count),
                    )


class BenchmarkSigmaSqOptimTest(BenchmarkOptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkSigmaSqOptimTest, cls).setUpClass()

    def test_sigma_sq_optim(self):
        mrse = 0.0
        for i, kwargs in enumerate(self.k_kwargs):
            for j, sigma_sq in enumerate(self.sigma_sqs):
                for k in range(self.its):
                    muygps = MuyGPS(**kwargs)
                    muygps = muygps_sigma_sq_optim(
                        muygps,
                        self.pairwise_diffs_list[i][j],
                        self.batch_nn_targets_list[i][j][k],
                        sigma_method="analytic",
                    )
                    estimate = muygps.sigma_sq()[0]

                    mrse += _sq_rel_err(sigma_sq, estimate)
        mrse /= self.its * self.model_count * self.ss_count
        print(f"optimizes with mean relative squared error {mrse}")
        self.assertLessEqual(mrse, self.sigma_tol)


class BenchmarkTensorsOptimTest(BenchmarkOptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTensorsOptimTest, cls).setUpClass()

    @parameterized.parameters(
        (
            (
                lm,
                om,
                opt_method_and_kwargs,
                sm,
            )
            for lm in ["lool"]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for sm in [None]
        )
    )
    def test_optim(
        self, loss_method, obj_method, opt_method_and_kwargs, sigma_method
    ):
        if True:
            _warn0(f"{self.__class__} is temporarily disabled.")
            return
        opt_method, opt_kwargs = opt_method_and_kwargs

        model_idx = 1
        ss_idx = 0
        mrse = 0.0
        for k in range(self.its):
            muygps = MuyGPS(**self.k_kwargs_opt)

            guess = muygps.kernel.hyperparameters["nu"]()

            batch_targets = _consistent_chunk_tensor(
                mm.array(
                    self.train_targets_list[model_idx][ss_idx][k][
                        self.batch_indices_list[model_idx][ss_idx], :
                    ]
                )
            )
            batch_nn_targets = _consistent_chunk_tensor(
                mm.array(
                    self.train_targets_list[model_idx][ss_idx][k][
                        self.batch_nn_indices_list[model_idx][ss_idx], :
                    ]
                )
            )
            self._check_ndarray(
                batch_targets,
                mm.ftype,
                shape=(self.batch_count, self.response_count),
            )
            self._check_ndarray(
                batch_nn_targets,
                mm.ftype,
                shape=(
                    self.batch_count,
                    self.nn_count,
                    self.response_count,
                ),
            )
            muygps = optimize_from_tensors(
                muygps,
                batch_targets,
                batch_nn_targets,
                self.crosswise_diffs_list[model_idx][ss_idx],
                self.pairwise_diffs_list[model_idx][ss_idx],
                loss_method=loss_method,
                obj_method=obj_method,
                opt_method=opt_method,
                sigma_method=sigma_method,
                **opt_kwargs,
            )
            estimate = muygps.kernel.hyperparameters["nu"]()
            print(
                f"iteration {k} found nu={estimate}, "
                f"(target={self.nu_target_list[model_idx]}) with initial guess "
                f"{guess}"
            )
            mrse += _sq_rel_err(self.nu_target_list[model_idx], estimate)
        mrse /= self.its
        print(f"optimizes with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertAlmostEqual(mrse, 0.0, 0)


if __name__ == "__main__":
    absltest.main()
