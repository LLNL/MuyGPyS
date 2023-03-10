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
    _exact_nn_kwarg_options,
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _sq_rel_err,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.distance import (
    pairwise_distances,
    crosswise_distances,
    make_train_tensors,
)
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import (
    sample_batch,
    sample_balanced_batch,
    full_filtered_batch,
)
from MuyGPyS.optimize.chassis import (
    optimize_from_tensors,
    optimize_from_indices,
)
from MuyGPyS.optimize.loss import get_loss_func
from MuyGPyS.optimize.objective import (
    make_loo_crossval_fn,
)
from MuyGPyS.optimize.sigma_sq import muygps_sigma_sq_optim, make_sigma_sq_optim


class BatchTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, b, nn_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for b in [10000, 1000, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_batch(
        self, data_count, feature_count, nn_count, batch_count, nn_kwargs
    ):
        data = _make_gaussian_matrix(data_count, feature_count)
        _check_ndarray(self.assertEqual, data, mm.ftype)
        nbrs_lookup = NN_Wrapper(data, nn_count, **nn_kwargs)
        indices, nn_indices = sample_batch(nbrs_lookup, batch_count, data_count)
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, indices, mm.itype)
        target_count = np.min((data_count, batch_count))
        self.assertEqual(indices.shape, (target_count,))
        self.assertEqual(nn_indices.shape, (target_count, nn_count))

    @parameterized.parameters(
        (
            (1000, f, r, nn, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_full_filtered_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = full_filtered_batch(nbrs_lookup, data["labels"])
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, _ in enumerate(indices):
            self.assertNotEqual(
                len(mm.unique(data["labels"][nn_indices[i, :]])), 1
            )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [10000, 1000, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["labels"], mm.itype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)
        self.assertEqual(indices.shape, (nn_indices.shape[0],))
        self.assertEqual(nn_indices.shape[1], nn_count)
        for i, _ in enumerate(indices):
            self.assertNotEqual(
                len(mm.unique(data["labels"][nn_indices[i, :]])), 1
            )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch_lo_dist(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["labels"], mm.itype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)

        hist, _ = np.array(
            np.histogram(data["labels"][indices], bins=response_count),
            dtype=object,
        )
        self.assertSequenceAlmostEqual(
            hist, (batch_count / response_count) * np.ones((response_count))
        )

    @parameterized.parameters(
        (
            (1000, f, r, nn, b, nn_kwargs)
            for f in [100, 10, 2]
            for r in [10, 2]
            for nn in [5, 10, 100]
            for b in [1000, 10000]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_sample_balanced_batch_hi_dist(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        batch_count,
        nn_kwargs,
    ):
        data = _make_gaussian_dict(data_count, feature_count, response_count)
        _check_ndarray(self.assertEqual, data["input"], mm.ftype)
        _check_ndarray(self.assertEqual, data["labels"], mm.itype)
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices, nn_indices = sample_balanced_batch(
            nbrs_lookup, data["labels"], batch_count
        )
        _check_ndarray(self.assertEqual, indices, mm.itype)
        _check_ndarray(self.assertEqual, nn_indices, mm.itype)

        target_count = np.min((data_count, batch_count))
        hist, _ = np.array(
            np.histogram(data["labels"][indices], bins=response_count),
            dtype=object,
        )
        self.assertGreaterEqual(
            np.mean(hist) + 0.1 * (target_count / response_count),
            target_count / response_count,
        )
        self.assertGreaterEqual(
            np.min(hist) + 0.45 * (target_count / response_count),
            target_count / response_count,
        )


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
                "kern": "matern",
                "metric": "l2",
                "nu": {"val": 0.5},
                "length_scale": {"val": 1e-2},
                "eps": {"val": 1e-5},
            },
            {
                "kern": "matern",
                "metric": "l2",
                "nu": {"val": 1.5},
                "length_scale": {"val": 1e-2},
                "eps": {"val": 1e-5},
            },
        )
        cls.k_kwargs_opt = {
            "kern": "matern",
            "metric": "l2",
            "nu": {"val": "sample", "bounds": (0.1, 5.0)},
            "length_scale": {"val": 1e-2},
            "eps": {"val": 1e-5},
        }
        cls.model_count = len(cls.k_kwargs)
        cls.ss_count = len(cls.sigma_sqs)
        cls.gps = list()
        cls.ys = list()
        cls.train_targets_list = list()
        cls.test_targets_list = list()
        for i, kwargs in enumerate(cls.k_kwargs):
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
                    self.x, metric=model.kernel.metric
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
        cls.crosswise_dists_list = list()
        cls.pairwise_dists_list = list()
        cls.batch_nn_targets_list = list()
        for i, kwargs in enumerate(cls.k_kwargs):
            cls.nu_target_list.append(kwargs["nu"]["val"])
            cls.batch_indices_list.append(list())
            cls.batch_nn_indices_list.append(list())
            cls.crosswise_dists_list.append(list())
            cls.pairwise_dists_list.append(list())
            cls.batch_nn_targets_list.append(list())
            for j, sigma_sq in enumerate(cls.sigma_sqs):
                batch_indices, batch_nn_indices = sample_batch(
                    nbrs_lookup, cls.batch_count, cls.train_count
                )
                cls.batch_indices_list[i].append(batch_indices)
                cls.batch_nn_indices_list[i].append(batch_nn_indices)
                cls.crosswise_dists_list[i].append(
                    crosswise_distances(
                        cls.train_features,
                        cls.train_features,
                        cls.batch_indices_list[i][j],
                        cls.batch_nn_indices_list[i][j],
                        metric=kwargs["metric"],
                    )
                )
                cls.pairwise_dists_list[i].append(
                    pairwise_distances(
                        cls.train_features,
                        cls.batch_nn_indices_list[i][j],
                        metric=kwargs["metric"],
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
                self._check_ndarray(self.batch_indices_list[i][j], mm.itype)
                self._check_ndarray(self.batch_nn_indices_list[i][j], mm.itype)
                self._check_ndarray(self.crosswise_dists_list[i][j], mm.ftype)
                self._check_ndarray(self.pairwise_dists_list[i][j], mm.ftype)
                self.assertEqual(
                    self.batch_indices_list[i][j].shape, (self.batch_count,)
                )
                self.assertEqual(
                    self.batch_nn_indices_list[i][j].shape,
                    (self.batch_count, self.nn_count),
                )
                self.assertEqual(
                    self.crosswise_dists_list[i][j].shape,
                    (self.batch_count, self.nn_count),
                )
                self.assertEqual(
                    self.pairwise_dists_list[i][j].shape,
                    (self.batch_count, self.nn_count, self.nn_count),
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
                        self.pairwise_dists_list[i][j],
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
                self.crosswise_dists_list[model_idx][ss_idx],
                self.pairwise_dists_list[model_idx][ss_idx],
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


# class GPIndicesOptimTest(parameterized.TestCase):
#     @parameterized.parameters(
#         (
#             (
#                 1001,
#                 b,
#                 n,
#                 nn_kwargs,
#                 loss_and_sigma_methods,
#                 om,
#                 opt_method_and_kwargs,
#                 k_kwargs,
#             )
#             for b in [250]
#             for n in [20]
#             for loss_and_sigma_methods in [["lool", None], ["mse", None]]
#             for om in ["loo_crossval"]
#             # for nn_kwargs in [_basic_nn_kwarg_options[0]]
#             # for opt_method_and_kwargs in [
#             #     _advanced_opt_method_and_kwarg_options[0]
#             # ]
#             for nn_kwargs in _basic_nn_kwarg_options
#             for opt_method_and_kwargs in _advanced_opt_method_and_kwarg_options
#             for k_kwargs in (
#                 (
#                     0.38,
#                     {
#                         "kern": "matern",
#                         "metric": "l2",
#                         "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
#                         "length_scale": {"val": 1.5},
#                         "eps": {"val": 1e-5},
#                     },
#                 ),
#             )
#         )
#     )
#     def test_hyper_optim_from_indices(
#         self,
#         data_count,
#         batch_count,
#         nn_count,
#         nn_kwargs,
#         loss_and_sigma_methods,
#         obj_method,
#         opt_method_and_kwargs,
#         k_kwargs,
#     ):
#         if config.state.backend == "torch" or config.state.backend == "jax":
#             _warn0(
#                 f"{self.__class__.__name__} uses {BenchmarkGP.__name__}, which "
#                 f"only supports numpy. Skipping"
#             )
#             return
#         target, kwargs = k_kwargs
#         loss_method, sigma_method = loss_and_sigma_methods
#         opt_method, opt_kwargs = opt_method_and_kwargs

#         # construct the observation locations
#         sim_train = dict()
#         sim_test = dict()
#         x = mm.linspace(-10.0, 10.0, data_count).reshape(data_count, 1)
#         sim_train["input"] = x[::2, :]
#         sim_test["input"] = x[1::2, :]
#         train_count = sim_train["input"].shape[0]
#         test_count = sim_test["input"].shape[0]

#         # compute nearest neighbor structure
#         nbrs_lookup = NN_Wrapper(sim_train["input"], nn_count, **nn_kwargs)
#         # nn_indices, _ = nbrs_lookup.get_nns(sim_test["input"])
#         batch_indices, batch_nn_indices = sample_batch(
#             nbrs_lookup, batch_count, train_count
#         )
#         # Make GP benchmark.
#         gp_kwargs = kwargs.copy()
#         gp_kwargs["nu"]["val"] = target
#         gp = BenchmarkGP(**gp_kwargs)

#         # Sample a response curve
#         y = benchmark_sample_full(gp, sim_test["input"], sim_train["input"])
#         sim_test["output"] = y[:test_count].reshape(test_count, 1)
#         sim_train["output"] = y[test_count:].reshape(train_count, 1)

#         # set up MuyGPS object
#         muygps = MuyGPS(**kwargs)

#         muygps = optimize_from_indices(
#             muygps,
#             batch_indices,
#             batch_nn_indices,
#             sim_train["input"],
#             sim_train["output"],
#             loss_method=loss_method,
#             obj_method=obj_method,
#             opt_method=opt_method,
#             sigma_method=sigma_method,
#             **opt_kwargs,
#         )

#         estimate = muygps.kernel.hyperparameters["nu"]()

#         rse = _sq_rel_err(target, estimate)
#         print(f"optimizes with relative squared error {rse}")
#         # Is this a strong enough guarantee?
#         self.assertAlmostEqual(rse, 0.0, 0)


class MethodsAgreementTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        data_count = 1001
        feature_count = 10
        response_count = 3
        batch_count = 250
        nn_count = 20

        data = _make_gaussian_dict(data_count, feature_count, response_count)
        nbrs_lookup = NN_Wrapper(
            data["input"], nn_count, **_exact_nn_kwarg_options[0]
        )
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, data_count
        )

        (
            cls.crosswise_dists,
            cls.pairwise_dists,
            cls.batch_targets,
            cls.batch_nn_targets,
        ) = make_train_tensors(
            "l2", batch_indices, batch_nn_indices, data["input"], data["output"]
        )

    def _make_x0(self, params):
        x0 = list()
        if "nu" in params:
            x0.append(params["nu"])
        if "length_scale" in params:
            x0.append(params["length_scale"])
        if "eps" in params:
            x0.append(params["eps"])
        return x0

    @parameterized.parameters(
        (
            (lm, k_kwargs)
            for lm in ["lool", "mse"]
            for k_kwargs in (
                (
                    {"nu": 0.38},
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {"val": 1.5},
                        "eps": {"val": 1e-5},
                    },
                ),
                (
                    {"nu": 0.38, "length_scale": 1.5, "eps": 1e-5},
                    {
                        "kern": "matern",
                        "metric": "l2",
                        "nu": {"val": "sample", "bounds": (1e-2, 1e0)},
                        "length_scale": {
                            "val": "sample",
                            "bounds": (1e-2, 1e0),
                        },
                        "eps": {"val": "sample", "bounds": (1e-6, 1e-3)},
                    },
                ),
            )
        )
    )
    def test_kernel_fn(self, loss_method, k_kwargs):
        if config.state.backend == "torch":
            _warn0(
                f"{self.__class__.__name__} does not support torch. Skipping"
            )
            return
        loss_fn = get_loss_func(loss_method)
        params, k_kwargs = k_kwargs
        muygps = MuyGPS(**k_kwargs)

        x0 = self._make_x0(params)

        array_kernel_fn = muygps.kernel.get_array_opt_fn()
        kwargs_kernel_fn = muygps.kernel.get_kwargs_opt_fn()

        K_array = array_kernel_fn(self.pairwise_dists, x0)
        Kcross_array = array_kernel_fn(self.crosswise_dists, x0)

        K_kwargs = kwargs_kernel_fn(self.pairwise_dists, **params)
        Kcross_kwargs = kwargs_kernel_fn(self.crosswise_dists, **params)

        self.assertTrue(mm.allclose(K_array, K_kwargs))
        self.assertTrue(mm.allclose(Kcross_array, Kcross_kwargs))

        array_mean_fn = muygps.get_array_opt_mean_fn()
        array_var_fn = muygps.get_array_opt_var_fn()
        array_sigma_fn = make_sigma_sq_optim("analytic", "scipy", muygps)
        kwargs_mean_fn = muygps.get_kwargs_opt_mean_fn()
        kwargs_var_fn = muygps.get_kwargs_opt_var_fn()
        kwargs_sigma_fn = make_sigma_sq_optim("analytic", "bayes", muygps)

        predictions_array = array_mean_fn(
            K_array, Kcross_array, self.batch_nn_targets, x0
        )
        predictions_kwargs = kwargs_mean_fn(
            K_kwargs, Kcross_kwargs, self.batch_nn_targets, **params
        )

        self.assertTrue(mm.allclose(predictions_array, predictions_kwargs))

        array_obj_fn = make_loo_crossval_fn(
            "scipy",
            loss_method,
            loss_fn,
            array_kernel_fn,
            array_mean_fn,
            array_var_fn,
            array_sigma_fn,
            self.pairwise_dists,
            self.crosswise_dists,
            self.batch_nn_targets,
            self.batch_targets,
        )
        kwargs_obj_fn = make_loo_crossval_fn(
            "bayes",
            loss_method,
            loss_fn,
            kwargs_kernel_fn,
            kwargs_mean_fn,
            kwargs_var_fn,
            kwargs_sigma_fn,
            self.pairwise_dists,
            self.crosswise_dists,
            self.batch_nn_targets,
            self.batch_targets,
        )

        array_val = array_obj_fn(x0)
        kwargs_val = kwargs_obj_fn(**params)

        self.assertAlmostEqual(array_val, -kwargs_val)


if __name__ == "__main__":
    absltest.main()
