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
from MuyGPyS._test.gp import (
    benchmark_sample_full,
    BenchmarkGP,
    get_analytic_sigma_sq,
)
from MuyGPyS._test.utils import (
    _advanced_opt_method_and_kwarg_options,
    _basic_opt_method_and_kwarg_options,
    _basic_nn_kwarg_options,
    _check_ndarray,
    _sq_rel_err,
)
from MuyGPyS.gp import MuyGPS

from MuyGPyS.gp.distortion import IsotropicDistortion, l2
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.gp.tensors import pairwise_tensor, crosswise_tensor
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import optimize_from_tensors
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.experimental.chassis import (
    optimize_from_tensors_mini_batch,
)
from MuyGPyS.optimize.sigma_sq import muygps_sigma_sq_optim

if config.state.backend != "numpy":
    raise ValueError(
        "optimize.py only supports the numpy backend at this time"
    )


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
        loss_method,
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
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=sigma_method,
            loss_kwargs=loss_kwargs,
            **opt_kwargs,
            verbose=False,  # TODO True,
        )
        estimate = muygps.kernel._hyperparameters[name]()
        return _sq_rel_err(self.params[name](), estimate)

    def _optim_chassis_mini_batch(
        self,
        muygps,
        name,
        itr,
        nn_count,
        batch_count,
        loss_method,
        obj_method,
        sigma_method,
        loss_kwargs,
        opt_kwargs,
    ) -> float:
        (
            muygps,
            _,
            _,
            _,
            _,
        ) = optimize_from_tensors_mini_batch(
            muygps,
            self.train_features,
            self.train_responses[itr, :, :],
            nn_count,
            batch_count,
            self.train_count,
            num_epochs=1,  # Optimizing over one epoch (for now)
            keep_state=False,
            probe_previous=False,
            batch_features=None,
            loss_method=loss_method,
            obj_method=obj_method,
            sigma_method=sigma_method,
            loss_kwargs=loss_kwargs,
            verbose=False,  # TODO True,
            **opt_kwargs,
        )
        estimate = muygps.kernel._hyperparameters[name]()
        return _sq_rel_err(self.params[name](), estimate)

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)


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
        K = self.gp.kernel(pairwise_diffs)
        +self.gp.eps() * mm.eye(self.feature_count)
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
                eps=HomoscedasticNoise(self.params["eps"]()),
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

            muygps = muygps_sigma_sq_optim(
                muygps,
                batch_pairwise_diffs,
                batch_nn_targets,
                sigma_method="analytic",
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
                loss_kwargs_and_sigma_methods,
                om,
                opt_method_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_kwargs_and_sigma_methods in [
                ["lool", dict(), "analytic"],
                ["mse", dict(), None],
                ["huber", {"boundary_scale": 1.5}, None],
                ["looph", {"boundary_scale": 1.5}, "analytic"],
            ]
            for om in ["loo_crossval"]
            # for nn_kwargs in _basic_nn_kwarg_options
            for opt_method_and_kwargs in _basic_opt_method_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
        )
    )
    def test_nu(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_kwargs_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
    ):
        loss_method, loss_kwargs, sigma_method = loss_kwargs_and_sigma_methods
        opt_method, opt_kwargs = opt_method_and_kwargs

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
                eps=HomoscedasticNoise(self.params["eps"]()),
            )

            mrse += self._optim_chassis(
                muygps,
                "nu",
                i,
                nbrs_lookup,
                batch_count,
                loss_method,
                obj_method,
                opt_method,
                sigma_method,
                opt_kwargs,
                loss_kwargs=loss_kwargs,
            )
        mrse /= self.its
        print(f"optimizes nu with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertLessEqual(mrse, self.nu_tol[loss_method])

    @parameterized.parameters(
        (
            (
                b,
                n,
                loss_kwargs_and_sigma_methods,
                om,
                opt_method_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_kwargs_and_sigma_methods in [
                ["lool", dict(), "analytic"],
                ["mse", dict(), None],
                ["huber", {"boundary_scale": 1.5}, None],
            ]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in [
                _basic_opt_method_and_kwarg_options[1]
            ]
        )
    )
    def test_nu_mini_batch(
        self,
        batch_count,
        nn_count,
        loss_kwargs_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
    ):
        loss_method, loss_kwargs, sigma_method = loss_kwargs_and_sigma_methods
        _, opt_kwargs = opt_method_and_kwargs

        mrse = 0.0

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
                eps=HomoscedasticNoise(self.params["eps"]()),
            )

            mrse += self._optim_chassis_mini_batch(
                muygps,
                "nu",
                i,
                nn_count,
                batch_count,
                loss_method,
                obj_method,
                sigma_method,
                loss_kwargs,
                opt_kwargs,
            )
        mrse /= self.its
        print(f"optimizes nu with mean relative squared error {mrse}")
        # Is this a strong enough guarantee?
        self.assertLessEqual(mrse, self.nu_tol[loss_method])


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
                loss_and_sigma_methods,
                om,
                opt_method_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_and_sigma_methods in [["lool", "analytic"]]
            for om in ["loo_crossval"]
            # for nn_kwargs in _basic_nn_kwarg_options
            for opt_method_and_kwargs in _advanced_opt_method_and_kwarg_options
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            # for opt_method_and_kwargs in [
            #     _basic_opt_method_and_kwarg_options[0]
            # ]
        )
    )
    def test_length_scale(
        self,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
    ):
        loss_method, sigma_method = loss_and_sigma_methods
        opt_method, opt_kwargs = opt_method_and_kwargs

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
                eps=HomoscedasticNoise(self.params["eps"]()),
            )

            error_vector[i] = self._optim_chassis(
                muygps,
                "length_scale",
                i,
                nbrs_lookup,
                batch_count,
                loss_method,
                obj_method,
                opt_method,
                sigma_method,
                opt_kwargs,
            )
        median_error = mm.median(error_vector)
        print(
            "optimizes length_scale with "
            f"median relative squared error {median_error}"
        )
        # Is this a strong enough guarantee?
        self.assertLessEqual(median_error, self.length_scale_tol[loss_method])

    @parameterized.parameters(
        (
            (
                b,
                n,
                loss_kwargs_and_sigma_methods,
                om,
                opt_method_and_kwargs,
            )
            for b in [250]
            for n in [20]
            for loss_kwargs_and_sigma_methods in [["lool", dict(), "analytic"]]
            for om in ["loo_crossval"]
            for opt_method_and_kwargs in [
                _advanced_opt_method_and_kwarg_options[1]
            ]
        )
    )
    def test_length_scale_mini_batch(
        self,
        batch_count,
        nn_count,
        loss_kwargs_and_sigma_methods,
        obj_method,
        opt_method_and_kwargs,
    ):
        loss_method, loss_kwargs, sigma_method = loss_kwargs_and_sigma_methods
        _, opt_kwargs = opt_method_and_kwargs

        error_vector = mm.zeros((self.its,))

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
                eps=HomoscedasticNoise(self.params["eps"]()),
            )

            error_vector[i] = self._optim_chassis_mini_batch(
                muygps,
                "length_scale",
                i,
                nn_count,
                batch_count,
                loss_method,
                obj_method,
                sigma_method,
                loss_kwargs,
                opt_kwargs,
            )
        median_error = mm.median(error_vector)
        print(
            "optimizes length_scale with "
            f"median relative squared error {median_error}"
        )
        # Is this a strong enough guarantee?
        self.assertLessEqual(median_error, self.length_scale_tol[loss_method])


if __name__ == "__main__":
    absltest.main()
