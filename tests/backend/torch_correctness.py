# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

if config.state.torch_enabled is False:
    raise ValueError(f"Bad attempt to run torch-only code with torch diabled.")

import torch
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from MuyGPyS._test.utils import (
    _make_gaussian_matrix,
    _make_gaussian_data,
    _exact_nn_kwarg_options,
)
from MuyGPyS.gp.muygps import MuyGPS
from MuyGPyS.gp.muygps import MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS._src.gp.distance.numpy import (
    _pairwise_distances as pairwise_distances_n,
    _crosswise_distances as crosswise_distances_n,
    _make_train_tensors as make_train_tensors_n,
    _make_fast_regress_tensors as make_fast_regress_tensors_n,
    _fast_nn_update as fast_nn_update_n,
)
from MuyGPyS._src.gp.distance.torch import (
    _pairwise_distances as pairwise_distances_t,
    _crosswise_distances as crosswise_distances_t,
    _make_train_tensors as make_train_tensors_t,
    _make_fast_regress_tensors as make_fast_regress_tensors_t,
    _fast_nn_update as fast_nn_update_t,
)
from MuyGPyS._src.gp.kernels.numpy import (
    _rbf_fn as rbf_fn_n,
    _matern_05_fn as matern_05_fn_n,
    _matern_15_fn as matern_15_fn_n,
    _matern_25_fn as matern_25_fn_n,
    _matern_inf_fn as matern_inf_fn_n,
    _matern_gen_fn as matern_gen_fn_n,
)
from MuyGPyS._src.gp.kernels.torch import (
    _rbf_fn as rbf_fn_t,
    _matern_05_fn as matern_05_fn_t,
    _matern_15_fn as matern_15_fn_t,
    _matern_25_fn as matern_25_fn_t,
    _matern_inf_fn as matern_inf_fn_t,
    _matern_gen_fn as matern_gen_fn_t,
)
from MuyGPyS._src.gp.muygps.numpy import (
    _muygps_compute_solve as muygps_compute_solve_n,
    _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_n,
    _muygps_fast_regress_solve as muygps_fast_regress_solve_n,
    _mmuygps_fast_regress_solve as mmuygps_fast_regress_solve_n,
    _muygps_fast_regress_precompute as muygps_fast_regress_precompute_n,
)
from MuyGPyS._src.gp.muygps.torch import (
    _muygps_compute_solve as muygps_compute_solve_t,
    _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_t,
    _muygps_fast_regress_solve as muygps_fast_regress_solve_t,
    _mmuygps_fast_regress_solve as mmuygps_fast_regress_solve_t,
    _muygps_fast_regress_precompute as muygps_fast_regress_precompute_t,
)
from MuyGPyS._src.gp.noise.numpy import (
    _homoscedastic_perturb as homoscedastic_perturb_n,
)
from MuyGPyS._src.gp.noise.torch import (
    _homoscedastic_perturb as homoscedastic_perturb_t,
)
from MuyGPyS._src.optimize.sigma_sq.numpy import (
    _analytic_sigma_sq_optim as analytic_sigma_sq_optim_n,
)
from MuyGPyS._src.optimize.sigma_sq.torch import (
    _analytic_sigma_sq_optim as analytic_sigma_sq_optim_t,
)
from MuyGPyS.optimize.sigma_sq import (
    make_kwargs_analytic_sigma_sq_optim,
    make_array_analytic_sigma_sq_optim,
)

from MuyGPyS._src.optimize.loss.numpy import (
    _mse_fn as mse_fn_n,
    _cross_entropy_fn as cross_entropy_fn_n,
    _lool_fn as lool_fn_n,
)
from MuyGPyS._src.optimize.loss.torch import (
    _mse_fn as mse_fn_t,
    _cross_entropy_fn as cross_entropy_fn_t,
    _lool_fn as lool_fn_t,
)
from MuyGPyS.optimize.objective import make_loo_crossval_fn


class DistanceTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(DistanceTestCase, cls).setUpClass()
        cls.train_count = 1000
        cls.test_count = 100
        cls.feature_count = 10
        cls.response_count = 1
        cls.nn_count = 40
        cls.batch_count = 500
        cls.length_scale = 1.0
        cls.nu = 0.5
        cls.nu_bounds = (1e-1, 1e1)
        cls.eps = 1e-3
        cls.k_kwargs = {
            "kern": "matern",
            "length_scale": {"val": cls.length_scale},
            "nu": {"val": cls.nu},
            "eps": {"val": cls.eps},
        }
        cls.train_features_n = _make_gaussian_matrix(
            cls.train_count, cls.feature_count
        )
        cls.train_features_t = torch.from_numpy(cls.train_features_n)
        cls.train_responses_n = _make_gaussian_matrix(
            cls.train_count, cls.response_count
        )
        cls.train_responses_t = torch.from_numpy(cls.train_responses_n)
        cls.test_features_n = _make_gaussian_matrix(
            cls.test_count, cls.feature_count
        )
        cls.test_features_t = torch.from_numpy(cls.test_features_n)
        cls.test_responses_n = _make_gaussian_matrix(
            cls.test_count, cls.response_count
        )
        cls.test_responses_t = torch.from_numpy(cls.test_responses_n)
        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features_n, cls.nn_count, **_exact_nn_kwarg_options[0]
        )
        cls.muygps = MuyGPS(**cls.k_kwargs)
        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_t = torch.from_numpy(cls.batch_indices_n)
        cls.batch_nn_indices_t = torch.from_numpy(cls.batch_nn_indices_n)


class DistanceTest(DistanceTestCase):
    @classmethod
    def setUpClass(cls):
        super(DistanceTest, cls).setUpClass()

    def test_pairwise_distances(self):
        self.assertTrue(
            np.allclose(
                pairwise_distances_n(
                    self.train_features_n, self.batch_nn_indices_n
                ),
                pairwise_distances_t(
                    self.train_features_t, self.batch_nn_indices_t
                ),
            )
        )

    def test_crosswise_distances(self):
        self.assertTrue(
            np.allclose(
                crosswise_distances_n(
                    self.train_features_n,
                    self.train_features_n,
                    self.batch_indices_n,
                    self.batch_nn_indices_n,
                ),
                crosswise_distances_t(
                    self.train_features_t,
                    self.train_features_t,
                    self.batch_indices_t,
                    self.batch_nn_indices_t,
                ),
            )
        )

    def test_make_train_tensors(self):
        (
            crosswise_dists_n,
            pairwise_dists_n,
            batch_targets_n,
            batch_nn_targets_n,
        ) = make_train_tensors_n(
            self.muygps.kernel.metric,
            self.batch_indices_n,
            self.batch_nn_indices_n,
            self.train_features_n,
            self.train_responses_n,
        )
        (
            crosswise_dists_t,
            pairwise_dists_t,
            batch_targets_t,
            batch_nn_targets_t,
        ) = make_train_tensors_t(
            self.muygps.kernel.metric,
            self.batch_indices_t,
            self.batch_nn_indices_t,
            self.train_features_t,
            self.train_responses_t,
        )
        self.assertTrue(np.allclose(crosswise_dists_n, crosswise_dists_t))
        self.assertTrue(np.allclose(pairwise_dists_n, pairwise_dists_t))
        self.assertTrue(np.allclose(batch_targets_n, batch_targets_t))
        self.assertTrue(np.allclose(batch_nn_targets_n, batch_nn_targets_t))


class KernelTestCase(DistanceTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()
        (
            cls.crosswise_dists_n,
            cls.pairwise_dists_n,
            cls.batch_targets_n,
            cls.batch_nn_targets_n,
        ) = make_train_tensors_n(
            cls.muygps.kernel.metric,
            cls.batch_indices_n,
            cls.batch_nn_indices_n,
            cls.train_features_n,
            cls.train_responses_n,
        )
        (
            cls.crosswise_dists_t,
            cls.pairwise_dists_t,
            cls.batch_targets_t,
            cls.batch_nn_targets_t,
        ) = make_train_tensors_t(
            cls.muygps.kernel.metric,
            cls.batch_indices_t,
            cls.batch_nn_indices_t,
            cls.train_features_t,
            cls.train_responses_t,
        )


class KernelTest(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTest, cls).setUpClass()

    def test_crosswise_rbf(self):
        self.assertTrue(
            np.allclose(
                rbf_fn_n(
                    self.crosswise_dists_n, length_scale=self.length_scale
                ),
                rbf_fn_t(
                    self.crosswise_dists_t, length_scale=self.length_scale
                ),
            )
        )

    def test_pairwise_rbf(self):
        self.assertTrue(
            np.allclose(
                rbf_fn_n(self.pairwise_dists_n, length_scale=self.length_scale),
                rbf_fn_t(self.pairwise_dists_t, length_scale=self.length_scale),
            )
        )

    def test_crosswise_matern(self):
        self.assertTrue(
            np.allclose(
                matern_05_fn_n(
                    self.crosswise_dists_n, length_scale=self.length_scale
                ),
                matern_05_fn_t(
                    self.crosswise_dists_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_15_fn_n(
                    self.crosswise_dists_n, length_scale=self.length_scale
                ),
                matern_15_fn_t(
                    self.crosswise_dists_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_25_fn_n(
                    self.crosswise_dists_n, length_scale=self.length_scale
                ),
                matern_25_fn_t(
                    self.crosswise_dists_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_inf_fn_n(
                    self.crosswise_dists_n, length_scale=self.length_scale
                ),
                matern_inf_fn_t(
                    self.crosswise_dists_t, length_scale=self.length_scale
                ),
            )
        )

    def test_pairwise_matern(self):
        self.assertTrue(
            np.allclose(
                matern_05_fn_n(
                    self.pairwise_dists_n, length_scale=self.length_scale
                ),
                matern_05_fn_t(
                    self.pairwise_dists_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_15_fn_n(
                    self.pairwise_dists_n, length_scale=self.length_scale
                ),
                matern_15_fn_t(
                    self.pairwise_dists_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_25_fn_n(
                    self.pairwise_dists_n, length_scale=self.length_scale
                ),
                matern_25_fn_t(
                    self.pairwise_dists_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_inf_fn_n(
                    self.pairwise_dists_n, length_scale=self.length_scale
                ),
                matern_inf_fn_t(
                    self.pairwise_dists_t, length_scale=self.length_scale
                ),
            )
        )


class MuyGPSTestCase(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTestCase, cls).setUpClass()
        cls.K_n = matern_05_fn_n(
            cls.pairwise_dists_n, length_scale=cls.length_scale
        )
        cls.homoscedastic_K_n = homoscedastic_perturb_n(
            cls.K_n, cls.muygps.eps()
        )

        cls.K_t = matern_05_fn_t(
            cls.pairwise_dists_t, length_scale=cls.length_scale
        )
        cls.homoscedastic_K_t = homoscedastic_perturb_t(
            cls.K_t, cls.muygps.eps()
        )

        cls.Kcross_n = matern_05_fn_n(
            cls.crosswise_dists_n, length_scale=cls.length_scale
        )
        cls.Kcross_t = matern_05_fn_t(
            cls.crosswise_dists_t, length_scale=cls.length_scale
        )


class MuyGPSTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTest, cls).setUpClass()

    def test_homoscedastic_noise(self):
        self.assertTrue(
            np.allclose(self.homoscedastic_K_n, self.homoscedastic_K_t)
        )

    def test_compute_solve(self):
        self.assertTrue(
            np.allclose(
                muygps_compute_solve_n(
                    self.homoscedastic_K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                muygps_compute_solve_t(
                    self.homoscedastic_K_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                ),
            )
        )

    def test_diagonal_variance(self):
        self.assertTrue(
            np.allclose(
                muygps_compute_diagonal_variance_n(
                    self.K_n, self.Kcross_n, self.muygps.eps()
                ),
                muygps_compute_diagonal_variance_t(
                    self.K_t, self.Kcross_t, self.muygps.eps()
                ),
            )
        )

    def test_sigma_sq_optim(self):
        self.assertTrue(
            np.allclose(
                analytic_sigma_sq_optim_n(
                    self.K_n,
                    self.batch_nn_targets_n,
                    self.muygps.eps(),
                ),
                analytic_sigma_sq_optim_t(
                    self.K_t,
                    self.batch_nn_targets_t,
                    self.muygps.eps(),
                ),
            )
        )


class FastPredictTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastPredictTestCase, cls).setUpClass()
        cls.nn_indices_all_n, _ = cls.nbrs_lookup.get_batch_nns(
            np.arange(0, cls.train_count)
        )
        cls.nn_indices_all_n = np.array(cls.nn_indices_all_n)
        (
            cls.K_fast_n,
            cls.train_nn_targets_fast_n,
        ) = make_fast_regress_tensors_n(
            cls.muygps.kernel.metric,
            cls.nn_indices_all_n,
            cls.train_features_n,
            cls.train_responses_n,
        )

        cls.fast_regress_coeffs_n = muygps_fast_regress_precompute_n(
            cls.K_fast_n, cls.muygps.eps(), cls.train_nn_targets_fast_n
        )

        cls.test_neighbors_n, _ = cls.nbrs_lookup.get_nns(cls.test_features_n)
        cls.closest_neighbor_n = cls.test_neighbors_n[:, 0]
        cls.closest_set_n = cls.nn_indices_all_n[cls.closest_neighbor_n]

        cls.new_nn_indices_n = fast_nn_update_n(cls.nn_indices_all_n)
        cls.closest_set_new_n = cls.new_nn_indices_n[
            cls.closest_neighbor_n
        ].astype(int)
        cls.crosswise_dists_fast_n = crosswise_distances_n(
            cls.test_features_n,
            cls.train_features_n,
            np.arange(0, cls.test_count),
            cls.closest_set_new_n,
        )

        kernel_func_n = matern_05_fn_n
        cls.Kcross_fast_n = kernel_func_n(
            cls.crosswise_dists_fast_n, cls.length_scale
        )

        cls.nn_indices_all_t, _ = cls.nbrs_lookup.get_batch_nns(
            torch.arange(0, cls.train_count)
        )
        cls.nn_indices_all_t = torch.from_numpy(cls.nn_indices_all_t)
        (
            cls.K_fast_t,
            cls.train_nn_targets_fast_t,
        ) = make_fast_regress_tensors_t(
            cls.muygps.kernel.metric,
            cls.nn_indices_all_t,
            cls.train_features_t,
            cls.train_responses_t,
        )

        cls.fast_regress_coeffs_t = muygps_fast_regress_precompute_t(
            cls.K_fast_t, cls.muygps.eps(), cls.train_nn_targets_fast_t
        )

        cls.test_neighbors_t, _ = cls.nbrs_lookup.get_nns(cls.test_features_t)
        cls.closest_neighbor_t = cls.test_neighbors_t[:, 0]
        cls.closest_set_t = cls.nn_indices_all_t[cls.closest_neighbor_t]

        cls.new_nn_indices_t = fast_nn_update_t(cls.nn_indices_all_t)
        cls.closest_set_new_t = cls.new_nn_indices_t[cls.closest_neighbor_t]
        cls.crosswise_dists_fast_t = crosswise_distances_t(
            cls.test_features_t,
            cls.train_features_t,
            torch.arange(0, cls.test_count),
            cls.closest_set_new_t,
        )

        kernel_func_t = matern_05_fn_t
        cls.Kcross_fast_t = kernel_func_t(
            cls.crosswise_dists_fast_t, cls.length_scale
        )

    def test_fast_nn_update(self):
        self.assertTrue(
            np.allclose(
                fast_nn_update_t(self.nn_indices_all_t),
                fast_nn_update_n(self.nn_indices_all_n),
            )
        )

    def test_make_fast_regress_tensors(self):
        self.assertTrue(np.allclose(self.K_fast_n, self.K_fast_t))
        self.assertTrue(
            np.allclose(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_t
            )
        )

    def test_fast_predict(self):
        self.assertTrue(
            np.allclose(
                muygps_fast_regress_solve_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                ),
                muygps_fast_regress_solve_t(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_t[self.closest_neighbor_t, :],
                ),
            )
        )

    def test_fast_predict_coeffs(self):
        self.assertTrue(
            np.allclose(
                self.fast_regress_coeffs_n,
                self.fast_regress_coeffs_t,
            )
        )


class FastMultivariatePredictTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastMultivariatePredictTestCase, cls).setUpClass()
        cls.train_count = 1000
        cls.test_count = 100
        cls.feature_count = 10
        cls.response_count = 2
        cls.nn_count = 40
        cls.batch_count = 500
        cls.length_scale = 1.0
        cls.nu = 0.5
        cls.nu_bounds = (1e-1, 1e1)
        cls.eps = 1e-3
        cls.k_kwargs_1 = {
            "length_scale": {"val": cls.length_scale},
            "nu": {"val": cls.nu},
            "eps": {"val": cls.eps},
        }
        cls.k_kwargs_2 = {
            "length_scale": {"val": cls.length_scale},
            "nu": {"val": cls.nu},
            "eps": {"val": cls.eps},
        }
        cls.k_kwargs = [cls.k_kwargs_1, cls.k_kwargs_2]
        cls.train_features_n = _make_gaussian_matrix(
            cls.train_count, cls.feature_count
        )
        cls.train_features_t = torch.from_numpy(cls.train_features_n)
        cls.train_responses_n = _make_gaussian_matrix(
            cls.train_count, cls.response_count
        )
        cls.train_responses_t = torch.from_numpy(cls.train_responses_n)
        cls.test_features_n = _make_gaussian_matrix(
            cls.test_count, cls.feature_count
        )
        cls.test_features_t = torch.from_numpy(cls.test_features_n)
        cls.test_responses_n = _make_gaussian_matrix(
            cls.test_count, cls.response_count
        )
        cls.test_responses_t = torch.from_numpy(cls.test_responses_n)
        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features_n, cls.nn_count, **_exact_nn_kwarg_options[0]
        )
        cls.muygps = MMuyGPS("matern", *cls.k_kwargs)
        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_t = torch.from_numpy(cls.batch_indices_n)
        cls.batch_nn_indices_t = torch.from_numpy(cls.batch_nn_indices_n)
        cls.nn_indices_all_n, _ = cls.nbrs_lookup.get_batch_nns(
            np.arange(0, cls.train_count)
        )
        cls.nn_indices_all_n = np.array(cls.nn_indices_all_n)
        (
            cls.K_fast_n,
            cls.train_nn_targets_fast_n,
        ) = make_fast_regress_tensors_n(
            cls.muygps.metric,
            cls.nn_indices_all_n,
            cls.train_features_n,
            cls.train_responses_n,
        )

        cls.fast_regress_coeffs_n = muygps_fast_regress_precompute_n(
            cls.K_fast_n, cls.eps, cls.train_nn_targets_fast_n
        )

        cls.test_neighbors_n, _ = cls.nbrs_lookup.get_nns(cls.test_features_n)
        cls.closest_neighbor_n = cls.test_neighbors_n[:, 0]
        cls.closest_set_n = cls.nn_indices_all_n[cls.closest_neighbor_n]

        cls.new_nn_indices_n = fast_nn_update_n(cls.nn_indices_all_n)
        cls.closest_set_new_n = cls.new_nn_indices_n[
            cls.closest_neighbor_n
        ].astype(int)
        cls.crosswise_dists_fast_n = crosswise_distances_n(
            cls.test_features_n,
            cls.train_features_n,
            np.arange(0, cls.test_count),
            cls.closest_set_new_n,
        )
        Kcross_fast_n = np.zeros(
            (cls.test_count, cls.nn_count, cls.response_count)
        )
        kernel_func_n = matern_05_fn_n
        for i, model in enumerate(cls.muygps.models):
            Kcross_fast_n[:, :, i] = kernel_func_n(
                cls.crosswise_dists_fast_n, cls.length_scale
            )
        cls.Kcross_fast_n = Kcross_fast_n

        cls.nn_indices_all_t, _ = cls.nbrs_lookup.get_batch_nns(
            torch.arange(0, cls.train_count)
        )
        cls.nn_indices_all_t = torch.from_numpy(cls.nn_indices_all_n)

        (
            cls.K_fast_t,
            cls.train_nn_targets_fast_t,
        ) = make_fast_regress_tensors_t(
            cls.muygps.metric,
            cls.nn_indices_all_t,
            cls.train_features_t,
            cls.train_responses_t,
        )

        cls.fast_regress_coeffs_t = muygps_fast_regress_precompute_t(
            cls.K_fast_t, cls.eps, cls.train_nn_targets_fast_t
        )

        cls.test_neighbors_t, _ = cls.nbrs_lookup.get_nns(cls.test_features_t)
        cls.closest_neighbor_t = cls.test_neighbors_t[:, 0]
        cls.closest_set_t = cls.nn_indices_all_t[cls.closest_neighbor_t]

        cls.new_nn_indices_t = fast_nn_update_t(cls.nn_indices_all_t)
        cls.closest_set_new_t = cls.new_nn_indices_t[cls.closest_neighbor_t]
        cls.crosswise_dists_fast_t = crosswise_distances_t(
            cls.test_features_t,
            cls.train_features_t,
            torch.arange(0, cls.test_count),
            cls.closest_set_new_t,
        )

        cls.Kcross_fast_t = torch.from_numpy(Kcross_fast_n)

    def test_make_fast_multivariate_regress_tensors(self):
        self.assertTrue(np.allclose(self.K_fast_n, self.K_fast_t))
        self.assertTrue(
            np.allclose(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_t
            )
        )

    def test_fast_multivariate_predict(self):
        self.assertTrue(
            np.allclose(
                mmuygps_fast_regress_solve_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                ),
                mmuygps_fast_regress_solve_t(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_t[self.closest_neighbor_t, :],
                ),
            )
        )

    def test_fast_multivariate_predict_coeffs(self):
        self.assertTrue(
            np.allclose(
                self.fast_regress_coeffs_n,
                self.fast_regress_coeffs_t,
            )
        )


class OptimTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTestCase, cls).setUpClass()
        cls.predictions_t = muygps_compute_solve_t(
            cls.homoscedastic_K_t, cls.Kcross_t, cls.batch_nn_targets_t
        )
        cls.variances_t = muygps_compute_diagonal_variance_t(
            cls.K_t, cls.Kcross_t, cls.muygps.eps()
        )
        cls.predictions_n = cls.predictions_t.detach().numpy()
        cls.variances_n = cls.variances_t.detach().numpy()
        cls.x0_names, cls.x0_n, cls.bounds = cls.muygps.get_optim_params()
        cls.x0_t = torch.from_numpy(cls.x0_n)
        cls.x0_map_n = {n: cls.x0_n[i] for i, n in enumerate(cls.x0_names)}
        cls.x0_map_t = {n: cls.x0_t[i] for i, n in enumerate(cls.x0_names)}

    def _get_array_kernel_fn_n(self):
        return self.muygps.kernel._get_array_opt_fn(
            matern_05_fn_n,
            matern_15_fn_n,
            matern_25_fn_n,
            matern_inf_fn_n,
            matern_gen_fn_n,
            self.muygps.kernel.nu,
            self.muygps.kernel.length_scale,
        )

    def _get_kwargs_kernel_fn_n(self):
        return self.muygps.kernel._get_kwargs_opt_fn(
            matern_05_fn_n,
            matern_15_fn_n,
            matern_25_fn_n,
            matern_inf_fn_n,
            matern_gen_fn_n,
            self.muygps.kernel.nu,
            self.muygps.kernel.length_scale,
        )

    def _get_array_kernel_fn_t(self):
        return self.muygps.kernel._get_array_opt_fn(
            matern_05_fn_t,
            matern_15_fn_t,
            matern_25_fn_t,
            matern_inf_fn_t,
            matern_gen_fn_t,
            self.muygps.kernel.nu,
            self.muygps.kernel.length_scale,
        )

    def _get_kwargs_kernel_fn_t(self):
        return self.muygps.kernel._get_kwargs_opt_fn(
            matern_05_fn_t,
            matern_15_fn_t,
            matern_25_fn_t,
            matern_inf_fn_t,
            matern_gen_fn_t,
            self.muygps.kernel.nu,
            self.muygps.kernel.length_scale,
        )

    def _get_array_mean_fn_n(self):
        return self.muygps._get_array_opt_mean_fn(
            muygps_compute_solve_n, homoscedastic_perturb_n, self.muygps.eps
        )

    def _get_kwargs_mean_fn_n(self):
        return self.muygps._get_kwargs_opt_mean_fn(
            muygps_compute_solve_n, homoscedastic_perturb_n, self.muygps.eps
        )

    def _get_array_var_fn_n(self):
        return self.muygps._get_array_opt_var_fn(
            muygps_compute_diagonal_variance_n, self.muygps.eps
        )

    def _get_kwargs_var_fn_n(self):
        return self.muygps._get_kwargs_opt_var_fn(
            muygps_compute_diagonal_variance_n, self.muygps.eps
        )

    def _get_array_sigma_sq_fn_n(self):
        return make_array_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_n
        )

    def _get_kwargs_sigma_sq_fn_n(self):
        return make_kwargs_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_n
        )

    def _get_array_mean_fn_t(self):
        return self.muygps._get_array_opt_mean_fn(
            muygps_compute_solve_t, homoscedastic_perturb_t, self.muygps.eps
        )

    def _get_kwargs_mean_fn_t(self):
        return self.muygps._get_kwargs_opt_mean_fn(
            muygps_compute_solve_t, homoscedastic_perturb_t, self.muygps.eps
        )

    def _get_array_var_fn_t(self):
        return self.muygps._get_array_opt_var_fn(
            muygps_compute_diagonal_variance_t, self.muygps.eps
        )

    def _get_kwargs_var_fn_t(self):
        return self.muygps._get_kwargs_opt_var_fn(
            muygps_compute_diagonal_variance_t, self.muygps.eps
        )

    def _get_array_sigma_sq_fn_t(self):
        return make_array_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_t
        )

    def _get_kwargs_sigma_sq_fn_t(self):
        return make_kwargs_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_t
        )

    def _get_array_obj_fn_n(self):
        return make_loo_crossval_fn(
            "scipy",
            "mse",
            mse_fn_n,
            self._get_array_kernel_fn_n(),
            self._get_array_mean_fn_n(),
            self._get_array_var_fn_n(),
            self._get_array_sigma_sq_fn_n(),
            self.pairwise_dists_n,
            self.crosswise_dists_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )

    def _get_kwargs_obj_fn_n(self):
        return make_loo_crossval_fn(
            "bayes",
            "mse",
            mse_fn_n,
            self._get_kwargs_kernel_fn_n(),
            self._get_kwargs_mean_fn_n(),
            self._get_kwargs_var_fn_n(),
            self._get_kwargs_sigma_sq_fn_n(),
            self.pairwise_dists_n,
            self.crosswise_dists_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )

    def _get_array_obj_fn_t(self):
        return make_loo_crossval_fn(
            "scipy",
            "mse",
            mse_fn_t,
            self._get_array_kernel_fn_t(),
            self._get_array_mean_fn_t(),
            self._get_array_var_fn_t(),
            self._get_array_sigma_sq_fn_t(),
            self.pairwise_dists_t,
            self.crosswise_dists_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )

    def _get_kwargs_obj_fn_t(self):
        return make_loo_crossval_fn(
            "bayes",
            "mse",
            mse_fn_t,
            self._get_kwargs_kernel_fn_t(),
            self._get_kwargs_mean_fn_t(),
            self._get_kwargs_var_fn_t(),
            self._get_kwargs_sigma_sq_fn_t(),
            self.pairwise_dists_t,
            self.crosswise_dists_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )

    def _get_array_obj_fn_h(self):
        return make_loo_crossval_fn(
            "scipy",
            "mse",
            mse_fn_t,
            self._get_array_kernel_fn_t(),
            self._get_array_mean_fn_n(),
            self._get_array_var_fn_n(),
            self._get_array_sigma_sq_fn_n(),
            self.pairwise_dists_t,
            self.crosswise_dists_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )

    def _get_kwargs_obj_fn_h(self):
        return make_loo_crossval_fn(
            "bayes",
            "mse",
            mse_fn_t,
            self._get_kwargs_kernel_fn_t(),
            self._get_kwargs_mean_fn_n(),
            self._get_kwargs_var_fn_n(),
            self._get_kwargs_sigma_sq_fn_n(),
            self.pairwise_dists_t,
            self.crosswise_dists_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )


class ObjectiveTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectiveTest, cls).setUpClass()

        cls.sigma_sq_n = cls.muygps.sigma_sq()
        cls.sigma_sq_t = torch.Tensor(cls.muygps.sigma_sq()).float()

    def test_mse(self):
        self.assertTrue(
            np.isclose(
                mse_fn_n(self.predictions_n, self.batch_targets_n),
                mse_fn_t(self.predictions_t, self.batch_targets_t),
            )
        )

    def test_lool(self):
        self.assertTrue(
            np.isclose(
                lool_fn_n(
                    self.predictions_n,
                    self.batch_targets_n,
                    self.variances_n,
                    self.sigma_sq_n,
                ),
                lool_fn_t(
                    self.predictions_t,
                    self.batch_targets_t,
                    self.variances_t,
                    self.sigma_sq_t,
                ),
            )
        )

    def test_cross_entropy(self):
        cat_predictions_n, cat_batch_targets_n = _make_gaussian_data(
            1000, 1000, 10, 2, categorical=True
        )
        cat_predictions_n = cat_predictions_n["output"]
        cat_batch_targets_n = cat_batch_targets_n["output"]
        cat_predictions_t = torch.from_numpy(cat_predictions_n)
        cat_batch_targets_t = torch.from_numpy(cat_batch_targets_n)
        self.assertTrue(
            np.all(
                (
                    np.allclose(cat_predictions_t, cat_predictions_n),
                    np.allclose(cat_batch_targets_t, cat_batch_targets_n),
                )
            )
        )
        self.assertTrue(
            np.allclose(
                cross_entropy_fn_n(
                    cat_predictions_n, cat_batch_targets_n, ll_eps=1e-6
                ),
                cross_entropy_fn_t(cat_predictions_t, cat_batch_targets_t),
            )
        )

    def test_kernel_fn(self):
        kernel_fn_n = self._get_array_kernel_fn_n()
        kernel_fn_t = self._get_array_kernel_fn_t()
        self.assertTrue(
            np.allclose(
                kernel_fn_n(self.pairwise_dists_n, self.x0_n),
                kernel_fn_t(self.pairwise_dists_t, self.x0_t),
            )
        )

    def test_kwargs_mean_fn(self):
        mean_fn_n = self._get_kwargs_mean_fn_n()
        mean_fn_t = self._get_kwargs_mean_fn_t()
        self.assertTrue(
            np.allclose(
                mean_fn_n(
                    self.K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_t(
                    self.K_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_array_mean_fn(self):
        mean_fn_n = self._get_array_mean_fn_n()
        mean_fn_t = self._get_array_mean_fn_t()
        self.assertTrue(
            np.allclose(
                mean_fn_n(
                    self.K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    self.muygps.eps(),
                ),
                mean_fn_t(
                    self.K_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                    self.muygps.eps(),
                ),
            )
        )

    def test_kwargs_var_fn(self):
        var_fn_n = self._get_kwargs_var_fn_n()
        var_fn_t = self._get_kwargs_var_fn_t()
        self.assertTrue(
            np.allclose(
                var_fn_n(
                    self.K_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_t(
                    self.K_t,
                    self.Kcross_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_array_var_fn(self):
        var_fn_n = self._get_array_var_fn_n()
        var_fn_t = self._get_array_var_fn_t()
        self.assertTrue(
            np.allclose(
                var_fn_n(
                    self.K_n,
                    self.Kcross_n,
                    self.muygps.eps(),
                ),
                var_fn_t(
                    self.K_t,
                    self.Kcross_t,
                    self.muygps.eps(),
                ),
            )
        )

    def test_kwargs_sigma_sq_fn(self):
        ss_fn_n = self._get_kwargs_sigma_sq_fn_n()
        ss_fn_t = self._get_kwargs_sigma_sq_fn_t()
        self.assertTrue(
            np.allclose(
                ss_fn_n(
                    self.K_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_t(
                    self.K_t,
                    self.batch_nn_targets_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_array_sigma_sq_fn(self):
        ss_fn_n = self._get_array_sigma_sq_fn_n()
        ss_fn_t = self._get_array_sigma_sq_fn_t()
        self.assertTrue(
            np.allclose(
                ss_fn_n(
                    self.K_n,
                    self.batch_nn_targets_n,
                    self.muygps.eps(),
                ),
                ss_fn_t(
                    self.K_t,
                    self.batch_nn_targets_t,
                    self.muygps.eps(),
                ),
            )
        )

    def test_loo_crossval(self):
        obj_fn_n = self._get_array_obj_fn_n()
        obj_fn_t = self._get_array_obj_fn_t()
        obj_fn_h = self._get_array_obj_fn_h()
        self.assertTrue(np.allclose(obj_fn_n(self.x0_n), obj_fn_t(self.x0_t)))
        self.assertTrue(np.allclose(obj_fn_n(self.x0_n), obj_fn_h(self.x0_t)))


if __name__ == "__main__":
    absltest.main()
