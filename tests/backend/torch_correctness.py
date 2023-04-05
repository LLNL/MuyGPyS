# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI

if config.state.torch_enabled is False:
    raise ValueError(f"Bad attempt to run torch-only code with torch diabled.")
if config.state.backend == "mpi":
    raise ValueError(f"Bad attempt to run non-MPI code in MPI mode.")
if config.state.backend != "numpy":
    import warnings

    warnings.warn(
        f"Backend correctness codes assume numpy mode, not "
        f"{config.state.backend}. "
        f"Force-switching MuyGPyS into numpy backend."
    )
    config.update("muygpys_backend", "numpy")

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
import MuyGPyS._src.math.torch as torch
from MuyGPyS._src.gp.tensors.numpy import (
    _pairwise_tensor as pairwise_tensor_n,
    _crosswise_tensor as crosswise_tensor_n,
    _make_train_tensors as make_train_tensors_n,
    _make_fast_predict_tensors as make_fast_predict_tensors_n,
    _fast_nn_update as fast_nn_update_n,
    _F2 as F2_n,
    _l2 as l2_n,
)
from MuyGPyS._src.gp.tensors.torch import (
    _pairwise_tensor as pairwise_tensor_t,
    _crosswise_tensor as crosswise_tensor_t,
    _make_train_tensors as make_train_tensors_t,
    _make_fast_predict_tensors as make_fast_predict_tensors_t,
    _fast_nn_update as fast_nn_update_t,
    _F2 as F2_t,
    _l2 as l2_t,
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
    _muygps_posterior_mean as muygps_posterior_mean_n,
    _muygps_diagonal_variance as muygps_diagonal_variance_n,
    _muygps_fast_posterior_mean as muygps_fast_posterior_mean_n,
    _mmuygps_fast_posterior_mean as mmuygps_fast_posterior_mean_n,
    _muygps_fast_posterior_mean_precompute as muygps_fast_posterior_mean_precompute_n,
)
from MuyGPyS._src.gp.muygps.torch import (
    _muygps_posterior_mean as muygps_posterior_mean_t,
    _muygps_diagonal_variance as muygps_diagonal_variance_t,
    _muygps_fast_posterior_mean as muygps_fast_posterior_mean_t,
    _mmuygps_fast_posterior_mean as mmuygps_fast_posterior_mean_t,
    _muygps_fast_posterior_mean_precompute as muygps_fast_posterior_mean_precompute_t,
)
from MuyGPyS._src.gp.noise.numpy import (
    _homoscedastic_perturb as homoscedastic_perturb_n,
    _heteroscedastic_perturb as heteroscedastic_perturb_n,
)
from MuyGPyS._src.gp.noise.torch import (
    _homoscedastic_perturb as homoscedastic_perturb_t,
    _heteroscedastic_perturb as heteroscedastic_perturb_t,
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
from MuyGPyS._src.optimize.sigma_sq.numpy import (
    _analytic_sigma_sq_optim as analytic_sigma_sq_optim_n,
)
from MuyGPyS._src.optimize.sigma_sq.torch import (
    _analytic_sigma_sq_optim as analytic_sigma_sq_optim_t,
)
from MuyGPyS._test.utils import (
    _exact_nn_kwarg_options,
    _make_gaussian_matrix,
    _make_gaussian_data,
    _make_heteroscedastic_test_nugget,
)
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.gp.distortion import apply_distortion, IsotropicDistortion
from MuyGPyS.gp.kernels import Hyperparameter, Matern
from MuyGPyS.gp.sigma_sq import sigma_sq_scale
from MuyGPyS.gp.noise import (
    HeteroscedasticNoise,
    HomoscedasticNoise,
    noise_perturb,
)
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.objective import make_loo_crossval_fn
from MuyGPyS.optimize.sigma_sq import make_analytic_sigma_sq_optim

rbf_fn_n = apply_distortion(F2_n)(rbf_fn_n)
matern_05_fn_n = apply_distortion(l2_n)(matern_05_fn_n)
matern_15_fn_n = apply_distortion(l2_n)(matern_15_fn_n)
matern_25_fn_n = apply_distortion(l2_n)(matern_25_fn_n)
matern_inf_fn_n = apply_distortion(l2_n)(matern_inf_fn_n)
matern_gen_fn_n = apply_distortion(l2_n)(matern_gen_fn_n)

rbf_fn_t = apply_distortion(F2_t)(rbf_fn_t)
matern_05_fn_t = apply_distortion(l2_t)(matern_05_fn_t)
matern_15_fn_t = apply_distortion(l2_t)(matern_15_fn_t)
matern_25_fn_t = apply_distortion(l2_t)(matern_25_fn_t)
matern_inf_fn_t = apply_distortion(l2_t)(matern_inf_fn_t)
matern_gen_fn_t = apply_distortion(l2_t)(matern_gen_fn_t)


def _allclose(x, y) -> bool:
    return np.allclose(
        x, y, atol=1e-5 if config.state.low_precision() else 1e-8
    )


class TensorsTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TensorsTestCase, cls).setUpClass()
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
        cls.eps_heteroscedastic_n = _make_heteroscedastic_test_nugget(
            cls.batch_count, cls.nn_count, cls.eps
        )
        cls.eps_heteroscedastic_train_n = _make_heteroscedastic_test_nugget(
            cls.train_count, cls.nn_count, cls.eps
        )
        cls.eps_heteroscedastic_t = torch.ndarray(cls.eps_heteroscedastic_n)
        cls.eps_heteroscedastic_train_t = torch.ndarray(
            cls.eps_heteroscedastic_train_n
        )
        cls.k_kwargs = {
            "kernel": Matern(
                nu=Hyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    "l2", length_scale=Hyperparameter(cls.length_scale)
                ),
            ),
            "eps": HomoscedasticNoise(cls.eps),
        }
        cls.k_kwargs_heteroscedastic = {
            "kernel": Matern(
                nu=Hyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    "l2", length_scale=Hyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_n),
        }
        cls.k_kwargs_heteroscedastic_train = {
            "kernel": Matern(
                nu=Hyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    "l2", length_scale=Hyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_train_n),
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
        cls.muygps_heteroscedastic = MuyGPS(**cls.k_kwargs_heteroscedastic)
        cls.muygps_heteroscedastic_train = MuyGPS(
            **cls.k_kwargs_heteroscedastic_train
        )
        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_t = torch.from_numpy(cls.batch_indices_n)
        cls.batch_nn_indices_t = torch.from_numpy(cls.batch_nn_indices_n)


class TensorsTest(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(TensorsTest, cls).setUpClass()

    def test_pairwise_tensor(self):
        self.assertTrue(
            np.allclose(
                pairwise_tensor_n(
                    self.train_features_n, self.batch_nn_indices_n
                ),
                pairwise_tensor_t(
                    self.train_features_t, self.batch_nn_indices_t
                ),
            )
        )

    def test_crosswise_tensor(self):
        self.assertTrue(
            np.allclose(
                crosswise_tensor_n(
                    self.train_features_n,
                    self.train_features_n,
                    self.batch_indices_n,
                    self.batch_nn_indices_n,
                ),
                crosswise_tensor_t(
                    self.train_features_t,
                    self.train_features_t,
                    self.batch_indices_t,
                    self.batch_nn_indices_t,
                ),
            )
        )

    def test_make_train_tensors(self):
        (
            crosswise_diffs_n,
            pairwise_diffs_n,
            batch_targets_n,
            batch_nn_targets_n,
        ) = make_train_tensors_n(
            self.batch_indices_n,
            self.batch_nn_indices_n,
            self.train_features_n,
            self.train_responses_n,
        )
        (
            crosswise_diffs_t,
            pairwise_diffs_t,
            batch_targets_t,
            batch_nn_targets_t,
        ) = make_train_tensors_t(
            self.batch_indices_t,
            self.batch_nn_indices_t,
            self.train_features_t,
            self.train_responses_t,
        )
        self.assertTrue(np.allclose(crosswise_diffs_n, crosswise_diffs_t))
        self.assertTrue(np.allclose(pairwise_diffs_n, pairwise_diffs_t))
        self.assertTrue(np.allclose(batch_targets_n, batch_targets_t))
        self.assertTrue(np.allclose(batch_nn_targets_n, batch_nn_targets_t))


class KernelTestCase(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()
        (
            cls.crosswise_diffs_n,
            cls.pairwise_diffs_n,
            cls.batch_targets_n,
            cls.batch_nn_targets_n,
        ) = make_train_tensors_n(
            cls.batch_indices_n,
            cls.batch_nn_indices_n,
            cls.train_features_n,
            cls.train_responses_n,
        )
        (
            cls.crosswise_diffs_t,
            cls.pairwise_diffs_t,
            cls.batch_targets_t,
            cls.batch_nn_targets_t,
        ) = make_train_tensors_t(
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
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                rbf_fn_t(
                    self.crosswise_diffs_t, length_scale=self.length_scale
                ),
            )
        )

    def test_pairwise_rbf(self):
        self.assertTrue(
            np.allclose(
                rbf_fn_n(self.pairwise_diffs_n, length_scale=self.length_scale),
                rbf_fn_t(self.pairwise_diffs_t, length_scale=self.length_scale),
            )
        )

    def test_crosswise_matern(self):
        self.assertTrue(
            np.allclose(
                matern_05_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_05_fn_t(
                    self.crosswise_diffs_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_15_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_15_fn_t(
                    self.crosswise_diffs_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_25_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_25_fn_t(
                    self.crosswise_diffs_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_inf_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_inf_fn_t(
                    self.crosswise_diffs_t, length_scale=self.length_scale
                ),
            )
        )

    def test_pairwise_matern(self):
        self.assertTrue(
            np.allclose(
                matern_05_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_05_fn_t(
                    self.pairwise_diffs_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_15_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_15_fn_t(
                    self.pairwise_diffs_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_25_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_25_fn_t(
                    self.pairwise_diffs_t, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                matern_inf_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_inf_fn_t(
                    self.pairwise_diffs_t, length_scale=self.length_scale
                ),
            )
        )


class MuyGPSTestCase(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTestCase, cls).setUpClass()
        cls.K_n = matern_05_fn_n(
            cls.pairwise_diffs_n, length_scale=cls.length_scale
        )
        cls.homoscedastic_K_n = homoscedastic_perturb_n(
            cls.K_n, cls.muygps.eps()
        )

        cls.heteroscedastic_K_n = heteroscedastic_perturb_n(
            cls.K_n, cls.muygps_heteroscedastic.eps()
        )

        cls.K_t = matern_05_fn_t(
            cls.pairwise_diffs_t, length_scale=cls.length_scale
        )
        cls.homoscedastic_K_t = homoscedastic_perturb_t(
            cls.K_t, cls.muygps.eps()
        )
        cls.heteroscedastic_K_t = heteroscedastic_perturb_t(
            cls.K_t, cls.muygps_heteroscedastic.eps()
        )

        cls.Kcross_n = matern_05_fn_n(
            cls.crosswise_diffs_n, length_scale=cls.length_scale
        )
        cls.Kcross_t = matern_05_fn_t(
            cls.crosswise_diffs_t, length_scale=cls.length_scale
        )


class MuyGPSTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTest, cls).setUpClass()

    def test_homoscedastic_noise(self):
        self.assertTrue(
            np.allclose(self.homoscedastic_K_n, self.homoscedastic_K_t)
        )

    def test_heteroscedastic_noise(self):
        self.assertTrue(
            np.allclose(self.heteroscedastic_K_n, self.heteroscedastic_K_t)
        )

    def test_posterior_mean(self):
        self.assertTrue(
            _allclose(
                muygps_posterior_mean_n(
                    self.homoscedastic_K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                muygps_posterior_mean_t(
                    self.homoscedastic_K_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                ),
            )
        )

    def test_posterior_mean_heteroscedastic(self):
        self.assertTrue(
            _allclose(
                muygps_posterior_mean_n(
                    self.heteroscedastic_K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                muygps_posterior_mean_t(
                    self.heteroscedastic_K_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                ),
            )
        )

    def test_diagonal_variance(self):
        self.assertTrue(
            np.allclose(
                muygps_diagonal_variance_n(
                    self.homoscedastic_K_n, self.Kcross_n
                ),
                muygps_diagonal_variance_t(
                    self.homoscedastic_K_t, self.Kcross_t
                ),
            )
        )

    def test_diagonal_variance_heteroscedastic(self):
        self.assertTrue(
            np.allclose(
                muygps_diagonal_variance_n(
                    self.heteroscedastic_K_n, self.Kcross_n
                ),
                muygps_diagonal_variance_t(
                    self.heteroscedastic_K_t, self.Kcross_t
                ),
            )
        )

    def test_sigma_sq_optim(self):
        self.assertTrue(
            np.allclose(
                analytic_sigma_sq_optim_n(
                    self.homoscedastic_K_n, self.batch_nn_targets_n
                ),
                analytic_sigma_sq_optim_t(
                    self.homoscedastic_K_t, self.batch_nn_targets_t
                ),
            )
        )

    def test_sigma_sq_optim_heteroscedastic(self):
        self.assertTrue(
            np.allclose(
                analytic_sigma_sq_optim_n(
                    self.heteroscedastic_K_n, self.batch_nn_targets_n
                ),
                analytic_sigma_sq_optim_t(
                    self.heteroscedastic_K_t, self.batch_nn_targets_t
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
        (
            cls.K_fast_n,
            cls.train_nn_targets_fast_n,
        ) = make_fast_predict_tensors_n(
            cls.nn_indices_all_n,
            cls.train_features_n,
            cls.train_responses_n,
        )

        cls.homoscedastic_K_fast_n = homoscedastic_perturb_n(
            l2_n(cls.K_fast_n), cls.muygps.eps()
        )
        cls.heteroscedastic_K_fast_n = heteroscedastic_perturb_n(
            l2_n(cls.K_fast_n), cls.muygps_heteroscedastic_train.eps()
        )
        cls.fast_regress_coeffs_n = muygps_fast_posterior_mean_precompute_n(
            cls.homoscedastic_K_fast_n, cls.train_nn_targets_fast_n
        )
        cls.fast_regress_coeffs_heteroscedastic_n = (
            muygps_fast_posterior_mean_precompute_n(
                cls.heteroscedastic_K_fast_n, cls.train_nn_targets_fast_n
            )
        )

        cls.test_neighbors_n, _ = cls.nbrs_lookup.get_nns(cls.test_features_n)
        cls.closest_neighbor_n = cls.test_neighbors_n[:, 0]
        cls.closest_set_n = cls.nn_indices_all_n[cls.closest_neighbor_n]

        cls.new_nn_indices_n = fast_nn_update_n(cls.nn_indices_all_n)
        cls.closest_set_new_n = cls.new_nn_indices_n[
            cls.closest_neighbor_n
        ].astype(int)
        cls.crosswise_diffs_fast_n = crosswise_tensor_n(
            cls.test_features_n,
            cls.train_features_n,
            np.arange(0, cls.test_count),
            cls.closest_set_new_n,
        )

        kernel_func_n = matern_05_fn_n
        cls.Kcross_fast_n = kernel_func_n(
            cls.crosswise_diffs_fast_n, cls.length_scale
        )

        cls.nn_indices_all_t, _ = cls.nbrs_lookup.get_batch_nns(
            torch.arange(0, cls.train_count)
        )
        cls.nn_indices_all_t = torch.from_numpy(cls.nn_indices_all_t)
        (
            cls.K_fast_t,
            cls.train_nn_targets_fast_t,
        ) = make_fast_predict_tensors_t(
            cls.nn_indices_all_t,
            cls.train_features_t,
            cls.train_responses_t,
        )

        cls.homoscedastic_K_fast_t = homoscedastic_perturb_t(
            l2_t(cls.K_fast_t), cls.muygps.eps()
        )
        cls.heteroscedastic_K_fast_t = heteroscedastic_perturb_t(
            l2_t(cls.K_fast_t), cls.muygps_heteroscedastic_train.eps()
        )
        cls.fast_regress_coeffs_t = muygps_fast_posterior_mean_precompute_t(
            cls.homoscedastic_K_fast_t, cls.train_nn_targets_fast_t
        )
        cls.fast_regress_coeffs_heteroscedastic_t = (
            muygps_fast_posterior_mean_precompute_t(
                cls.heteroscedastic_K_fast_t, cls.train_nn_targets_fast_t
            )
        )

        cls.test_neighbors_t, _ = cls.nbrs_lookup.get_nns(cls.test_features_t)
        cls.closest_neighbor_t = cls.test_neighbors_t[:, 0]
        cls.closest_set_t = cls.nn_indices_all_t[cls.closest_neighbor_t]

        cls.new_nn_indices_t = fast_nn_update_t(cls.nn_indices_all_t)
        cls.closest_set_new_t = cls.new_nn_indices_t[cls.closest_neighbor_t]
        cls.crosswise_diffs_fast_t = crosswise_tensor_t(
            cls.test_features_t,
            cls.train_features_t,
            torch.arange(0, cls.test_count),
            cls.closest_set_new_t,
        )

        kernel_func_t = matern_05_fn_t
        cls.Kcross_fast_t = kernel_func_t(
            cls.crosswise_diffs_fast_t, cls.length_scale
        )

    def test_fast_nn_update(self):
        self.assertTrue(
            np.allclose(
                fast_nn_update_t(self.nn_indices_all_t),
                fast_nn_update_n(self.nn_indices_all_n),
            )
        )

    def test_make_fast_predict_tensors(self):
        self.assertTrue(np.allclose(self.K_fast_n, self.K_fast_t))
        self.assertTrue(
            np.allclose(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_t
            )
        )

    def test_homoscedastic_kernel_tensors(self):
        self.assertTrue(
            np.allclose(
                self.homoscedastic_K_fast_n, self.homoscedastic_K_fast_t
            )
        )

    def test_heteroscedastic_kernel_tensors(self):
        self.assertTrue(
            np.allclose(
                self.heteroscedastic_K_fast_n, self.heteroscedastic_K_fast_t
            )
        )

    def test_fast_predict(self):
        self.assertTrue(
            _allclose(
                muygps_fast_posterior_mean_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                ),
                muygps_fast_posterior_mean_t(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_t[self.closest_neighbor_t, :],
                ),
            )
        )

    def test_fast_predict_heteroscedastic(self):
        self.assertTrue(
            _allclose(
                muygps_fast_posterior_mean_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_heteroscedastic_n[
                        self.closest_neighbor_n, :
                    ],
                ),
                muygps_fast_posterior_mean_t(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_heteroscedastic_t[
                        self.closest_neighbor_t, :
                    ],
                ),
            )
        )

    def test_fast_predict_coeffs(self):
        self.assertTrue(
            _allclose(
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
        cls.eps_heteroscedastic_n = _make_heteroscedastic_test_nugget(
            cls.batch_count, cls.nn_count, cls.eps
        )
        cls.eps_heteroscedastic_train_n = _make_heteroscedastic_test_nugget(
            cls.train_count, cls.nn_count, cls.eps
        )
        cls.eps_heteroscedastic_t = torch.ndarray(cls.eps_heteroscedastic_n)
        cls.eps_heteroscedastic_train_t = torch.ndarray(
            cls.eps_heteroscedastic_train_n
        )
        cls.k_kwargs_1 = {
            "kernel": Matern(
                nu=Hyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    "l2", length_scale=Hyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_train_n),
        }
        cls.k_kwargs_2 = {
            "kernel": Matern(
                nu=Hyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    "l2", length_scale=Hyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_train_n),
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
        cls.muygps = MMuyGPS(*cls.k_kwargs)
        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_t = torch.from_numpy(cls.batch_indices_n)
        cls.batch_nn_indices_t = torch.from_numpy(cls.batch_nn_indices_n)
        cls.nn_indices_all_n, _ = cls.nbrs_lookup.get_batch_nns(
            np.arange(0, cls.train_count)
        )
        (
            cls.K_fast_n,
            cls.train_nn_targets_fast_n,
        ) = make_fast_predict_tensors_n(
            cls.nn_indices_all_n,
            cls.train_features_n,
            cls.train_responses_n,
        )

        cls.homoscedastic_K_fast_n = homoscedastic_perturb_n(
            l2_n(cls.K_fast_n), cls.eps
        )
        cls.heteroscedastic_K_fast_n = heteroscedastic_perturb_n(
            l2_n(cls.K_fast_n), cls.eps_heteroscedastic_train_n
        )

        cls.fast_regress_coeffs_n = muygps_fast_posterior_mean_precompute_n(
            cls.homoscedastic_K_fast_n, cls.train_nn_targets_fast_n
        )

        cls.test_neighbors_n, _ = cls.nbrs_lookup.get_nns(cls.test_features_n)
        cls.closest_neighbor_n = cls.test_neighbors_n[:, 0]
        cls.closest_set_n = cls.nn_indices_all_n[cls.closest_neighbor_n]

        cls.new_nn_indices_n = fast_nn_update_n(cls.nn_indices_all_n)
        cls.closest_set_new_n = cls.new_nn_indices_n[
            cls.closest_neighbor_n
        ].astype(int)
        cls.crosswise_diffs_fast_n = crosswise_tensor_n(
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
                cls.crosswise_diffs_fast_n, cls.length_scale
            )
        cls.Kcross_fast_n = Kcross_fast_n

        cls.nn_indices_all_t, _ = cls.nbrs_lookup.get_batch_nns(
            torch.arange(0, cls.train_count)
        )
        cls.nn_indices_all_t = torch.from_numpy(cls.nn_indices_all_n)

        (
            cls.K_fast_t,
            cls.train_nn_targets_fast_t,
        ) = make_fast_predict_tensors_t(
            cls.nn_indices_all_t,
            cls.train_features_t,
            cls.train_responses_t,
        )

        cls.homoscedastic_K_fast_t = homoscedastic_perturb_t(
            l2_t(cls.K_fast_t), cls.eps
        )

        cls.heteroscedastic_K_fast_t = heteroscedastic_perturb_t(
            l2_t(cls.K_fast_t), cls.eps_heteroscedastic_train_t
        )

        cls.fast_regress_coeffs_t = muygps_fast_posterior_mean_precompute_t(
            cls.homoscedastic_K_fast_t, cls.train_nn_targets_fast_t
        )

        cls.fast_regress_coeffs_heteroscedastic_t = (
            muygps_fast_posterior_mean_precompute_t(
                cls.heteroscedastic_K_fast_t, cls.train_nn_targets_fast_t
            )
        )

        cls.test_neighbors_t, _ = cls.nbrs_lookup.get_nns(cls.test_features_t)
        cls.closest_neighbor_t = cls.test_neighbors_t[:, 0]
        cls.closest_set_t = cls.nn_indices_all_t[cls.closest_neighbor_t]

        cls.new_nn_indices_t = fast_nn_update_t(cls.nn_indices_all_t)
        cls.closest_set_new_t = cls.new_nn_indices_t[cls.closest_neighbor_t]
        cls.crosswise_diffs_fast_t = crosswise_tensor_t(
            cls.test_features_t,
            cls.train_features_t,
            torch.arange(0, cls.test_count),
            cls.closest_set_new_t,
        )

        cls.Kcross_fast_t = torch.from_numpy(Kcross_fast_n)

    def test_make_fast_multivariate_predict_tensors(self):
        self.assertTrue(np.allclose(self.K_fast_n, self.K_fast_t))
        self.assertTrue(
            np.allclose(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_t
            )
        )

    def test_fast_multivariate_predict(self):
        self.assertTrue(
            _allclose(
                mmuygps_fast_posterior_mean_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                ),
                mmuygps_fast_posterior_mean_t(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_t[self.closest_neighbor_t, :],
                ),
            )
        )

    def test_fast_multivariate_predict_coeffs(self):
        self.assertTrue(
            _allclose(
                self.fast_regress_coeffs_n,
                self.fast_regress_coeffs_t,
            )
        )


class OptimTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTestCase, cls).setUpClass()
        cls.predictions_t = muygps_posterior_mean_t(
            cls.homoscedastic_K_t, cls.Kcross_t, cls.batch_nn_targets_t
        )
        cls.variances_t = muygps_diagonal_variance_t(
            cls.homoscedastic_K_t, cls.Kcross_t
        )
        cls.predictions_heteroscedastic_t = muygps_posterior_mean_t(
            cls.heteroscedastic_K_t, cls.Kcross_t, cls.batch_nn_targets_t
        )
        cls.variances_heteroscedastic_t = muygps_diagonal_variance_t(
            cls.heteroscedastic_K_t, cls.Kcross_t
        )
        cls.predictions_n = cls.predictions_t.detach().numpy()
        cls.variances_n = cls.variances_t.detach().numpy()
        cls.x0_names, cls.x0_n, cls.bounds = cls.muygps.get_optim_params()
        cls.x0_t = torch.from_numpy(cls.x0_n)
        cls.x0_map_n = {n: cls.x0_n[i] for i, n in enumerate(cls.x0_names)}
        cls.x0_map_t = {n: cls.x0_t[i] for i, n in enumerate(cls.x0_names)}

    def _get_kernel_fn_n(self):
        return self.muygps.kernel._get_opt_fn(
            matern_05_fn_n,
            self.muygps.kernel.nu,
            self.muygps.kernel.length_scale,
        )

    def _get_kernel_fn_t(self):
        return self.muygps.kernel._get_opt_fn(
            matern_05_fn_t,
            self.muygps.kernel.nu,
            self.muygps.kernel.length_scale,
        )

    def _get_mean_fn_n(self):
        return self.muygps._mean_fn._get_opt_fn(
            noise_perturb(homoscedastic_perturb_n)(muygps_posterior_mean_n),
            self.muygps.eps,
        )

    def _get_mean_fn_heteroscedastic_n(self):
        return self.muygps_heteroscedastic._mean_fn._get_opt_fn(
            noise_perturb(heteroscedastic_perturb_n)(muygps_posterior_mean_n),
            self.muygps_heteroscedastic.eps,
        )

    def _get_var_fn_n(self):
        return self.muygps._var_fn._get_opt_fn(
            sigma_sq_scale(
                noise_perturb(homoscedastic_perturb_n)(
                    muygps_diagonal_variance_n
                )
            ),
            self.muygps.eps,
            self.muygps.sigma_sq,
        )

    def _get_var_fn_heteroscedastic_n(self):
        return self.muygps_heteroscedastic._var_fn._get_opt_fn(
            sigma_sq_scale(
                noise_perturb(heteroscedastic_perturb_n)(
                    muygps_diagonal_variance_n
                )
            ),
            self.muygps_heteroscedastic.eps,
            self.muygps_heteroscedastic.sigma_sq,
        )

    def _get_sigma_sq_fn_n(self):
        return make_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_n, homoscedastic_perturb_n
        )

    def _get_sigma_sq_fn_heteroscedastic_n(self):
        return make_analytic_sigma_sq_optim(
            self.muygps_heteroscedastic,
            analytic_sigma_sq_optim_n,
            heteroscedastic_perturb_n,
        )

    def _get_mean_fn_t(self):
        return self.muygps._mean_fn._get_opt_fn(
            noise_perturb(homoscedastic_perturb_t)(muygps_posterior_mean_t),
            self.muygps.eps,
        )

    def _get_mean_fn_heteroscedastic_t(self):
        return self.muygps_heteroscedastic._mean_fn._get_opt_fn(
            noise_perturb(heteroscedastic_perturb_t)(muygps_posterior_mean_t),
            self.muygps_heteroscedastic.eps,
        )

    def _get_var_fn_t(self):
        return self.muygps._var_fn._get_opt_fn(
            sigma_sq_scale(
                noise_perturb(homoscedastic_perturb_t)(
                    muygps_diagonal_variance_t
                )
            ),
            self.muygps.eps,
            self.muygps.sigma_sq,
        )

    def _get_var_fn_heteroscedastic_t(self):
        return self.muygps_heteroscedastic._var_fn._get_opt_fn(
            sigma_sq_scale(
                noise_perturb(heteroscedastic_perturb_t)(
                    muygps_diagonal_variance_t
                )
            ),
            self.muygps_heteroscedastic.eps,
            self.muygps_heteroscedastic.sigma_sq,
        )

    def _get_sigma_sq_fn_t(self):
        return make_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_t, homoscedastic_perturb_t
        )

    def _get_sigma_sq_fn_heteroscedastic_t(self):
        return make_analytic_sigma_sq_optim(
            self.muygps_heteroscedastic,
            analytic_sigma_sq_optim_t,
            heteroscedastic_perturb_t,
        )

    def _get_obj_fn_n(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_n,
            self._get_kernel_fn_n(),
            self._get_mean_fn_n(),
            self._get_var_fn_n(),
            self._get_sigma_sq_fn_n(),
            self.pairwise_diffs_n,
            self.crosswise_diffs_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )

    def _get_obj_fn_heteroscedastic_n(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_n,
            self._get_kernel_fn_n(),
            self._get_mean_fn_heteroscedastic_n(),
            self._get_var_fn_heteroscedastic_n(),
            self._get_sigma_sq_fn_heteroscedastic_n(),
            self.pairwise_diffs_n,
            self.crosswise_diffs_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )

    def _get_obj_fn_t(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_t,
            self._get_kernel_fn_t(),
            self._get_mean_fn_t(),
            self._get_var_fn_t(),
            self._get_sigma_sq_fn_t(),
            self.pairwise_diffs_t,
            self.crosswise_diffs_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )

    def _get_obj_fn_heteroscedastic_t(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_t,
            self._get_kernel_fn_t(),
            self._get_mean_fn_heteroscedastic_t(),
            self._get_var_fn_heteroscedastic_t(),
            self._get_sigma_sq_fn_heteroscedastic_t(),
            self.pairwise_diffs_t,
            self.crosswise_diffs_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )


class ObjectiveTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectiveTest, cls).setUpClass()

        cls.sigma_sq_n = cls.muygps.sigma_sq()
        cls.sigma_sq_t = torch.array(cls.muygps.sigma_sq()).float()

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
        kernel_fn_n = self._get_kernel_fn_n()
        kernel_fn_t = self._get_kernel_fn_t()
        self.assertTrue(
            np.allclose(
                kernel_fn_n(self.pairwise_diffs_n, **self.x0_map_n),
                kernel_fn_t(self.pairwise_diffs_t, **self.x0_map_t),
            )
        )

    def test_mean_fn(self):
        mean_fn_n = self._get_mean_fn_n()
        mean_fn_t = self._get_mean_fn_t()
        self.assertTrue(
            _allclose(
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

    def test_mean_heteroscedastic_fn(self):
        mean_fn_n = self._get_mean_fn_heteroscedastic_n()
        mean_fn_t = self._get_mean_fn_heteroscedastic_t()
        self.assertTrue(
            _allclose(
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

    def test_var_fn(self):
        var_fn_n = self._get_var_fn_n()
        var_fn_t = self._get_var_fn_t()
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

    def test_var_heteroscedastic_fn(self):
        var_fn_n = self._get_var_fn_heteroscedastic_n()
        var_fn_t = self._get_var_fn_heteroscedastic_t()
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

    def test_sigma_sq_fn(self):
        ss_fn_n = self._get_sigma_sq_fn_n()
        ss_fn_t = self._get_sigma_sq_fn_t()
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

    def test_sigma_sq_heteroscedastic_fn(self):
        ss_fn_n = self._get_sigma_sq_fn_heteroscedastic_n()
        ss_fn_t = self._get_sigma_sq_fn_heteroscedastic_t()
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

    def test_loo_crossval(self):
        obj_fn_n = self._get_obj_fn_n()
        obj_fn_t = self._get_obj_fn_t()
        self.assertTrue(
            np.allclose(obj_fn_n(**self.x0_map_n), obj_fn_t(**self.x0_map_t))
        )

    def test_loo_crossval_heteroscedastic(self):
        obj_fn_n = self._get_obj_fn_heteroscedastic_n()
        obj_fn_t = self._get_obj_fn_heteroscedastic_t()
        self.assertTrue(
            np.allclose(obj_fn_n(**self.x0_map_n), obj_fn_t(**self.x0_map_t))
        )


if __name__ == "__main__":
    absltest.main()
