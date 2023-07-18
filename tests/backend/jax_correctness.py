# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.jax as jnp
import MuyGPyS._src.math.numpy as np
from MuyGPyS import config, jax_config
from MuyGPyS._src.gp.tensors.numpy import (
    _crosswise_tensor as crosswise_tensor_n,
    _fast_nn_update as fast_nn_update_n,
    _F2 as F2_n,
    _l2 as l2_n,
    _pairwise_tensor as pairwise_tensor_n,
    _make_train_tensors as make_train_tensors_n,
    _make_fast_predict_tensors as make_fast_predict_tensors_n,
)
from MuyGPyS._src.gp.tensors.jax import (
    _pairwise_tensor as pairwise_tensor_j,
    _crosswise_tensor as crosswise_tensor_j,
    _make_train_tensors as make_train_tensors_j,
    _make_fast_predict_tensors as make_fast_predict_tensors_j,
    _fast_nn_update as fast_nn_update_j,
    _F2 as F2_j,
    _l2 as l2_j,
)
from MuyGPyS._src.gp.kernels.numpy import (
    _rbf_fn as rbf_fn_n,
    _matern_05_fn as matern_05_fn_n,
    _matern_15_fn as matern_15_fn_n,
    _matern_25_fn as matern_25_fn_n,
    _matern_inf_fn as matern_inf_fn_n,
    _matern_gen_fn as matern_gen_fn_n,
)
from MuyGPyS._src.gp.kernels.jax import (
    _rbf_fn as rbf_fn_j,
    _matern_05_fn as matern_05_fn_j,
    _matern_15_fn as matern_15_fn_j,
    _matern_25_fn as matern_25_fn_j,
    _matern_inf_fn as matern_inf_fn_j,
    _matern_gen_fn as matern_gen_fn_j,
)
from MuyGPyS._src.gp.muygps.numpy import (
    _muygps_posterior_mean as muygps_posterior_mean_n,
    _muygps_diagonal_variance as muygps_diagonal_variance_n,
    _muygps_fast_posterior_mean as muygps_fast_posterior_mean_n,
    _mmuygps_fast_posterior_mean as mmuygps_fast_posterior_mean_n,
    _muygps_fast_posterior_mean_precompute as muygps_fast_posterior_mean_precompute_n,
)
from MuyGPyS._src.gp.muygps.jax import (
    _muygps_posterior_mean as muygps_posterior_mean_j,
    _muygps_diagonal_variance as muygps_diagonal_variance_j,
    _muygps_fast_posterior_mean as muygps_fast_posterior_mean_j,
    _mmuygps_fast_posterior_mean as mmuygps_fast_posterior_mean_j,
    _muygps_fast_posterior_mean_precompute as muygps_fast_posterior_mean_precompute_j,
)
from MuyGPyS._src.gp.noise.numpy import (
    _homoscedastic_perturb as homoscedastic_perturb_n,
    _heteroscedastic_perturb as heteroscedastic_perturb_n,
)
from MuyGPyS._src.gp.noise.jax import (
    _homoscedastic_perturb as homoscedastic_perturb_j,
    _heteroscedastic_perturb as heteroscedastic_perturb_j,
)
from MuyGPyS._src.optimize.chassis.numpy import (
    _scipy_optimize as scipy_optimize_n,
    _bayes_opt_optimize as bayes_optimize_n,
)
from MuyGPyS._src.optimize.chassis.jax import (
    _scipy_optimize as scipy_optimize_j,
    _bayes_opt_optimize as bayes_optimize_j,
)
from MuyGPyS._src.optimize.loss.numpy import (
    _mse_fn as mse_fn_n,
    _cross_entropy_fn as cross_entropy_fn_n,
    _lool_fn as lool_fn_n,
    _pseudo_huber_fn as pseudo_huber_fn_n,
    _looph_fn as looph_fn_n,
)
from MuyGPyS._src.optimize.loss.jax import (
    _mse_fn as mse_fn_j,
    _cross_entropy_fn as cross_entropy_fn_j,
    _lool_fn as lool_fn_j,
    _pseudo_huber_fn as pseudo_huber_fn_j,
    _looph_fn as looph_fn_j,
)
from MuyGPyS._src.optimize.sigma_sq.numpy import (
    _analytic_sigma_sq_optim as analytic_sigma_sq_optim_n,
)
from MuyGPyS._src.optimize.sigma_sq.jax import (
    _analytic_sigma_sq_optim as analytic_sigma_sq_optim_j,
)
from MuyGPyS._test.utils import (
    _check_ndarray,
    _exact_nn_kwarg_options,
    _make_gaussian_matrix,
    _make_gaussian_data,
    _make_heteroscedastic_test_nugget,
)
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.gp.distortion import (
    apply_distortion,
    AnisotropicDistortion,
    IsotropicDistortion,
)
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels import Matern
from MuyGPyS.gp.noise import (
    HeteroscedasticNoise,
    HomoscedasticNoise,
    noise_perturb,
)
from MuyGPyS.gp.sigma_sq import sigma_sq_scale
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.objective import make_loo_crossval_fn
from MuyGPyS.optimize.sigma_sq import make_analytic_sigma_sq_optim

if config.state.jax_enabled is False:
    raise ValueError("Bad attempt to run jax-only code with jax diabled.")
if config.state.backend == "mpi":
    raise ValueError("Bad attempt to run non-MPI code in MPI mode.")
if config.state.backend != "numpy":
    raise ValueError(
        f"torch_correctness.py must be run in numpy mode, not "
        f"{config.state.backend} mode."
    )


def isotropic_F2_n(diffs, length_scale):
    return F2_n(diffs / length_scale)


def isotropic_l2_n(diffs, length_scale):
    return l2_n(diffs / length_scale)


def isotropic_F2_j(diffs, length_scale):
    return F2_j(diffs / length_scale)


def isotropic_l2_j(diffs, length_scale):
    return l2_j(diffs / length_scale)


def anisotropic_F2_n(diffs, **length_scales):
    length_scale_array = AnisotropicDistortion._get_length_scale_array(
        np.array, diffs.shape, **length_scales
    )
    return F2_n(diffs / length_scale_array)


def anisotropic_l2_n(diffs, **length_scales):
    length_scale_array = AnisotropicDistortion._get_length_scale_array(
        np.array, diffs.shape, **length_scales
    )
    return l2_n(diffs / length_scale_array)


def anisotropic_F2_j(diffs, **length_scales):
    length_scale_array = AnisotropicDistortion._get_length_scale_array(
        jnp.array, diffs.shape, **length_scales
    )
    return F2_j(diffs / length_scale_array)


def anisotropic_l2_j(diffs, **length_scales):
    length_scale_array = AnisotropicDistortion._get_length_scale_array(
        jnp.array, diffs.shape, **length_scales
    )
    return l2_j(diffs / length_scale_array)


rbf_isotropic_fn_n = apply_distortion(isotropic_F2_n, length_scale=1.0)(
    rbf_fn_n
)
matern_05_isotropic_fn_n = apply_distortion(isotropic_l2_n, length_scale=1.0)(
    matern_05_fn_n
)
matern_15_isotropic_fn_n = apply_distortion(isotropic_l2_n, length_scale=1.0)(
    matern_15_fn_n
)
matern_25_isotropic_fn_n = apply_distortion(isotropic_l2_n, length_scale=1.0)(
    matern_25_fn_n
)
matern_inf_isotropic_fn_n = apply_distortion(isotropic_l2_n, length_scale=1.0)(
    matern_inf_fn_n
)
matern_gen_isotropic_fn_n = apply_distortion(isotropic_l2_n, length_scale=1.0)(
    matern_gen_fn_n
)

rbf_isotropic_fn_j = apply_distortion(isotropic_F2_j, length_scale=1.0)(
    rbf_fn_j
)
matern_05_isotropic_fn_j = apply_distortion(isotropic_l2_j, length_scale=1.0)(
    matern_05_fn_j
)
matern_15_isotropic_fn_j = apply_distortion(isotropic_l2_j, length_scale=1.0)(
    matern_15_fn_j
)
matern_25_isotropic_fn_j = apply_distortion(isotropic_l2_j, length_scale=1.0)(
    matern_25_fn_j
)
matern_inf_isotropic_fn_j = apply_distortion(isotropic_l2_j, length_scale=1.0)(
    matern_inf_fn_j
)
matern_gen_isotropic_fn_j = apply_distortion(isotropic_l2_j, length_scale=1.0)(
    matern_gen_fn_j
)

rbf_anisotropic_fn_n = apply_distortion(anisotropic_F2_n, length_scale0=1.0)(
    rbf_fn_n
)
matern_05_anisotropic_fn_n = apply_distortion(
    anisotropic_l2_n, length_scale0=1.0
)(matern_05_fn_n)
matern_15_anisotropic_fn_n = apply_distortion(
    anisotropic_l2_n, length_scale0=1.0
)(matern_15_fn_n)
matern_25_anisotropic_fn_n = apply_distortion(
    anisotropic_l2_n, length_scale0=1.0
)(matern_25_fn_n)
matern_inf_anisotropic_fn_n = apply_distortion(
    anisotropic_l2_n, length_scale0=1.0
)(matern_inf_fn_n)
matern_gen_anisotropic_fn_n = apply_distortion(
    anisotropic_l2_n, length_scale0=1.0
)(matern_gen_fn_n)

rbf_anisotropic_fn_j = apply_distortion(anisotropic_F2_j, length_scale0=1.0)(
    rbf_fn_j
)
matern_05_anisotropic_fn_j = apply_distortion(
    anisotropic_l2_j, length_scale0=1.0
)(matern_05_fn_j)
matern_15_anisotropic_fn_j = apply_distortion(
    anisotropic_l2_j, length_scale0=1.0
)(matern_15_fn_j)
matern_25_anisotropic_fn_j = apply_distortion(
    anisotropic_l2_j, length_scale0=1.0
)(matern_25_fn_j)
matern_inf_anisotropic_fn_j = apply_distortion(
    anisotropic_l2_j, length_scale0=1.0
)(matern_inf_fn_j)
matern_gen_anisotropic_fn_j = apply_distortion(
    anisotropic_l2_j, length_scale0=1.0
)(matern_gen_fn_j)


def allclose_gen(a: np.ndarray, b: np.ndarray) -> bool:
    if jax_config.x64_enabled:  # type: ignore
        return np.allclose(a, b)
    else:
        return np.allclose(a, b, atol=1e-7)


def allclose_var(a: np.ndarray, b: np.ndarray) -> bool:
    if jax_config.x64_enabled:  # type: ignore
        return np.allclose(a, b)
    else:
        return np.allclose(a, b, atol=1e-6)


def allclose_inv(a: np.ndarray, b: np.ndarray) -> bool:
    if jax_config.x64_enabled:  # type: ignore
        return np.allclose(a, b)
    else:
        return np.allclose(a, b, atol=1e-3)


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
        cls.nu = 0.55
        cls.nu_bounds = (1e-1, 1e1)
        cls.eps = 1e-3
        cls.eps_heteroscedastic_n = cls.eps * np.ones(
            (cls.batch_count, cls.nn_count)
        )
        cls.eps_heteroscedastic_train_n = cls.eps * np.ones(
            (cls.train_count, cls.nn_count)
        )

        cls.eps_heteroscedastic_j = jnp.array(cls.eps_heteroscedastic_n)
        cls.eps_heteroscedastic_train_j = jnp.array(
            cls.eps_heteroscedastic_train_n
        )
        cls.k_kwargs = {
            "kernel": Matern(
                nu=ScalarHyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    l2_n, length_scale=ScalarHyperparameter(cls.length_scale)
                ),
            ),
            "eps": HomoscedasticNoise(cls.eps),
        }
        cls.k_kwargs_heteroscedastic = {
            "kernel": Matern(
                nu=ScalarHyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    l2_n, length_scale=ScalarHyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_n),
        }
        cls.train_features_n = _make_gaussian_matrix(
            cls.train_count, cls.feature_count
        )
        cls.train_features_j = jnp.array(cls.train_features_n)
        cls.train_responses_n = _make_gaussian_matrix(
            cls.train_count, cls.response_count
        )
        cls.train_responses_j = jnp.array(cls.train_responses_n)
        cls.test_features_n = _make_gaussian_matrix(
            cls.test_count, cls.feature_count
        )
        cls.test_features_j = jnp.array(cls.test_features_n)
        cls.test_responses_n = _make_gaussian_matrix(
            cls.test_count, cls.response_count
        )
        cls.test_responses_j = jnp.array(cls.test_responses_n)
        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features_n, cls.nn_count, **_exact_nn_kwarg_options[0]
        )
        cls.muygps = MuyGPS(**cls.k_kwargs)
        cls.muygps_heteroscedastic = MuyGPS(**cls.k_kwargs_heteroscedastic)
        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_j = jnp.iarray(cls.batch_indices_n)
        cls.batch_nn_indices_j = jnp.iarray(cls.batch_nn_indices_n)

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)


class TensorsTest(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(TensorsTest, cls).setUpClass()

    def test_types(self):
        self._check_ndarray(self.batch_indices_n, np.itype, ctype=np.ndarray)
        self._check_ndarray(self.batch_nn_indices_n, np.itype, ctype=np.ndarray)
        self._check_ndarray(self.train_features_n, np.ftype, ctype=np.ndarray)
        self._check_ndarray(self.train_responses_n, np.ftype, ctype=np.ndarray)
        self._check_ndarray(self.batch_indices_j, jnp.itype, ctype=jnp.ndarray)
        self._check_ndarray(
            self.batch_nn_indices_j, jnp.itype, ctype=jnp.ndarray
        )
        self._check_ndarray(self.train_features_j, jnp.ftype, ctype=jnp.ndarray)
        self._check_ndarray(
            self.train_responses_j, jnp.ftype, ctype=jnp.ndarray
        )

    def test_pairwise_tensor(self):
        self.assertTrue(
            allclose_gen(
                pairwise_tensor_n(
                    self.train_features_n, self.batch_nn_indices_n
                ),
                pairwise_tensor_j(
                    self.train_features_j, self.batch_nn_indices_j
                ),
            )
        )

    def test_crosswise_tensor(self):
        self.assertTrue(
            allclose_gen(
                crosswise_tensor_n(
                    self.train_features_n,
                    self.train_features_n,
                    self.batch_indices_n,
                    self.batch_nn_indices_n,
                ),
                crosswise_tensor_j(
                    self.train_features_j,
                    self.train_features_j,
                    self.batch_indices_j,
                    self.batch_nn_indices_j,
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
            crosswise_diffs_j,
            pairwise_diffs_j,
            batch_targets_j,
            batch_nn_targets_j,
        ) = make_train_tensors_j(
            self.batch_indices_j,
            self.batch_nn_indices_j,
            self.train_features_j,
            self.train_responses_j,
        )
        self.assertTrue(allclose_gen(crosswise_diffs_n, crosswise_diffs_j))
        self.assertTrue(allclose_gen(pairwise_diffs_n, pairwise_diffs_j))
        self.assertTrue(allclose_gen(batch_targets_n, batch_targets_j))
        self.assertTrue(allclose_gen(batch_nn_targets_n, batch_nn_targets_j))


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
            cls.crosswise_diffs_j,
            cls.pairwise_diffs_j,
            cls.batch_targets_j,
            cls.batch_nn_targets_j,
        ) = make_train_tensors_j(
            cls.batch_indices_j,
            cls.batch_nn_indices_j,
            cls.train_features_j,
            cls.train_responses_j,
        )

    def _check_ndarray(self, *args, **kwargs):
        return _check_ndarray(self.assertEqual, *args, **kwargs)


class KernelTest(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTest, cls).setUpClass()

    def _test_types(
        self,
        crosswise_diffs,
        pairwise_diffs,
        batch_targets,
        batch_nn_targets,
        ftype,
        ctype,
    ):
        self._check_ndarray(
            crosswise_diffs,
            ftype,
            ctype=ctype,
            shape=(self.batch_count, self.nn_count, self.feature_count),
        )
        self._check_ndarray(
            pairwise_diffs,
            ftype,
            ctype=ctype,
            shape=(
                self.batch_count,
                self.nn_count,
                self.nn_count,
                self.feature_count,
            ),
        )
        self._check_ndarray(
            batch_targets,
            ftype,
            ctype=ctype,
            shape=(self.batch_count, self.response_count),
        )
        self._check_ndarray(
            batch_nn_targets,
            ftype,
            ctype=ctype,
            shape=(self.batch_count, self.nn_count, self.response_count),
        )

    def test_types(self):
        self._test_types(
            self.crosswise_diffs_j,
            self.pairwise_diffs_j,
            self.batch_targets_j,
            self.batch_nn_targets_j,
            jnp.ftype,
            jnp.ndarray,
        )
        self._test_types(
            self.crosswise_diffs_n,
            self.pairwise_diffs_n,
            self.batch_targets_n,
            self.batch_nn_targets_n,
            np.ftype,
            np.ndarray,
        )

    def test_crosswise_rbf(self):
        self.assertTrue(
            allclose_gen(
                rbf_isotropic_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                rbf_isotropic_fn_j(
                    self.crosswise_diffs_j, length_scale=self.length_scale
                ),
            )
        )

    def test_pairwise_rbf(self):
        self.assertTrue(
            allclose_gen(
                rbf_isotropic_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                rbf_isotropic_fn_j(
                    self.pairwise_diffs_j, length_scale=self.length_scale
                ),
            )
        )

    def test_crosswise_matern(self):
        self.assertTrue(
            allclose_gen(
                matern_05_isotropic_fn_n(self.crosswise_diffs_n),
                matern_05_isotropic_fn_j(self.crosswise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_05_anisotropic_fn_n(
                    self.crosswise_diffs_n, length_scale0=1.0
                ),
                matern_05_anisotropic_fn_j(
                    self.crosswise_diffs_j, length_scale0=1.0
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_15_isotropic_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_15_isotropic_fn_j(
                    self.crosswise_diffs_j, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_15_anisotropic_fn_n(
                    self.crosswise_diffs_n, length_scale0=self.length_scale
                ),
                matern_15_anisotropic_fn_j(
                    self.crosswise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_25_isotropic_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_25_isotropic_fn_j(
                    self.crosswise_diffs_j, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_25_anisotropic_fn_n(
                    self.crosswise_diffs_n, length_scale0=self.length_scale
                ),
                matern_25_anisotropic_fn_j(
                    self.crosswise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_inf_isotropic_fn_n(
                    self.crosswise_diffs_n, length_scale=self.length_scale
                ),
                matern_inf_isotropic_fn_j(
                    self.crosswise_diffs_j, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_inf_anisotropic_fn_n(
                    self.crosswise_diffs_n, length_scale0=self.length_scale
                ),
                matern_inf_anisotropic_fn_j(
                    self.crosswise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )

        self.assertTrue(
            allclose_gen(
                matern_gen_isotropic_fn_n(
                    self.crosswise_diffs_n,
                    nu=self.nu,
                    length_scale=self.length_scale,
                ),
                matern_gen_isotropic_fn_j(
                    self.crosswise_diffs_j,
                    nu=self.nu,
                    length_scale=self.length_scale,
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_gen_anisotropic_fn_n(
                    self.crosswise_diffs_n,
                    nu=self.nu,
                    length_scale0=self.length_scale,
                ),
                matern_gen_anisotropic_fn_j(
                    self.crosswise_diffs_j,
                    nu=self.nu,
                    length_scale0=self.length_scale,
                ),
            )
        )

    def test_pairwise_matern(self):
        self.assertTrue(
            allclose_gen(
                matern_05_isotropic_fn_n(self.pairwise_diffs_n),
                matern_05_isotropic_fn_j(self.pairwise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_05_anisotropic_fn_n(
                    self.pairwise_diffs_n, length_scale0=self.length_scale
                ),
                matern_05_anisotropic_fn_j(
                    self.pairwise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_15_isotropic_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_15_isotropic_fn_j(
                    self.pairwise_diffs_j, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_15_anisotropic_fn_n(
                    self.pairwise_diffs_n, length_scale0=self.length_scale
                ),
                matern_15_anisotropic_fn_j(
                    self.pairwise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_25_isotropic_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_25_isotropic_fn_j(
                    self.pairwise_diffs_j, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_25_anisotropic_fn_n(
                    self.pairwise_diffs_n, length_scale0=self.length_scale
                ),
                matern_25_anisotropic_fn_j(
                    self.pairwise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_inf_isotropic_fn_n(
                    self.pairwise_diffs_n, length_scale=self.length_scale
                ),
                matern_inf_isotropic_fn_j(
                    self.pairwise_diffs_j, length_scale=self.length_scale
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_inf_anisotropic_fn_n(
                    self.pairwise_diffs_n, length_scale0=self.length_scale
                ),
                matern_inf_anisotropic_fn_j(
                    self.pairwise_diffs_j, length_scale0=self.length_scale
                ),
            )
        )

        self.assertTrue(
            allclose_gen(
                matern_gen_isotropic_fn_n(
                    self.pairwise_diffs_n,
                    nu=self.nu,
                    length_scale=self.length_scale,
                ),
                matern_gen_isotropic_fn_j(
                    self.pairwise_diffs_j,
                    nu=self.nu,
                    length_scale=self.length_scale,
                ),
            )
        )
        self.assertTrue(
            allclose_gen(
                matern_gen_anisotropic_fn_n(
                    self.pairwise_diffs_n,
                    nu=self.nu,
                    length_scale0=self.length_scale,
                ),
                matern_gen_anisotropic_fn_j(
                    self.pairwise_diffs_j,
                    nu=self.nu,
                    length_scale0=self.length_scale,
                ),
            )
        )


class MuyGPSTestCase(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTestCase, cls).setUpClass()
        cls.K_n = matern_gen_isotropic_fn_n(
            cls.pairwise_diffs_n, nu=cls.nu, length_scale=cls.length_scale
        )
        cls.K_j = jnp.array(cls.K_n)
        cls.homoscedastic_K_n = homoscedastic_perturb_n(cls.K_n, cls.eps)
        cls.homoscedastic_K_j = homoscedastic_perturb_j(cls.K_j, cls.eps)
        cls.heteroscedastic_K_n = heteroscedastic_perturb_n(
            cls.K_n, cls.eps_heteroscedastic_n
        )
        cls.heteroscedastic_K_j = heteroscedastic_perturb_j(
            cls.K_j, cls.eps_heteroscedastic_j
        )
        cls.Kcross_n = matern_gen_isotropic_fn_n(
            cls.crosswise_diffs_n, nu=cls.nu, length_scale=cls.length_scale
        )
        cls.Kcross_j = jnp.array(cls.Kcross_n)


class MuyGPSTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTest, cls).setUpClass()

    def test_homoscedastic_perturb(self):
        self.assertTrue(
            allclose_gen(self.homoscedastic_K_n, self.homoscedastic_K_j)
        )

    def test_heteroscedastic_perturb(self):
        self.assertTrue(
            allclose_gen(self.heteroscedastic_K_n, self.heteroscedastic_K_j)
        )

    def test_posterior_mean(self):
        self.assertTrue(
            allclose_inv(
                muygps_posterior_mean_n(
                    self.homoscedastic_K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                muygps_posterior_mean_j(
                    self.homoscedastic_K_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                ),
            )
        )

    def test_posterior_mean_heteroscedastic(self):
        self.assertTrue(
            allclose_inv(
                muygps_posterior_mean_n(
                    self.heteroscedastic_K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                muygps_posterior_mean_j(
                    self.heteroscedastic_K_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                ),
            )
        )

    def test_diagonal_variance(self):
        self.assertTrue(
            allclose_var(
                muygps_diagonal_variance_n(
                    self.homoscedastic_K_n, self.Kcross_n
                ),
                muygps_diagonal_variance_j(
                    self.homoscedastic_K_j, self.Kcross_j
                ),
            )
        )

    def test_diagonal_variance_heteroscedastic(self):
        self.assertTrue(
            allclose_var(
                muygps_diagonal_variance_n(
                    self.heteroscedastic_K_n, self.Kcross_n
                ),
                muygps_diagonal_variance_j(
                    self.heteroscedastic_K_j, self.Kcross_j
                ),
            )
        )

    def test_sigma_sq_optim(self):
        self.assertTrue(
            allclose_inv(
                analytic_sigma_sq_optim_n(
                    self.homoscedastic_K_n, self.batch_nn_targets_n
                ),
                analytic_sigma_sq_optim_j(
                    self.homoscedastic_K_j, self.batch_nn_targets_j
                ),
            )
        )

    def test_sigma_sq_optim_heteroscedastic(self):
        self.assertTrue(
            allclose_inv(
                analytic_sigma_sq_optim_n(
                    self.heteroscedastic_K_n, self.batch_nn_targets_n
                ),
                analytic_sigma_sq_optim_j(
                    self.heteroscedastic_K_j, self.batch_nn_targets_j
                ),
            )
        )


class FastPredictTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastPredictTest, cls).setUpClass()
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
        cls.Kcross_fast_n = matern_gen_isotropic_fn_n(
            cls.crosswise_diffs_fast_n,
            nu=cls.nu,
            length_scale=cls.length_scale,
        )

        cls.nn_indices_all_j, _ = cls.nbrs_lookup.get_batch_nns(
            np.arange(0, cls.train_count)
        )
        cls.nn_indices_all_j = jnp.iarray(cls.nn_indices_all_j)

        (
            cls.K_fast_j,
            cls.train_nn_targets_fast_j,
        ) = make_fast_predict_tensors_j(
            cls.nn_indices_all_j,
            cls.train_features_j,
            cls.train_responses_j,
        )

        cls.homoscedastic_K_fast_j = homoscedastic_perturb_j(
            l2_j(cls.K_fast_j), cls.eps
        )

        cls.heteroscedastic_K_fast_j = heteroscedastic_perturb_j(
            l2_n(cls.K_fast_j), cls.eps_heteroscedastic_train_j
        )

        cls.fast_regress_coeffs_j = muygps_fast_posterior_mean_precompute_j(
            cls.homoscedastic_K_fast_j, cls.train_nn_targets_fast_j
        )

        cls.fast_regress_coeffs_heteroscedastic_j = (
            muygps_fast_posterior_mean_precompute_j(
                cls.heteroscedastic_K_fast_j, cls.train_nn_targets_fast_j
            )
        )

        cls.test_neighbors_j, _ = cls.nbrs_lookup.get_nns(cls.test_features_j)
        cls.closest_neighbor_j = cls.test_neighbors_j[:, 0]
        cls.closest_set_j = cls.nn_indices_all_j[cls.closest_neighbor_j].astype(
            int
        )

        cls.new_nn_indices_j = fast_nn_update_j(cls.nn_indices_all_j)
        cls.closest_set_new_j = cls.new_nn_indices_j[cls.closest_neighbor_j]
        cls.crosswise_diffs_fast_j = crosswise_tensor_j(
            cls.test_features_j,
            cls.train_features_j,
            np.arange(0, cls.test_count),
            cls.closest_set_new_j,
        )
        cls.Kcross_fast_j = matern_gen_isotropic_fn_j(
            cls.crosswise_diffs_fast_j,
            nu=cls.nu,
            length_scale=cls.length_scale,
        )

    def test_fast_nn_update(self):
        self.assertTrue(
            allclose_gen(
                fast_nn_update_j(self.nn_indices_all_j),
                fast_nn_update_n(self.nn_indices_all_n),
            )
        )

    def test_make_fast_predict_tensors(self):
        self.assertTrue(allclose_gen(self.K_fast_n, self.K_fast_j))
        self.assertTrue(
            allclose_gen(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_j
            )
        )

    def test_homoscedastic_kernel_tensors(self):
        self.assertTrue(
            allclose_inv(
                self.homoscedastic_K_fast_n, self.homoscedastic_K_fast_j
            )
        )

    def test_heteroscedastic_kernel_tensors(self):
        self.assertTrue(
            allclose_inv(
                self.heteroscedastic_K_fast_n, self.heteroscedastic_K_fast_j
            )
        )

    def test_fast_predict(self):
        self.assertTrue(
            allclose_inv(
                muygps_fast_posterior_mean_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                ),
                muygps_fast_posterior_mean_j(
                    self.Kcross_fast_j,
                    self.fast_regress_coeffs_j[self.closest_neighbor_j, :],
                ),
            )
        )

    def test_fast_predict_coeffs(self):
        self.assertTrue(
            allclose_inv(
                self.fast_regress_coeffs_n,
                self.fast_regress_coeffs_j,
            )
        )


class FastMultivariatePredictTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(FastMultivariatePredictTest, cls).setUpClass()
        cls.train_count = 1000
        cls.test_count = 100
        cls.feature_count = 10
        cls.response_count = 2
        cls.nn_count = 40
        cls.batch_count = 500
        cls.length_scale = 1.0
        cls.nu = 0.55
        cls.nu_bounds = (1e-1, 1e1)
        cls.eps = 1e-3
        cls.eps_heteroscedastic_n = _make_heteroscedastic_test_nugget(
            cls.batch_count, cls.nn_count, cls.eps
        )
        cls.eps_heteroscedastic_train_n = _make_heteroscedastic_test_nugget(
            cls.train_count, cls.nn_count, cls.eps
        )
        cls.eps_heteroscedastic_j = jnp.array(cls.eps_heteroscedastic_n)
        cls.eps_heteroscedastic_train_j = jnp.array(
            cls.eps_heteroscedastic_train_n
        )
        cls.k_kwargs_1 = {
            "kernel": Matern(
                nu=ScalarHyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    l2_n, length_scale=ScalarHyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_train_n),
        }
        cls.k_kwargs_2 = {
            "kernel": Matern(
                nu=ScalarHyperparameter(cls.nu, cls.nu_bounds),
                metric=IsotropicDistortion(
                    l2_n, length_scale=ScalarHyperparameter(cls.length_scale)
                ),
            ),
            "eps": HeteroscedasticNoise(cls.eps_heteroscedastic_train_n),
        }
        cls.k_kwargs = [cls.k_kwargs_1, cls.k_kwargs_2]
        cls.train_features_n = _make_gaussian_matrix(
            cls.train_count, cls.feature_count
        )
        cls.train_features_j = jnp.array(cls.train_features_n)
        cls.train_responses_n = _make_gaussian_matrix(
            cls.train_count, cls.response_count
        )
        cls.train_responses_j = jnp.array(cls.train_responses_n)
        cls.test_features_n = _make_gaussian_matrix(
            cls.test_count, cls.feature_count
        )
        cls.test_features_j = jnp.array(cls.test_features_n)
        cls.test_responses_n = _make_gaussian_matrix(
            cls.test_count, cls.response_count
        )
        cls.test_responses_j = jnp.array(cls.test_responses_n)
        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features_n, cls.nn_count, **_exact_nn_kwarg_options[0]
        )
        cls.muygps = MMuyGPS(*cls.k_kwargs)
        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_j = jnp.iarray(cls.batch_indices_n)
        cls.batch_nn_indices_j = jnp.iarray(cls.batch_nn_indices_n)
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
        cls.fast_regress_coeffs_n = muygps_fast_posterior_mean_precompute_n(
            cls.homoscedastic_K_fast_n, cls.train_nn_targets_fast_n
        )

        cls.heteroscedastic_K_fast_n = heteroscedastic_perturb_n(
            l2_n(cls.K_fast_n), cls.eps_heteroscedastic_train_n
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
        Kcross_fast_n = np.zeros(
            (cls.test_count, cls.nn_count, cls.response_count)
        )
        for i, model in enumerate(cls.muygps.models):
            Kcross_fast_n[:, :, i] = model.kernel(cls.crosswise_diffs_fast_n)
        cls.Kcross_fast_n = Kcross_fast_n

        cls.nn_indices_all_j, _ = cls.nbrs_lookup.get_batch_nns(
            np.arange(0, cls.train_count)
        )
        cls.nn_indices_all_j = jnp.iarray(cls.nn_indices_all_j)

        (
            cls.K_fast_j,
            cls.train_nn_targets_fast_j,
        ) = make_fast_predict_tensors_j(
            cls.nn_indices_all_j,
            cls.train_features_j,
            cls.train_responses_j,
        )

        cls.homoscedastic_K_fast_j = homoscedastic_perturb_j(
            l2_j(cls.K_fast_j), cls.eps
        )
        cls.fast_regress_coeffs_j = muygps_fast_posterior_mean_precompute_j(
            cls.homoscedastic_K_fast_j, cls.train_nn_targets_fast_j
        )

        cls.heteroscedastic_K_fast_j = heteroscedastic_perturb_j(
            l2_n(cls.K_fast_j), cls.eps_heteroscedastic_train_j
        )

        cls.fast_regress_coeffs_heteroscedastic_j = (
            muygps_fast_posterior_mean_precompute_j(
                cls.heteroscedastic_K_fast_j, cls.train_nn_targets_fast_j
            )
        )

        cls.test_neighbors_j, _ = cls.nbrs_lookup.get_nns(cls.test_features_j)
        cls.closest_neighbor_j = cls.test_neighbors_j[:, 0]
        cls.closest_set_j = cls.nn_indices_all_j[cls.closest_neighbor_j].astype(
            int
        )

        cls.new_nn_indices_j = fast_nn_update_j(cls.nn_indices_all_j)
        cls.closest_set_new_j = cls.new_nn_indices_j[cls.closest_neighbor_j]
        cls.crosswise_diffs_fast_j = crosswise_tensor_j(
            cls.test_features_j,
            cls.train_features_j,
            np.arange(0, cls.test_count),
            cls.closest_set_new_j,
        )

        cls.Kcross_fast_j = jnp.array(Kcross_fast_n)

    def test_make_fast_homoscedastic_multivariate_predict_tensors(self):
        self.assertTrue(
            allclose_inv(
                self.homoscedastic_K_fast_n, self.homoscedastic_K_fast_j
            )
        )
        self.assertTrue(
            allclose_inv(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_j
            )
        )

    def test_make_fast_heteroscedastic_multivariate_predict_tensors(self):
        self.assertTrue(
            allclose_inv(
                self.heteroscedastic_K_fast_n, self.heteroscedastic_K_fast_j
            )
        )
        self.assertTrue(
            allclose_inv(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_j
            )
        )

    def test_fast_multivariate_predict(self):
        self.assertTrue(
            allclose_inv(
                mmuygps_fast_posterior_mean_n(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                ),
                mmuygps_fast_posterior_mean_j(
                    self.Kcross_fast_j,
                    self.fast_regress_coeffs_j[self.closest_neighbor_j, :],
                ),
            )
        )

    def test_fast_multivariate_predict_coeffs(self):
        self.assertTrue(
            allclose_inv(
                self.fast_regress_coeffs_n,
                self.fast_regress_coeffs_j,
            )
        )

    def test_fast_multivariate_heteroscedastic_predict_coeffs(self):
        self.assertTrue(
            allclose_inv(
                self.fast_regress_coeffs_heteroscedastic_n,
                self.fast_regress_coeffs_heteroscedastic_j,
            )
        )


class OptimTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTestCase, cls).setUpClass()
        cls.predictions_n = muygps_posterior_mean_n(
            cls.homoscedastic_K_n, cls.Kcross_n, cls.batch_nn_targets_n
        )
        cls.variances_n = muygps_diagonal_variance_n(
            cls.homoscedastic_K_n, cls.Kcross_n
        )
        cls.predictions_heteroscedastic_n = muygps_posterior_mean_n(
            cls.heteroscedastic_K_n, cls.Kcross_n, cls.batch_nn_targets_n
        )
        cls.variances_heteroscedastic_n = muygps_diagonal_variance_n(
            cls.heteroscedastic_K_n, cls.Kcross_n
        )
        cls.predictions_j = jnp.array(cls.predictions_n)
        cls.variances_j = jnp.array(cls.variances_n)
        cls.predictions_heteroscedastic_j = jnp.array(
            cls.predictions_heteroscedastic_n
        )
        cls.variances_heteroscedastic_j = jnp.array(
            cls.variances_heteroscedastic_n
        )
        cls.x0_names, cls.x0_n, cls.bounds = cls.muygps.get_opt_params()
        cls.x0_j = jnp.array(cls.x0_n)
        cls.x0_map_n = {n: cls.x0_n[i] for i, n in enumerate(cls.x0_names)}
        cls.x0_map_j = {n: cls.x0_j[i] for i, n in enumerate(cls.x0_names)}

    def _get_kernel_fn_n(self):
        return self.muygps.kernel._get_opt_fn(
            matern_gen_isotropic_fn_n,
            IsotropicDistortion(
                l2_n, length_scale=ScalarHyperparameter(self.length_scale)
            ),
            self.muygps.kernel.nu,
        )

    def _get_kernel_fn_j(self):
        return self.muygps.kernel._get_opt_fn(
            matern_gen_isotropic_fn_j,
            IsotropicDistortion(
                l2_j, length_scale=ScalarHyperparameter(self.length_scale)
            ),
            self.muygps.kernel.nu,
        )

    def _get_mean_fn_n(self):
        return self.muygps._mean_fn._get_opt_fn(
            noise_perturb(homoscedastic_perturb_n)(muygps_posterior_mean_n),
            self.muygps.eps,
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

    def _get_sigma_sq_fn_n(self):
        return make_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_n, homoscedastic_perturb_n
        )

    def _get_mean_fn_heteroscedastic_n(self):
        return self.muygps_heteroscedastic._mean_fn._get_opt_fn(
            noise_perturb(heteroscedastic_perturb_n)(muygps_posterior_mean_n),
            self.muygps_heteroscedastic.eps,
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

    def _get_sigma_sq_fn_heteroscedastic_n(self):
        return make_analytic_sigma_sq_optim(
            self.muygps_heteroscedastic,
            analytic_sigma_sq_optim_n,
            heteroscedastic_perturb_n,
        )

    def _get_mean_fn_j(self):
        return self.muygps._mean_fn._get_opt_fn(
            noise_perturb(homoscedastic_perturb_j)(muygps_posterior_mean_j),
            self.muygps.eps,
        )

    def _get_var_fn_j(self):
        return self.muygps._var_fn._get_opt_fn(
            sigma_sq_scale(
                noise_perturb(homoscedastic_perturb_j)(
                    muygps_diagonal_variance_j
                )
            ),
            self.muygps.eps,
            self.muygps.sigma_sq,
        )

    def _get_sigma_sq_fn_j(self):
        return make_analytic_sigma_sq_optim(
            self.muygps, analytic_sigma_sq_optim_j, homoscedastic_perturb_j
        )

    def _get_mean_fn_heteroscedastic_j(self):
        return self.muygps_heteroscedastic._mean_fn._get_opt_fn(
            noise_perturb(heteroscedastic_perturb_j)(muygps_posterior_mean_j),
            self.muygps_heteroscedastic.eps,
        )

    def _get_var_fn_heteroscedastic_j(self):
        return self.muygps_heteroscedastic._var_fn._get_opt_fn(
            sigma_sq_scale(
                noise_perturb(heteroscedastic_perturb_j)(
                    muygps_diagonal_variance_j
                )
            ),
            self.muygps_heteroscedastic.eps,
            self.muygps_heteroscedastic.sigma_sq,
        )

    def _get_sigma_sq_fn_heteroscedastic_j(self):
        return make_analytic_sigma_sq_optim(
            self.muygps_heteroscedastic,
            analytic_sigma_sq_optim_j,
            heteroscedastic_perturb_j,
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

    def _get_obj_fn_j(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_j,
            self._get_kernel_fn_j(),
            self._get_mean_fn_j(),
            self._get_var_fn_j(),
            self._get_sigma_sq_fn_j(),
            self.pairwise_diffs_j,
            self.crosswise_diffs_j,
            self.batch_nn_targets_j,
            self.batch_targets_j,
        )

    def _get_obj_fn_h(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_j,
            self._get_kernel_fn_j(),
            self._get_mean_fn_n(),
            self._get_var_fn_n(),
            self._get_sigma_sq_fn_n(),
            self.pairwise_diffs_j,
            self.crosswise_diffs_j,
            self.batch_nn_targets_j,
            self.batch_targets_j,
        )

    def _get_obj_fn_heteroscedastic_j(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_j,
            self._get_kernel_fn_j(),
            self._get_mean_fn_heteroscedastic_j(),
            self._get_var_fn_heteroscedastic_j(),
            self._get_sigma_sq_fn_heteroscedastic_j(),
            self.pairwise_diffs_j,
            self.crosswise_diffs_j,
            self.batch_nn_targets_j,
            self.batch_targets_j,
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

    def _get_obj_fn_heteroscedastic_h(self):
        return make_loo_crossval_fn(
            "mse",
            mse_fn_j,
            self._get_kernel_fn_j(),
            self._get_mean_fn_heteroscedastic_j(),
            self._get_var_fn_heteroscedastic_j(),
            self._get_sigma_sq_fn_heteroscedastic_j(),
            self.pairwise_diffs_n,
            self.crosswise_diffs_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )


class ObjectiveTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectiveTest, cls).setUpClass()

        cls.sigma_sq_n = cls.muygps.sigma_sq()
        cls.sigma_sq_j = jnp.array(cls.muygps.sigma_sq())

    def test_mse(self):
        self.assertTrue(
            np.isclose(
                mse_fn_n(self.predictions_n, self.batch_targets_n),
                mse_fn_j(self.predictions_j, self.batch_targets_j),
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
                lool_fn_j(
                    self.predictions_j,
                    self.batch_targets_j,
                    self.variances_j,
                    self.sigma_sq_j,
                ),
            )
        )

    @parameterized.parameters(bs for bs in [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_pseudo_huber(self, boundary_scale):
        self.assertTrue(
            np.isclose(
                pseudo_huber_fn_n(
                    self.predictions_n, self.batch_targets_n, boundary_scale
                ),
                pseudo_huber_fn_j(
                    self.predictions_j, self.batch_targets_j, boundary_scale
                ),
            )
        )

    @parameterized.parameters(bs for bs in [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_looph(self, boundary_scale):
        self.assertTrue(
            np.isclose(
                looph_fn_n(
                    self.predictions_n,
                    self.batch_targets_n,
                    self.variances_n,
                    self.sigma_sq_n,
                    boundary_scale=boundary_scale,
                ),
                looph_fn_j(
                    self.predictions_j,
                    self.batch_targets_j,
                    self.variances_j,
                    self.sigma_sq_j,
                    boundary_scale=boundary_scale,
                ),
            )
        )

    def test_cross_entropy(self):
        cat_predictions_n, cat_batch_targets_n = _make_gaussian_data(
            1000, 1000, 10, 2, categorical=True
        )
        cat_predictions_n = cat_predictions_n["output"]
        cat_batch_targets_n = cat_batch_targets_n["output"]
        cat_predictions_j = jnp.array(cat_predictions_n)
        cat_batch_targets_j = jnp.array(cat_batch_targets_n)
        self.assertTrue(
            np.all(
                (
                    np.all(cat_predictions_j == cat_predictions_n),
                    np.all(cat_batch_targets_j == cat_batch_targets_n),
                )
            )
        )
        self.assertTrue(
            allclose_gen(
                cross_entropy_fn_n(
                    cat_predictions_n, cat_batch_targets_n, ll_eps=1e-6
                ),
                cross_entropy_fn_j(
                    cat_predictions_j, cat_batch_targets_j, ll_eps=1e-6
                ),
            )
        )

    def test_kernel_fn(self):
        kernel_fn_n = self._get_kernel_fn_n()
        kernel_fn_j = self._get_kernel_fn_j()
        self.assertTrue(
            allclose_gen(
                kernel_fn_n(self.pairwise_diffs_n, **self.x0_map_n),
                kernel_fn_j(self.pairwise_diffs_j, **self.x0_map_j),
            )
        )

    def test_mean_fn(self):
        mean_fn_n = self._get_mean_fn_n()
        mean_fn_j = self._get_mean_fn_j()
        self.assertTrue(
            allclose_inv(
                mean_fn_n(
                    self.K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_j(
                    self.K_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_mean_fn_heteroscedastic(self):
        mean_fn_heteroscedastic_n = self._get_mean_fn_heteroscedastic_n()
        mean_fn_heteroscedastic_j = self._get_mean_fn_heteroscedastic_j()
        self.assertTrue(
            allclose_inv(
                mean_fn_heteroscedastic_n(
                    self.K_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_heteroscedastic_j(
                    self.K_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_var_fn(self):
        var_fn_n = self._get_var_fn_n()
        var_fn_j = self._get_var_fn_j()
        self.assertTrue(
            allclose_inv(
                var_fn_n(
                    self.K_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_j(
                    self.K_j,
                    self.Kcross_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_var_fn_heteroscedastic(self):
        var_fn_heteroscedastic_n = self._get_var_fn_heteroscedastic_n()
        var_fn_heteroscedastic_j = self._get_var_fn_heteroscedastic_j()
        self.assertTrue(
            allclose_inv(
                var_fn_heteroscedastic_n(
                    self.K_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_heteroscedastic_j(
                    self.K_j,
                    self.Kcross_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_sigma_sq_fn(self):
        ss_fn_n = self._get_sigma_sq_fn_n()
        ss_fn_j = self._get_sigma_sq_fn_j()
        self.assertTrue(
            allclose_inv(
                ss_fn_n(
                    self.K_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_j(
                    self.K_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_sigma_sq_heteroscedastic_fn(self):
        ss_fn_heteroscedastic_n = self._get_sigma_sq_fn_heteroscedastic_n()
        ss_fn_heteroscedastic_j = self._get_sigma_sq_fn_heteroscedastic_j()
        self.assertTrue(
            allclose_inv(
                ss_fn_heteroscedastic_n(
                    self.K_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_heteroscedastic_j(
                    self.K_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_loo_crossval(self):
        obj_fn_n = self._get_obj_fn_n()
        obj_fn_j = self._get_obj_fn_j()
        obj_fn_h = self._get_obj_fn_h()
        obj_fn_heteroscedastic_n = self._get_obj_fn_heteroscedastic_n()
        obj_fn_heteroscedastic_j = self._get_obj_fn_heteroscedastic_j()
        obj_fn_heteroscedastic_h = self._get_obj_fn_heteroscedastic_h()
        self.assertTrue(
            allclose_inv(obj_fn_n(**self.x0_map_n), obj_fn_j(**self.x0_map_j))
        )
        self.assertTrue(
            allclose_inv(obj_fn_n(**self.x0_map_n), obj_fn_h(**self.x0_map_j))
        )
        self.assertTrue(
            allclose_inv(
                obj_fn_heteroscedastic_n(**self.x0_map_n),
                obj_fn_heteroscedastic_h(**self.x0_map_j),
            )
        )
        self.assertTrue(
            allclose_inv(
                obj_fn_heteroscedastic_n(**self.x0_map_n),
                obj_fn_heteroscedastic_j(**self.x0_map_j),
            )
        )


class OptimTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTest, cls).setUpClass()
        cls.sopt_kwargs = {"verbose": False}

    def test_scipy_optimize(self):
        obj_fn_n = self._get_obj_fn_n()
        obj_fn_j = self._get_obj_fn_j()
        obj_fn_h = self._get_obj_fn_h()
        obj_fn_het_j = self._get_obj_fn_heteroscedastic_j()
        obj_fn_het_n = self._get_obj_fn_heteroscedastic_n()
        obj_fn_het_h = self._get_obj_fn_heteroscedastic_h()

        mopt_n = scipy_optimize_n(self.muygps, obj_fn_n, **self.sopt_kwargs)
        mopt_j = scipy_optimize_j(self.muygps, obj_fn_j, **self.sopt_kwargs)
        mopt_h = scipy_optimize_j(self.muygps, obj_fn_h, **self.sopt_kwargs)
        mopt_het_j = scipy_optimize_j(
            self.muygps_heteroscedastic, obj_fn_het_j, **self.sopt_kwargs
        )
        mopt_het_n = scipy_optimize_n(
            self.muygps_heteroscedastic, obj_fn_het_n, **self.sopt_kwargs
        )
        mopt_het_h = scipy_optimize_j(
            self.muygps_heteroscedastic, obj_fn_het_h, **self.sopt_kwargs
        )
        self.assertTrue(allclose_gen(mopt_n.kernel.nu(), mopt_j.kernel.nu()))
        self.assertTrue(allclose_gen(mopt_n.kernel.nu(), mopt_h.kernel.nu()))
        self.assertTrue(
            allclose_inv(mopt_het_n.kernel.nu(), mopt_het_j.kernel.nu())
        )
        self.assertTrue(
            allclose_inv(mopt_het_n.kernel.nu(), mopt_het_h.kernel.nu())
        )


class BayesOptimTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(BayesOptimTest, cls).setUpClass()
        cls.bopt_kwargs = {
            "verbose": False,
            "random_state": 1,
            "init_points": 5,
            "n_iter": 5,
            "allow_duplicate_points": True,
        }

    def test_optimize(self):
        if config.state.ftype == "32":
            import warnings

            warnings.warn("Bayesopt does not support JAX in 32bit mode.")
            return
        obj_fn_n = self._get_obj_fn_n()
        obj_fn_j = self._get_obj_fn_j()
        obj_fn_h = self._get_obj_fn_h()
        obj_fn_het_j = self._get_obj_fn_heteroscedastic_j()
        obj_fn_het_n = self._get_obj_fn_heteroscedastic_n()
        obj_fn_het_h = self._get_obj_fn_heteroscedastic_h()

        mopt_n = bayes_optimize_n(self.muygps, obj_fn_n, **self.bopt_kwargs)
        mopt_j = bayes_optimize_j(self.muygps, obj_fn_j, **self.bopt_kwargs)
        mopt_h = bayes_optimize_j(self.muygps, obj_fn_h, **self.bopt_kwargs)
        mopt_het_j = bayes_optimize_j(
            self.muygps_heteroscedastic, obj_fn_het_j, **self.bopt_kwargs
        )
        mopt_het_n = bayes_optimize_n(
            self.muygps_heteroscedastic, obj_fn_het_n, **self.bopt_kwargs
        )
        mopt_het_h = bayes_optimize_j(
            self.muygps_heteroscedastic, obj_fn_het_h, **self.bopt_kwargs
        )
        self.assertTrue(allclose_inv(mopt_n.kernel.nu(), mopt_j.kernel.nu()))
        self.assertTrue(allclose_inv(mopt_n.kernel.nu(), mopt_h.kernel.nu()))
        self.assertTrue(
            allclose_inv(mopt_het_n.kernel.nu(), mopt_het_j.kernel.nu())
        )
        self.assertTrue(
            allclose_inv(mopt_het_n.kernel.nu(), mopt_het_h.kernel.nu())
        )


if __name__ == "__main__":
    absltest.main()
