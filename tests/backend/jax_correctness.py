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
    _F2 as _F2_n,
    _l2 as _l2_n,
    _pairwise_tensor as pairwise_tensor_n,
    _make_fast_predict_tensors as make_fast_predict_tensors_n,
)
from MuyGPyS._src.gp.tensors.jax import (
    _pairwise_tensor as pairwise_tensor_j,
    _crosswise_tensor as crosswise_tensor_j,
    _make_fast_predict_tensors as make_fast_predict_tensors_j,
    _fast_nn_update as fast_nn_update_j,
    _F2 as _F2_j,
    _l2 as _l2_j,
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
    _muygps_fast_posterior_mean_precompute as muygps_fast_posterior_mean_precompute_n,
)
from MuyGPyS._src.gp.muygps.jax import (
    _muygps_posterior_mean as muygps_posterior_mean_j,
    _muygps_diagonal_variance as muygps_diagonal_variance_j,
    _muygps_fast_posterior_mean as muygps_fast_posterior_mean_j,
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
from MuyGPyS._src.optimize.loss.jax import (
    _mse_fn as mse_fn_j,
    _cross_entropy_fn as cross_entropy_fn_j,
    _lool_fn as lool_fn_j,
    _pseudo_huber_fn as pseudo_huber_fn_j,
    _looph_fn as looph_fn_j,
)
from MuyGPyS._src.optimize.scale.numpy import (
    _analytic_scale_optim as analytic_scale_optim_n,
)
from MuyGPyS._src.optimize.scale.jax import (
    _analytic_scale_optim as analytic_scale_optim_j,
)
from MuyGPyS._test.utils import (
    _check_ndarray,
    _exact_nn_kwarg_options,
    _make_gaussian_matrix,
    _make_gaussian_data,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import Anisotropy, Isotropy, MetricFn
from MuyGPyS.gp.hyperparameter import AnalyticScale, ScalarParam, VectorParam
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.noise import HeteroscedasticNoise, HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import (
    LossFn,
    mse_fn as mse_fn_n,
    cross_entropy_fn as cross_entropy_fn_n,
    lool_fn as lool_fn_n,
    pseudo_huber_fn as pseudo_huber_fn_n,
    looph_fn as looph_fn_n,
    make_raw_predict_and_loss_fn,
    make_var_predict_and_loss_fn,
)
from MuyGPyS.optimize.objective import make_loo_crossval_fn

if config.state.jax_enabled is False:
    raise ValueError("Bad attempt to run jax-only code with jax diabled.")
if config.state.backend == "mpi":
    raise ValueError("Bad attempt to run non-MPI code in MPI mode.")
if config.state.backend != "numpy":
    raise ValueError(
        f"torch_correctness.py must be run in numpy mode, not "
        f"{config.state.backend} mode."
    )

# make jax loss functor aliases
mse_fn_j = LossFn(mse_fn_j, make_raw_predict_and_loss_fn)
cross_entropy_fn_j = LossFn(cross_entropy_fn_j, make_raw_predict_and_loss_fn)
lool_fn_j = LossFn(lool_fn_j, make_var_predict_and_loss_fn)
pseudo_huber_fn_j = LossFn(pseudo_huber_fn_j, make_raw_predict_and_loss_fn)
looph_fn_j = LossFn(looph_fn_j, make_var_predict_and_loss_fn)

l2_n = MetricFn(
    differences_metric_fn=_l2_n,
    crosswise_differences_fn=crosswise_tensor_n,
    pairwise_differences_fn=pairwise_tensor_n,
    apply_length_scale_fn=lambda x, y: x / y,
)
l2_j = MetricFn(
    differences_metric_fn=_l2_j,
    crosswise_differences_fn=crosswise_tensor_j,
    pairwise_differences_fn=pairwise_tensor_j,
    apply_length_scale_fn=lambda x, y: x / y,
)
F2_n = MetricFn(
    differences_metric_fn=_F2_n,
    crosswise_differences_fn=crosswise_tensor_n,
    pairwise_differences_fn=pairwise_tensor_n,
    apply_length_scale_fn=lambda x, y: x / y**2,
)
F2_j = MetricFn(
    differences_metric_fn=_F2_j,
    crosswise_differences_fn=crosswise_tensor_j,
    pairwise_differences_fn=pairwise_tensor_j,
    apply_length_scale_fn=lambda x, y: x / y**2,
)


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
    def _make_muygps_rbf_n(cls):
        return MuyGPS(
            kernel=RBF(
                deformation=Isotropy(
                    F2_n,
                    length_scale=ScalarParam(cls.length_scale),
                ),
                _backend_fn=rbf_fn_n,
                _backend_ones=np.ones,
                _backend_zeros=np.zeros,
                _backend_squeeze=np.squeeze,
            ),
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_n
            ),
            scale=AnalyticScale(_backend_fn=analytic_scale_optim_n),
            _backend_mean_fn=muygps_posterior_mean_n,
            _backend_var_fn=muygps_diagonal_variance_n,
        )

    @classmethod
    def _make_muygps_n(
        cls, smoothness, noise, deformation, smoothness_bounds="fixed"
    ):
        return MuyGPS(
            kernel=Matern(
                smoothness=ScalarParam(smoothness, smoothness_bounds),
                deformation=deformation,
                _backend_05_fn=matern_05_fn_n,
                _backend_15_fn=matern_15_fn_n,
                _backend_25_fn=matern_25_fn_n,
                _backend_inf_fn=matern_inf_fn_n,
                _backend_gen_fn=matern_gen_fn_n,
                _backend_ones=np.ones,
                _backend_zeros=np.zeros,
                _backend_squeeze=np.squeeze,
            ),
            noise=noise,
            scale=AnalyticScale(_backend_fn=analytic_scale_optim_n),
            _backend_mean_fn=muygps_posterior_mean_n,
            _backend_var_fn=muygps_diagonal_variance_n,
            _backend_fast_mean_fn=muygps_fast_posterior_mean_n,
            _backend_fast_precompute_fn=muygps_fast_posterior_mean_precompute_n,
        )

    @classmethod
    def _make_homoscedastic_muygps_n(cls, smoothness, deformation, **kwargs):
        return cls._make_muygps_n(
            smoothness,
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_n
            ),
            deformation=deformation,
            **kwargs,
        )

    @classmethod
    def _make_isotropic_muygps_n(cls, smoothness, **kwargs):
        return cls._make_homoscedastic_muygps_n(
            smoothness,
            deformation=Isotropy(
                l2_n,
                length_scale=ScalarParam(cls.length_scale),
            ),
            **kwargs,
        )

    @classmethod
    def _make_anisotropic_muygps_n(cls, smoothness, **kwargs):
        return cls._make_homoscedastic_muygps_n(
            smoothness,
            deformation=Anisotropy(
                l2_n,
                length_scale=VectorParam(
                    ScalarParam(cls.length_scale), ScalarParam(cls.length_scale)
                ),
            ),
            **kwargs,
        )

    @classmethod
    def _make_heteroscedastic_muygps_n(cls, smoothness, noise, **kwargs):
        return cls._make_muygps_n(
            smoothness,
            noise=HeteroscedasticNoise(
                noise, _backend_fn=heteroscedastic_perturb_n
            ),
            deformation=Isotropy(
                l2_n,
                length_scale=ScalarParam(cls.length_scale),
            ),
            **kwargs,
        )

    @classmethod
    def _make_muygps_rbf_j(cls):
        return MuyGPS(
            kernel=RBF(
                deformation=Isotropy(
                    F2_j,
                    length_scale=ScalarParam(cls.length_scale),
                ),
                _backend_fn=rbf_fn_j,
                _backend_ones=jnp.ones,
                _backend_zeros=jnp.zeros,
                _backend_squeeze=jnp.squeeze,
            ),
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_j
            ),
            scale=AnalyticScale(_backend_fn=analytic_scale_optim_j),
            _backend_mean_fn=muygps_posterior_mean_j,
            _backend_var_fn=muygps_diagonal_variance_j,
        )

    @classmethod
    def _make_muygps_j(
        cls, smoothness, noise, deformation, smoothness_bounds="fixed"
    ):
        return MuyGPS(
            kernel=Matern(
                smoothness=ScalarParam(smoothness, smoothness_bounds),
                deformation=deformation,
                _backend_05_fn=matern_05_fn_j,
                _backend_15_fn=matern_15_fn_j,
                _backend_25_fn=matern_25_fn_j,
                _backend_inf_fn=matern_inf_fn_j,
                _backend_gen_fn=matern_gen_fn_j,
                _backend_ones=jnp.ones,
                _backend_zeros=jnp.zeros,
                _backend_squeeze=jnp.squeeze,
            ),
            noise=noise,
            scale=AnalyticScale(_backend_fn=analytic_scale_optim_j),
            _backend_mean_fn=muygps_posterior_mean_j,
            _backend_var_fn=muygps_diagonal_variance_j,
            _backend_fast_mean_fn=muygps_fast_posterior_mean_j,
            _backend_fast_precompute_fn=muygps_fast_posterior_mean_precompute_j,
        )

    @classmethod
    def _make_homoscedastic_muygps_j(cls, smoothness, deformation, **kwargs):
        return cls._make_muygps_j(
            smoothness,
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_j
            ),
            deformation=deformation,
            **kwargs,
        )

    @classmethod
    def _make_isotropic_muygps_j(cls, smoothness, **kwargs):
        return cls._make_homoscedastic_muygps_j(
            smoothness,
            Isotropy(
                l2_j,
                length_scale=ScalarParam(cls.length_scale),
            ),
            **kwargs,
        )

    @classmethod
    def _make_anisotropic_muygps_j(cls, smoothness, **kwargs):
        return cls._make_homoscedastic_muygps_j(
            smoothness,
            Anisotropy(
                l2_j,
                length_scale=VectorParam(
                    ScalarParam(cls.length_scale), ScalarParam(cls.length_scale)
                ),
            ),
            **kwargs,
        )

    @classmethod
    def _make_heteroscedastic_muygps_j(cls, smoothness, noise, **kwargs):
        return cls._make_muygps_j(
            smoothness,
            noise=HeteroscedasticNoise(
                noise, _backend_fn=heteroscedastic_perturb_j
            ),
            deformation=Isotropy(
                l2_j,
                length_scale=ScalarParam(cls.length_scale),
            ),
            **kwargs,
        )

    @classmethod
    def setUpClass(cls):
        super(TensorsTestCase, cls).setUpClass()
        cls.train_count = 1000
        cls.test_count = 100
        cls.feature_count = 2
        cls.response_count = 1
        cls.nn_count = 40
        cls.batch_count = 500
        cls.length_scale = 1.0
        cls.smoothness = 0.55
        cls.smoothness_bounds = (1e-1, 1e1)
        cls.noise = 1e-3
        cls.noise_heteroscedastic_n = cls.noise * np.ones(
            (cls.batch_count, cls.nn_count)
        )
        cls.noise_heteroscedastic_train_n = cls.noise * np.ones(
            (cls.train_count, cls.nn_count)
        )

        cls.noise_heteroscedastic_j = jnp.array(cls.noise_heteroscedastic_n)
        cls.noise_heteroscedastic_train_j = jnp.array(
            cls.noise_heteroscedastic_train_n
        )
        cls.train_features_n = _make_gaussian_matrix(
            cls.train_count, cls.feature_count
        )
        cls.train_features_j = jnp.array(cls.train_features_n)
        cls.train_responses_n = np.squeeze(
            _make_gaussian_matrix(cls.train_count, cls.response_count)
        )
        cls.train_responses_j = jnp.array(cls.train_responses_n)
        cls.test_features_n = _make_gaussian_matrix(
            cls.test_count, cls.feature_count
        )
        cls.test_features_j = jnp.array(cls.test_features_n)
        cls.test_responses_n = np.squeeze(
            _make_gaussian_matrix(cls.test_count, cls.response_count)
        )
        cls.test_responses_j = jnp.array(cls.test_responses_n)
        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features_n, cls.nn_count, **_exact_nn_kwarg_options[0]
        )

        cls.muygps_rbf_n = cls._make_muygps_rbf_n()
        cls.muygps_rbf_j = cls._make_muygps_rbf_j()

        cls.muygps_gen_n = cls._make_isotropic_muygps_n(
            cls.smoothness, smoothness_bounds=cls.smoothness_bounds
        )
        cls.muygps_05_n = cls._make_isotropic_muygps_n(0.5)
        cls.muygps_15_n = cls._make_isotropic_muygps_n(1.5)
        cls.muygps_25_n = cls._make_isotropic_muygps_n(2.5)
        cls.muygps_inf_n = cls._make_isotropic_muygps_n(np.inf)
        cls.muygps_gen_j = cls._make_isotropic_muygps_j(
            cls.smoothness, smoothness_bounds=cls.smoothness_bounds
        )
        cls.muygps_05_j = cls._make_isotropic_muygps_j(0.5)
        cls.muygps_15_j = cls._make_isotropic_muygps_j(1.5)
        cls.muygps_25_j = cls._make_isotropic_muygps_j(2.5)
        cls.muygps_inf_j = cls._make_isotropic_muygps_j(jnp.inf)

        cls.muygps_05_anisotropic_n = cls._make_anisotropic_muygps_n(0.5)
        cls.muygps_15_anisotropic_n = cls._make_anisotropic_muygps_n(1.5)
        cls.muygps_25_anisotropic_n = cls._make_anisotropic_muygps_n(2.5)
        cls.muygps_inf_anisotropic_n = cls._make_anisotropic_muygps_n(np.inf)
        cls.muygps_gen_anisotropic_n = cls._make_anisotropic_muygps_n(
            cls.smoothness
        )
        cls.muygps_05_anisotropic_j = cls._make_anisotropic_muygps_j(0.5)
        cls.muygps_15_anisotropic_j = cls._make_anisotropic_muygps_j(1.5)
        cls.muygps_25_anisotropic_j = cls._make_anisotropic_muygps_j(2.5)
        cls.muygps_inf_anisotropic_j = cls._make_anisotropic_muygps_j(jnp.inf)
        cls.muygps_gen_anisotropic_j = cls._make_anisotropic_muygps_j(
            cls.smoothness
        )

        cls.muygps_heteroscedastic_n = cls._make_heteroscedastic_muygps_n(
            cls.smoothness,
            cls.noise_heteroscedastic_n,
            smoothness_bounds=cls.smoothness_bounds,
        )
        cls.muygps_heteroscedastic_train_n = cls._make_heteroscedastic_muygps_n(
            cls.smoothness,
            cls.noise_heteroscedastic_train_n,
            smoothness_bounds=cls.smoothness_bounds,
        )
        cls.muygps_heteroscedastic_j = cls._make_heteroscedastic_muygps_j(
            cls.smoothness,
            cls.noise_heteroscedastic_j,
            smoothness_bounds=cls.smoothness_bounds,
        )
        cls.muygps_heteroscedastic_train_j = cls._make_heteroscedastic_muygps_j(
            cls.smoothness,
            cls.noise_heteroscedastic_train_j,
            smoothness_bounds=cls.smoothness_bounds,
        )

        cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
            cls.nbrs_lookup, cls.batch_count, cls.train_count
        )
        cls.batch_indices_j = jnp.iarray(cls.batch_indices_n)
        cls.batch_nn_indices_j = jnp.iarray(cls.batch_nn_indices_n)

    def _check_ndarray(self, *args, shape=None, **kwargs):
        if shape is not None:
            shape = tuple(x for x in shape if x > 1)
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
                self.muygps_gen_n.kernel.deformation.pairwise_tensor(
                    self.train_features_n, self.batch_nn_indices_n
                ),
                self.muygps_gen_j.kernel.deformation.pairwise_tensor(
                    self.train_features_j, self.batch_nn_indices_j
                ),
            )
        )

    def test_crosswise_tensor(self):
        self.assertTrue(
            allclose_gen(
                self.muygps_gen_n.kernel.deformation.crosswise_tensor(
                    self.train_features_n,
                    self.train_features_n,
                    self.batch_indices_n,
                    self.batch_nn_indices_n,
                ),
                self.muygps_gen_j.kernel.deformation.crosswise_tensor(
                    self.train_features_j,
                    self.train_features_j,
                    self.batch_indices_j,
                    self.batch_nn_indices_j,
                ),
            )
        )

    def test_make_train_tensors(self):
        (
            crosswise_dists_n,
            pairwise_dists_n,
            batch_targets_n,
            batch_nn_targets_n,
        ) = self.muygps_gen_n.make_train_tensors(
            self.batch_indices_n,
            self.batch_nn_indices_n,
            self.train_features_n,
            self.train_responses_n,
        )
        (
            crosswise_dists_j,
            pairwise_dists_j,
            batch_targets_j,
            batch_nn_targets_j,
        ) = self.muygps_gen_j.make_train_tensors(
            self.batch_indices_j,
            self.batch_nn_indices_j,
            self.train_features_j,
            self.train_responses_j,
        )
        self.assertTrue(allclose_gen(crosswise_dists_n, crosswise_dists_j))
        self.assertTrue(allclose_gen(pairwise_dists_n, pairwise_dists_j))
        self.assertTrue(allclose_gen(batch_targets_n, batch_targets_j))
        self.assertTrue(allclose_gen(batch_nn_targets_n, batch_nn_targets_j))
        crosswise_diffs_n, pairwise_diffs_n, _, _ = (
            self.muygps_gen_n.make_train_tensors(
                self.batch_indices_n,
                self.batch_nn_indices_n,
                self.train_features_n,
                self.train_responses_n,
            )
        )
        crosswise_diffs_j, pairwise_diffs_j, _, _ = (
            self.muygps_gen_j.make_train_tensors(
                self.batch_indices_j,
                self.batch_nn_indices_j,
                self.train_features_j,
                self.train_responses_j,
            )
        )
        self.assertTrue(allclose_gen(crosswise_diffs_n, crosswise_diffs_j))
        self.assertTrue(allclose_gen(pairwise_diffs_n, pairwise_diffs_j))


class KernelTestCase(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()
        (
            cls.crosswise_dists_n,
            cls.pairwise_dists_n,
            cls.batch_targets_n,
            cls.batch_nn_targets_n,
        ) = cls.muygps_gen_n.make_train_tensors(
            cls.batch_indices_n,
            cls.batch_nn_indices_n,
            cls.train_features_n,
            cls.train_responses_n,
        )
        (
            cls.crosswise_dists_j,
            cls.pairwise_dists_j,
            cls.batch_targets_j,
            cls.batch_nn_targets_j,
        ) = cls.muygps_gen_j.make_train_tensors(
            cls.batch_indices_j,
            cls.batch_nn_indices_j,
            cls.train_features_j,
            cls.train_responses_j,
        )
        cls.crosswise_diffs_n, cls.pairwise_diffs_n, _, _ = (
            cls.muygps_gen_anisotropic_n.make_train_tensors(
                cls.batch_indices_n,
                cls.batch_nn_indices_n,
                cls.train_features_n,
                cls.train_responses_n,
            )
        )
        cls.crosswise_diffs_j, cls.pairwise_diffs_j, _, _ = (
            cls.muygps_gen_anisotropic_j.make_train_tensors(
                cls.batch_indices_j,
                cls.batch_nn_indices_j,
                cls.train_features_j,
                cls.train_responses_j,
            )
        )

    def _check_ndarray(self, *args, shape=None, **kwargs):
        if shape is not None:
            shape = tuple(x for x in shape if x > 1)
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
                self.muygps_rbf_n.kernel(self.crosswise_dists_n),
                self.muygps_rbf_j.kernel(self.crosswise_dists_j),
            )
        )

    def test_pairwise_rbf(self):
        self.assertTrue(
            allclose_gen(
                self.muygps_rbf_n.kernel(self.pairwise_dists_n),
                self.muygps_rbf_j.kernel(self.pairwise_dists_j),
            )
        )

    def test_crosswise_matern(self):
        self.assertTrue(
            allclose_gen(
                self.muygps_05_n.kernel(self.crosswise_dists_n),
                self.muygps_05_j.kernel(self.crosswise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_05_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_05_anisotropic_j.kernel(self.crosswise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_15_n.kernel(self.crosswise_dists_n),
                self.muygps_15_j.kernel(self.crosswise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_15_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_15_anisotropic_j.kernel(self.crosswise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_25_n.kernel(self.crosswise_dists_n),
                self.muygps_25_j.kernel(self.crosswise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_25_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_25_anisotropic_j.kernel(self.crosswise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_inf_n.kernel(self.crosswise_dists_n),
                self.muygps_inf_j.kernel(self.crosswise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_inf_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_inf_anisotropic_j.kernel(self.crosswise_diffs_j),
            )
        )

        self.assertTrue(
            allclose_gen(
                self.muygps_gen_n.kernel(self.crosswise_dists_n),
                self.muygps_gen_j.kernel(self.crosswise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_gen_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_gen_anisotropic_j.kernel(self.crosswise_diffs_j),
            )
        )

    def test_pairwise_matern(self):
        self.assertTrue(
            allclose_gen(
                self.muygps_05_n.kernel(self.pairwise_dists_n),
                self.muygps_05_j.kernel(self.pairwise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_05_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_05_anisotropic_j.kernel(self.pairwise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_15_n.kernel(self.pairwise_dists_n),
                self.muygps_15_j.kernel(self.pairwise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_15_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_15_anisotropic_j.kernel(self.pairwise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_25_n.kernel(self.pairwise_dists_n),
                self.muygps_25_j.kernel(self.pairwise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_25_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_25_anisotropic_j.kernel(self.pairwise_diffs_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_inf_n.kernel(self.pairwise_dists_n),
                self.muygps_inf_j.kernel(self.pairwise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_inf_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_inf_anisotropic_j.kernel(self.pairwise_diffs_j),
            )
        )

        self.assertTrue(
            allclose_gen(
                self.muygps_gen_n.kernel(self.pairwise_dists_n),
                self.muygps_gen_j.kernel(self.pairwise_dists_j),
            )
        )
        self.assertTrue(
            allclose_gen(
                self.muygps_gen_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_gen_anisotropic_j.kernel(self.pairwise_diffs_j),
            )
        )


class MuyGPSTestCase(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTestCase, cls).setUpClass()
        cls.Kin_n = cls.muygps_gen_n.kernel(cls.pairwise_dists_n)
        cls.Kin_j = cls.muygps_gen_j.kernel(cls.pairwise_dists_j)

        cls.homoscedastic_Kin_n = cls.muygps_gen_n.noise.perturb(cls.Kin_n)
        cls.homoscedastic_Kin_j = cls.muygps_gen_j.noise.perturb(cls.Kin_j)
        cls.heteroscedastic_Kin_n = cls.muygps_heteroscedastic_n.noise.perturb(
            cls.Kin_n,
        )
        cls.heteroscedastic_Kin_j = cls.muygps_heteroscedastic_j.noise.perturb(
            cls.Kin_j
        )

        cls.Kcross_n = cls.muygps_gen_n.kernel(cls.crosswise_dists_n)
        cls.Kcross_j = cls.muygps_gen_j.kernel(cls.crosswise_dists_j)


class MuyGPSTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTest, cls).setUpClass()

    def test_homoscedastic_perturb(self):
        self.assertTrue(
            allclose_gen(self.homoscedastic_Kin_n, self.homoscedastic_Kin_j)
        )

    def test_heteroscedastic_perturb(self):
        self.assertTrue(
            allclose_gen(self.heteroscedastic_Kin_n, self.heteroscedastic_Kin_j)
        )

    def test_posterior_mean(self):
        self.assertTrue(
            allclose_inv(
                self.muygps_gen_n.posterior_mean(
                    self.homoscedastic_Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                self.muygps_gen_j.posterior_mean(
                    self.homoscedastic_Kin_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                ),
            )
        )

    def test_posterior_mean_heteroscedastic(self):
        self.assertTrue(
            allclose_inv(
                self.muygps_heteroscedastic_n.posterior_mean(
                    self.heteroscedastic_Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                self.muygps_heteroscedastic_j.posterior_mean(
                    self.heteroscedastic_Kin_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                ),
            )
        )

    def test_diagonal_variance(self):
        self.assertTrue(
            allclose_var(
                self.muygps_gen_n.posterior_variance(
                    self.homoscedastic_Kin_n, self.Kcross_n
                ),
                self.muygps_gen_j.posterior_variance(
                    self.homoscedastic_Kin_j, self.Kcross_j
                ),
            )
        )

    def test_diagonal_variance_heteroscedastic(self):
        self.assertTrue(
            allclose_var(
                self.muygps_heteroscedastic_n.posterior_variance(
                    self.heteroscedastic_Kin_n, self.Kcross_n
                ),
                self.muygps_heteroscedastic_j.posterior_variance(
                    self.heteroscedastic_Kin_j, self.Kcross_j
                ),
            )
        )

    def test_scale_optim(self):
        self.assertTrue(
            allclose_inv(
                self.muygps_gen_n.scale.get_opt_fn(self.muygps_gen_n)(
                    self.homoscedastic_Kin_n, self.batch_nn_targets_n
                ),
                self.muygps_gen_j.scale.get_opt_fn(self.muygps_gen_j)(
                    self.homoscedastic_Kin_j, self.batch_nn_targets_j
                ),
            )
        )

    def test_scale_optim_heteroscedastic(self):
        self.assertTrue(
            allclose_inv(
                self.muygps_heteroscedastic_n.scale.get_opt_fn(
                    self.muygps_heteroscedastic_n
                )(self.heteroscedastic_Kin_n, self.batch_nn_targets_n),
                self.muygps_heteroscedastic_j.scale.get_opt_fn(
                    self.muygps_heteroscedastic_j
                )(self.heteroscedastic_Kin_j, self.batch_nn_targets_j),
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
            cls.Kin_fast_n,
            cls.train_nn_targets_fast_n,
        ) = make_fast_predict_tensors_n(
            cls.nn_indices_all_n,
            cls.train_features_n,
            cls.train_responses_n,
        )

        cls.homoscedastic_Kin_fast_n = cls.muygps_gen_n.noise.perturb(
            l2_n(cls.Kin_fast_n),
        )

        cls.heteroscedastic_Kin_fast_n = (
            cls.muygps_heteroscedastic_train_n.noise.perturb(
                l2_n(cls.Kin_fast_n)
            )
        )

        cls.fast_regress_coeffs_n = cls.muygps_gen_n.fast_coefficients(
            cls.homoscedastic_Kin_fast_n, cls.train_nn_targets_fast_n
        )

        cls.fast_regress_coeffs_heteroscedastic_n = (
            cls.muygps_heteroscedastic_train_n.fast_coefficients(
                cls.heteroscedastic_Kin_fast_n, cls.train_nn_targets_fast_n
            )
        )

        cls.test_neighbors_n, _ = cls.nbrs_lookup.get_nns(cls.test_features_n)
        cls.closest_neighbor_n = cls.test_neighbors_n[:, 0]
        cls.closest_set_n = cls.nn_indices_all_n[cls.closest_neighbor_n]

        cls.new_nn_indices_n = fast_nn_update_n(cls.nn_indices_all_n)
        cls.closest_set_new_n = cls.new_nn_indices_n[
            cls.closest_neighbor_n
        ].astype(int)
        cls.crosswise_dists_fast_n = (
            cls.muygps_gen_n.kernel.deformation.crosswise_tensor(
                cls.test_features_n,
                cls.train_features_n,
                np.arange(0, cls.test_count),
                cls.closest_set_new_n,
            )
        )
        cls.Kcross_fast_n = cls.muygps_gen_n.kernel(cls.crosswise_dists_fast_n)

        cls.nn_indices_all_j, _ = cls.nbrs_lookup.get_batch_nns(
            np.arange(0, cls.train_count)
        )
        cls.nn_indices_all_j = jnp.iarray(cls.nn_indices_all_j)

        (
            cls.Kin_fast_j,
            cls.train_nn_targets_fast_j,
        ) = make_fast_predict_tensors_j(
            cls.nn_indices_all_j,
            cls.train_features_j,
            cls.train_responses_j,
        )

        cls.homoscedastic_Kin_fast_j = cls.muygps_gen_j.noise.perturb(
            l2_j(cls.Kin_fast_j),
        )

        cls.heteroscedastic_Kin_fast_j = (
            cls.muygps_heteroscedastic_train_j.noise.perturb(
                l2_j(cls.Kin_fast_j),
            )
        )

        cls.fast_regress_coeffs_j = cls.muygps_gen_j.fast_coefficients(
            cls.homoscedastic_Kin_fast_j, cls.train_nn_targets_fast_j
        )

        cls.fast_regress_coeffs_heteroscedastic_j = (
            cls.muygps_heteroscedastic_train_j.fast_coefficients(
                cls.heteroscedastic_Kin_fast_j, cls.train_nn_targets_fast_j
            )
        )

        cls.test_neighbors_j, _ = cls.nbrs_lookup.get_nns(cls.test_features_j)
        cls.closest_neighbor_j = cls.test_neighbors_j[:, 0]
        cls.closest_set_j = cls.nn_indices_all_j[cls.closest_neighbor_j].astype(
            int
        )

        cls.new_nn_indices_j = fast_nn_update_j(cls.nn_indices_all_j)
        cls.closest_set_new_j = cls.new_nn_indices_j[cls.closest_neighbor_j]
        cls.crosswise_dists_fast_j = (
            cls.muygps_gen_j.kernel.deformation.crosswise_tensor(
                cls.test_features_j,
                cls.train_features_j,
                np.arange(0, cls.test_count),
                cls.closest_set_new_j,
            )
        )
        cls.Kcross_fast_j = cls.muygps_gen_j.kernel(cls.crosswise_dists_fast_j)

    def test_fast_nn_update(self):
        self.assertTrue(
            allclose_gen(
                fast_nn_update_j(self.nn_indices_all_j),
                fast_nn_update_n(self.nn_indices_all_n),
            )
        )

    def test_make_fast_predict_tensors(self):
        self.assertTrue(allclose_gen(self.Kin_fast_n, self.Kin_fast_j))
        self.assertTrue(
            allclose_gen(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_j
            )
        )

    def test_homoscedastic_kernel_tensors(self):
        self.assertTrue(
            allclose_inv(
                self.homoscedastic_Kin_fast_n, self.homoscedastic_Kin_fast_j
            )
        )

    def test_heteroscedastic_kernel_tensors(self):
        self.assertTrue(
            allclose_inv(
                self.heteroscedastic_Kin_fast_n,
                self.heteroscedastic_Kin_fast_j,
            )
        )

    def test_fast_predict(self):
        self.assertTrue(
            allclose_inv(
                self.muygps_gen_n.fast_posterior_mean(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n],
                ),
                self.muygps_gen_j.fast_posterior_mean(
                    self.Kcross_fast_j,
                    self.fast_regress_coeffs_j[self.closest_neighbor_j],
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


class OptimTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTestCase, cls).setUpClass()
        cls.predictions_n = cls.muygps_05_n.posterior_mean(
            cls.homoscedastic_Kin_n, cls.Kcross_n, cls.batch_nn_targets_n
        )
        cls.variances_n = cls.muygps_05_n.posterior_variance(
            cls.homoscedastic_Kin_n, cls.Kcross_n
        )
        cls.predictions_heteroscedastic_n = (
            cls.muygps_heteroscedastic_n.posterior_mean(
                cls.heteroscedastic_Kin_n, cls.Kcross_n, cls.batch_nn_targets_n
            )
        )
        cls.variances_heteroscedastic_n = (
            cls.muygps_heteroscedastic_n.posterior_variance(
                cls.heteroscedastic_Kin_n, cls.Kcross_n
            )
        )
        cls.predictions_j = jnp.array(cls.predictions_n)
        cls.variances_j = jnp.array(cls.variances_n)
        cls.predictions_heteroscedastic_j = jnp.array(
            cls.predictions_heteroscedastic_n
        )
        cls.variances_heteroscedastic_j = jnp.array(
            cls.variances_heteroscedastic_n
        )
        cls.x0_names, cls.x0_n, cls.bounds = cls.muygps_gen_n.get_opt_params()
        cls.x0_j = jnp.array(cls.x0_n)
        cls.x0_map_n = {n: cls.x0_n[i] for i, n in enumerate(cls.x0_names)}
        cls.x0_map_j = {n: cls.x0_j[i] for i, n in enumerate(cls.x0_names)}

    def _get_scale_fn_n(self):
        return self.muygps_gen_n.scale.get_opt_fn(self.muygps_gen_n)

    def _get_scale_fn_heteroscedastic_n(self):
        return self.muygps_heteroscedastic_n.scale.get_opt_fn(
            self.muygps_heteroscedastic_n
        )

    def _get_scale_fn_j(self):
        return self.muygps_gen_j.scale.get_opt_fn(self.muygps_gen_j)

    def _get_scale_fn_heteroscedastic_j(self):
        return self.muygps_heteroscedastic_j.scale.get_opt_fn(
            self.muygps_heteroscedastic_j
        )

    def _get_obj_fn_n(self):
        return make_loo_crossval_fn(
            mse_fn_n,
            self.muygps_gen_n.kernel.get_opt_fn(),
            self.muygps_gen_n.get_opt_mean_fn(),
            self.muygps_gen_n.get_opt_var_fn(),
            self._get_scale_fn_n(),
            self.pairwise_dists_n,
            self.crosswise_dists_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )

    def _get_obj_fn_j(self):
        return make_loo_crossval_fn(
            mse_fn_j,
            self.muygps_gen_j.kernel.get_opt_fn(),
            self.muygps_gen_j.get_opt_mean_fn(),
            self.muygps_gen_j.get_opt_var_fn(),
            self._get_scale_fn_j(),
            self.pairwise_dists_j,
            self.crosswise_dists_j,
            self.batch_nn_targets_j,
            self.batch_targets_j,
        )

    def _get_obj_fn_heteroscedastic_j(self):
        return make_loo_crossval_fn(
            mse_fn_j,
            self.muygps_heteroscedastic_j.kernel.get_opt_fn(),
            self.muygps_heteroscedastic_j.get_opt_mean_fn(),
            self.muygps_heteroscedastic_j.get_opt_var_fn(),
            self._get_scale_fn_heteroscedastic_j(),
            self.pairwise_dists_j,
            self.crosswise_dists_j,
            self.batch_nn_targets_j,
            self.batch_targets_j,
        )

    def _get_obj_fn_heteroscedastic_n(self):
        return make_loo_crossval_fn(
            mse_fn_n,
            self.muygps_heteroscedastic_n.kernel.get_opt_fn(),
            self.muygps_heteroscedastic_n.get_opt_mean_fn(),
            self.muygps_heteroscedastic_n.get_opt_var_fn(),
            self._get_scale_fn_heteroscedastic_n(),
            self.pairwise_dists_n,
            self.crosswise_dists_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
        )


class LossTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(LossTest, cls).setUpClass()

        cls.scale_n = cls.muygps_gen_n.scale()
        cls.scale_j = cls.muygps_gen_j.scale()

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
                    self.scale_n,
                ),
                lool_fn_j(
                    self.predictions_j,
                    self.batch_targets_j,
                    self.variances_j,
                    self.scale_j,
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
                    self.scale_n,
                    boundary_scale=boundary_scale,
                ),
                looph_fn_j(
                    self.predictions_j,
                    self.batch_targets_j,
                    self.variances_j,
                    self.scale_j,
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
                cross_entropy_fn_n(cat_predictions_n, cat_batch_targets_n),
                cross_entropy_fn_j(cat_predictions_j, cat_batch_targets_j),
            )
        )


class ObjectivePartsTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectivePartsTest, cls).setUpClass()

        cls.scale_n = cls.muygps_gen_n.scale()
        cls.scale_j = cls.muygps_gen_j.scale()

    def test_kernel_fn(self):
        self.assertTrue(
            allclose_gen(
                self.muygps_gen_n.kernel.get_opt_fn()(
                    self.pairwise_dists_n, **self.x0_map_n
                ),
                self.muygps_gen_j.kernel.get_opt_fn()(
                    self.pairwise_dists_j, **self.x0_map_j
                ),
            )
        )

    def test_mean_fn(self):
        mean_fn_n = self.muygps_gen_n.get_opt_mean_fn()
        mean_fn_j = self.muygps_gen_j.get_opt_mean_fn()
        self.assertTrue(
            allclose_inv(
                mean_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_j(
                    self.Kin_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_mean_fn_heteroscedastic(self):
        mean_fn_n = self.muygps_heteroscedastic_n.get_opt_mean_fn()
        mean_fn_j = self.muygps_heteroscedastic_j.get_opt_mean_fn()
        self.assertTrue(
            allclose_inv(
                mean_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_j(
                    self.Kin_j,
                    self.Kcross_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_var_fn(self):
        var_fn_n = self.muygps_gen_n.get_opt_var_fn()
        var_fn_j = self.muygps_gen_j.get_opt_var_fn()
        self.assertTrue(
            allclose_inv(
                var_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_j(
                    self.Kin_j,
                    self.Kcross_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_var_fn_heteroscedastic(self):
        var_fn_n = self.muygps_heteroscedastic_n.get_opt_var_fn()
        var_fn_j = self.muygps_heteroscedastic_j.get_opt_var_fn()
        self.assertTrue(
            allclose_inv(
                var_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_j(
                    self.Kin_j,
                    self.Kcross_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_scale_fn(self):
        ss_fn_n = self._get_scale_fn_n()
        ss_fn_j = self._get_scale_fn_j()
        self.assertTrue(
            allclose_inv(
                ss_fn_n(
                    self.Kin_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_j(
                    self.Kin_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )

    def test_scale_heteroscedastic_fn(self):
        ss_fn_heteroscedastic_n = self._get_scale_fn_heteroscedastic_n()
        ss_fn_heteroscedastic_j = self._get_scale_fn_heteroscedastic_j()
        self.assertTrue(
            allclose_inv(
                ss_fn_heteroscedastic_n(
                    self.Kin_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_heteroscedastic_j(
                    self.Kin_j,
                    self.batch_nn_targets_j,
                    **self.x0_map_j,
                ),
            )
        )


class ObjectiveTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectiveTest, cls).setUpClass()

        cls.scale_n = cls.muygps_gen_n.scale()
        cls.scale_j = cls.muygps_gen_j.scale()

    def test_loo_crossval(self):
        obj_fn_n = self._get_obj_fn_n()
        obj_fn_j = self._get_obj_fn_j()
        obj_fn_heteroscedastic_n = self._get_obj_fn_heteroscedastic_n()
        obj_fn_heteroscedastic_j = self._get_obj_fn_heteroscedastic_j()
        self.assertTrue(
            allclose_inv(obj_fn_n(**self.x0_map_n), obj_fn_j(**self.x0_map_j))
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
        obj_fn_het_j = self._get_obj_fn_heteroscedastic_j()
        obj_fn_het_n = self._get_obj_fn_heteroscedastic_n()

        mopt_n = scipy_optimize_n(
            self.muygps_gen_n, obj_fn_n, **self.sopt_kwargs
        )
        mopt_j = scipy_optimize_j(
            self.muygps_gen_j, obj_fn_j, **self.sopt_kwargs
        )
        mopt_het_j = scipy_optimize_j(
            self.muygps_heteroscedastic_j, obj_fn_het_j, **self.sopt_kwargs
        )
        mopt_het_n = scipy_optimize_n(
            self.muygps_heteroscedastic_n, obj_fn_het_n, **self.sopt_kwargs
        )
        self.assertTrue(
            allclose_gen(mopt_n.kernel.smoothness(), mopt_j.kernel.smoothness())
        )
        self.assertTrue(
            allclose_inv(
                mopt_het_n.kernel.smoothness(), mopt_het_j.kernel.smoothness()
            )
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
        obj_fn_het_j = self._get_obj_fn_heteroscedastic_j()
        obj_fn_het_n = self._get_obj_fn_heteroscedastic_n()

        mopt_n = bayes_optimize_n(
            self.muygps_gen_n, obj_fn_n, **self.bopt_kwargs
        )
        mopt_j = bayes_optimize_j(
            self.muygps_gen_j, obj_fn_j, **self.bopt_kwargs
        )
        mopt_het_j = bayes_optimize_j(
            self.muygps_heteroscedastic_j, obj_fn_het_j, **self.bopt_kwargs
        )
        mopt_het_n = bayes_optimize_n(
            self.muygps_heteroscedastic_n, obj_fn_het_n, **self.bopt_kwargs
        )
        self.assertTrue(
            allclose_inv(mopt_n.kernel.smoothness(), mopt_j.kernel.smoothness())
        )
        self.assertTrue(
            allclose_inv(
                mopt_het_n.kernel.smoothness(), mopt_het_j.kernel.smoothness()
            )
        )


if __name__ == "__main__":
    absltest.main()
