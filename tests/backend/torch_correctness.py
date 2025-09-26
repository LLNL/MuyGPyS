# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
import MuyGPyS._src.math.torch as torch
from MuyGPyS import config
from MuyGPyS._src.gp.tensors.numpy import (
    _pairwise_tensor as pairwise_tensor_n,
    _crosswise_tensor as crosswise_tensor_n,
    _make_fast_predict_tensors as make_fast_predict_tensors_n,
    _fast_nn_update as fast_nn_update_n,
    _F2 as _F2_n,
    _l2 as _l2_n,
)
from MuyGPyS._src.gp.tensors.torch import (
    _pairwise_tensor as pairwise_tensor_t,
    _crosswise_tensor as crosswise_tensor_t,
    _make_fast_predict_tensors as make_fast_predict_tensors_t,
    _fast_nn_update as fast_nn_update_t,
    _F2 as _F2_t,
    _l2 as _l2_t,
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
    _muygps_fast_posterior_mean_precompute as muygps_fast_posterior_mean_precompute_n,
)
from MuyGPyS._src.gp.muygps.torch import (
    _muygps_posterior_mean as muygps_posterior_mean_t,
    _muygps_diagonal_variance as muygps_diagonal_variance_t,
    _muygps_fast_posterior_mean as muygps_fast_posterior_mean_t,
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
from MuyGPyS._src.optimize.loss.torch import (
    _mse_fn as mse_fn_t,
    _cross_entropy_fn as cross_entropy_fn_t,
    _lool_fn as lool_fn_t,
    _pseudo_huber_fn as pseudo_huber_fn_t,
    _looph_fn as looph_fn_t,
)
from MuyGPyS._src.optimize.scale.numpy import (
    _analytic_scale_optim as analytic_scale_optim_n,
)
from MuyGPyS._src.optimize.scale.torch import (
    _analytic_scale_optim as analytic_scale_optim_t,
)
from MuyGPyS._test.utils import (
    _exact_nn_kwarg_options,
    _make_gaussian_matrix,
    _make_gaussian_data,
    _make_heteroscedastic_test_nugget,
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

if config.state.torch_enabled is False:
    raise ValueError("Bad attempt to run torch-only code with torch diabled.")
if config.state.backend == "mpi":
    raise ValueError("Bad attempt to run non-MPI code in MPI mode.")
if config.state.backend != "numpy":
    raise ValueError(
        f"torch_correctness.py must be run in numpy mode, not "
        f"{config.state.backend} mode."
    )

# make torch loss functor aliases
mse_fn_t = LossFn(mse_fn_t, make_raw_predict_and_loss_fn)
cross_entropy_fn_t = LossFn(cross_entropy_fn_t, make_raw_predict_and_loss_fn)
lool_fn_t = LossFn(lool_fn_t, make_var_predict_and_loss_fn)
pseudo_huber_fn_t = LossFn(pseudo_huber_fn_t, make_raw_predict_and_loss_fn)
looph_fn_t = LossFn(looph_fn_t, make_var_predict_and_loss_fn)

l2_n = MetricFn(
    differences_metric_fn=_l2_n,
    crosswise_differences_fn=crosswise_tensor_n,
    pairwise_differences_fn=pairwise_tensor_n,
    apply_length_scale_fn=lambda x, y: x / y,
)
l2_t = MetricFn(
    differences_metric_fn=_l2_t,
    crosswise_differences_fn=crosswise_tensor_t,
    pairwise_differences_fn=pairwise_tensor_t,
    apply_length_scale_fn=lambda x, y: x / y,
)
F2_n = MetricFn(
    differences_metric_fn=_F2_n,
    crosswise_differences_fn=crosswise_tensor_n,
    pairwise_differences_fn=pairwise_tensor_n,
    apply_length_scale_fn=lambda x, y: x / y**2,
)
F2_t = MetricFn(
    differences_metric_fn=_F2_t,
    crosswise_differences_fn=crosswise_tensor_t,
    pairwise_differences_fn=pairwise_tensor_t,
    apply_length_scale_fn=lambda x, y: x / y**2,
)


def _allclose(x, y) -> bool:
    return np.allclose(
        x, y, atol=1e-5 if config.state.low_precision() else 1e-8
    )


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
    def _make_muygps_n(cls, smoothness, noise, deformation):
        return MuyGPS(
            kernel=Matern(
                smoothness=ScalarParam(smoothness),
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
    def _make_homoscedastic_muygps_n(cls, smoothness, deformation):
        return cls._make_muygps_n(
            smoothness,
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_n
            ),
            deformation=deformation,
        )

    @classmethod
    def _make_isotropic_muygps_n(cls, smoothness):
        return cls._make_homoscedastic_muygps_n(
            smoothness,
            deformation=Isotropy(
                l2_n,
                length_scale=ScalarParam(cls.length_scale),
            ),
        )

    @classmethod
    def _make_anisotropic_muygps_n(cls, smoothness):
        return cls._make_homoscedastic_muygps_n(
            smoothness,
            Anisotropy(
                l2_n,
                length_scale=VectorParam(
                    ScalarParam(cls.length_scale), ScalarParam(cls.length_scale)
                ),
            ),
        )

    @classmethod
    def _make_heteroscedastic_muygps_n(cls, smoothness, noise):
        return cls._make_muygps_n(
            smoothness,
            noise=HeteroscedasticNoise(
                noise, _backend_fn=heteroscedastic_perturb_n
            ),
            deformation=Isotropy(
                l2_n,
                length_scale=ScalarParam(cls.length_scale),
            ),
        )

    @classmethod
    def _make_muygps_rbf_t(cls):
        return MuyGPS(
            kernel=RBF(
                deformation=Isotropy(
                    F2_t,
                    length_scale=ScalarParam(cls.length_scale),
                ),
                _backend_fn=rbf_fn_t,
                _backend_ones=torch.ones,
                _backend_zeros=torch.zeros,
                _backend_squeeze=torch.squeeze,
            ),
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_t
            ),
            scale=AnalyticScale(_backend_fn=analytic_scale_optim_t),
            _backend_mean_fn=muygps_posterior_mean_t,
            _backend_var_fn=muygps_diagonal_variance_t,
        )

    @classmethod
    def _make_muygps_t(cls, smoothness, noise, deformation):
        return MuyGPS(
            kernel=Matern(
                smoothness=ScalarParam(smoothness),
                deformation=deformation,
                _backend_05_fn=matern_05_fn_t,
                _backend_15_fn=matern_15_fn_t,
                _backend_25_fn=matern_25_fn_t,
                _backend_inf_fn=matern_inf_fn_t,
                _backend_gen_fn=matern_gen_fn_t,
                _backend_ones=torch.ones,
                _backend_zeros=torch.zeros,
                _backend_squeeze=torch.squeeze,
            ),
            noise=noise,
            scale=AnalyticScale(_backend_fn=analytic_scale_optim_t),
            _backend_mean_fn=muygps_posterior_mean_t,
            _backend_var_fn=muygps_diagonal_variance_t,
            _backend_fast_mean_fn=muygps_fast_posterior_mean_t,
            _backend_fast_precompute_fn=muygps_fast_posterior_mean_precompute_t,
        )

    @classmethod
    def _make_homoscedastic_muygps_t(cls, smoothness, deformation):
        return cls._make_muygps_t(
            smoothness,
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_t
            ),
            deformation=deformation,
        )

    @classmethod
    def _make_isotropic_muygps_t(cls, smoothness):
        return cls._make_homoscedastic_muygps_t(
            smoothness,
            Isotropy(
                l2_t,
                length_scale=ScalarParam(cls.length_scale),
            ),
        )

    @classmethod
    def _make_anisotropic_muygps_t(cls, smoothness):
        return cls._make_homoscedastic_muygps_t(
            smoothness,
            Anisotropy(
                l2_t,
                length_scale=VectorParam(
                    ScalarParam(cls.length_scale), ScalarParam(cls.length_scale)
                ),
            ),
        )

    @classmethod
    def _make_heteroscedastic_muygps_t(cls, smoothness, noise):
        return cls._make_muygps_t(
            smoothness,
            noise=HeteroscedasticNoise(
                noise, _backend_fn=heteroscedastic_perturb_t
            ),
            deformation=Isotropy(
                l2_t,
                length_scale=ScalarParam(cls.length_scale),
            ),
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
        cls.smoothness = 0.5
        cls.smoothness_bounds = (1e-1, 1e1)
        cls.noise = 1e-3
        cls.noise_heteroscedastic_n = _make_heteroscedastic_test_nugget(
            cls.batch_count, cls.nn_count, cls.noise
        )
        cls.noise_heteroscedastic_train_n = _make_heteroscedastic_test_nugget(
            cls.train_count, cls.nn_count, cls.noise
        )
        cls.noise_heteroscedastic_t = torch.ndarray(cls.noise_heteroscedastic_n)
        cls.noise_heteroscedastic_train_t = torch.ndarray(
            cls.noise_heteroscedastic_train_n
        )
        cls.train_features_n = _make_gaussian_matrix(
            cls.train_count, cls.feature_count
        )
        cls.train_features_t = torch.from_numpy(cls.train_features_n)
        cls.train_responses_n = np.squeeze(
            _make_gaussian_matrix(cls.train_count, cls.response_count)
        )
        cls.train_responses_t = torch.from_numpy(cls.train_responses_n)
        cls.test_features_n = _make_gaussian_matrix(
            cls.test_count, cls.feature_count
        )
        cls.test_features_t = torch.from_numpy(cls.test_features_n)
        cls.test_responses_n = np.squeeze(
            _make_gaussian_matrix(cls.test_count, cls.response_count)
        )
        cls.test_responses_t = torch.from_numpy(cls.test_responses_n)
        cls.nbrs_lookup = NN_Wrapper(
            cls.train_features_n, cls.nn_count, **_exact_nn_kwarg_options[0]
        )

        cls.muygps_rbf_n = cls._make_muygps_rbf_n()
        cls.muygps_rbf_t = cls._make_muygps_rbf_t()

        cls.muygps_05_n = cls._make_isotropic_muygps_n(cls.smoothness)
        cls.muygps_15_n = cls._make_isotropic_muygps_n(1.5)
        cls.muygps_25_n = cls._make_isotropic_muygps_n(2.5)
        cls.muygps_inf_n = cls._make_isotropic_muygps_n(np.inf)
        cls.muygps_05_t = cls._make_isotropic_muygps_t(cls.smoothness)
        cls.muygps_15_t = cls._make_isotropic_muygps_t(1.5)
        cls.muygps_25_t = cls._make_isotropic_muygps_t(2.5)
        cls.muygps_inf_t = cls._make_isotropic_muygps_t(torch.inf)

        cls.muygps_05_anisotropic_n = cls._make_anisotropic_muygps_n(
            cls.smoothness
        )
        cls.muygps_15_anisotropic_n = cls._make_anisotropic_muygps_n(1.5)
        cls.muygps_25_anisotropic_n = cls._make_anisotropic_muygps_n(2.5)
        cls.muygps_inf_anisotropic_n = cls._make_anisotropic_muygps_n(np.inf)
        cls.muygps_05_anisotropic_t = cls._make_anisotropic_muygps_t(
            cls.smoothness
        )
        cls.muygps_15_anisotropic_t = cls._make_anisotropic_muygps_t(1.5)
        cls.muygps_25_anisotropic_t = cls._make_anisotropic_muygps_t(2.5)
        cls.muygps_inf_anisotropic_t = cls._make_anisotropic_muygps_t(torch.inf)

        cls.muygps_heteroscedastic_n = cls._make_heteroscedastic_muygps_n(
            cls.smoothness, cls.noise_heteroscedastic_n
        )
        cls.muygps_heteroscedastic_t = cls._make_heteroscedastic_muygps_t(
            cls.smoothness, cls.noise_heteroscedastic_t
        )
        cls.muygps_heteroscedastic_train_n = cls._make_heteroscedastic_muygps_n(
            cls.smoothness, cls.noise_heteroscedastic_train_n
        )
        cls.muygps_heteroscedastic_train_t = cls._make_heteroscedastic_muygps_t(
            cls.smoothness, cls.noise_heteroscedastic_train_t
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
                self.muygps_05_n.kernel.deformation.pairwise_tensor(
                    self.train_features_n, self.batch_nn_indices_n
                ),
                self.muygps_05_t.kernel.deformation.pairwise_tensor(
                    self.train_features_t, self.batch_nn_indices_t
                ),
            )
        )

    def test_crosswise_tensor(self):
        self.assertTrue(
            np.allclose(
                self.muygps_05_n.kernel.deformation.crosswise_tensor(
                    self.train_features_n,
                    self.train_features_n,
                    self.batch_indices_n,
                    self.batch_nn_indices_n,
                ),
                self.muygps_05_t.kernel.deformation.crosswise_tensor(
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
        ) = self.muygps_05_n.make_train_tensors(
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
        ) = self.muygps_05_t.make_train_tensors(
            self.batch_indices_t,
            self.batch_nn_indices_t,
            self.train_features_t,
            self.train_responses_t,
        )
        self.assertTrue(np.allclose(crosswise_dists_n, crosswise_dists_t))
        self.assertTrue(np.allclose(pairwise_dists_n, pairwise_dists_t))
        self.assertTrue(np.allclose(batch_targets_n, batch_targets_t))
        self.assertTrue(np.allclose(batch_nn_targets_n, batch_nn_targets_t))

        crosswise_diffs_n, pairwise_diffs_n, _, _ = (
            self.muygps_05_anisotropic_n.make_train_tensors(
                self.batch_indices_n,
                self.batch_nn_indices_n,
                self.train_features_n,
                self.train_responses_n,
            )
        )
        crosswise_diffs_t, pairwise_diffs_t, _, _ = (
            self.muygps_05_anisotropic_t.make_train_tensors(
                self.batch_indices_t,
                self.batch_nn_indices_t,
                self.train_features_t,
                self.train_responses_t,
            )
        )
        self.assertTrue(np.allclose(crosswise_diffs_n, crosswise_diffs_t))
        self.assertTrue(np.allclose(pairwise_diffs_n, pairwise_diffs_t))


class KernelTestCase(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()
        (
            cls.crosswise_dists_n,
            cls.pairwise_dists_n,
            cls.batch_targets_n,
            cls.batch_nn_targets_n,
        ) = cls.muygps_05_n.make_train_tensors(
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
        ) = cls.muygps_05_t.make_train_tensors(
            cls.batch_indices_t,
            cls.batch_nn_indices_t,
            cls.train_features_t,
            cls.train_responses_t,
        )
        cls.crosswise_diffs_n, cls.pairwise_diffs_n, _, _ = (
            cls.muygps_05_anisotropic_n.make_train_tensors(
                cls.batch_indices_n,
                cls.batch_nn_indices_n,
                cls.train_features_n,
                cls.train_responses_n,
            )
        )
        cls.crosswise_diffs_t, cls.pairwise_diffs_t, _, _ = (
            cls.muygps_05_anisotropic_t.make_train_tensors(
                cls.batch_indices_t,
                cls.batch_nn_indices_t,
                cls.train_features_t,
                cls.train_responses_t,
            )
        )


class KernelTest(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTest, cls).setUpClass()

    def test_crosswise_rbf(self):
        self.assertTrue(
            np.allclose(
                self.muygps_rbf_n.kernel(self.crosswise_dists_n),
                self.muygps_rbf_t.kernel(self.crosswise_dists_t),
            )
        )

    def test_pairwise_rbf(self):
        self.assertTrue(
            np.allclose(
                self.muygps_rbf_n.kernel(self.pairwise_dists_n),
                self.muygps_rbf_t.kernel(self.pairwise_dists_t),
            )
        )

    def test_crosswise_matern(self):
        self.assertTrue(
            np.allclose(
                self.muygps_05_n.kernel(self.crosswise_dists_n),
                self.muygps_05_t.kernel(self.crosswise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_05_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_05_anisotropic_t.kernel(self.crosswise_diffs_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_15_n.kernel(self.crosswise_dists_n),
                self.muygps_15_t.kernel(self.crosswise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_15_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_15_anisotropic_t.kernel(self.crosswise_diffs_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_25_n.kernel(self.crosswise_dists_n),
                self.muygps_25_t.kernel(self.crosswise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_25_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_25_anisotropic_t.kernel(self.crosswise_diffs_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_inf_n.kernel(self.crosswise_dists_n),
                self.muygps_inf_t.kernel(self.crosswise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_inf_anisotropic_n.kernel(self.crosswise_diffs_n),
                self.muygps_inf_anisotropic_t.kernel(self.crosswise_diffs_t),
            )
        )

    def test_pairwise_matern(self):
        self.assertTrue(
            np.allclose(
                self.muygps_05_n.kernel(self.pairwise_dists_n),
                self.muygps_05_t.kernel(self.pairwise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_05_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_05_anisotropic_t.kernel(self.pairwise_diffs_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_15_n.kernel(self.pairwise_dists_n),
                self.muygps_15_t.kernel(self.pairwise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_15_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_15_anisotropic_t.kernel(self.pairwise_diffs_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_25_n.kernel(self.pairwise_dists_n),
                self.muygps_25_t.kernel(self.pairwise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_25_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_25_anisotropic_t.kernel(self.pairwise_diffs_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_inf_n.kernel(self.pairwise_dists_n),
                self.muygps_inf_t.kernel(self.pairwise_dists_t),
            )
        )
        self.assertTrue(
            np.allclose(
                self.muygps_inf_anisotropic_n.kernel(self.pairwise_diffs_n),
                self.muygps_inf_anisotropic_t.kernel(self.pairwise_diffs_t),
            )
        )


class MuyGPSTestCase(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTestCase, cls).setUpClass()
        cls.Kin_n = cls.muygps_05_n.kernel(cls.pairwise_dists_n)
        cls.Kin_t = cls.muygps_05_t.kernel(cls.pairwise_dists_t)
        cls.homoscedastic_Kin_n = cls.muygps_05_n.noise.perturb(cls.Kin_n)

        cls.heteroscedastic_Kin_n = cls.muygps_heteroscedastic_n.noise.perturb(
            cls.Kin_n,
        )
        cls.homoscedastic_Kin_t = cls.muygps_05_t.noise.perturb(cls.Kin_t)
        cls.heteroscedastic_Kin_t = cls.muygps_heteroscedastic_t.noise.perturb(
            cls.Kin_t
        )

        cls.Kcross_n = cls.muygps_05_n.kernel(cls.crosswise_dists_n)
        cls.Kcross_t = cls.muygps_05_t.kernel(cls.crosswise_dists_t)


class MuyGPSTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTest, cls).setUpClass()

    def test_homoscedastic_noise(self):
        self.assertTrue(
            np.allclose(self.homoscedastic_Kin_n, self.homoscedastic_Kin_t)
        )

    def test_heteroscedastic_noise(self):
        self.assertTrue(
            np.allclose(self.heteroscedastic_Kin_n, self.heteroscedastic_Kin_t)
        )

    def test_posterior_mean(self):
        self.assertTrue(
            _allclose(
                self.muygps_05_n.posterior_mean(
                    self.homoscedastic_Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                self.muygps_05_t.posterior_mean(
                    self.homoscedastic_Kin_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                ),
            )
        )

    def test_posterior_mean_heteroscedastic(self):
        self.assertTrue(
            _allclose(
                self.muygps_heteroscedastic_n.posterior_mean(
                    self.heteroscedastic_Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                ),
                self.muygps_heteroscedastic_t.posterior_mean(
                    self.heteroscedastic_Kin_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                ),
            )
        )

    def test_diagonal_variance(self):
        self.assertTrue(
            np.allclose(
                self.muygps_05_n.posterior_variance(
                    self.homoscedastic_Kin_n, self.Kcross_n
                ),
                self.muygps_05_t.posterior_variance(
                    self.homoscedastic_Kin_t, self.Kcross_t
                ),
            )
        )

    def test_diagonal_variance_heteroscedastic(self):
        self.assertTrue(
            np.allclose(
                self.muygps_heteroscedastic_n.posterior_variance(
                    self.heteroscedastic_Kin_n, self.Kcross_n
                ),
                self.muygps_heteroscedastic_t.posterior_variance(
                    self.heteroscedastic_Kin_t, self.Kcross_t
                ),
            )
        )

    def test_scale_optim(self):
        self.assertTrue(
            np.allclose(
                self.muygps_rbf_n.scale.get_opt_fn(self.muygps_rbf_n)(
                    self.homoscedastic_Kin_n, self.batch_nn_targets_n
                ),
                self.muygps_rbf_t.scale.get_opt_fn(self.muygps_rbf_t)(
                    self.homoscedastic_Kin_t, self.batch_nn_targets_t
                ),
            )
        )

    def test_scale_optim_heteroscedastic(self):
        self.assertTrue(
            np.allclose(
                self.muygps_heteroscedastic_n.scale.get_opt_fn(
                    self.muygps_heteroscedastic_n
                )(self.heteroscedastic_Kin_n, self.batch_nn_targets_n),
                self.muygps_heteroscedastic_t.scale.get_opt_fn(
                    self.muygps_heteroscedastic_t
                )(self.heteroscedastic_Kin_t, self.batch_nn_targets_t),
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

        cls.homoscedastic_Kin_fast_n = cls.muygps_05_n.noise.perturb(
            l2_n(cls.Kin_fast_n)
        )
        cls.heteroscedastic_Kin_fast_n = (
            cls.muygps_heteroscedastic_train_n.noise.perturb(
                l2_n(cls.Kin_fast_n)
            )
        )
        cls.fast_regress_coeffs_n = cls.muygps_05_n.fast_coefficients(
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
            cls.muygps_05_n.kernel.deformation.crosswise_tensor(
                cls.test_features_n,
                cls.train_features_n,
                np.arange(0, cls.test_count),
                cls.closest_set_new_n,
            )
        )

        cls.Kcross_fast_n = cls.muygps_05_n.kernel(cls.crosswise_dists_fast_n)

        cls.nn_indices_all_t, _ = cls.nbrs_lookup.get_batch_nns(
            torch.arange(0, cls.train_count)
        )
        cls.nn_indices_all_t = torch.from_numpy(cls.nn_indices_all_t)
        (
            cls.Kin_fast_t,
            cls.train_nn_targets_fast_t,
        ) = make_fast_predict_tensors_t(
            cls.nn_indices_all_t,
            cls.train_features_t,
            cls.train_responses_t,
        )

        cls.homoscedastic_Kin_fast_t = cls.muygps_05_t.noise.perturb(
            l2_t(cls.Kin_fast_t)
        )
        cls.heteroscedastic_Kin_fast_t = (
            cls.muygps_heteroscedastic_train_t.noise.perturb(
                l2_t(cls.Kin_fast_t)
            )
        )
        cls.fast_regress_coeffs_t = cls.muygps_05_t.fast_coefficients(
            cls.homoscedastic_Kin_fast_t, cls.train_nn_targets_fast_t
        )
        cls.fast_regress_coeffs_heteroscedastic_t = (
            cls.muygps_heteroscedastic_train_t.fast_coefficients(
                cls.heteroscedastic_Kin_fast_t, cls.train_nn_targets_fast_t
            )
        )

        cls.test_neighbors_t, _ = cls.nbrs_lookup.get_nns(cls.test_features_t)
        cls.closest_neighbor_t = cls.test_neighbors_t[:, 0]
        cls.closest_set_t = cls.nn_indices_all_t[cls.closest_neighbor_t]

        cls.new_nn_indices_t = fast_nn_update_t(cls.nn_indices_all_t)
        cls.closest_set_new_t = cls.new_nn_indices_t[cls.closest_neighbor_t]
        cls.crosswise_dists_fast_t = (
            cls.muygps_05_t.kernel.deformation.crosswise_tensor(
                cls.test_features_t,
                cls.train_features_t,
                np.arange(0, cls.test_count),
                cls.closest_set_new_t,
            )
        )
        cls.Kcross_fast_t = cls.muygps_05_t.kernel(cls.crosswise_dists_fast_t)

    def test_fast_nn_update(self):
        self.assertTrue(
            np.allclose(
                fast_nn_update_t(self.nn_indices_all_t),
                fast_nn_update_n(self.nn_indices_all_n),
            )
        )

    def test_make_fast_predict_tensors(self):
        self.assertTrue(np.allclose(self.Kin_fast_n, self.Kin_fast_t))
        self.assertTrue(
            np.allclose(
                self.train_nn_targets_fast_n, self.train_nn_targets_fast_t
            )
        )

    def test_homoscedastic_kernel_tensors(self):
        self.assertTrue(
            np.allclose(
                self.homoscedastic_Kin_fast_n, self.homoscedastic_Kin_fast_t
            )
        )

    def test_heteroscedastic_kernel_tensors(self):
        self.assertTrue(
            np.allclose(
                self.heteroscedastic_Kin_fast_n,
                self.heteroscedastic_Kin_fast_t,
            )
        )

    def test_fast_predict(self):
        self.assertTrue(
            _allclose(
                self.muygps_05_n.fast_posterior_mean(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_n[self.closest_neighbor_n],
                ),
                self.muygps_05_t.fast_posterior_mean(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_t[self.closest_neighbor_t],
                ),
            )
        )

    def test_fast_predict_heteroscedastic(self):
        self.assertTrue(
            _allclose(
                self.muygps_heteroscedastic_train_n.fast_posterior_mean(
                    self.Kcross_fast_n,
                    self.fast_regress_coeffs_heteroscedastic_n[
                        self.closest_neighbor_n
                    ],
                ),
                self.muygps_heteroscedastic_train_t.fast_posterior_mean(
                    self.Kcross_fast_t,
                    self.fast_regress_coeffs_heteroscedastic_t[
                        self.closest_neighbor_t
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


class OptimTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTestCase, cls).setUpClass()
        cls.predictions_t = cls.muygps_05_t.posterior_mean(
            cls.homoscedastic_Kin_t, cls.Kcross_t, cls.batch_nn_targets_t
        )
        cls.variances_t = cls.muygps_05_t.posterior_variance(
            cls.homoscedastic_Kin_t, cls.Kcross_t
        )
        cls.predictions_heteroscedastic_t = (
            cls.muygps_heteroscedastic_t.posterior_mean(
                cls.heteroscedastic_Kin_t, cls.Kcross_t, cls.batch_nn_targets_t
            )
        )
        cls.variances_heteroscedastic_t = (
            cls.muygps_heteroscedastic_t.posterior_variance(
                cls.heteroscedastic_Kin_t, cls.Kcross_t
            )
        )
        cls.predictions_n = cls.predictions_t.detach().numpy()
        cls.variances_n = cls.variances_t.detach().numpy()
        cls.x0_names, cls.x0_n, cls.bounds = cls.muygps_05_n.get_opt_params()
        cls.x0_t = torch.from_numpy(cls.x0_n)
        cls.x0_map_n = {n: cls.x0_n[i] for i, n in enumerate(cls.x0_names)}
        cls.x0_map_t = {n: cls.x0_t[i] for i, n in enumerate(cls.x0_names)}

    def _get_scale_fn_n(self):
        return self.muygps_05_n.scale.get_opt_fn(self.muygps_05_n)

    def _get_scale_fn_heteroscedastic_n(self):
        return self.muygps_heteroscedastic_n.scale.get_opt_fn(
            self.muygps_heteroscedastic_n
        )

    def _get_scale_fn_t(self):
        return self.muygps_05_t.scale.get_opt_fn(self.muygps_05_t)

    def _get_scale_fn_heteroscedastic_t(self):
        return self.muygps_heteroscedastic_t.scale.get_opt_fn(
            self.muygps_heteroscedastic_t
        )

    def _get_obj_fn_n(self):
        return make_loo_crossval_fn(
            mse_fn_n,
            self.muygps_05_n.kernel.get_opt_fn(),
            self.muygps_05_n.get_opt_mean_fn(),
            self.muygps_05_n.get_opt_var_fn(),
            self._get_scale_fn_n(),
            self.pairwise_dists_n,
            self.crosswise_dists_n,
            self.batch_nn_targets_n,
            self.batch_targets_n,
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

    def _get_obj_fn_t(self):
        return make_loo_crossval_fn(
            mse_fn_t,
            self.muygps_05_t.kernel.get_opt_fn(),
            self.muygps_05_t.get_opt_mean_fn(),
            self.muygps_05_t.get_opt_var_fn(),
            self._get_scale_fn_t(),
            self.pairwise_dists_t,
            self.crosswise_dists_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )

    def _get_obj_fn_heteroscedastic_t(self):
        return make_loo_crossval_fn(
            mse_fn_t,
            self.muygps_heteroscedastic_t.kernel.get_opt_fn(),
            self.muygps_heteroscedastic_t.get_opt_mean_fn(),
            self.muygps_heteroscedastic_t.get_opt_var_fn(),
            self._get_scale_fn_heteroscedastic_t(),
            self.pairwise_dists_t,
            self.crosswise_dists_t,
            self.batch_nn_targets_t,
            self.batch_targets_t,
        )


class ObjectiveTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectiveTest, cls).setUpClass()

        cls.scale_n = cls.muygps_05_n.scale()
        cls.scale_t = cls.muygps_05_t.scale()

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
                    self.scale_n,
                ),
                lool_fn_t(
                    self.predictions_t,
                    self.batch_targets_t,
                    self.variances_t,
                    self.scale_t,
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
                pseudo_huber_fn_t(
                    self.predictions_t, self.batch_targets_t, boundary_scale
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
                looph_fn_t(
                    self.predictions_t,
                    self.batch_targets_t,
                    self.variances_t,
                    self.scale_t,
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
                cross_entropy_fn_n(cat_predictions_n, cat_batch_targets_n),
                cross_entropy_fn_t(cat_predictions_t, cat_batch_targets_t),
            )
        )

    def test_kernel_fn(self):
        kernel_fn_n = self.muygps_05_n.kernel.get_opt_fn()
        kernel_fn_t = self.muygps_05_t.kernel.get_opt_fn()
        self.assertTrue(
            np.allclose(
                kernel_fn_n(self.pairwise_dists_n, **self.x0_map_n),
                kernel_fn_t(self.pairwise_dists_t, **self.x0_map_t),
            )
        )

    def test_mean_fn(self):
        mean_fn_n = self.muygps_05_n.get_opt_mean_fn()
        mean_fn_t = self.muygps_05_t.get_opt_mean_fn()
        self.assertTrue(
            _allclose(
                mean_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_t(
                    self.Kin_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_mean_heteroscedastic_fn(self):
        mean_fn_n = self.muygps_heteroscedastic_n.get_opt_mean_fn()
        mean_fn_t = self.muygps_heteroscedastic_t.get_opt_mean_fn()
        self.assertTrue(
            _allclose(
                mean_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                mean_fn_t(
                    self.Kin_t,
                    self.Kcross_t,
                    self.batch_nn_targets_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_var_fn(self):
        var_fn_n = self.muygps_05_n.get_opt_var_fn()
        var_fn_t = self.muygps_05_t.get_opt_var_fn()
        self.assertTrue(
            np.allclose(
                var_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_t(
                    self.Kin_t,
                    self.Kcross_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_var_heteroscedastic_fn(self):
        var_fn_n = self.muygps_heteroscedastic_n.get_opt_var_fn()
        var_fn_t = self.muygps_heteroscedastic_t.get_opt_var_fn()
        self.assertTrue(
            np.allclose(
                var_fn_n(
                    self.Kin_n,
                    self.Kcross_n,
                    **self.x0_map_n,
                ),
                var_fn_t(
                    self.Kin_t,
                    self.Kcross_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_scale_fn(self):
        ss_fn_n = self._get_scale_fn_n()
        ss_fn_t = self._get_scale_fn_t()
        self.assertTrue(
            np.allclose(
                ss_fn_n(
                    self.Kin_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_t(
                    self.Kin_t,
                    self.batch_nn_targets_t,
                    **self.x0_map_t,
                ),
            )
        )

    def test_scale_heteroscedastic_fn(self):
        ss_fn_n = self._get_scale_fn_heteroscedastic_n()
        ss_fn_t = self._get_scale_fn_heteroscedastic_t()
        self.assertTrue(
            np.allclose(
                ss_fn_n(
                    self.Kin_n,
                    self.batch_nn_targets_n,
                    **self.x0_map_n,
                ),
                ss_fn_t(
                    self.Kin_t,
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
