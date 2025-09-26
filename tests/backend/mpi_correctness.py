# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._src.gp.tensors.numpy import (
    _F2 as _F2_n,
    _l2 as _l2_n,
    _crosswise_tensor as crosswise_tensor_n,
    _pairwise_tensor as pairwise_tensor_n,
)
from MuyGPyS._src.gp.kernels.numpy import (
    _rbf_fn as rbf_fn_n,
    _matern_05_fn as matern_05_fn_n,
    _matern_15_fn as matern_15_fn_n,
    _matern_25_fn as matern_25_fn_n,
    _matern_inf_fn as matern_inf_fn_n,
    _matern_gen_fn as matern_gen_fn_n,
)

from MuyGPyS._src.mpi_utils import _chunk_tensor
from MuyGPyS._src.gp.muygps.numpy import (
    _muygps_diagonal_variance as muygps_diagonal_variance_n,
    _muygps_posterior_mean as muygps_posterior_mean_n,
)
from MuyGPyS._src.gp.noise.numpy import (
    _homoscedastic_perturb as homoscedastic_perturb_n,
)
from MuyGPyS._src.gp.noise.mpi import (
    _homoscedastic_perturb as homoscedastic_perturb_m,
)
from MuyGPyS._src.optimize.scale.numpy import (
    _analytic_scale_optim as analytic_scale_optim_n,
)
from MuyGPyS._src.optimize.scale.mpi import (
    _analytic_scale_optim as analytic_scale_optim_m,
)
from MuyGPyS._src.optimize.loss.numpy import (
    _cross_entropy_fn as cross_entropy_fn_n,
    _lool_fn as lool_fn_n,
    _mse_fn as mse_fn_n,
    _pseudo_huber_fn as pseudo_huber_fn_n,
    _looph_fn as looph_fn_n,
)
from MuyGPyS._src.optimize.loss.mpi import (
    _cross_entropy_fn as cross_entropy_fn_m,
    _lool_fn as lool_fn_m,
    _mse_fn as mse_fn_m,
    _pseudo_huber_fn as pseudo_huber_fn_m,
    _looph_fn as looph_fn_m,
)
from MuyGPyS._src.optimize.chassis.numpy import (
    _bayes_opt_optimize as bayes_optimize_n,
    _scipy_optimize as scipy_optimize_n,
)
from MuyGPyS._src.optimize.chassis.mpi import (
    _bayes_opt_optimize as bayes_optimize_m,
    _scipy_optimize as scipy_optimize_m,
)
from MuyGPyS._test.utils import (
    _exact_nn_kwarg_options,
    _make_gaussian_data,
    _make_gaussian_matrix,
    _make_uniform_matrix,
)
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import Anisotropy, Isotropy, MetricFn
from MuyGPyS.gp.hyperparameter import AnalyticScale, ScalarParam, VectorParam
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize.loss import (
    LossFn,
    make_raw_predict_and_loss_fn,
    make_var_predict_and_loss_fn,
)
from MuyGPyS.optimize.objective import make_loo_crossval_fn

if config.state.mpi_enabled is False:
    raise ValueError("Bad attempt to run mpi-only code with mpi diabled.")

if config.state.backend != "mpi":
    raise ValueError(
        "MPI correctness test must be run in MPI mode, not "
        f"{config.state.backend}. mode"
    )


# make numpy loss functor aliases
mse_fn_n = LossFn(mse_fn_n, make_raw_predict_and_loss_fn)
cross_entropy_fn_n = LossFn(cross_entropy_fn_n, make_raw_predict_and_loss_fn)
lool_fn_n = LossFn(lool_fn_n, make_var_predict_and_loss_fn)
pseudo_huber_fn_n = LossFn(pseudo_huber_fn_n, make_raw_predict_and_loss_fn)
looph_fn_n = LossFn(looph_fn_n, make_var_predict_and_loss_fn)


# make mpi loss functor aliases
mse_fn_m = LossFn(mse_fn_m, make_raw_predict_and_loss_fn)
cross_entropy_fn_m = LossFn(cross_entropy_fn_m, make_raw_predict_and_loss_fn)
lool_fn_m = LossFn(lool_fn_m, make_var_predict_and_loss_fn)
pseudo_huber_fn_m = LossFn(pseudo_huber_fn_m, make_raw_predict_and_loss_fn)
looph_fn_m = LossFn(looph_fn_m, make_var_predict_and_loss_fn)

l2_n = MetricFn(
    differences_metric_fn=_l2_n,
    crosswise_differences_fn=crosswise_tensor_n,
    pairwise_differences_fn=pairwise_tensor_n,
    apply_length_scale_fn=lambda x, y: x / y,
)
F2_n = MetricFn(
    differences_metric_fn=_F2_n,
    crosswise_differences_fn=crosswise_tensor_n,
    pairwise_differences_fn=pairwise_tensor_n,
    apply_length_scale_fn=lambda x, y: x / y**2,
)


world = config.mpi_state.comm_world
rank = world.Get_rank()
size = world.Get_size()


class TensorsTestCase(parameterized.TestCase):
    @classmethod
    def _make_muygps(cls, smoothness, deformation, smoothness_bounds="fixed"):
        return MuyGPS(
            kernel=Matern(
                smoothness=ScalarParam(smoothness, smoothness_bounds),
                deformation=deformation,
                _backend_05_fn=matern_05_fn_n,
                _backend_15_fn=matern_15_fn_n,
                _backend_25_fn=matern_25_fn_n,
                _backend_inf_fn=matern_inf_fn_n,
                _backend_gen_fn=matern_gen_fn_n,
            ),
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_n
            ),
            scale=AnalyticScale(),
            _backend_mean_fn=muygps_posterior_mean_n,
            _backend_var_fn=muygps_diagonal_variance_n,
        )

    @classmethod
    def _make_isotropic_muygps(cls, smoothness, **kwargs):
        return cls._make_muygps(
            smoothness,
            deformation=Isotropy(
                l2_n,
                length_scale=ScalarParam(cls.length_scale),
            ),
            **kwargs,
        )

    @classmethod
    def _make_anisotropic_muygps(cls, smoothness, **kwargs):
        return cls._make_muygps(
            smoothness,
            Anisotropy(
                l2_n,
                length_scale=VectorParam(
                    ScalarParam(cls.length_scale), ScalarParam(cls.length_scale)
                ),
            ),
            **kwargs,
        )

    @classmethod
    def _make_muygps_rbf(cls, deformation):
        return MuyGPS(
            kernel=RBF(
                deformation=deformation,
                _backend_fn=rbf_fn_n,
            ),
            noise=HomoscedasticNoise(
                cls.noise, _backend_fn=homoscedastic_perturb_n
            ),
            scale=AnalyticScale(
                _backend_ones=np.ones,
                _backend_ndarray=np.ndarray,
                _backend_ftype=np.ftype,
                _backend_farray=np.farray,
                _backend_outer=np.outer,
            ),
            _backend_mean_fn=muygps_posterior_mean_n,
            _backend_var_fn=muygps_diagonal_variance_n,
        )

    @classmethod
    def _make_isotropic_muygps_rbf(cls):
        return cls._make_muygps_rbf(
            Isotropy(
                F2_n,
                length_scale=ScalarParam(cls.length_scale),
            )
        )

    @classmethod
    def _make_anisotropic_muygps_rbf(cls):
        return cls._make_muygps_rbf(
            Anisotropy(
                F2_n,
                length_scale=VectorParam(
                    ScalarParam(cls.length_scale), ScalarParam(cls.length_scale)
                ),
            )
        )

    @classmethod
    def setUpClass(cls):
        super(TensorsTestCase, cls).setUpClass()
        cls.train_count = 1000
        cls.test_count = 100
        cls.feature_count = 2
        cls.response_count = 1
        cls.nn_count = 10
        cls.batch_count = 500
        cls.length_scale = 1.0
        cls.smoothness = 0.5
        cls.smoothness_bounds = (1e-1, 2)
        cls.noise = 1e-3
        cls.muygps_gen = cls._make_isotropic_muygps(
            cls.smoothness, smoothness_bounds=cls.smoothness_bounds
        )
        cls.muygps_05 = cls._make_isotropic_muygps(0.5)
        cls.muygps_15 = cls._make_isotropic_muygps(1.5)
        cls.muygps_25 = cls._make_isotropic_muygps(2.5)
        cls.muygps_inf = cls._make_isotropic_muygps(np.inf)
        cls.muygps_rbf = cls._make_isotropic_muygps_rbf()

        cls.muygps_anisotropic_gen = cls._make_anisotropic_muygps(
            cls.smoothness, smoothness_bounds=cls.smoothness_bounds
        )
        cls.muygps_anisotropic_05 = cls._make_anisotropic_muygps(0.5)
        cls.muygps_anisotropic_15 = cls._make_anisotropic_muygps(1.5)
        cls.muygps_anisotropic_25 = cls._make_anisotropic_muygps(2.5)
        cls.muygps_anisotropic_inf = cls._make_anisotropic_muygps(np.inf)
        cls.muygps_anisotropic_rbf = cls._make_anisotropic_muygps_rbf()

        cls.measurement_noise = _make_uniform_matrix(cls.train_count, 1)

        if rank == 0:
            cls.train_features = _make_gaussian_matrix(
                cls.train_count, cls.feature_count
            )
            cls.train_responses = np.squeeze(
                _make_gaussian_matrix(cls.train_count, cls.response_count)
            )
            cls.test_features = _make_gaussian_matrix(
                cls.test_count, cls.feature_count
            )
            cls.test_responses = np.squeeze(
                _make_gaussian_matrix(cls.test_count, cls.response_count)
            )
            nbrs_lookup = NN_Wrapper(
                cls.train_features,
                cls.nn_count,
                **_exact_nn_kwarg_options[0],
            )
            batch_indices, batch_nn_indices = sample_batch(
                nbrs_lookup, cls.batch_count, cls.train_count
            )

            (
                cls.batch_crosswise_dists,
                cls.batch_pairwise_dists,
                cls.batch_targets,
                cls.batch_nn_targets,
            ) = cls.muygps_gen.make_train_tensors(
                batch_indices,
                batch_nn_indices,
                cls.train_features,
                cls.train_responses,
                disable_mpi=True,
            )
            cls.batch_crosswise_diffs, cls.batch_pairwise_diffs, _, _ = (
                cls.muygps_anisotropic_gen.make_train_tensors(
                    batch_indices,
                    batch_nn_indices,
                    cls.train_features,
                    cls.train_responses,
                    disable_mpi=True,
                )
            )

            test_nn_indices, _ = nbrs_lookup.get_nns(cls.test_features)

            (
                cls.test_crosswise_dists,
                cls.test_pairwise_dists,
                cls.test_nn_targets,
            ) = cls.muygps_gen.make_predict_tensors(
                np.arange(cls.test_count),
                test_nn_indices,
                cls.test_features,
                cls.train_features,
                cls.train_responses,
                disable_mpi=True,
            )
            cls.test_crosswise_diffs, cls.test_pairwise_diffs, _ = (
                cls.muygps_anisotropic_gen.make_predict_tensors(
                    np.arange(cls.test_count),
                    test_nn_indices,
                    cls.test_features,
                    cls.train_features,
                    cls.train_responses,
                    disable_mpi=True,
                )
            )

        else:
            cls.train_features = None
            cls.train_responses = None
            cls.test_features = None
            cls.test_responses = None
            batch_indices = None
            batch_nn_indices = None
            test_nn_indices = None

            cls.batch_crosswise_dists = None
            cls.batch_pairwise_dists = None
            cls.batch_crosswise_diffs = None
            cls.batch_pairwise_diffs = None
            cls.batch_targets = None
            cls.batch_nn_targets = None

            cls.test_crosswise_dists = None
            cls.test_pairwise_dists = None
            cls.test_crosswise_diffs = None
            cls.test_pairwise_diffs = None
            cls.test_nn_targets = None

        (
            cls.batch_crosswise_dists_chunk,
            cls.batch_pairwise_dists_chunk,
            cls.batch_targets_chunk,
            cls.batch_nn_targets_chunk,
        ) = cls.muygps_gen.make_train_tensors(  # MPI version
            batch_indices,
            batch_nn_indices,
            cls.train_features,
            cls.train_responses,
        )
        (
            cls.batch_crosswise_diffs_chunk,
            cls.batch_pairwise_diffs_chunk,
            _,
            _,
        ) = cls.muygps_anisotropic_gen.make_train_tensors(  # MPI version
            batch_indices,
            batch_nn_indices,
            cls.train_features,
            cls.train_responses,
        )
        (
            cls.test_crosswise_dists_chunk,
            cls.test_pairwise_dists_chunk,
            cls.test_nn_targets_chunk,
        ) = cls.muygps_gen.make_predict_tensors(  # MPI version
            np.arange(cls.test_count),
            test_nn_indices,
            cls.test_features,
            cls.train_features,
            cls.train_responses,
        )
        cls.test_crosswise_diffs_chunk, cls.test_pairwise_diffs_chunk, _ = (
            cls.muygps_anisotropic_gen.make_predict_tensors(  # MPI version
                np.arange(cls.test_count),
                test_nn_indices,
                cls.test_features,
                cls.train_features,
                cls.train_responses,
            )
        )
        cls.test_responses_chunk = _chunk_tensor(cls.test_responses)

    def _compare_tensors(self, tensor, tensor_chunks):
        recovered_tensor = world.gather(tensor_chunks, root=0)
        if rank == 0:
            recovered_tensor = np.array(recovered_tensor)
            shape = recovered_tensor.shape
            recovered_tensor = recovered_tensor.reshape(
                (shape[0] * shape[1],) + shape[2:]
            )
            self.assertTrue(np.allclose(recovered_tensor, tensor))


class TensorsTest(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(TensorsTest, cls).setUpClass()

    def test_batch_pairwise_dists(self):
        self._compare_tensors(
            self.batch_pairwise_dists, self.batch_pairwise_dists_chunk
        )

    def test_test_pairwise_dists(self):
        self._compare_tensors(
            self.test_pairwise_dists, self.test_pairwise_dists_chunk
        )

    def test_batch_pairwise_diffs(self):
        self._compare_tensors(
            self.batch_pairwise_diffs, self.batch_pairwise_diffs_chunk
        )

    def test_test_pairwise_diffs(self):
        self._compare_tensors(
            self.test_pairwise_diffs, self.test_pairwise_diffs_chunk
        )

    def test_batch_crosswise_dists(self):
        self._compare_tensors(
            self.batch_crosswise_dists, self.batch_crosswise_dists_chunk
        )

    def test_test_crosswise_dists(self):
        self._compare_tensors(
            self.test_crosswise_dists, self.test_crosswise_dists_chunk
        )

    def test_batch_crosswise_diffs(self):
        self._compare_tensors(
            self.batch_crosswise_diffs, self.batch_crosswise_diffs_chunk
        )

    def test_test_crosswise_diffs(self):
        self._compare_tensors(
            self.test_crosswise_diffs, self.test_crosswise_diffs_chunk
        )

    def test_batch_targets(self):
        self._compare_tensors(self.batch_targets, self.batch_targets_chunk)

    def test_batch_nn_targets(self):
        self._compare_tensors(
            self.batch_nn_targets, self.batch_nn_targets_chunk
        )

    def test_test_nn_targets(self):
        self._compare_tensors(self.test_nn_targets, self.test_nn_targets_chunk)


class KernelTestCase(TensorsTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()
        if rank == 0:
            cls.batch_covariance_rbf = cls.muygps_rbf.kernel(
                cls.batch_pairwise_dists
            )
            cls.batch_covariance_05 = cls.muygps_05.kernel(
                cls.batch_pairwise_dists
            )
            cls.batch_covariance_15 = cls.muygps_15.kernel(
                cls.batch_pairwise_dists
            )
            cls.batch_covariance_25 = cls.muygps_25.kernel(
                cls.batch_pairwise_dists
            )
            cls.batch_covariance_inf = cls.muygps_inf.kernel(
                cls.batch_pairwise_dists
            )
            cls.batch_covariance_gen = cls.muygps_gen.kernel(
                cls.batch_pairwise_dists
            )
            cls.batch_crosscov_rbf = cls.muygps_rbf.kernel(
                cls.batch_crosswise_dists
            )
            cls.batch_crosscov_05 = cls.muygps_05.kernel(
                cls.batch_crosswise_dists
            )
            cls.batch_crosscov_15 = cls.muygps_15.kernel(
                cls.batch_crosswise_dists
            )
            cls.batch_crosscov_25 = cls.muygps_25.kernel(
                cls.batch_crosswise_dists
            )
            cls.batch_crosscov_inf = cls.muygps_inf.kernel(
                cls.batch_crosswise_dists
            )
            cls.batch_crosscov_gen = cls.muygps_gen.kernel(
                cls.batch_crosswise_dists
            )
            cls.test_covariance_rbf = cls.muygps_rbf.kernel(
                cls.test_pairwise_dists
            )
            cls.test_covariance_05 = cls.muygps_05.kernel(
                cls.test_pairwise_dists
            )
            cls.test_covariance_15 = cls.muygps_15.kernel(
                cls.test_pairwise_dists
            )
            cls.test_covariance_25 = cls.muygps_25.kernel(
                cls.test_pairwise_dists
            )
            cls.test_covariance_inf = cls.muygps_inf.kernel(
                cls.test_pairwise_dists
            )
            cls.test_covariance_gen = cls.muygps_gen.kernel(
                cls.test_pairwise_dists
            )
            cls.test_crosscov_rbf = cls.muygps_rbf.kernel(
                cls.test_crosswise_dists
            )
            cls.test_crosscov_05 = cls.muygps_05.kernel(
                cls.test_crosswise_dists
            )
            cls.test_crosscov_15 = cls.muygps_15.kernel(
                cls.test_crosswise_dists
            )
            cls.test_crosscov_25 = cls.muygps_25.kernel(
                cls.test_crosswise_dists
            )
            cls.test_crosscov_inf = cls.muygps_inf.kernel(
                cls.test_crosswise_dists
            )
            cls.test_crosscov_gen = cls.muygps_gen.kernel(
                cls.test_crosswise_dists
            )
            cls.batch_covariance_anisotropic_rbf = (
                cls.muygps_anisotropic_rbf.kernel(cls.batch_pairwise_diffs)
            )
            cls.batch_covariance_anisotropic_05 = (
                cls.muygps_anisotropic_05.kernel(cls.batch_pairwise_diffs)
            )
            cls.batch_covariance_anisotropic_15 = (
                cls.muygps_anisotropic_15.kernel(cls.batch_pairwise_diffs)
            )
            cls.batch_covariance_anisotropic_25 = (
                cls.muygps_anisotropic_25.kernel(cls.batch_pairwise_diffs)
            )
            cls.batch_covariance_anisotropic_inf = (
                cls.muygps_anisotropic_inf.kernel(cls.batch_pairwise_diffs)
            )
            cls.batch_covariance_anisotropic_gen = (
                cls.muygps_anisotropic_gen.kernel(cls.batch_pairwise_diffs)
            )
            cls.batch_crosscov_anisotropic_rbf = (
                cls.muygps_anisotropic_rbf.kernel(cls.batch_crosswise_diffs)
            )
            cls.batch_crosscov_anisotropic_05 = (
                cls.muygps_anisotropic_05.kernel(cls.batch_crosswise_diffs)
            )
            cls.batch_crosscov_anisotropic_15 = (
                cls.muygps_anisotropic_15.kernel(cls.batch_crosswise_diffs)
            )
            cls.batch_crosscov_anisotropic_25 = (
                cls.muygps_anisotropic_25.kernel(cls.batch_crosswise_diffs)
            )
            cls.batch_crosscov_anisotropic_inf = (
                cls.muygps_anisotropic_inf.kernel(cls.batch_crosswise_diffs)
            )
            cls.batch_crosscov_anisotropic_gen = (
                cls.muygps_anisotropic_gen.kernel(cls.batch_crosswise_diffs)
            )
            cls.test_covariance_anisotropic_rbf = (
                cls.muygps_anisotropic_rbf.kernel(cls.test_pairwise_diffs)
            )
            cls.test_covariance_anisotropic_05 = (
                cls.muygps_anisotropic_05.kernel(cls.test_pairwise_diffs)
            )
            cls.test_covariance_anisotropic_15 = (
                cls.muygps_anisotropic_15.kernel(cls.test_pairwise_diffs)
            )
            cls.test_covariance_anisotropic_25 = (
                cls.muygps_anisotropic_25.kernel(cls.test_pairwise_diffs)
            )
            cls.test_covariance_anisotropic_inf = (
                cls.muygps_anisotropic_inf.kernel(cls.test_pairwise_diffs)
            )
            cls.test_covariance_anisotropic_gen = (
                cls.muygps_anisotropic_gen.kernel(cls.test_pairwise_diffs)
            )
            cls.test_crosscov_anisotropic_rbf = (
                cls.muygps_anisotropic_rbf.kernel(cls.test_crosswise_diffs)
            )
            cls.test_crosscov_anisotropic_05 = cls.muygps_anisotropic_05.kernel(
                cls.test_crosswise_diffs
            )
            cls.test_crosscov_anisotropic_15 = cls.muygps_anisotropic_15.kernel(
                cls.test_crosswise_diffs
            )
            cls.test_crosscov_anisotropic_25 = cls.muygps_anisotropic_25.kernel(
                cls.test_crosswise_diffs
            )
            cls.test_crosscov_anisotropic_inf = (
                cls.muygps_anisotropic_inf.kernel(cls.test_crosswise_diffs)
            )
            cls.test_crosscov_anisotropic_gen = (
                cls.muygps_anisotropic_gen.kernel(cls.test_crosswise_diffs)
            )
        else:
            cls.batch_covariance_rbf = None
            cls.batch_covariance_05 = None
            cls.batch_covariance_15 = None
            cls.batch_covariance_25 = None
            cls.batch_covariance_inf = None
            cls.batch_covariance_gen = None
            cls.batch_crosscov_rbf = None
            cls.batch_crosscov_05 = None
            cls.batch_crosscov_15 = None
            cls.batch_crosscov_25 = None
            cls.batch_crosscov_inf = None
            cls.batch_crosscov_gen = None
            cls.test_covariance_rbf = None
            cls.test_covariance_05 = None
            cls.test_covariance_15 = None
            cls.test_covariance_25 = None
            cls.test_covariance_inf = None
            cls.test_covariance_gen = None
            cls.test_crosscov_rbf = None
            cls.test_crosscov_05 = None
            cls.test_crosscov_15 = None
            cls.test_crosscov_25 = None
            cls.test_crosscov_inf = None
            cls.test_crosscov_gen = None
            cls.batch_covariance_anisotropic_rbf = None
            cls.batch_covariance_anisotropic_05 = None
            cls.batch_covariance_anisotropic_15 = None
            cls.batch_covariance_anisotropic_25 = None
            cls.batch_covariance_anisotropic_inf = None
            cls.batch_covariance_anisotropic_gen = None
            cls.batch_crosscov_anisotropic_rbf = None
            cls.batch_crosscov_anisotropic_05 = None
            cls.batch_crosscov_anisotropic_15 = None
            cls.batch_crosscov_anisotropic_25 = None
            cls.batch_crosscov_anisotropic_inf = None
            cls.batch_crosscov_anisotropic_gen = None
            cls.test_covariance_anisotropic_rbf = None
            cls.test_covariance_anisotropic_05 = None
            cls.test_covariance_anisotropic_15 = None
            cls.test_covariance_anisotropic_25 = None
            cls.test_covariance_anisotropic_inf = None
            cls.test_covariance_anisotropic_gen = None
            cls.test_crosscov_anisotropic_rbf = None
            cls.test_crosscov_anisotropic_05 = None
            cls.test_crosscov_anisotropic_15 = None
            cls.test_crosscov_anisotropic_25 = None
            cls.test_crosscov_anisotropic_inf = None
            cls.test_crosscov_anisotropic_gen = None

        cls.batch_covariance_rbf_chunk = cls.muygps_rbf.kernel(
            cls.batch_pairwise_dists_chunk
        )
        cls.batch_covariance_05_chunk = cls.muygps_05.kernel(
            cls.batch_pairwise_dists_chunk
        )
        cls.batch_covariance_15_chunk = cls.muygps_15.kernel(
            cls.batch_pairwise_dists_chunk
        )
        cls.batch_covariance_25_chunk = cls.muygps_25.kernel(
            cls.batch_pairwise_dists_chunk
        )
        cls.batch_covariance_inf_chunk = cls.muygps_inf.kernel(
            cls.batch_pairwise_dists_chunk
        )
        cls.batch_covariance_gen_chunk = cls.muygps_gen.kernel(
            cls.batch_pairwise_dists_chunk
        )
        cls.batch_crosscov_rbf_chunk = cls.muygps_rbf.kernel(
            cls.batch_crosswise_dists_chunk
        )
        cls.batch_crosscov_05_chunk = cls.muygps_05.kernel(
            cls.batch_crosswise_dists_chunk
        )
        cls.batch_crosscov_15_chunk = cls.muygps_15.kernel(
            cls.batch_crosswise_dists_chunk
        )
        cls.batch_crosscov_25_chunk = cls.muygps_25.kernel(
            cls.batch_crosswise_dists_chunk
        )
        cls.batch_crosscov_inf_chunk = cls.muygps_inf.kernel(
            cls.batch_crosswise_dists_chunk
        )
        cls.batch_crosscov_gen_chunk = cls.muygps_gen.kernel(
            cls.batch_crosswise_dists_chunk
        )
        cls.test_covariance_rbf_chunk = cls.muygps_rbf.kernel(
            cls.test_pairwise_dists_chunk
        )
        cls.test_covariance_05_chunk = cls.muygps_05.kernel(
            cls.test_pairwise_dists_chunk
        )
        cls.test_covariance_15_chunk = cls.muygps_15.kernel(
            cls.test_pairwise_dists_chunk
        )
        cls.test_covariance_25_chunk = cls.muygps_25.kernel(
            cls.test_pairwise_dists_chunk
        )
        cls.test_covariance_inf_chunk = cls.muygps_inf.kernel(
            cls.test_pairwise_dists_chunk
        )
        cls.test_covariance_gen_chunk = cls.muygps_gen.kernel(
            cls.test_pairwise_dists_chunk
        )
        cls.test_crosscov_rbf_chunk = cls.muygps_rbf.kernel(
            cls.test_crosswise_dists_chunk
        )
        cls.test_crosscov_05_chunk = cls.muygps_05.kernel(
            cls.test_crosswise_dists_chunk
        )
        cls.test_crosscov_15_chunk = cls.muygps_15.kernel(
            cls.test_crosswise_dists_chunk
        )
        cls.test_crosscov_25_chunk = cls.muygps_25.kernel(
            cls.test_crosswise_dists_chunk
        )
        cls.test_crosscov_inf_chunk = cls.muygps_inf.kernel(
            cls.test_crosswise_dists_chunk
        )
        cls.test_crosscov_gen_chunk = cls.muygps_gen.kernel(
            cls.test_crosswise_dists_chunk
        )

        cls.batch_covariance_anisotropic_rbf_chunk = (
            cls.muygps_anisotropic_rbf.kernel(cls.batch_pairwise_diffs_chunk)
        )
        cls.batch_covariance_anisotropic_05_chunk = (
            cls.muygps_anisotropic_05.kernel(cls.batch_pairwise_diffs_chunk)
        )
        cls.batch_covariance_anisotropic_15_chunk = (
            cls.muygps_anisotropic_15.kernel(cls.batch_pairwise_diffs_chunk)
        )
        cls.batch_covariance_anisotropic_25_chunk = (
            cls.muygps_anisotropic_25.kernel(cls.batch_pairwise_diffs_chunk)
        )
        cls.batch_covariance_anisotropic_inf_chunk = (
            cls.muygps_anisotropic_inf.kernel(cls.batch_pairwise_diffs_chunk)
        )
        cls.batch_covariance_anisotropic_gen_chunk = (
            cls.muygps_anisotropic_gen.kernel(cls.batch_pairwise_diffs_chunk)
        )
        cls.batch_crosscov_anisotropic_rbf_chunk = (
            cls.muygps_anisotropic_rbf.kernel(cls.batch_crosswise_diffs_chunk)
        )
        cls.batch_crosscov_anisotropic_05_chunk = (
            cls.muygps_anisotropic_05.kernel(cls.batch_crosswise_diffs_chunk)
        )
        cls.batch_crosscov_anisotropic_15_chunk = (
            cls.muygps_anisotropic_15.kernel(cls.batch_crosswise_diffs_chunk)
        )
        cls.batch_crosscov_anisotropic_25_chunk = (
            cls.muygps_anisotropic_25.kernel(cls.batch_crosswise_diffs_chunk)
        )
        cls.batch_crosscov_anisotropic_inf_chunk = (
            cls.muygps_anisotropic_inf.kernel(cls.batch_crosswise_diffs_chunk)
        )
        cls.batch_crosscov_anisotropic_gen_chunk = (
            cls.muygps_anisotropic_gen.kernel(cls.batch_crosswise_diffs_chunk)
        )
        cls.test_covariance_anisotropic_rbf_chunk = (
            cls.muygps_anisotropic_rbf.kernel(cls.test_pairwise_diffs_chunk)
        )
        cls.test_covariance_anisotropic_05_chunk = (
            cls.muygps_anisotropic_05.kernel(cls.test_pairwise_diffs_chunk)
        )
        cls.test_covariance_anisotropic_15_chunk = (
            cls.muygps_anisotropic_15.kernel(cls.test_pairwise_diffs_chunk)
        )
        cls.test_covariance_anisotropic_25_chunk = (
            cls.muygps_anisotropic_25.kernel(cls.test_pairwise_diffs_chunk)
        )
        cls.test_covariance_anisotropic_inf_chunk = (
            cls.muygps_anisotropic_inf.kernel(cls.test_pairwise_diffs_chunk)
        )
        cls.test_covariance_anisotropic_gen_chunk = (
            cls.muygps_anisotropic_gen.kernel(cls.test_pairwise_diffs_chunk)
        )
        cls.test_crosscov_anisotropic_rbf_chunk = (
            cls.muygps_anisotropic_rbf.kernel(cls.test_crosswise_diffs_chunk)
        )
        cls.test_crosscov_anisotropic_05_chunk = (
            cls.muygps_anisotropic_05.kernel(cls.test_crosswise_diffs_chunk)
        )
        cls.test_crosscov_anisotropic_15_chunk = (
            cls.muygps_anisotropic_15.kernel(cls.test_crosswise_diffs_chunk)
        )
        cls.test_crosscov_anisotropic_25_chunk = (
            cls.muygps_anisotropic_25.kernel(cls.test_crosswise_diffs_chunk)
        )
        cls.test_crosscov_anisotropic_inf_chunk = (
            cls.muygps_anisotropic_inf.kernel(cls.test_crosswise_diffs_chunk)
        )
        cls.test_crosscov_anisotropic_gen_chunk = (
            cls.muygps_anisotropic_gen.kernel(cls.test_crosswise_diffs_chunk)
        )


class KernelTest(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTest, cls).setUpClass()

    def test_batch_covariance_rbf(self):
        self._compare_tensors(
            self.batch_covariance_rbf, self.batch_covariance_rbf_chunk
        )

    def test_batch_covariance_05(self):
        self._compare_tensors(
            self.batch_covariance_05, self.batch_covariance_05_chunk
        )

    def test_batch_covariance_15(self):
        self._compare_tensors(
            self.batch_covariance_15, self.batch_covariance_15_chunk
        )

    def test_batch_covariance_25(self):
        self._compare_tensors(
            self.batch_covariance_25, self.batch_covariance_25_chunk
        )

    def test_batch_covariance_inf(self):
        self._compare_tensors(
            self.batch_covariance_inf, self.batch_covariance_inf_chunk
        )

    def test_batch_covariance_gen(self):
        self._compare_tensors(
            self.batch_covariance_gen, self.batch_covariance_gen_chunk
        )

    def test_batch_crosscov_rbf(self):
        self._compare_tensors(
            self.batch_crosscov_rbf, self.batch_crosscov_rbf_chunk
        )

    def test_batch_crosscov_05(self):
        self._compare_tensors(
            self.batch_crosscov_05, self.batch_crosscov_05_chunk
        )

    def test_batch_crosscov_15(self):
        self._compare_tensors(
            self.batch_crosscov_15, self.batch_crosscov_15_chunk
        )

    def test_batch_crosscov_25(self):
        self._compare_tensors(
            self.batch_crosscov_25, self.batch_crosscov_25_chunk
        )

    def test_batch_crosscov_inf(self):
        self._compare_tensors(
            self.batch_crosscov_inf, self.batch_crosscov_inf_chunk
        )

    def test_batch_crosscov_gen(self):
        self._compare_tensors(
            self.batch_crosscov_gen, self.batch_crosscov_gen_chunk
        )

    def test_test_covariance_rbf(self):
        self._compare_tensors(
            self.test_covariance_rbf, self.test_covariance_rbf_chunk
        )

    def test_test_covariance_05(self):
        self._compare_tensors(
            self.test_covariance_05, self.test_covariance_05_chunk
        )

    def test_test_covariance_15(self):
        self._compare_tensors(
            self.test_covariance_15, self.test_covariance_15_chunk
        )

    def test_test_covariance_25(self):
        self._compare_tensors(
            self.test_covariance_25, self.test_covariance_25_chunk
        )

    def test_test_covariance_inf(self):
        self._compare_tensors(
            self.test_covariance_inf, self.test_covariance_inf_chunk
        )

    def test_test_covariance_gen(self):
        self._compare_tensors(
            self.test_covariance_gen, self.test_covariance_gen_chunk
        )

    def test_test_crosscov_rbf(self):
        self._compare_tensors(
            self.test_crosscov_rbf, self.test_crosscov_rbf_chunk
        )

    def test_test_crosscov_05(self):
        self._compare_tensors(
            self.test_crosscov_05, self.test_crosscov_05_chunk
        )

    def test_test_crosscov_15(self):
        self._compare_tensors(
            self.test_crosscov_15, self.test_crosscov_15_chunk
        )

    def test_test_crosscov_25(self):
        self._compare_tensors(
            self.test_crosscov_25, self.test_crosscov_25_chunk
        )

    def test_test_crosscov_inf(self):
        self._compare_tensors(
            self.test_crosscov_inf, self.test_crosscov_inf_chunk
        )

    def test_test_crosscov_gen(self):
        self._compare_tensors(
            self.test_crosscov_gen, self.test_crosscov_gen_chunk
        )

    def test_batch_covariance_anisotropic_rbf(self):
        self._compare_tensors(
            self.batch_covariance_anisotropic_rbf,
            self.batch_covariance_anisotropic_rbf_chunk,
        )

    def test_batch_covariance_anisotropic_05(self):
        self._compare_tensors(
            self.batch_covariance_anisotropic_05,
            self.batch_covariance_anisotropic_05_chunk,
        )

    def test_batch_covariance_anisotropic_15(self):
        self._compare_tensors(
            self.batch_covariance_anisotropic_15,
            self.batch_covariance_anisotropic_15_chunk,
        )

    def test_batch_covariance_anisotropic_25(self):
        self._compare_tensors(
            self.batch_covariance_anisotropic_25,
            self.batch_covariance_anisotropic_25_chunk,
        )

    def test_batch_covariance_anisotropic_inf(self):
        self._compare_tensors(
            self.batch_covariance_anisotropic_inf,
            self.batch_covariance_anisotropic_inf_chunk,
        )

    def test_batch_covariance_anisotropic_gen(self):
        self._compare_tensors(
            self.batch_covariance_anisotropic_gen,
            self.batch_covariance_anisotropic_gen_chunk,
        )

    def test_batch_crosscov_anisotropic_rbf(self):
        self._compare_tensors(
            self.batch_crosscov_anisotropic_rbf,
            self.batch_crosscov_anisotropic_rbf_chunk,
        )

    def test_batch_crosscov_anisotropic_05(self):
        self._compare_tensors(
            self.batch_crosscov_anisotropic_05,
            self.batch_crosscov_anisotropic_05_chunk,
        )

    def test_batch_crosscov_anisotropic_15(self):
        self._compare_tensors(
            self.batch_crosscov_anisotropic_15,
            self.batch_crosscov_anisotropic_15_chunk,
        )

    def test_batch_crosscov_anisotropic_25(self):
        self._compare_tensors(
            self.batch_crosscov_anisotropic_25,
            self.batch_crosscov_anisotropic_25_chunk,
        )

    def test_batch_crosscov_anisotropic_inf(self):
        self._compare_tensors(
            self.batch_crosscov_anisotropic_inf,
            self.batch_crosscov_anisotropic_inf_chunk,
        )

    def test_batch_crosscov_anisotropic_gen(self):
        self._compare_tensors(
            self.batch_crosscov_anisotropic_gen,
            self.batch_crosscov_anisotropic_gen_chunk,
        )

    def test_test_covariance_anisotropic_rbf(self):
        self._compare_tensors(
            self.test_covariance_anisotropic_rbf,
            self.test_covariance_anisotropic_rbf_chunk,
        )

    def test_test_covariance_anisotropic_05(self):
        self._compare_tensors(
            self.test_covariance_anisotropic_05,
            self.test_covariance_anisotropic_05_chunk,
        )

    def test_test_covariance_anisotropic_15(self):
        self._compare_tensors(
            self.test_covariance_anisotropic_15,
            self.test_covariance_anisotropic_15_chunk,
        )

    def test_test_covariance_anisotropic_25(self):
        self._compare_tensors(
            self.test_covariance_anisotropic_25,
            self.test_covariance_anisotropic_25_chunk,
        )

    def test_test_covariance_anisotropic_inf(self):
        self._compare_tensors(
            self.test_covariance_anisotropic_inf,
            self.test_covariance_anisotropic_inf_chunk,
        )

    def test_test_covariance_anisotropic_gen(self):
        self._compare_tensors(
            self.test_covariance_anisotropic_gen,
            self.test_covariance_anisotropic_gen_chunk,
        )

    def test_test_crosscov_anisotropic_rbf(self):
        self._compare_tensors(
            self.test_crosscov_anisotropic_rbf,
            self.test_crosscov_anisotropic_rbf_chunk,
        )

    def test_test_crosscov_anisotropic_05(self):
        self._compare_tensors(
            self.test_crosscov_anisotropic_05,
            self.test_crosscov_anisotropic_05_chunk,
        )

    def test_test_crosscov_anisotropic_15(self):
        self._compare_tensors(
            self.test_crosscov_anisotropic_15,
            self.test_crosscov_anisotropic_15_chunk,
        )

    def test_test_crosscov_anisotropic_25(self):
        self._compare_tensors(
            self.test_crosscov_anisotropic_25,
            self.test_crosscov_anisotropic_25_chunk,
        )

    def test_test_crosscov_anisotropic_inf(self):
        self._compare_tensors(
            self.test_crosscov_anisotropic_inf,
            self.test_crosscov_anisotropic_inf_chunk,
        )

    def test_test_crosscov_anisotropic_gen(self):
        self._compare_tensors(
            self.test_crosscov_anisotropic_gen,
            self.test_crosscov_anisotropic_gen_chunk,
        )


class MuyGPSTestCase(KernelTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTestCase, cls).setUpClass()
        if rank == 0:
            cls.batch_homoscedastic_covariance_gen = homoscedastic_perturb_n(
                cls.batch_covariance_gen, cls.muygps_gen.noise()
            )
            cls.batch_prediction = cls.muygps_gen.posterior_mean(
                cls.batch_homoscedastic_covariance_gen,
                cls.batch_crosscov_gen,
                cls.batch_nn_targets,
            )
            cls.batch_variance = cls.muygps_gen.posterior_variance(
                cls.batch_homoscedastic_covariance_gen,
                cls.batch_crosscov_gen,
            )
        else:
            cls.batch_homoscedastic_covariance_gen = None
            cls.batch_prediction = None
            cls.batch_variance = None

        cls.batch_homoscedastic_covariance_gen_chunk = homoscedastic_perturb_m(
            cls.batch_covariance_gen_chunk, cls.muygps_gen.noise()
        )
        cls.batch_prediction_chunk = cls.muygps_gen.posterior_mean(
            cls.batch_homoscedastic_covariance_gen_chunk,
            cls.batch_crosscov_gen_chunk,
            cls.batch_nn_targets_chunk,
        )
        cls.batch_variance_chunk = cls.muygps_gen.posterior_variance(
            cls.batch_homoscedastic_covariance_gen_chunk,
            cls.batch_crosscov_gen_chunk,
        )


class MuyGPSTest(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(MuyGPSTest, cls).setUpClass()

    def test_homoscedastic_perturb(self):
        self._compare_tensors(
            self.batch_homoscedastic_covariance_gen,
            self.batch_homoscedastic_covariance_gen_chunk,
        )

    def test_batch_posterior_mean(self):
        self._compare_tensors(
            self.batch_prediction, self.batch_prediction_chunk
        )

    def test_batch_diagonal_variance(self):
        self._compare_tensors(self.batch_variance, self.batch_variance_chunk)

    def test_scale_optim(self):
        parallel_scale = analytic_scale_optim_m(
            self.batch_homoscedastic_covariance_gen_chunk,
            self.batch_nn_targets_chunk,
        )

        if rank == 0:
            serial_scale = analytic_scale_optim_n(
                self.batch_homoscedastic_covariance_gen,
                self.batch_nn_targets,
            )
            self.assertAlmostEqual(serial_scale, parallel_scale)


class OptimTestCase(MuyGPSTestCase):
    @classmethod
    def setUpClass(cls):
        super(OptimTestCase, cls).setUpClass()
        cls.x0_names, cls.x0, bounds = cls.muygps_gen.get_opt_params()
        cls.x0_map = {n: cls.x0[i] for i, n in enumerate(cls.x0_names)}
        cls.sopt_kwargs = {"verbose": False}
        cls.bopt_kwargs = {
            "verbose": False,
            "random_state": 1,
            "init_points": 5,
            "n_iter": 5,
            "allow_duplicate_points": True,
        }

    def _get_scale_fn_n(self):
        return self.muygps_gen.scale.get_opt_fn(self.muygps_gen)

    def _get_scale_fn_m(self):
        return self.muygps_gen.scale.get_opt_fn(self.muygps_gen)

    # Numpy objective functions
    def _get_obj_fn_n(self):
        return make_loo_crossval_fn(
            mse_fn_n,
            self.muygps_gen.kernel.get_opt_fn(),
            self.muygps_gen.get_opt_mean_fn(),
            self.muygps_gen.get_opt_var_fn(),
            self._get_scale_fn_n(),
            self.batch_pairwise_dists,
            self.batch_crosswise_dists,
            self.batch_nn_targets,
            self.batch_targets,
        )

    def _get_obj_fn_anisotropic_n(self):
        return make_loo_crossval_fn(
            mse_fn_n,
            self.muygps_anisotropic_gen.kernel.get_opt_fn(),
            self.muygps_anisotropic_gen.get_opt_mean_fn(),
            self.muygps_anisotropic_gen.get_opt_var_fn(),
            self._get_scale_fn_n(),
            self.batch_pairwise_diffs,
            self.batch_crosswise_diffs,
            self.batch_nn_targets,
            self.batch_targets,
        )

    # MPI objective functions
    def _get_obj_fn_m(self):
        return make_loo_crossval_fn(
            mse_fn_m,
            self.muygps_gen.kernel.get_opt_fn(),
            self.muygps_gen.get_opt_mean_fn(),
            self.muygps_gen.get_opt_var_fn(),
            self._get_scale_fn_m(),
            self.batch_pairwise_dists_chunk,
            self.batch_crosswise_dists_chunk,
            self.batch_nn_targets_chunk,
            self.batch_targets_chunk,
        )

    def _get_obj_fn_anisotropic_m(self):
        return make_loo_crossval_fn(
            mse_fn_m,
            self.muygps_anisotropic_gen.kernel.get_opt_fn(),
            self.muygps_anisotropic_gen.get_opt_mean_fn(),
            self.muygps_anisotropic_gen.get_opt_var_fn(),
            self._get_scale_fn_m(),
            self.batch_pairwise_diffs_chunk,
            self.batch_crosswise_diffs_chunk,
            self.batch_nn_targets_chunk,
            self.batch_targets_chunk,
        )


class CrossEntropyTest(parameterized.TestCase):
    @parameterized.parameters((10, 10, f, r) for f in [100] for r in [2])
    def test_cross_entropy(
        self, train_count, test_count, feature_count, response_count
    ):
        if rank == 0:
            train, test = _make_gaussian_data(
                train_count,
                test_count,
                feature_count,
                response_count,
                categorical=True,
            )
            targets = train["output"]
            predictions = test["output"]
            serial_cross_entropy = cross_entropy_fn_n(predictions, targets)
        else:
            targets = None
            predictions = None
            serial_cross_entropy = None

        targets_chunk = _chunk_tensor(targets)
        predictions_chunk = _chunk_tensor(predictions)

        parallel_cross_entropy = cross_entropy_fn_m(
            predictions_chunk, targets_chunk
        )
        if rank == 0:
            self.assertAlmostEqual(serial_cross_entropy, parallel_cross_entropy)


class LossTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(LossTest, cls).setUpClass()
        cls.batch_scale = cls.muygps_gen.scale()

    def test_mse(self):
        parallel_mse = mse_fn_m(
            self.batch_prediction_chunk, self.batch_targets_chunk
        )

        if rank == 0:
            serial_mse = mse_fn_n(self.batch_prediction, self.batch_targets)
            self.assertAlmostEqual(serial_mse, parallel_mse)

    @parameterized.parameters(bs for bs in [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_pseudo_huber(self, boundary_scale):
        parallel_pseudo_huber = pseudo_huber_fn_m(
            self.batch_prediction_chunk,
            self.batch_targets_chunk,
            boundary_scale,
        )
        if rank == 0:
            serial_pseduo_huber = pseudo_huber_fn_n(
                self.batch_prediction, self.batch_targets, boundary_scale
            )
            self.assertAlmostEqual(serial_pseduo_huber, parallel_pseudo_huber)

    @parameterized.parameters(bs for bs in [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_looph(self, boundary_scale):
        parallel_looph = looph_fn_m(
            self.batch_prediction_chunk,
            self.batch_targets_chunk,
            self.batch_variance_chunk,
            self.batch_scale,
            boundary_scale=boundary_scale,
        )
        if rank == 0:
            serial_looph = looph_fn_n(
                self.batch_prediction,
                self.batch_targets,
                self.batch_variance,
                self.batch_scale,
                boundary_scale=boundary_scale,
            )
            self.assertAlmostEqual(serial_looph, parallel_looph)

    def test_lool(self):
        parallel_lool = lool_fn_m(
            self.batch_prediction_chunk,
            self.batch_targets_chunk,
            self.batch_variance_chunk,
            self.batch_scale,
        )

        if rank == 0:
            serial_lool = lool_fn_n(
                self.batch_prediction,
                self.batch_targets,
                self.batch_variance,
                self.batch_scale,
            )
            self.assertAlmostEqual(serial_lool, parallel_lool)


class ObjectiveTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ObjectiveTest, cls).setUpClass()

    def test_kernel_fn(self):
        if rank == 0:
            kernel = self.muygps_gen.kernel.get_opt_fn()(
                self.batch_pairwise_dists, **self.x0_map
            )
        else:
            kernel = None

        kernel_chunk = self.muygps_gen.kernel.get_opt_fn()(
            self.batch_pairwise_dists_chunk, **self.x0_map
        )

        self._compare_tensors(kernel, kernel_chunk)

    def test_mean_fn(self):
        if rank == 0:
            mean_fn_n = self.muygps_gen.get_opt_mean_fn()
            mean = mean_fn_n(
                self.batch_covariance_gen,
                self.batch_crosscov_gen,
                self.batch_nn_targets,
                **self.x0_map,
            )
        else:
            mean = None

        mean_fn_m = self.muygps_gen.get_opt_mean_fn()
        mean_chunk = mean_fn_m(
            self.batch_covariance_gen_chunk,
            self.batch_crosscov_gen_chunk,
            self.batch_nn_targets_chunk,
            **self.x0_map,
        )

        self._compare_tensors(mean, mean_chunk)

    def test_var_fn(self):
        if rank == 0:
            var_fn_n = self.muygps_gen.get_opt_var_fn()
            var = var_fn_n(
                self.batch_covariance_gen,
                self.batch_crosscov_gen,
                **self.x0_map,
            )
        else:
            var = None

        var_fn_m = self.muygps_gen.get_opt_var_fn()
        var_chunk = var_fn_m(
            self.batch_covariance_gen_chunk,
            self.batch_crosscov_gen_chunk,
            **self.x0_map,
        )

        self._compare_tensors(var, var_chunk)

    def test_loo_crossval(self):
        obj_fn_m = self._get_obj_fn_m()
        obj_m = obj_fn_m(**self.x0_map)

        if rank == 0:
            obj_fn_n = self._get_obj_fn_n()
            obj_n = obj_fn_n(**self.x0_map)
            self.assertAlmostEqual(obj_n, obj_m)


# Note[MWP]: this is breaking for reasons unknown. will need to revisit
# class ScaleTest(OptimTestCase):
#     @classmethod
#     def setUpClass(cls):
#         super(ScaleTest, cls).setUpClass()

#     def test_scale_fn(self):
#         if rank == 0:
#             ss_fn_n = self._get_scale_fn_n()
#             ss = ss_fn_n(
#                 self.batch_covariance_gen,
#                 self.batch_nn_targets,
#                 **self.x0_map,
#             )
#         else:
#             ss = None

#         ss_fn_m = self._get_scale_fn_m()
#         ss_chunk = ss_fn_m(
#             self.batch_covariance_gen_chunk,
#             self.batch_nn_targets_chunk,
#             **self.x0_map,
#         )

#         self._compare_tensors(ss, ss_chunk)


class ScipyOptimTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(ScipyOptimTest, cls).setUpClass()

    def test_scipy_optimize(self):
        obj_fn_m = self._get_obj_fn_m()
        opt_m = scipy_optimize_m(self.muygps_gen, obj_fn_m, **self.sopt_kwargs)

        if rank == 0:
            obj_fn_n = self._get_obj_fn_n()
            opt_n = scipy_optimize_n(
                self.muygps_gen, obj_fn_n, **self.sopt_kwargs
            )
            self.assertAlmostEqual(
                opt_m.kernel.smoothness(), opt_n.kernel.smoothness()
            )


class BayesOptimTest(OptimTestCase):
    @classmethod
    def setUpClass(cls):
        super(BayesOptimTest, cls).setUpClass()

    def test_bayes_optimize(self):
        obj_fn_m = self._get_obj_fn_m()
        model_m = bayes_optimize_m(
            self.muygps_gen, obj_fn_m, **self.bopt_kwargs
        )

        if rank == 0:
            obj_fn_n = self._get_obj_fn_n()
            model_n = bayes_optimize_n(
                self.muygps_gen, obj_fn_n, **self.bopt_kwargs
            )
            self.assertAlmostEqual(
                model_m.kernel.smoothness(), model_n.kernel.smoothness()
            )


if __name__ == "__main__":
    absltest.main()
