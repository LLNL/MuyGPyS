# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config

# config.jax_enable_x64()

if config.jax_enabled() is True:
    import jax.numpy as jnp
    import numpy as np

    from absl.testing import absltest
    from absl.testing import parameterized

    from MuyGPyS._test.utils import (
        _make_gaussian_matrix,
        _make_gaussian_data,
        _exact_nn_kwarg_options,
    )
    from MuyGPyS.gp.muygps import MuyGPS
    from MuyGPyS.neighbors import NN_Wrapper
    from MuyGPyS.optimize.batch import sample_batch

    def allclose_gen(a: np.ndarray, b: np.ndarray) -> bool:
        if config.x64_enabled():
            return np.allclose(a, b)
        else:
            return np.allclose(a, b, atol=1e-7)

    def allclose_var(a: np.ndarray, b: np.ndarray) -> bool:
        if config.x64_enabled():
            return np.allclose(a, b)
        else:
            return np.allclose(a, b, atol=1e-6)

    def allclose_inv(a: np.ndarray, b: np.ndarray) -> bool:
        if config.x64_enabled():
            return np.allclose(a, b)
        else:
            return np.allclose(a, b, atol=1e-3)

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
            cls.nu = 0.55
            cls.nu_bounds = (1e-1, 1e1)
            cls.eps = 1e-3
            cls.k_kwargs = {
                "kern": "matern",
                "length_scale": {"val": cls.length_scale},
                "nu": {"val": cls.nu, "bounds": cls.nu_bounds},
                "eps": {"val": cls.eps},
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
            cls.batch_indices_n, cls.batch_nn_indices_n = sample_batch(
                cls.nbrs_lookup, cls.batch_count, cls.train_count
            )
            cls.batch_indices_j = jnp.array(cls.batch_indices_n)
            cls.batch_nn_indices_j = jnp.array(cls.batch_nn_indices_n)

    from MuyGPyS._src.gp.numpy_distance import (
        _pairwise_distances as pairwise_distances_n,
        _crosswise_distances as crosswise_distances_n,
        _make_train_tensors as make_train_tensors_n,
    )
    from MuyGPyS._src.gp.jax_distance import (
        _pairwise_distances as pairwise_distances_j,
        _crosswise_distances as crosswise_distances_j,
        _make_train_tensors as make_train_tensors_j,
    )

    class DistanceTest(DistanceTestCase):
        @classmethod
        def setUpClass(cls):
            super(DistanceTest, cls).setUpClass()

        def test_pairwise_distances(self):
            self.assertTrue(
                allclose_gen(
                    pairwise_distances_n(
                        self.train_features_n, self.batch_nn_indices_n
                    ),
                    pairwise_distances_j(
                        self.train_features_j, self.batch_nn_indices_j
                    ),
                )
            )

        def test_crosswise_distances(self):
            self.assertTrue(
                allclose_gen(
                    crosswise_distances_n(
                        self.train_features_n,
                        self.train_features_n,
                        self.batch_indices_n,
                        self.batch_nn_indices_n,
                    ),
                    crosswise_distances_j(
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
            ) = make_train_tensors_n(
                self.muygps.kernel.metric,
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
            ) = make_train_tensors_j(
                self.muygps.kernel.metric,
                self.batch_indices_j,
                self.batch_nn_indices_j,
                self.train_features_j,
                self.train_responses_j,
            )
            self.assertTrue(allclose_gen(crosswise_dists_n, crosswise_dists_j))
            self.assertTrue(allclose_gen(pairwise_dists_n, pairwise_dists_j))
            self.assertTrue(allclose_gen(batch_targets_n, batch_targets_j))
            self.assertTrue(
                allclose_gen(batch_nn_targets_n, batch_nn_targets_j)
            )

    from MuyGPyS._src.gp.numpy_kernels import (
        _rbf_fn as rbf_fn_n,
        _matern_05_fn as matern_05_fn_n,
        _matern_15_fn as matern_15_fn_n,
        _matern_25_fn as matern_25_fn_n,
        _matern_inf_fn as matern_inf_fn_n,
        _matern_gen_fn as matern_gen_fn_n,
    )
    from MuyGPyS._src.gp.jax_kernels import (
        _rbf_fn as rbf_fn_j,
        _matern_05_fn as matern_05_fn_j,
        _matern_15_fn as matern_15_fn_j,
        _matern_25_fn as matern_25_fn_j,
        _matern_inf_fn as matern_inf_fn_j,
        _matern_gen_fn as matern_gen_fn_j,
    )

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
                cls.crosswise_dists_j,
                cls.pairwise_dists_j,
                cls.batch_targets_j,
                cls.batch_nn_targets_j,
            ) = make_train_tensors_j(
                cls.muygps.kernel.metric,
                cls.batch_indices_j,
                cls.batch_nn_indices_j,
                cls.train_features_j,
                cls.train_responses_j,
            )

    class KernelTest(KernelTestCase):
        @classmethod
        def setUpClass(cls):
            super(KernelTest, cls).setUpClass()

        def test_crosswise_rbf(self):
            self.assertTrue(
                allclose_gen(
                    rbf_fn_n(
                        self.crosswise_dists_n, length_scale=self.length_scale
                    ),
                    rbf_fn_j(
                        self.crosswise_dists_j, length_scale=self.length_scale
                    ),
                )
            )

        def test_pairwise_rbf(self):
            self.assertTrue(
                allclose_gen(
                    rbf_fn_n(
                        self.pairwise_dists_n, length_scale=self.length_scale
                    ),
                    rbf_fn_j(
                        self.pairwise_dists_j, length_scale=self.length_scale
                    ),
                )
            )

        def test_crosswise_matern(self):
            self.assertTrue(
                allclose_gen(
                    matern_05_fn_n(
                        self.crosswise_dists_n, length_scale=self.length_scale
                    ),
                    matern_05_fn_j(
                        self.crosswise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_15_fn_n(
                        self.crosswise_dists_n, length_scale=self.length_scale
                    ),
                    matern_15_fn_j(
                        self.crosswise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_25_fn_n(
                        self.crosswise_dists_n, length_scale=self.length_scale
                    ),
                    matern_25_fn_j(
                        self.crosswise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_inf_fn_n(
                        self.crosswise_dists_n, length_scale=self.length_scale
                    ),
                    matern_inf_fn_j(
                        self.crosswise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_gen_fn_n(
                        self.crosswise_dists_n,
                        nu=self.nu,
                        length_scale=self.length_scale,
                    ),
                    matern_gen_fn_j(
                        self.crosswise_dists_j,
                        nu=self.nu,
                        length_scale=self.length_scale,
                    ),
                )
            )

        def test_pairwise_matern(self):
            self.assertTrue(
                allclose_gen(
                    matern_05_fn_n(
                        self.pairwise_dists_n, length_scale=self.length_scale
                    ),
                    matern_05_fn_j(
                        self.pairwise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_15_fn_n(
                        self.pairwise_dists_n, length_scale=self.length_scale
                    ),
                    matern_15_fn_j(
                        self.pairwise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_25_fn_n(
                        self.pairwise_dists_n, length_scale=self.length_scale
                    ),
                    matern_25_fn_j(
                        self.pairwise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_inf_fn_n(
                        self.pairwise_dists_n, length_scale=self.length_scale
                    ),
                    matern_inf_fn_j(
                        self.pairwise_dists_j, length_scale=self.length_scale
                    ),
                )
            )
            self.assertTrue(
                allclose_gen(
                    matern_gen_fn_n(
                        self.pairwise_dists_n,
                        nu=self.nu,
                        length_scale=self.length_scale,
                    ),
                    matern_gen_fn_j(
                        self.pairwise_dists_j,
                        nu=self.nu,
                        length_scale=self.length_scale,
                    ),
                )
            )

    from MuyGPyS._src.gp.numpy_muygps import (
        _muygps_compute_solve as muygps_compute_solve_n,
        _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_n,
        _muygps_sigma_sq_optim as muygps_sigma_sq_optim_n,
    )
    from MuyGPyS._src.gp.jax_muygps import (
        _muygps_compute_solve as muygps_compute_solve_j,
        _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_j,
        _muygps_sigma_sq_optim as muygps_sigma_sq_optim_j,
    )

    class MuyGPSTestCase(KernelTestCase):
        @classmethod
        def setUpClass(cls):
            super(MuyGPSTestCase, cls).setUpClass()
            cls.K_n = matern_gen_fn_n(
                cls.pairwise_dists_n, nu=cls.nu, length_scale=cls.length_scale
            )
            cls.K_j = jnp.array(cls.K_n)
            cls.Kcross_n = matern_gen_fn_n(
                cls.crosswise_dists_n, nu=cls.nu, length_scale=cls.length_scale
            )
            cls.Kcross_j = jnp.array(cls.Kcross_n)

    class MuyGPSTest(MuyGPSTestCase):
        @classmethod
        def setUpClass(cls):
            super(MuyGPSTest, cls).setUpClass()

        def test_compute_solve(self):
            self.assertTrue(
                allclose_inv(
                    muygps_compute_solve_n(
                        self.K_n,
                        self.Kcross_n,
                        self.batch_nn_targets_n,
                        self.muygps.eps(),
                    ),
                    muygps_compute_solve_j(
                        self.K_j,
                        self.Kcross_j,
                        self.batch_nn_targets_j,
                        self.muygps.eps(),
                    ),
                )
            )

        def test_diagonal_variance(self):
            self.assertTrue(
                allclose_var(
                    muygps_compute_diagonal_variance_n(
                        self.K_n, self.Kcross_n, self.muygps.eps()
                    ),
                    muygps_compute_diagonal_variance_j(
                        self.K_j, self.Kcross_j, self.muygps.eps()
                    ),
                )
            )

        def test_sigma_sq_optim(self):
            self.assertTrue(
                allclose_inv(
                    muygps_sigma_sq_optim_n(
                        self.K_n,
                        self.batch_nn_indices_n,
                        self.train_responses_n,
                        self.muygps.eps(),
                    ),
                    muygps_sigma_sq_optim_j(
                        self.K_j,
                        self.batch_nn_indices_j,
                        self.train_responses_j,
                        self.muygps.eps(),
                    ),
                )
            )

    from MuyGPyS._src.optimize.numpy_objective import (
        _mse_fn as mse_fn_n,
        _cross_entropy_fn as cross_entropy_fn_n,
    )
    from MuyGPyS._src.optimize.jax_objective import (
        _mse_fn as mse_fn_j,
        _cross_entropy_fn as cross_entropy_fn_j,
    )
    from MuyGPyS.optimize.objective import make_loo_crossval_fn
    from MuyGPyS._src.optimize.numpy_chassis import (
        _scipy_optimize_from_tensors as scipy_optimize_from_tensors_n,
    )
    from MuyGPyS._src.optimize.jax_chassis import (
        _scipy_optimize_from_tensors as scipy_optimize_from_tensors_j,
    )

    class OptimTestCase(MuyGPSTestCase):
        @classmethod
        def setUpClass(cls):
            super(OptimTestCase, cls).setUpClass()
            cls.predictions_n = cls.muygps._regress(
                cls.K_n,
                cls.Kcross_n,
                cls.batch_nn_targets_n,
                cls.muygps.eps(),
                cls.muygps.sigma_sq(),
            )
            cls.predictions_j = jnp.array(cls.predictions_n)
            cls.x0_names, cls.x0_n, cls.bounds = cls.muygps.get_optim_params()
            cls.x0_j = jnp.array(cls.x0_n)

        def _get_kernel_fn_n(self):
            return self.muygps.kernel._get_opt_fn(
                matern_05_fn_n,
                matern_15_fn_n,
                matern_25_fn_n,
                matern_inf_fn_n,
                matern_gen_fn_n,
                self.muygps.kernel.nu,
                self.muygps.kernel.length_scale,
            )

        def _get_kernel_fn_j(self):
            return self.muygps.kernel._get_opt_fn(
                matern_05_fn_j,
                matern_15_fn_j,
                matern_25_fn_j,
                matern_inf_fn_j,
                matern_gen_fn_j,
                self.muygps.kernel.nu,
                self.muygps.kernel.length_scale,
            )

        def _get_predict_fn_n(self):
            return self.muygps._get_opt_fn(
                muygps_compute_solve_n, self.muygps.eps
            )

        def _get_predict_fn_j(self):
            return self.muygps._get_opt_fn(
                muygps_compute_solve_j, self.muygps.eps
            )

        def _get_obj_fn_n(self):
            return make_loo_crossval_fn(
                mse_fn_n,
                self._get_kernel_fn_n(),
                self._get_predict_fn_n(),
                self.pairwise_dists_n,
                self.crosswise_dists_n,
                self.batch_nn_targets_n,
                self.batch_targets_n,
            )

        def _get_obj_fn_j(self):
            return make_loo_crossval_fn(
                mse_fn_j,
                self._get_kernel_fn_j(),
                self._get_predict_fn_j(),
                self.pairwise_dists_j,
                self.crosswise_dists_j,
                self.batch_nn_targets_j,
                self.batch_targets_j,
            )

        def _get_obj_fn_h(self):
            return make_loo_crossval_fn(
                mse_fn_j,
                self._get_kernel_fn_j(),
                self._get_predict_fn_n(),
                self.pairwise_dists_j,
                self.crosswise_dists_j,
                self.batch_nn_targets_j,
                self.batch_targets_j,
            )

    class ObjectiveTest(OptimTestCase):
        @classmethod
        def setUpClass(cls):
            super(ObjectiveTest, cls).setUpClass()

        def test_mse(self):
            self.assertTrue(
                np.isclose(
                    mse_fn_n(self.predictions_n, self.batch_targets_n),
                    mse_fn_j(self.predictions_j, self.batch_targets_j),
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
                    kernel_fn_n(self.pairwise_dists_n, self.x0_n),
                    kernel_fn_j(self.pairwise_dists_j, self.x0_j),
                )
            )

        def test_predict_fn(self):
            predict_fn_n = self._get_predict_fn_n()
            predict_fn_j = self._get_predict_fn_j()
            self.assertTrue(
                allclose_inv(
                    predict_fn_n(
                        self.K_n,
                        self.Kcross_n,
                        self.batch_nn_targets_n,
                        self.x0_n[-1],
                    ),
                    predict_fn_j(
                        self.K_j,
                        self.Kcross_j,
                        self.batch_nn_targets_j,
                        self.x0_j[-1],
                    ),
                )
            )

        def test_loo_crossval(self):
            obj_fn_n = self._get_obj_fn_n()
            obj_fn_j = self._get_obj_fn_j()
            obj_fn_h = self._get_obj_fn_h()
            self.assertTrue(
                allclose_inv(obj_fn_n(self.x0_n), obj_fn_j(self.x0_j))
            )
            self.assertTrue(
                allclose_inv(obj_fn_n(self.x0_n), obj_fn_h(self.x0_j))
            )

    class OptimTest(OptimTestCase):
        @classmethod
        def setUpClass(cls):
            super(OptimTest, cls).setUpClass()

        def test_scipy_optimize_from_tensors(self):
            obj_fn_n = self._get_obj_fn_n()
            obj_fn_j = self._get_obj_fn_j()
            obj_fn_h = self._get_obj_fn_h()

            mopt_n = scipy_optimize_from_tensors_n(
                self.muygps,
                obj_fn_n,
                verbose=False,
            )
            mopt_j = scipy_optimize_from_tensors_j(
                self.muygps,
                obj_fn_j,
                verbose=False,
            )
            mopt_h = scipy_optimize_from_tensors_j(
                self.muygps,
                obj_fn_h,
                verbose=False,
            )
            self.assertTrue(
                allclose_gen(mopt_n.kernel.nu(), mopt_j.kernel.nu())
            )
            self.assertTrue(
                allclose_gen(mopt_n.kernel.nu(), mopt_h.kernel.nu())
            )

    if __name__ == "__main__":
        absltest.main()

else:
    if __name__ == "__main__":
        pass
