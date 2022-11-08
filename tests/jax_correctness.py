# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config, jax_config

config.parse_flags_with_absl()  # Affords option setting from CLI


if config.muygpys_jax_enabled is True:  # type: ignore
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
    from MuyGPyS._src.gp.distance.numpy import (
        _pairwise_distances as pairwise_distances_n,
        _crosswise_distances as crosswise_distances_n,
        _make_train_tensors as make_train_tensors_n,
        _make_fast_regress_tensors as make_fast_regress_tensors_n,
    )
    from MuyGPyS._src.gp.distance.jax import (
        _pairwise_distances as pairwise_distances_j,
        _crosswise_distances as crosswise_distances_j,
        _make_train_tensors as make_train_tensors_j,
        _make_fast_regress_tensors as make_fast_regress_tensors_j,
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
        _muygps_compute_solve as muygps_compute_solve_n,
        _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_n,
    )
    from MuyGPyS._src.gp.muygps.jax import (
        _muygps_compute_solve as muygps_compute_solve_j,
        _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_j,
    )
    from MuyGPyS._src.optimize.sigma_sq.numpy import (
        _analytic_sigma_sq_optim as analytic_sigma_sq_optim_n,
    )
    from MuyGPyS._src.optimize.sigma_sq.jax import (
        _analytic_sigma_sq_optim as analytic_sigma_sq_optim_j,
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
    from MuyGPyS._src.optimize.loss.jax import (
        _mse_fn as mse_fn_j,
        _cross_entropy_fn as cross_entropy_fn_j,
        _lool_fn as lool_fn_j,
    )
    from MuyGPyS.optimize.objective import make_loo_crossval_fn
    from MuyGPyS._src.optimize.chassis.numpy import (
        _scipy_optimize as scipy_optimize_n,
        _bayes_opt_optimize as bayes_optimize_n,
    )
    from MuyGPyS._src.optimize.chassis.jax import (
        _scipy_optimize as scipy_optimize_j,
        _bayes_opt_optimize as bayes_optimize_j,
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
                    analytic_sigma_sq_optim_n(
                        self.K_n,
                        self.batch_nn_targets_n,
                        self.muygps.eps(),
                    ),
                    analytic_sigma_sq_optim_j(
                        self.K_j,
                        self.batch_nn_targets_j,
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
            cls.fast_regress_coeffs_n = cls.muygps._build_fast_regress_coeffs(
                cls.K_fast_n, cls.muygps.eps(), cls.train_nn_targets_fast_n
            )
            # cls.fast_regress_coeffs_n = cls.muygps.build_fast_regress_coeffs(
            #     cls.train_features_n,
            #     cls.nn_indices_all_n,
            #     cls.train_responses_n,
            # )

            cls.test_neighbors_n, _ = cls.nbrs_lookup.get_nns(
                cls.test_features_n
            )
            cls.closest_neighbor_n = cls.test_neighbors_n[:, 0]
            cls.closest_set_n = cls.nn_indices_all_n[cls.closest_neighbor_n]

            cls.nn_indices_with_self_n = np.zeros(
                (cls.train_count, cls.nn_count + 1)
            )
            cls.nn_indices_with_self_n[
                :, 1 : cls.nn_count + 1
            ] = cls.nn_indices_all_n
            cls.nn_indices_with_self_n[:, 0] = np.arange(0, cls.train_count)
            cls.new_nn_indices_n = cls.nn_indices_with_self_n[:, :-1]

            cls.closest_set_new_n = cls.new_nn_indices_n[cls.closest_neighbor_n]
            cls.crosswise_dists_fast_n = crosswise_distances_n(
                cls.test_features_n,
                cls.train_features_n,
                np.arange(0, cls.test_count),
                cls.closest_set_new_n,
            )
            cls.Kcross_fast_n = cls.muygps.kernel(cls.crosswise_dists_fast_n)

            cls.nn_indices_all_j, _ = cls.nbrs_lookup.get_batch_nns(
                jnp.arange(0, cls.train_count)
            )
            cls.nn_indices_all_j = jnp.array(cls.nn_indices_all_j)

            (
                cls.K_fast_j,
                cls.train_nn_targets_fast_j,
            ) = make_fast_regress_tensors_j(
                cls.muygps.kernel.metric,
                cls.nn_indices_all_j,
                cls.train_features_j,
                cls.train_responses_j,
            )
            cls.fast_regress_coeffs_j = cls.muygps._build_fast_regress_coeffs(
                cls.K_fast_j, cls.muygps.eps(), cls.train_nn_targets_fast_j
            )

            # cls.fast_regress_coeffs_j = cls.muygps.build_fast_regress_coeffs(
            #     cls.train_features_j,
            #     cls.nn_indices_all_j,
            #     cls.train_responses_j,
            # )

            cls.test_neighbors_j, _ = cls.nbrs_lookup.get_nns(
                cls.test_features_j
            )
            cls.closest_neighbor_j = cls.test_neighbors_j[:, 0]
            cls.closest_set_j = cls.nn_indices_all_j[cls.closest_neighbor_j]

            print(cls.nn_indices_all_j.shape)
            cls.new_nn_indices_j = jnp.concatenate(
                (
                    jnp.expand_dims(jnp.arange(0, cls.train_count), axis=1),
                    cls.nn_indices_all_j[:, :-1],
                ),
                axis=1,
            )
            cls.closest_set_new_j = cls.new_nn_indices_j[cls.closest_neighbor_j]
            cls.crosswise_dists_fast_j = crosswise_distances_j(
                cls.test_features_j,
                cls.train_features_j,
                jnp.arange(0, cls.test_count),
                cls.closest_set_new_j,
            )
            cls.Kcross_fast_j = cls.muygps.kernel(cls.crosswise_dists_fast_j)

        def test_fast_predict(self):
            self.assertTrue(
                allclose_inv(
                    self.muygps._fast_regress(
                        self.Kcross_fast_n,
                        self.fast_regress_coeffs_n[self.closest_neighbor_n, :],
                    ),
                    self.muygps._fast_regress(
                        self.Kcross_fast_j,
                        self.fast_regress_coeffs_j[self.closest_neighbor_j, :],
                    ),
                )
            )

    class OptimTestCase(MuyGPSTestCase):
        @classmethod
        def setUpClass(cls):
            super(OptimTestCase, cls).setUpClass()
            cls.predictions_n, cls.variances_n = cls.muygps._regress(
                cls.K_n,
                cls.Kcross_n,
                cls.batch_nn_targets_n,
                cls.muygps.eps(),
                cls.muygps.sigma_sq(),
                variance_mode="diagonal",
                apply_sigma_sq=False,
            )
            cls.predictions_j = jnp.array(cls.predictions_n)
            cls.variances_j = jnp.array(cls.variances_n)
            cls.x0_names, cls.x0_n, cls.bounds = cls.muygps.get_optim_params()
            cls.x0_j = jnp.array(cls.x0_n)
            cls.x0_map_n = {n: cls.x0_n[i] for i, n in enumerate(cls.x0_names)}
            cls.x0_map_j = {n: cls.x0_j[i] for i, n in enumerate(cls.x0_names)}

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

        def _get_array_kernel_fn_j(self):
            return self.muygps.kernel._get_array_opt_fn(
                matern_05_fn_j,
                matern_15_fn_j,
                matern_25_fn_j,
                matern_inf_fn_j,
                matern_gen_fn_j,
                self.muygps.kernel.nu,
                self.muygps.kernel.length_scale,
            )

        def _get_kwargs_kernel_fn_j(self):
            return self.muygps.kernel._get_kwargs_opt_fn(
                matern_05_fn_j,
                matern_15_fn_j,
                matern_25_fn_j,
                matern_inf_fn_j,
                matern_gen_fn_j,
                self.muygps.kernel.nu,
                self.muygps.kernel.length_scale,
            )

        def _get_array_mean_fn_n(self):
            return self.muygps._get_array_opt_mean_fn(
                muygps_compute_solve_n, self.muygps.eps
            )

        def _get_kwargs_mean_fn_n(self):
            return self.muygps._get_kwargs_opt_mean_fn(
                muygps_compute_solve_n, self.muygps.eps
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

        def _get_array_mean_fn_j(self):
            return self.muygps._get_array_opt_mean_fn(
                muygps_compute_solve_j, self.muygps.eps
            )

        def _get_kwargs_mean_fn_j(self):
            return self.muygps._get_kwargs_opt_mean_fn(
                muygps_compute_solve_j, self.muygps.eps
            )

        def _get_array_var_fn_j(self):
            return self.muygps._get_array_opt_var_fn(
                muygps_compute_diagonal_variance_j, self.muygps.eps
            )

        def _get_kwargs_var_fn_j(self):
            return self.muygps._get_kwargs_opt_var_fn(
                muygps_compute_diagonal_variance_j, self.muygps.eps
            )

        def _get_array_sigma_sq_fn_j(self):
            return make_array_analytic_sigma_sq_optim(
                self.muygps, analytic_sigma_sq_optim_j
            )

        def _get_kwargs_sigma_sq_fn_j(self):
            return make_kwargs_analytic_sigma_sq_optim(
                self.muygps, analytic_sigma_sq_optim_j
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

        def _get_array_obj_fn_j(self):
            return make_loo_crossval_fn(
                "scipy",
                "mse",
                mse_fn_j,
                self._get_array_kernel_fn_j(),
                self._get_array_mean_fn_j(),
                self._get_array_var_fn_j(),
                self._get_array_sigma_sq_fn_j(),
                self.pairwise_dists_j,
                self.crosswise_dists_j,
                self.batch_nn_targets_j,
                self.batch_targets_j,
            )

        def _get_kwargs_obj_fn_j(self):
            return make_loo_crossval_fn(
                "bayes",
                "mse",
                mse_fn_j,
                self._get_kwargs_kernel_fn_j(),
                self._get_kwargs_mean_fn_j(),
                self._get_kwargs_var_fn_j(),
                self._get_kwargs_sigma_sq_fn_j(),
                self.pairwise_dists_j,
                self.crosswise_dists_j,
                self.batch_nn_targets_j,
                self.batch_targets_j,
            )

        def _get_array_obj_fn_h(self):
            return make_loo_crossval_fn(
                "scipy",
                "mse",
                mse_fn_j,
                self._get_array_kernel_fn_j(),
                self._get_array_mean_fn_n(),
                self._get_array_var_fn_n(),
                self._get_array_sigma_sq_fn_n(),
                self.pairwise_dists_j,
                self.crosswise_dists_j,
                self.batch_nn_targets_j,
                self.batch_targets_j,
            )

        def _get_kwargs_obj_fn_h(self):
            return make_loo_crossval_fn(
                "bayes",
                "mse",
                mse_fn_j,
                self._get_kwargs_kernel_fn_j(),
                self._get_kwargs_mean_fn_n(),
                self._get_kwargs_var_fn_n(),
                self._get_kwargs_sigma_sq_fn_n(),
                self.pairwise_dists_j,
                self.crosswise_dists_j,
                self.batch_nn_targets_j,
                self.batch_targets_j,
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
            kernel_fn_n = self._get_array_kernel_fn_n()
            kernel_fn_j = self._get_array_kernel_fn_j()
            self.assertTrue(
                allclose_gen(
                    kernel_fn_n(self.pairwise_dists_n, self.x0_n),
                    kernel_fn_j(self.pairwise_dists_j, self.x0_j),
                )
            )

        def test_kwargs_mean_fn(self):
            mean_fn_n = self._get_kwargs_mean_fn_n()
            mean_fn_j = self._get_kwargs_mean_fn_j()
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

        def test_array_mean_fn(self):
            mean_fn_n = self._get_array_mean_fn_n()
            mean_fn_j = self._get_array_mean_fn_j()
            self.assertTrue(
                allclose_inv(
                    mean_fn_n(
                        self.K_n,
                        self.Kcross_n,
                        self.batch_nn_targets_n,
                        self.x0_n[-1],
                    ),
                    mean_fn_j(
                        self.K_j,
                        self.Kcross_j,
                        self.batch_nn_targets_j,
                        self.x0_j[-1],
                    ),
                )
            )

        def test_kwargs_var_fn(self):
            var_fn_n = self._get_kwargs_var_fn_n()
            var_fn_j = self._get_kwargs_var_fn_j()
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

        def test_array_var_fn(self):
            var_fn_n = self._get_array_var_fn_n()
            var_fn_j = self._get_array_var_fn_j()
            self.assertTrue(
                allclose_inv(
                    var_fn_n(
                        self.K_n,
                        self.Kcross_n,
                        self.x0_n[-1],
                    ),
                    var_fn_j(
                        self.K_j,
                        self.Kcross_j,
                        self.x0_j[-1],
                    ),
                )
            )

        def test_kwargs_sigma_sq_fn(self):
            ss_fn_n = self._get_kwargs_sigma_sq_fn_n()
            ss_fn_j = self._get_kwargs_sigma_sq_fn_j()
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

        def test_array_sigma_sq_fn(self):
            ss_fn_n = self._get_array_sigma_sq_fn_n()
            ss_fn_j = self._get_array_sigma_sq_fn_j()
            self.assertTrue(
                allclose_inv(
                    ss_fn_n(
                        self.K_n,
                        self.batch_nn_targets_n,
                        self.x0_n[-1],
                    ),
                    ss_fn_j(
                        self.K_j,
                        self.batch_nn_targets_j,
                        self.x0_j[-1],
                    ),
                )
            )

        def test_loo_crossval(self):
            obj_fn_n = self._get_array_obj_fn_n()
            obj_fn_j = self._get_array_obj_fn_j()
            obj_fn_h = self._get_array_obj_fn_h()
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
            cls.sopt_kwargs = {"verbose": False}

        def test_scipy_optimize(self):
            obj_fn_n = self._get_array_obj_fn_n()
            obj_fn_j = self._get_array_obj_fn_j()
            obj_fn_h = self._get_array_obj_fn_h()

            mopt_n = scipy_optimize_n(self.muygps, obj_fn_n, **self.sopt_kwargs)
            mopt_j = scipy_optimize_j(self.muygps, obj_fn_j, **self.sopt_kwargs)
            mopt_h = scipy_optimize_j(self.muygps, obj_fn_h, **self.sopt_kwargs)
            self.assertTrue(
                allclose_gen(mopt_n.kernel.nu(), mopt_j.kernel.nu())
            )
            self.assertTrue(
                allclose_gen(mopt_n.kernel.nu(), mopt_h.kernel.nu())
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
            }

        def test_optimize(self):
            obj_fn_n = self._get_kwargs_obj_fn_n()
            obj_fn_j = self._get_kwargs_obj_fn_j()
            obj_fn_h = self._get_kwargs_obj_fn_h()

            mopt_n = bayes_optimize_n(self.muygps, obj_fn_n, **self.bopt_kwargs)
            mopt_j = bayes_optimize_j(self.muygps, obj_fn_j, **self.bopt_kwargs)
            mopt_h = bayes_optimize_j(self.muygps, obj_fn_h, **self.bopt_kwargs)
            self.assertTrue(
                allclose_inv(mopt_n.kernel.nu(), mopt_j.kernel.nu())
            )
            self.assertTrue(
                allclose_inv(mopt_n.kernel.nu(), mopt_h.kernel.nu())
            )

    if __name__ == "__main__":
        absltest.main()

else:
    if __name__ == "__main__":
        pass
