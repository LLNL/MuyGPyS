# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config

config.parse_flags_with_absl()  # Affords option setting from CLI


if config.muygpys_mpi_enabled is True:  # type: ignore
    from MuyGPyS._test.utils import (
        _make_gaussian_matrix,
        _make_gaussian_data,
        _exact_nn_kwarg_options,
    )
    from MuyGPyS.gp.muygps import MuyGPS
    from MuyGPyS.neighbors import NN_Wrapper
    from MuyGPyS.optimize.batch import sample_batch
    from MuyGPyS._src.gp.distance.numpy import (
        _make_regress_tensors as make_regress_tensors_n,
        _make_train_tensors as make_train_tensors_n,
    )
    from MuyGPyS._src.gp.distance.mpi import (
        _make_regress_tensors as make_regress_tensors_m,
        _make_train_tensors as make_train_tensors_m,
    )
    from MuyGPyS._src.mpi_utils import _chunk_tensor

    from MuyGPyS._src.gp.kernels.numpy import (
        _rbf_fn as rbf_fn_n,
        _matern_05_fn as matern_05_fn_n,
        _matern_15_fn as matern_15_fn_n,
        _matern_25_fn as matern_25_fn_n,
        _matern_inf_fn as matern_inf_fn_n,
        _matern_gen_fn as matern_gen_fn_n,
    )
    from MuyGPyS._src.gp.kernels.mpi import (
        _rbf_fn as rbf_fn_m,
        _matern_05_fn as matern_05_fn_m,
        _matern_15_fn as matern_15_fn_m,
        _matern_25_fn as matern_25_fn_m,
        _matern_inf_fn as matern_inf_fn_m,
        _matern_gen_fn as matern_gen_fn_m,
    )
    from MuyGPyS._src.gp.muygps.numpy import (
        _muygps_compute_solve as muygps_compute_solve_n,
        _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_n,
    )
    from MuyGPyS._src.gp.muygps.mpi import (
        _muygps_compute_solve as muygps_compute_solve_m,
        _muygps_compute_diagonal_variance as muygps_compute_diagonal_variance_m,
    )
    from MuyGPyS._src.optimize.sigma_sq.numpy import (
        _analytic_sigma_sq_optim as analytic_sigma_sq_optim_n,
    )
    from MuyGPyS._src.optimize.sigma_sq.mpi import (
        _analytic_sigma_sq_optim as analytic_sigma_sq_optim_m,
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
    from MuyGPyS._src.optimize.loss.mpi import (
        _mse_fn as mse_fn_m,
        _cross_entropy_fn as cross_entropy_fn_m,
        _lool_fn as lool_fn_m,
    )
    from MuyGPyS.optimize.objective import make_loo_crossval_fn

    from MuyGPyS._src.optimize.chassis.numpy import (
        _scipy_optimize as scipy_optimize_n,
        _bayes_opt_optimize as bayes_optimize_n,
    )
    from MuyGPyS._src.optimize.chassis.mpi import (
        _scipy_optimize as scipy_optimize_m,
        _bayes_opt_optimize as bayes_optimize_m,
    )

    from absl.testing import absltest
    from absl.testing import parameterized
    from mpi4py import MPI

    import numpy as np

    world = config.mpi_state.comm_world
    rank = world.Get_rank()
    size = world.Get_size()

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
            cls.nu_bounds = (1e-1, 2)
            cls.eps = 1e-3
            cls.k_kwargs = {
                "kern": "matern",
                "length_scale": {"val": cls.length_scale},
                "nu": {"val": cls.nu, "bounds": cls.nu_bounds},
                "eps": {"val": cls.eps},
            }
            cls.muygps = MuyGPS(**cls.k_kwargs)
            cls.kernel_kwargs = {
                "nu": cls.muygps.kernel.nu(),
                "length_scale": cls.muygps.kernel.length_scale(),
            }
            if rank == 0:
                cls.train_features = _make_gaussian_matrix(
                    cls.train_count, cls.feature_count
                )
                cls.train_responses = _make_gaussian_matrix(
                    cls.train_count, cls.response_count
                )
                cls.test_features = _make_gaussian_matrix(
                    cls.test_count, cls.feature_count
                )
                cls.test_responses = _make_gaussian_matrix(
                    cls.test_count, cls.response_count
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
                ) = make_train_tensors_n(
                    cls.muygps.kernel.metric,
                    batch_indices,
                    batch_nn_indices,
                    cls.train_features,
                    cls.train_responses,
                )

                test_nn_indices, _ = nbrs_lookup.get_nns(cls.test_features)

                (
                    cls.test_crosswise_dists,
                    cls.test_pairwise_dists,
                    cls.test_nn_targets,
                ) = make_regress_tensors_n(
                    cls.muygps.kernel.metric,
                    np.arange(cls.test_count),
                    test_nn_indices,
                    cls.test_features,
                    cls.train_features,
                    cls.train_responses,
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
                cls.batch_targets = None
                cls.batch_nn_targets = None

                cls.test_crosswise_dists = None
                cls.test_pairwise_dists = None
                cls.test_nn_targets = None

            (
                cls.batch_crosswise_dists_chunk,
                cls.batch_pairwise_dists_chunk,
                cls.batch_targets_chunk,
                cls.batch_nn_targets_chunk,
            ) = make_train_tensors_m(
                cls.muygps.kernel.metric,
                batch_indices,
                batch_nn_indices,
                cls.train_features,
                cls.train_responses,
            )
            (
                cls.test_crosswise_dists_chunk,
                cls.test_pairwise_dists_chunk,
                cls.test_nn_targets_chunk,
            ) = make_regress_tensors_m(
                cls.muygps.kernel.metric,
                np.arange(cls.test_count),
                test_nn_indices,
                cls.test_features,
                cls.train_features,
                cls.train_responses,
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

    class DistanceTest(DistanceTestCase):
        @classmethod
        def setUpClass(cls):
            super(DistanceTest, cls).setUpClass()

        def test_batch_pairwise_dists(self):
            self._compare_tensors(
                self.batch_pairwise_dists, self.batch_pairwise_dists_chunk
            )

        def test_test_pairwise_dists(self):
            self._compare_tensors(
                self.test_pairwise_dists, self.test_pairwise_dists_chunk
            )

        def test_batch_crosswise_dists(self):
            self._compare_tensors(
                self.batch_crosswise_dists, self.batch_crosswise_dists_chunk
            )

        def test_test_crosswise_dists(self):
            self._compare_tensors(
                self.test_crosswise_dists, self.test_crosswise_dists_chunk
            )

        def test_batch_targets(self):
            self._compare_tensors(self.batch_targets, self.batch_targets_chunk)

        def test_batch_nn_targets(self):
            self._compare_tensors(
                self.batch_nn_targets, self.batch_nn_targets_chunk
            )

        def test_test_nn_targets(self):
            self._compare_tensors(
                self.test_nn_targets, self.test_nn_targets_chunk
            )

    class KernelTestCase(DistanceTestCase):
        @classmethod
        def setUpClass(cls):
            super(KernelTestCase, cls).setUpClass()
            ls = cls.kernel_kwargs["length_scale"]
            if rank == 0:
                cls.batch_covariance_rbf = rbf_fn_n(
                    cls.batch_pairwise_dists, length_scale=ls
                )
                cls.batch_covariance_05 = matern_05_fn_n(
                    cls.batch_pairwise_dists, length_scale=ls
                )
                cls.batch_covariance_15 = matern_15_fn_n(
                    cls.batch_pairwise_dists, length_scale=ls
                )
                cls.batch_covariance_25 = matern_25_fn_n(
                    cls.batch_pairwise_dists, length_scale=ls
                )
                cls.batch_covariance_inf = matern_inf_fn_n(
                    cls.batch_pairwise_dists, length_scale=ls
                )
                cls.batch_covariance_gen = matern_gen_fn_n(
                    cls.batch_pairwise_dists, **cls.kernel_kwargs
                )
                cls.batch_crosscov_rbf = rbf_fn_n(
                    cls.batch_crosswise_dists, length_scale=ls
                )
                cls.batch_crosscov_05 = matern_05_fn_n(
                    cls.batch_crosswise_dists, length_scale=ls
                )
                cls.batch_crosscov_15 = matern_15_fn_n(
                    cls.batch_crosswise_dists, length_scale=ls
                )
                cls.batch_crosscov_25 = matern_25_fn_n(
                    cls.batch_crosswise_dists, length_scale=ls
                )
                cls.batch_crosscov_inf = matern_inf_fn_n(
                    cls.batch_crosswise_dists, length_scale=ls
                )
                cls.batch_crosscov_gen = matern_gen_fn_n(
                    cls.batch_crosswise_dists, **cls.kernel_kwargs
                )
                cls.test_covariance_rbf = rbf_fn_n(
                    cls.test_pairwise_dists, length_scale=ls
                )
                cls.test_covariance_05 = matern_05_fn_n(
                    cls.test_pairwise_dists, length_scale=ls
                )
                cls.test_covariance_15 = matern_15_fn_n(
                    cls.test_pairwise_dists, length_scale=ls
                )
                cls.test_covariance_25 = matern_25_fn_n(
                    cls.test_pairwise_dists, length_scale=ls
                )
                cls.test_covariance_inf = matern_inf_fn_n(
                    cls.test_pairwise_dists, length_scale=ls
                )
                cls.test_covariance_gen = matern_gen_fn_n(
                    cls.test_pairwise_dists, **cls.kernel_kwargs
                )
                cls.test_crosscov_rbf = rbf_fn_n(
                    cls.test_crosswise_dists, length_scale=ls
                )
                cls.test_crosscov_05 = matern_05_fn_n(
                    cls.test_crosswise_dists, length_scale=ls
                )
                cls.test_crosscov_15 = matern_15_fn_n(
                    cls.test_crosswise_dists, length_scale=ls
                )
                cls.test_crosscov_25 = matern_25_fn_n(
                    cls.test_crosswise_dists, length_scale=ls
                )
                cls.test_crosscov_inf = matern_inf_fn_n(
                    cls.test_crosswise_dists, length_scale=ls
                )
                cls.test_crosscov_gen = matern_gen_fn_n(
                    cls.test_crosswise_dists, **cls.kernel_kwargs
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

            cls.batch_covariance_rbf_chunk = rbf_fn_m(
                cls.batch_pairwise_dists_chunk, length_scale=ls
            )
            cls.batch_covariance_05_chunk = matern_05_fn_m(
                cls.batch_pairwise_dists_chunk, length_scale=ls
            )
            cls.batch_covariance_15_chunk = matern_15_fn_m(
                cls.batch_pairwise_dists_chunk, length_scale=ls
            )
            cls.batch_covariance_25_chunk = matern_25_fn_m(
                cls.batch_pairwise_dists_chunk, length_scale=ls
            )
            cls.batch_covariance_inf_chunk = matern_inf_fn_m(
                cls.batch_pairwise_dists_chunk, length_scale=ls
            )
            cls.batch_covariance_gen_chunk = matern_gen_fn_m(
                cls.batch_pairwise_dists_chunk, **cls.kernel_kwargs
            )
            cls.batch_crosscov_rbf_chunk = rbf_fn_m(
                cls.batch_crosswise_dists_chunk, length_scale=ls
            )
            cls.batch_crosscov_05_chunk = matern_05_fn_n(
                cls.batch_crosswise_dists_chunk, length_scale=ls
            )
            cls.batch_crosscov_15_chunk = matern_15_fn_n(
                cls.batch_crosswise_dists_chunk, length_scale=ls
            )
            cls.batch_crosscov_25_chunk = matern_25_fn_n(
                cls.batch_crosswise_dists_chunk, length_scale=ls
            )
            cls.batch_crosscov_inf_chunk = matern_inf_fn_n(
                cls.batch_crosswise_dists_chunk, length_scale=ls
            )
            cls.batch_crosscov_gen_chunk = matern_gen_fn_n(
                cls.batch_crosswise_dists_chunk, **cls.kernel_kwargs
            )
            cls.test_covariance_rbf_chunk = rbf_fn_m(
                cls.test_pairwise_dists_chunk, length_scale=ls
            )
            cls.test_covariance_05_chunk = matern_05_fn_m(
                cls.test_pairwise_dists_chunk, length_scale=ls
            )
            cls.test_covariance_15_chunk = matern_15_fn_m(
                cls.test_pairwise_dists_chunk, length_scale=ls
            )
            cls.test_covariance_25_chunk = matern_25_fn_m(
                cls.test_pairwise_dists_chunk, length_scale=ls
            )
            cls.test_covariance_inf_chunk = matern_inf_fn_m(
                cls.test_pairwise_dists_chunk, length_scale=ls
            )
            cls.test_covariance_gen_chunk = matern_gen_fn_m(
                cls.test_pairwise_dists_chunk, **cls.kernel_kwargs
            )
            cls.test_crosscov_rbf_chunk = rbf_fn_m(
                cls.test_crosswise_dists_chunk, length_scale=ls
            )
            cls.test_crosscov_05_chunk = matern_05_fn_n(
                cls.test_crosswise_dists_chunk, length_scale=ls
            )
            cls.test_crosscov_15_chunk = matern_15_fn_n(
                cls.test_crosswise_dists_chunk, length_scale=ls
            )
            cls.test_crosscov_25_chunk = matern_25_fn_n(
                cls.test_crosswise_dists_chunk, length_scale=ls
            )
            cls.test_crosscov_inf_chunk = matern_inf_fn_n(
                cls.test_crosswise_dists_chunk, length_scale=ls
            )
            cls.test_crosscov_gen_chunk = matern_gen_fn_n(
                cls.test_crosswise_dists_chunk, **cls.kernel_kwargs
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

    class MuyGPSTestCase(KernelTestCase):
        @classmethod
        def setUpClass(cls):
            super(MuyGPSTestCase, cls).setUpClass()
            if rank == 0:
                cls.batch_prediction = muygps_compute_solve_n(
                    cls.batch_covariance_gen,
                    cls.batch_crosscov_gen,
                    cls.batch_nn_targets,
                    cls.muygps.eps(),
                )
                cls.batch_variance = muygps_compute_diagonal_variance_n(
                    cls.batch_covariance_gen,
                    cls.batch_crosscov_gen,
                    cls.muygps.eps(),
                )
            else:
                cls.batch_prediction = None
                cls.batch_variance = None

            cls.batch_prediction_chunk = muygps_compute_solve_m(
                cls.batch_covariance_gen_chunk,
                cls.batch_crosscov_gen_chunk,
                cls.batch_nn_targets_chunk,
                cls.muygps.eps(),
            )
            cls.batch_variance_chunk = muygps_compute_diagonal_variance_m(
                cls.batch_covariance_gen_chunk,
                cls.batch_crosscov_gen_chunk,
                cls.muygps.eps(),
            )

    class MuyGPSTest(MuyGPSTestCase):
        @classmethod
        def setUpClass(cls):
            super(MuyGPSTest, cls).setUpClass()

        def test_batch_compute_solve(self):
            self._compare_tensors(
                self.batch_prediction, self.batch_prediction_chunk
            )

        def test_batch_diagonal_variance(self):
            self._compare_tensors(
                self.batch_variance, self.batch_variance_chunk
            )

        def test_sigma_sq_optim(self):
            parallel_sigma_sq = analytic_sigma_sq_optim_m(
                self.batch_covariance_gen_chunk,
                self.batch_nn_targets_chunk,
                self.muygps.eps(),
            )

            if rank == 0:
                serial_sigma_sq = analytic_sigma_sq_optim_n(
                    self.batch_covariance_gen,
                    self.batch_nn_targets,
                    self.muygps.eps(),
                )
                self.assertAlmostEqual(serial_sigma_sq[0], parallel_sigma_sq[0])

    class OptimTestCase(MuyGPSTestCase):
        @classmethod
        def setUpClass(cls):
            super(OptimTestCase, cls).setUpClass()
            cls.x0_names, cls.x0, bounds = cls.muygps.get_optim_params()
            cls.x0_map = {n: cls.x0[i] for i, n in enumerate(cls.x0_names)}
            cls.sopt_kwargs = {"verbose": False}
            cls.bopt_kwargs = {
                "verbose": False,
                "random_state": 1,
                "init_points": 5,
                "n_iter": 5,
            }

        # Numpy kernel functions
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

        # MPI kernel functions
        def _get_array_kernel_fn_m(self):
            return self.muygps.kernel._get_array_opt_fn(
                matern_05_fn_m,
                matern_15_fn_m,
                matern_25_fn_m,
                matern_inf_fn_m,
                matern_gen_fn_m,
                self.muygps.kernel.nu,
                self.muygps.kernel.length_scale,
            )

        def _get_kwargs_kernel_fn_m(self):
            return self.muygps.kernel._get_kwargs_opt_fn(
                matern_05_fn_m,
                matern_15_fn_m,
                matern_25_fn_m,
                matern_inf_fn_m,
                matern_gen_fn_m,
                self.muygps.kernel.nu,
                self.muygps.kernel.length_scale,
            )

        # Numpy predict functions
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

        # MPI predict functions
        def _get_array_mean_fn_m(self):
            return self.muygps._get_array_opt_mean_fn(
                muygps_compute_solve_m, self.muygps.eps
            )

        def _get_kwargs_mean_fn_m(self):
            return self.muygps._get_kwargs_opt_mean_fn(
                muygps_compute_solve_m, self.muygps.eps
            )

        def _get_array_var_fn_m(self):
            return self.muygps._get_array_opt_var_fn(
                muygps_compute_diagonal_variance_m, self.muygps.eps
            )

        def _get_kwargs_var_fn_m(self):
            return self.muygps._get_kwargs_opt_var_fn(
                muygps_compute_diagonal_variance_m, self.muygps.eps
            )

        def _get_array_sigma_sq_fn_m(self):
            return make_array_analytic_sigma_sq_optim(
                self.muygps, analytic_sigma_sq_optim_m
            )

        def _get_kwargs_sigma_sq_fn_m(self):
            return make_kwargs_analytic_sigma_sq_optim(
                self.muygps, analytic_sigma_sq_optim_m
            )

        # Numpy objective functions
        def _get_array_obj_fn_n(self):
            return make_loo_crossval_fn(
                "scipy",
                "mse",
                mse_fn_n,
                self._get_array_kernel_fn_n(),
                self._get_array_mean_fn_n(),
                self._get_array_var_fn_n(),
                self._get_array_sigma_sq_fn_n(),
                self.batch_pairwise_dists,
                self.batch_crosswise_dists,
                self.batch_nn_targets,
                self.batch_targets,
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
                self.batch_pairwise_dists,
                self.batch_crosswise_dists,
                self.batch_nn_targets,
                self.batch_targets,
            )

        # MPI objective functions
        def _get_array_obj_fn_m(self):
            return make_loo_crossval_fn(
                "scipy",
                "mse",
                mse_fn_m,
                self._get_array_kernel_fn_m(),
                self._get_array_mean_fn_m(),
                self._get_array_var_fn_m(),
                self._get_array_sigma_sq_fn_m(),
                self.batch_pairwise_dists_chunk,
                self.batch_crosswise_dists_chunk,
                self.batch_nn_targets_chunk,
                self.batch_targets_chunk,
            )

        def _get_kwargs_obj_fn_m(self):
            return make_loo_crossval_fn(
                "bayes",
                "mse",
                mse_fn_m,
                self._get_kwargs_kernel_fn_m(),
                self._get_kwargs_mean_fn_m(),
                self._get_kwargs_var_fn_m(),
                self._get_kwargs_sigma_sq_fn_m(),
                self.batch_pairwise_dists_chunk,
                self.batch_crosswise_dists_chunk,
                self.batch_nn_targets_chunk,
                self.batch_targets_chunk,
            )

    class LossTest(parameterized.TestCase):
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
                self.assertAlmostEqual(
                    serial_cross_entropy, parallel_cross_entropy
                )

    class LOOLTest(OptimTestCase):
        @classmethod
        def setUpClass(cls):
            super(LOOLTest, cls).setUpClass()
            cls.batch_sigma_sq = cls.muygps.sigma_sq()

        def test_lool(self):
            parallel_mse = lool_fn_m(
                self.batch_prediction_chunk,
                self.batch_targets_chunk,
                self.batch_variance_chunk,
                self.batch_sigma_sq,
            )

            if rank == 0:
                serial_mse = lool_fn_n(
                    self.batch_prediction,
                    self.batch_targets,
                    self.batch_variance,
                    self.batch_sigma_sq,
                )
                self.assertAlmostEqual(serial_mse, parallel_mse)

    class ObjectiveTest(OptimTestCase):
        @classmethod
        def setUpClass(cls):
            super(ObjectiveTest, cls).setUpClass()

            cls.batch_sigma_sq = cls.muygps.sigma_sq()
            cls.batch_sigma_sq_chunk = _chunk_tensor(cls.muygps.sigma_sq())
            cls.batch_variance_chunk

        def test_mse(self):
            parallel_mse = mse_fn_m(
                self.batch_prediction_chunk, self.batch_targets_chunk
            )

            if rank == 0:
                serial_mse = mse_fn_n(self.batch_prediction, self.batch_targets)
                self.assertAlmostEqual(serial_mse, parallel_mse)

        def test_lool(self):
            parallel_mse = lool_fn_m(
                self.batch_prediction_chunk,
                self.batch_targets_chunk,
                self.batch_variance_chunk,
                self.batch_sigma_sq_chunk,
            )

            if rank == 0:
                serial_mse = lool_fn_n(
                    self.batch_prediction,
                    self.batch_targets,
                    self.batch_variance,
                    self.batch_sigma_sq,
                )
                self.assertAlmostEqual(serial_mse, parallel_mse)

        # def test_cross_entropy(self):
        #     # TODO[bwp]: implement
        #     pass

        def test_array_kernel_fn(self):
            if rank == 0:
                kernel_fn_n = self._get_array_kernel_fn_n()
                kernel = kernel_fn_n(self.batch_pairwise_dists, self.x0)
            else:
                kernel = None

            kernel_fn_m = self._get_array_kernel_fn_m()
            kernel_chunk = kernel_fn_m(self.batch_pairwise_dists_chunk, self.x0)

            self._compare_tensors(kernel, kernel_chunk)

        def test_kwargs_kernel_fn(self):
            if rank == 0:
                kernel_fn_n = self._get_kwargs_kernel_fn_n()
                kernel = kernel_fn_n(self.batch_pairwise_dists, **self.x0_map)
            else:
                kernel = None

            kernel_fn_m = self._get_kwargs_kernel_fn_m()
            kernel_chunk = kernel_fn_m(
                self.batch_pairwise_dists_chunk, **self.x0_map
            )

            self._compare_tensors(kernel, kernel_chunk)

        def test_array_mean_fn(self):
            if rank == 0:
                mean_fn_n = self._get_array_mean_fn_n()
                mean = mean_fn_n(
                    self.batch_covariance_gen,
                    self.batch_crosscov_gen,
                    self.batch_nn_targets,
                    self.x0,
                )
            else:
                mean = None

            mean_fn_m = self._get_array_mean_fn_m()
            mean_chunk = mean_fn_m(
                self.batch_covariance_gen_chunk,
                self.batch_crosscov_gen_chunk,
                self.batch_nn_targets_chunk,
                self.x0,
            )

            self._compare_tensors(mean, mean_chunk)

        def test_kwargs_mean_fn(self):
            if rank == 0:
                mean_fn_n = self._get_kwargs_mean_fn_n()
                mean = mean_fn_n(
                    self.batch_covariance_gen,
                    self.batch_crosscov_gen,
                    self.batch_nn_targets,
                    **self.x0_map,
                )
            else:
                mean = None

            mean_fn_m = self._get_kwargs_mean_fn_m()
            mean_chunk = mean_fn_m(
                self.batch_covariance_gen_chunk,
                self.batch_crosscov_gen_chunk,
                self.batch_nn_targets_chunk,
                **self.x0_map,
            )

            self._compare_tensors(mean, mean_chunk)

        def test_array_var_fn(self):
            if rank == 0:
                var_fn_n = self._get_array_var_fn_n()
                var = var_fn_n(
                    self.batch_covariance_gen,
                    self.batch_crosscov_gen,
                    self.x0,
                )
            else:
                var = None

            var_fn_m = self._get_array_var_fn_m()
            var_chunk = var_fn_m(
                self.batch_covariance_gen_chunk,
                self.batch_crosscov_gen_chunk,
                self.x0,
            )

            self._compare_tensors(var, var_chunk)

        def test_kwargs_var_fn(self):
            if rank == 0:
                var_fn_n = self._get_kwargs_var_fn_n()
                var = var_fn_n(
                    self.batch_covariance_gen,
                    self.batch_crosscov_gen,
                    **self.x0_map,
                )
            else:
                var = None

            var_fn_m = self._get_kwargs_var_fn_m()
            var_chunk = var_fn_m(
                self.batch_covariance_gen_chunk,
                self.batch_crosscov_gen_chunk,
                **self.x0_map,
            )

            self._compare_tensors(var, var_chunk)

        def test_kwargs_sigma_sq_fn(self):
            if rank == 0:
                ss_fn_n = self._get_kwargs_sigma_sq_fn_n()
                ss = ss_fn_n(
                    self.batch_covariance_gen,
                    self.batch_nn_targets,
                    **self.x0_map,
                )
            else:
                ss = None

            ss_fn_m = self._get_kwargs_sigma_sq_fn_m()
            ss_chunk = ss_fn_m(
                self.batch_covariance_gen_chunk,
                self.batch_nn_targets_chunk,
                **self.x0_map,
            )

            self._compare_tensors(ss, ss_chunk)

        def test_array_sigma_sq_fn(self):
            if rank == 0:
                ss_fn_n = self._get_array_sigma_sq_fn_n()
                ss = ss_fn_n(
                    self.batch_covariance_gen,
                    self.batch_nn_targets,
                    self.x0,
                )
            else:
                ss = None

            ss_fn_m = self._get_array_sigma_sq_fn_m()
            ss_chunk = ss_fn_m(
                self.batch_covariance_gen_chunk,
                self.batch_nn_targets_chunk,
                self.x0,
            )

            self._compare_tensors(ss, ss_chunk)

        def test_array_loo_crossval(self):
            obj_fn_m = self._get_array_obj_fn_m()
            obj_m = obj_fn_m(self.x0)

            if rank == 0:
                obj_fn_n = self._get_array_obj_fn_n()
                obj_n = obj_fn_n(self.x0)
                self.assertAlmostEqual(obj_n, obj_m)

        def test_kwargs_loo_crossval(self):
            obj_fn_m = self._get_kwargs_obj_fn_m()
            obj_m = obj_fn_m(**self.x0_map)

            if rank == 0:
                obj_fn_n = self._get_kwargs_obj_fn_n()
                obj_n = obj_fn_n(**self.x0_map)
                self.assertAlmostEqual(obj_n, obj_m)

    class ScipyOptimTest(OptimTestCase):
        @classmethod
        def setUpClass(cls):
            super(ScipyOptimTest, cls).setUpClass()

        def test_scipy_optimize(self):
            obj_fn_m = self._get_array_obj_fn_m()
            opt_m = scipy_optimize_m(self.muygps, obj_fn_m, **self.sopt_kwargs)

            if rank == 0:
                obj_fn_n = self._get_array_obj_fn_n()
                opt_n = scipy_optimize_n(
                    self.muygps, obj_fn_n, **self.sopt_kwargs
                )
                self.assertAlmostEqual(opt_m.kernel.nu(), opt_n.kernel.nu())

    class BayesOptimTest(OptimTestCase):
        @classmethod
        def setUpClass(cls):
            super(BayesOptimTest, cls).setUpClass()

        def test_bayes_optimize(self):
            obj_fn_m = self._get_kwargs_obj_fn_m()
            model_m = bayes_optimize_m(
                self.muygps, obj_fn_m, **self.bopt_kwargs
            )

            if rank == 0:
                obj_fn_n = self._get_kwargs_obj_fn_n()
                model_n = bayes_optimize_n(
                    self.muygps, obj_fn_n, **self.bopt_kwargs
                )
                self.assertAlmostEqual(model_m.kernel.nu(), model_n.kernel.nu())

    if __name__ == "__main__":
        absltest.main()

else:
    if __name__ == "__main__":
        pass
