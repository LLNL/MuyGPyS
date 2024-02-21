# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest

import MuyGPyS._src.math as mm

from MuyGPyS.gp.kernels.experimental import ShearKernel
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS._test.utils import _check_ndarray
from MuyGPyS._test.shear import BenchmarkTestCase
from MuyGPyS._test.shear import original_shear, conventional_mean, conventional_variance

from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize import Bayes_optimize
from MuyGPyS.optimize.loss import mse_fn


class ShearKernelTest(BenchmarkTestCase):

    def test_flat_shear_kernel(self):

        data_count = self.features.shape[0]

        diffs = self.dist_fn.pairwise_tensor(self.features, np.arange(data_count))

        analytic_kernel = original_shear(self.features, length_scale=self.length_scale)

        library_kernel = ShearKernel(deformation=self.dist_fn)(diffs).reshape(3 * data_count, 3 * data_count)

        _check_ndarray(
            self.assertEqual,
            library_kernel,
            mm.ftype,
            shape=analytic_kernel.shape,
        )

        self.assertTrue(mm.allclose(library_kernel, analytic_kernel))

    def test_K_cross(self):

        split = 200
        X1 = self.features[:split]
        X2 = self.features[split:]
        n1, _ = X1.shape
        n2, _ = X2.shape

        crosswise_diffs = self.dist_fn.crosswise_tensor(
            X1, X2, np.arange(n1), np.arange(n2)
        )

        library_Kcross = ShearKernel(deformation=self.dist_fn)(crosswise_diffs, adjust=False)

        analytic_Kcross = original_shear(X1, X2, length_scale=self.length_scale)

        library_Kcross_flat = library_Kcross.reshape(n1 * 3, n2 * 3)

        _check_ndarray(
            self.assertEqual,
            library_Kcross_flat,
            mm.ftype,
            shape=analytic_Kcross.shape,
        )

        self.assertTrue(mm.allclose(library_Kcross_flat, analytic_Kcross))

    def test_flat_mean(self):

        train_ratio = 0.2

        data_count = self.features.shape[0]

        rng = np.random.default_rng(seed=1)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            train_mask[idx] = True
        test_mask = np.invert(train_mask)
        train_count = np.count_nonzero(train_mask)
        test_count = np.count_nonzero(test_mask)

        train_targets = self.targets[train_mask, :]
        train_features = self.features[train_mask, :]
        test_features = self.features[test_mask, :]

        train_targets_flat = train_targets.swapaxes(0, 1).reshape(3 * train_count)

        Kin_analytic = original_shear(train_features, train_features, length_scale=self.length_scale)
        Kcross_analytic = original_shear(test_features, train_features, length_scale=self.length_scale)

        pairwise_diffs = self.library_shear.kernel.deformation.pairwise_tensor(
            train_features, np.arange(train_count)
        )
        crosswise_diffs = self.library_shear.kernel.deformation.crosswise_tensor(
            test_features, train_features, np.arange(test_count), np.arange(train_count)
        )
        library_Kin = self.library_shear.kernel(pairwise_diffs, adjust=False)
        library_Kcross = self.library_shear.kernel(crosswise_diffs, adjust=False)
        library_Kin_flat = library_Kin.reshape(3 * train_count, 3 * train_count)
        library_Kcross_flat = library_Kcross.reshape(3 * test_count, 3 * train_count)

        posterior_mean_analytic = conventional_mean(
            Kin_analytic,
            Kcross_analytic,
            train_targets_flat,
            self.noise_prior
        )
        posterior_mean_flat = conventional_mean(
            library_Kin_flat,
            library_Kcross_flat,
            train_targets_flat,
            self.noise_prior
        )

        self.assertTrue(mm.allclose(posterior_mean_analytic, posterior_mean_flat))

    def test_library_mean(self):
        train_ratio = 0.2

        data_count = self.features.shape[0]

        rng = np.random.default_rng(seed=1)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            train_mask[idx] = True
        test_mask = np.invert(train_mask)
        test_count = np.count_nonzero(test_mask)

        train_targets = self.targets[train_mask, :]
        train_features = self.features[train_mask, :]
        test_features = self.features[test_mask, :]

        indices = np.arange(test_count)

        nbrs_lookup = NN_Wrapper(train_features, self.nn_count, nn_method='exact', algorithm='ball_tree')
        nn_indices, _ = nbrs_lookup.get_nns(test_features)

        (
            crosswise_diffs,
            pairwise_diffs,
            nn_targets,
        ) = self.library_shear.make_predict_tensors(
            indices,
            nn_indices,
            test_features,
            train_features,
            train_targets,
        )

        nn_targets = nn_targets.swapaxes(-2, -1)

        Kcross = self.library_shear.kernel(crosswise_diffs)
        Kin = self.library_shear.kernel(pairwise_diffs)

        library_posterior_mean = self.library_shear.posterior_mean(Kin, Kcross, nn_targets)

        Kin_flat = Kin.reshape(test_count, 3 * self.nn_count, 3 * self.nn_count)
        Kcross_flat = Kcross.reshape(test_count, 3 * self.nn_count, 3)
        nn_targets_flat = nn_targets.reshape(test_count, 3 * self.nn_count)

        kappa_mean_flat = np.squeeze(conventional_mean(
            Kin_flat[0],
            Kcross_flat[0].swapaxes(-2, -1),
            nn_targets_flat[0],
            self.noise_prior
        ))

        self.assertTrue(mm.allclose(library_posterior_mean[0], kappa_mean_flat))

    def test_flat_variance(self):

        train_ratio = 0.2

        data_count = self.features.shape[0]

        rng = np.random.default_rng(seed=1)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            train_mask[idx] = True
        test_mask = np.invert(train_mask)
        train_count = np.count_nonzero(train_mask)
        test_count = np.count_nonzero(test_mask)

        train_features = self.features[train_mask, :]
        test_features = self.features[test_mask, :]

        pairwise_diffs = self.dist_fn.pairwise_tensor(
            train_features, np.arange(train_count)
        )
        crosswise_diffs = self.dist_fn.crosswise_tensor(
            test_features,
            train_features,
            np.arange(test_count),
            np.arange(train_count),
        )
        # test diffs
        pairwise_diffs_test = self.dist_fn.pairwise_tensor(
            test_features, np.arange(test_count)
        )

        library_Kin_test = self.library_shear.kernel(pairwise_diffs_test, adjust=False)
        library_Kin = self.library_shear.kernel(pairwise_diffs, adjust=False)
        library_Kcross = self.library_shear.kernel(crosswise_diffs, adjust=False)

        library_Kin_flat = library_Kin.reshape(3 * train_count, 3 * train_count)
        library_Kcross_flat = library_Kcross.reshape(3 * test_count, 3 * train_count)
        library_Kin_test_flat = library_Kin_test.reshape(3 * test_count, 3 * test_count)

        Kin_an = original_shear(
            train_features,
            length_scale=self.length_scale,
        )
        Kcross_an = original_shear(
            test_features,
            train_features,
            length_scale=self.length_scale,
        )
        Kin_test_an = original_shear(
            test_features,
            length_scale=self.length_scale
        )

        conventional_var_analytic_flat = conventional_variance(
            Kin_an,
            Kcross_an,
            Kin_test_an,
            self.noise_prior
        )
        library_conventional_var_flat = conventional_variance(
            library_Kin_flat,
            library_Kcross_flat,
            library_Kin_test_flat,
            self.noise_prior
        )

        self.assertTrue(mm.allclose(library_conventional_var_flat, conventional_var_analytic_flat))

    """
    def test_libary_variance(self):
        train_ratio = 0.2

        data_count = self.features.shape[0]

        nn_count = self.nn_count

        rng = np.random.default_rng(seed=1)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            train_mask[idx] = True
        test_mask = np.invert(train_mask)
        train_count = np.count_nonzero(train_mask)
        test_count = np.count_nonzero(test_mask)

        train_targets = self.targets[train_mask, :]
        test_targets = self.targets[test_mask, :]
        train_features = self.features[train_mask, :]
        test_features = self.features[test_mask, :]

        Kin_an = original_shear(
            train_features,
            length_scale=self.length_scale,
        )
        Kcross_an = original_shear(
            test_features,
            train_features,
            length_scale=self.length_scale,
        )
        # Construct the tensors K(X*,X*) and K(X,X*),
        # although not sure that that the explicit K(X,X*)
        # is necessary
        Kin_test_an = original_shear(
            test_features,
            length_scale=self.length_scale
            )

        conventional_var_analytic_flat = conventional_variance(
            Kin_an,
            Kcross_an,
            Kin_test_an,
            self.noise_prior
        )
        print(conventional_var_analytic_flat.shape)

        posterior_var_an = np.zeros((500,3,3))
        for i in range(3):
            for j in range(3):
                posterior_var_an[:,i,j] = np.diagonal(conventional_var_analytic_flat[500*i:500*(i+1), 500*j:500*(j+1)])

        indices = np.arange(test_count)

        if nn_count == train_count:
            nn_indices = np.array([
                np.arange(train_count) for _ in range(test_count)
            ])
        else:
            nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method='exact', algorithm='ball_tree')
            nn_indices, _ = nbrs_lookup.get_nns(test_features)

        (
            crosswise_diffs,
            pairwise_diffs,
            nn_targets,
        ) = self.library_shear.make_predict_tensors(
            indices,
            nn_indices,
            test_features,
            train_features,
            train_targets,
        )

        nn_targets= nn_targets.swapaxes(-2, -1)

        Kcross = self.library_shear.kernel(crosswise_diffs)
        Kin = self.library_shear.kernel(pairwise_diffs)

        library_posterior_var = self.library_shear.posterior_variance(Kin, Kcross)

        var_residual = np.abs(posterior_var_an - library_posterior_var)


        print(
            "Min Resid = ",
            np.min(var_residual),
            ", Max Resid = ",
            np.max(var_residual), ",
            Avg Residual = ",
            np.mean(var_residual)
        )

        self.assertTrue(mm.allclose(posterior_var_an, library_posterior_var, atol=1e-10))
    """

    def test_ls_optimization(self):

        train_ratio = 0.2

        data_count = self.features.shape[0]

        rng = np.random.default_rng(seed=1)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            train_mask[idx] = True

        train_targets = self.targets[train_mask, :]
        train_features = self.features[train_mask, :]

        train_features_count = train_features.shape[0]

        nn_count = 50
        nbrs_lookup = NN_Wrapper(train_features, nn_count, nn_method='exact', algorithm='ball_tree')

        batch_count = 500
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, train_features_count
        )

        batch_crosswise_diffs = self.optimize_model.kernel.deformation.crosswise_tensor(
            train_features,
            train_features,
            batch_indices,
            batch_nn_indices,
        )

        batch_pairwise_diffs = self.optimize_model.kernel.deformation.pairwise_tensor(
            train_features, batch_nn_indices
        )

        batch_targets = train_targets[batch_indices]
        batch_nn_targets = train_targets[batch_nn_indices].swapaxes(-2, -1)

        shear_mse_optimized = Bayes_optimize(
            self.optimize_model,
            batch_targets,
            batch_nn_targets,
            batch_crosswise_diffs,
            batch_pairwise_diffs,
            train_targets,
            loss_fn=mse_fn,
            verbose=False,
            init_points=5,
            n_iter=20,
        )

        self.assertTrue(mm.allclose(self.length_scale, shear_mse_optimized.kernel.deformation.length_scale(), atol=0.015))


if __name__ == "__main__":
    absltest.main()
