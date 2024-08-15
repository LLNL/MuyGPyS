# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from absl.testing import absltest

import MuyGPyS._src.math as mm

from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS._test.utils import _check_ndarray
from MuyGPyS._test.shear import (
    BenchmarkTestCase,
    conventional_Kout,
    conventional_mean,
    conventional_mean33,
    conventional_shear,
    conventional_variance,
    conventional_variance33,
)

from MuyGPyS.optimize.batch import sample_batch
from MuyGPyS.optimize import Bayes_optimize
from MuyGPyS.optimize.loss import mse_fn


class KernelTestCase(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(KernelTestCase, cls).setUpClass()
        split = 200
        cls.X1 = cls.features[:split]
        cls.X2 = cls.features[split:]
        cls.n1, _ = cls.X1.shape
        cls.n2, _ = cls.X2.shape
        cls.pairwise_diffs = cls.model33.kernel.deformation.pairwise_tensor(
            cls.X1, np.arange(cls.n1)
        )
        cls.crosswise_diffs = cls.model33.kernel.deformation.crosswise_tensor(
            cls.X1, cls.X2, np.arange(cls.n1), np.arange(cls.n2)
        )
        cls.Kin_analytic = conventional_shear(
            cls.X1, length_scale=cls.length_scale
        )
        cls.Kcross_analytic = conventional_shear(
            cls.X1, cls.X2, length_scale=cls.length_scale
        )

    def _Kin_chassis(self, Kin_analytic, Kin_fn, in_dim=3, **kwargs):
        Kin_muygps = Kin_fn(
            self.pairwise_diffs, length_scale=self.length_scale, **kwargs
        )
        Kin_flat = Kin_muygps.reshape(in_dim * self.n1, in_dim * self.n1)
        _check_ndarray(
            self.assertEqual,
            Kin_muygps,
            mm.ftype,
            shape=(in_dim, self.n1, in_dim, self.n1),
        )
        self.assertEqual(Kin_flat.shape, Kin_analytic.shape)
        self.assertTrue(np.allclose(Kin_flat, Kin_analytic))

    def _Kcross_chassis(
        self, Kcross_analytic, Kcross_fn, in_dim=3, out_dim=3, **kwargs
    ):
        Kcross_muygps = Kcross_fn(
            self.crosswise_diffs, length_scale=self.length_scale, **kwargs
        )
        Kcross_flat = Kcross_muygps.reshape(in_dim * self.n1, out_dim * self.n2)
        _check_ndarray(
            self.assertEqual,
            Kcross_muygps,
            mm.ftype,
            shape=(in_dim, self.n1, out_dim, self.n2),
        )
        self.assertEqual(Kcross_flat.shape, Kcross_analytic.shape)
        self.assertTrue(mm.allclose(Kcross_flat, Kcross_analytic))


class KernelTest(KernelTestCase):
    def test_Kin33(self):
        self._Kin_chassis(self.Kin_analytic, self.model33.kernel)

    def test_Kin23(self):
        Kin_analytic = self.Kin_analytic[self.n1 :, self.n1 :]
        self._Kin_chassis(Kin_analytic, self.model23.kernel, in_dim=2)

    def test_Kcross33(self):
        self._Kcross_chassis(
            self.Kcross_analytic, self.model33.kernel, adjust=False
        )

    def test_Kcross23(self):
        self._Kcross_chassis(
            self.Kcross_analytic[self.n1 :, :],
            self.model23.kernel,
            in_dim=2,
            force_Kcross=True,
        )


class DataTestCase(BenchmarkTestCase):
    @classmethod
    def setUpClass(cls):
        super(DataTestCase, cls).setUpClass()
        train_ratio = 0.2

        data_count = cls.features.shape[0]

        rng = np.random.default_rng(seed=1)
        interval_count = int(data_count * train_ratio)
        interval = int(data_count / interval_count)
        sfl = rng.permutation(np.arange(data_count))
        train_mask = np.zeros(data_count, dtype=bool)
        for i in range(interval_count):
            idx = np.random.choice(sfl[i * interval : (i + 1) * interval])
            train_mask[idx] = True
        test_mask = np.invert(train_mask)
        cls.train_count = np.count_nonzero(train_mask)
        cls.test_count = np.count_nonzero(test_mask)

        cls.train_targets = cls.targets[train_mask, :]
        cls.train_features = cls.features[train_mask, :]
        cls.test_features = cls.features[test_mask, :]


class FlatTestCase(DataTestCase):
    @classmethod
    def setUpClass(cls):
        super(FlatTestCase, cls).setUpClass()
        cls.train_targets_flat = cls.train_targets.swapaxes(0, 1).reshape(
            3 * cls.train_count
        )
        cls.Kin_analytic = conventional_shear(
            cls.train_features,
            cls.train_features,
            length_scale=cls.length_scale,
        )
        cls.Kcross_analytic = conventional_shear(
            cls.train_features, cls.test_features, length_scale=cls.length_scale
        )
        cls.pairwise_diffs = cls.model33.kernel.deformation.pairwise_tensor(
            cls.train_features, np.arange(cls.train_count)
        )
        cls.crosswise_diffs = cls.model33.kernel.deformation.crosswise_tensor(
            cls.train_features,
            cls.test_features,
            np.arange(cls.train_count),
            np.arange(cls.test_count),
        )

    def _mean_chassis(
        self,
        Kin_analytic,
        Kcross_analytic,
        train_targets_flat,
        kernel_fn,
        in_dim=3,
        out_dim=3,
        **kwargs,
    ):
        Kin_muygps = kernel_fn(
            self.pairwise_diffs, length_scale=self.length_scale
        )
        Kcross_muygps = kernel_fn(
            self.crosswise_diffs, length_scale=self.length_scale, **kwargs
        )
        Kin_flat = Kin_muygps.reshape(
            in_dim * self.train_count, in_dim * self.train_count
        )
        Kcross_flat = Kcross_muygps.reshape(
            in_dim * self.train_count, out_dim * self.test_count
        )
        self.assertTrue(mm.allclose(Kin_flat, Kin_analytic))
        self.assertTrue(mm.allclose(Kcross_flat, Kcross_analytic))
        posterior_mean_analytic = conventional_mean(
            Kin_analytic,
            Kcross_analytic.T,
            train_targets_flat,
            self.noise_prior,
        )
        posterior_mean_flat = conventional_mean(
            Kin_flat, Kcross_flat.T, train_targets_flat, self.noise_prior
        )
        self.assertTrue(
            mm.allclose(posterior_mean_analytic, posterior_mean_flat)
        )

    def _variance_chassis(
        self,
        Kin_analytic,
        Kcross_analytic,
        kernel_fn,
        in_dim=3,
        out_dim=3,
        **kwargs,
    ):
        Kin_muygps = kernel_fn(
            self.pairwise_diffs, length_scale=self.length_scale
        )
        Kcross_muygps = kernel_fn(
            self.crosswise_diffs, length_scale=self.length_scale, **kwargs
        )
        Kin_flat = Kin_muygps.reshape(
            in_dim * self.train_count, in_dim * self.train_count
        )
        Kcross_flat = Kcross_muygps.reshape(
            in_dim * self.train_count, out_dim * self.test_count
        )
        Kout_flat = conventional_Kout(kernel_fn, self.test_count)
        self.assertTrue(mm.allclose(Kin_flat, Kin_analytic))
        self.assertTrue(mm.allclose(Kcross_flat, Kcross_analytic))
        posterior_variance_analytic = conventional_variance(
            Kin_analytic, Kcross_analytic.T, Kout_flat, self.noise_prior
        )
        posterior_variance_flat = conventional_variance(
            Kin_flat, Kcross_flat.T, Kout_flat, self.noise_prior
        )
        self.assertTrue(
            mm.allclose(posterior_variance_analytic, posterior_variance_flat)
        )


class FlatTest(FlatTestCase):
    def test_mean33(self):
        self._mean_chassis(
            self.Kin_analytic,
            self.Kcross_analytic,
            self.train_targets_flat,
            self.model33.kernel,
            adjust=False,
        )

    def test_mean23(self):
        Kin_analytic = self.Kin_analytic[self.train_count :, self.train_count :]
        Kcross_analytic = self.Kcross_analytic[self.train_count :, :]
        train_targets_flat = self.train_targets_flat[self.train_count :]
        self._mean_chassis(
            Kin_analytic,
            Kcross_analytic,
            train_targets_flat,
            self.model23.kernel,
            in_dim=2,
            force_Kcross=True,
        )

    def test_variance33(self):
        self._variance_chassis(
            self.Kin_analytic,
            self.Kcross_analytic,
            self.model33.kernel,
            adjust=False,
        )

    def test_variance23(self):
        Kin_analytic = self.Kin_analytic[self.train_count :, self.train_count :]
        Kcross_analytic = self.Kcross_analytic[self.train_count :, :]
        self._variance_chassis(
            Kin_analytic,
            Kcross_analytic,
            self.model23.kernel,
            in_dim=2,
            force_Kcross=True,
        )


class LibraryTestCase(DataTestCase):
    @classmethod
    def setUpClass(cls):
        super(LibraryTestCase, cls).setUpClass()
        indices = np.arange(cls.test_count)

        nbrs_lookup = NN_Wrapper(
            cls.train_features,
            cls.nn_count,
            nn_method="exact",
            algorithm="ball_tree",
        )
        nn_indices, _ = nbrs_lookup.get_nns(cls.test_features)

        (
            cls.crosswise_diffs,
            cls.pairwise_diffs,
            cls.nn_targets,
        ) = cls.model33.make_predict_tensors(
            indices,
            nn_indices,
            cls.test_features,
            cls.train_features,
            cls.train_targets,
        )
        cls.nn_targets = cls.nn_targets.swapaxes(-2, -1)

        batch_count = 500
        batch_indices, batch_nn_indices = sample_batch(
            nbrs_lookup, batch_count, cls.train_count
        )

        (
            cls.batch_crosswise_diffs,
            cls.batch_pairwise_diffs,
            cls.batch_targets,
            cls.batch_nn_targets,
        ) = cls.model33.make_train_tensors(
            batch_indices,
            batch_nn_indices,
            cls.train_features,
            cls.train_targets,
        )
        cls.batch_nn_targets = cls.batch_nn_targets.swapaxes(-2, -1)

    def _mean_chassis(
        self, nn_targets, model, in_dim=3, out_dim=3, mean_fn=conventional_mean
    ):
        Kcross = model.kernel(self.crosswise_diffs)
        Kin = model.kernel(self.pairwise_diffs)
        posterior_mean = model.posterior_mean(Kin, Kcross, nn_targets)
        Kin_flat = Kin.reshape(
            self.test_count, in_dim * self.nn_count, in_dim * self.nn_count
        )
        Kcross_flat = Kcross.reshape(
            self.test_count, in_dim * self.nn_count, out_dim
        )
        nn_targets_flat = nn_targets.reshape(
            self.test_count, in_dim * self.nn_count
        )
        mean_flat = np.squeeze(
            mean_fn(
                Kin_flat[0],
                Kcross_flat[0].swapaxes(-2, -1),
                nn_targets_flat[0],
                self.noise_prior,
            )
        )

        self.assertTrue(mm.allclose(posterior_mean[0], mean_flat))

    def _mean_chassis33(self, nn_targets, model, in_dim=3, out_dim=3):
        self._mean_chassis(
            nn_targets,
            model,
            in_dim=in_dim,
            out_dim=out_dim,
            mean_fn=conventional_mean33,
        )

    def _variance_chassis(
        self, model, in_dim=3, out_dim=3, variance_fn=conventional_variance
    ):
        Kcross = model.kernel(self.crosswise_diffs)
        Kin = model.kernel(self.pairwise_diffs)
        posterior_variance = model.posterior_variance(Kin, Kcross)
        Kin_flat = Kin.reshape(
            self.test_count, in_dim * self.nn_count, in_dim * self.nn_count
        )
        Kcross_flat = Kcross.reshape(
            self.test_count, in_dim * self.nn_count, out_dim
        )
        variance_flat = np.squeeze(
            variance_fn(
                Kin_flat[0],
                Kcross_flat[0].swapaxes(-2, -1),
                model.kernel.Kout(),
                self.noise_prior,
            )
        )

        self.assertTrue(mm.allclose(posterior_variance[0], variance_flat))

    def _variance_chassis33(self, model, in_dim=3, out_dim=3):
        self._variance_chassis(
            model,
            in_dim=in_dim,
            out_dim=out_dim,
            variance_fn=conventional_variance33,
        )

    def _opt_chassis(
        self, opt_model, batch_targets, batch_nn_targets, **kwargs
    ):
        shear_mse_optimized = Bayes_optimize(
            opt_model,
            batch_targets,
            batch_nn_targets,
            self.batch_crosswise_diffs,
            self.batch_pairwise_diffs,
            loss_fn=mse_fn,
            verbose=False,
            init_points=5,
            n_iter=20,
            **kwargs,
        )

        ls = shear_mse_optimized.kernel.deformation.length_scale()
        print(f"finds length scale: {ls}")
        self.assertTrue(mm.allclose(self.length_scale, ls, atol=0.015))


class LibraryTest(LibraryTestCase):
    def test_mean33(self):
        self._mean_chassis33(self.nn_targets, self.model33)

    def test_mean23(self):
        self._mean_chassis(self.nn_targets[:, 1:, :], self.model23, in_dim=2)

    def test_variance33(self):
        self._variance_chassis33(self.model33)

    def test_variance23(self):
        self._variance_chassis(self.model23, in_dim=2)

    def test_opt33(self):
        self._opt_chassis(
            self.model33, self.batch_targets, self.batch_nn_targets
        )

    def test_opt23(self):
        batch_targets = self.batch_targets[:, 1:]
        batch_nn_targets = self.batch_nn_targets[:, 1:, :]
        self._opt_chassis(
            self.model23,
            batch_targets,
            batch_nn_targets,
            target_mask=(1, 2),
        )

    def test_convergence(self):
        length_scale = self.model33.kernel.deformation.length_scale()
        self.assertEqual(
            self.model23.kernel.deformation.length_scale(), length_scale
        )
        # Kcross23 = self.model23.kernel(self.crosswise_diffs)
        # Kin23 = self.model23.kernel(self.pairwise_diffs)
        # posterior_mean23 = self.model23.posterior_mean(
        #     Kin23, Kcross23, self.nn_targets[:, 1:, :]
        # )
        # Kcross33 = self.model33.kernel(self.crosswise_diffs)
        # Kin33 = self.model33.kernel(self.pairwise_diffs)
        # posterior_mean33 = self.model33.posterior_mean(
        #     Kin33, Kcross33, self.nn_targets
        # )

        Kin_analytic33 = conventional_shear(
            self.train_features, length_scale=length_scale
        )
        Kcross_analytic33 = conventional_shear(
            self.train_features, self.test_features, length_scale=length_scale
        )
        targets33 = self.train_targets.T.reshape(3 * self.train_count)
        Kin_analytic23 = Kin_analytic33[self.train_count :, self.train_count :]
        Kcross_analytic23 = Kcross_analytic33[self.train_count :, :]
        targets23 = targets33[self.train_count :]
        # print(
        #     self.train_features.shape,
        #     self.test_features.shape,
        #     targets.shape,
        # )
        # print(Kin_analytic.shape, Kcross_analytic.T.shape, targets.shape)
        posterior_mean_analytic33 = conventional_mean(
            Kin_analytic33, Kcross_analytic33.T, targets33, self.noise_prior
        )
        posterior_mean_analytic23 = conventional_mean(
            Kin_analytic23, Kcross_analytic23.T, targets23, self.noise_prior
        )
        # print(posterior_mean_analytic.shape)

        residual = np.abs(posterior_mean_analytic33 - posterior_mean_analytic23)
        min_res = np.min(residual, axis=0)
        max_res = np.max(residual, axis=0)
        avg_res = np.mean(residual, axis=0)

        print("\tconvergence\t\tshear 1\t\t\tshear 2")
        print(f"min\t{min_res[0]}\t{min_res[1]}\t{min_res[2]}")
        print(f"max\t{max_res[0]}\t{max_res[1]}\t{max_res[2]}")
        print(f"mean\t{avg_res[0]}\t{avg_res[1]}\t{avg_res[2]}")
        print()


if __name__ == "__main__":
    absltest.main()
