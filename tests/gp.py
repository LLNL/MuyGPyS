# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._src.gp.noise import (
    _homoscedastic_perturb,
    _heteroscedastic_perturb,
)
from MuyGPyS._src.mpi_utils import (
    _consistent_unchunk_tensor,
    _warn0,
)
from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _basic_opt_fn_and_kwarg_options,
    _check_ndarray,
    _consistent_assert,
    _get_scale_series,
    _make_gaussian_data,
    _make_gaussian_dict,
    _make_heteroscedastic_test_nugget,
    _precision_assert,
)
from MuyGPyS.examples.regress import make_regressor
from MuyGPyS.examples.classify import make_classifier
from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import (
    Isotropy,
    Anisotropy,
    F2,
    l2,
)
from MuyGPyS.gp.hyperparameter import AnalyticScale, FixedScale, ScalarParam
from MuyGPyS.gp.kernels import Matern, RBF
from MuyGPyS.gp.noise import HomoscedasticNoise, HeteroscedasticNoise
from MuyGPyS.gp.tensors import (
    make_train_tensors,
    make_predict_tensors,
)
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.loss import mse_fn


class GPInitTest(parameterized.TestCase):
    @parameterized.parameters(
        (kernel, noise, gp)
        for kernel in (
            Matern(smoothness=ScalarParam(1.0)),
            RBF(),
        )
        for noise in ((HomoscedasticNoise(1e-5),))
        for gp in [MuyGPS]
        # for gp in (MuyGPS, BenchmarkGP)
    )
    def test_bounds_defaults_init(self, kernel, noise, gp_type):
        muygps = gp_type(kernel=kernel, noise=noise)
        for name, param in kernel._hyperparameters.items():
            self.assertEqual(
                param(),
                muygps.kernel._hyperparameters[name](),
            )
            self.assertTrue(
                muygps.kernel._hyperparameters[name].fixed(),
            )
        self.assertEqual(noise(), muygps.noise())
        self.assertTrue(muygps.noise.fixed())
        if gp_type == MuyGPS:
            self.assertFalse(muygps.scale.trained)
            self.assertEqual(mm.array([1.0]), muygps.scale())
        # elif gp_type == BenchmarkGP:
        #     self.assertFalse(muygps.scale.trained)
        #     self.assertEqual(mm.array([1.0]), muygps.scale())

    @parameterized.parameters(
        (kernel, e, gp)
        for kernel in (
            Matern(
                smoothness=ScalarParam(1.0, (1e-2, 5e4)),
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                ),
            ),
            Matern(
                smoothness=ScalarParam(1.0),
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                ),
            ),
            Matern(
                smoothness=ScalarParam(1.0, (1e-2, 5e4)),
                deformation=Anisotropy(
                    l2,
                    length_scale0=ScalarParam(2.0, (0.0, 3.0)),
                    length_scale1=ScalarParam(2.0, (0.0, 3.0)),
                ),
            ),
            Matern(
                smoothness=ScalarParam(1.0),
                deformation=Anisotropy(
                    l2,
                    length_scale0=ScalarParam(2.0, (0.0, 3.0)),
                    length_scale1=ScalarParam(2.0, (0.0, 3.0)),
                ),
            ),
            RBF(
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                )
            ),
            RBF(
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                )
            ),
            RBF(
                deformation=Anisotropy(
                    F2,
                    length_scale0=ScalarParam(2.0, (0.0, 3.0)),
                    length_scale1=ScalarParam(2.0, (0.0, 3.0)),
                )
            ),
        )
        for e in (
            (
                HomoscedasticNoise(1e-5, (1e-8, 1e-2)),
                HomoscedasticNoise(1e-5, "fixed"),
            )
        )
        for gp in [MuyGPS]
        # for gp in (MuyGPS, BenchmarkGP)
    )
    def test_full_init(self, kernel, noise, gp_type):
        muygps = gp_type(kernel=kernel, noise=noise)
        for name, param in kernel._hyperparameters.items():
            self.assertEqual(
                param(),
                muygps.kernel._hyperparameters[name](),
            )
            if param.fixed() is True:
                self.assertTrue(muygps.kernel._hyperparameters[name].fixed())
            else:
                self.assertFalse(muygps.kernel._hyperparameters[name].fixed())
                self.assertEqual(
                    param.get_bounds(),
                    muygps.kernel._hyperparameters[name].get_bounds(),
                )
        self.assertEqual(noise(), muygps.noise())
        if noise.fixed() is True:
            self.assertTrue(muygps.noise.fixed())
        else:
            self.assertFalse(muygps.noise.fixed())
            self.assertEqual(noise.get_bounds(), muygps.noise.get_bounds())
        if gp_type == MuyGPS:
            self.assertFalse(muygps.scale.trained)
            self.assertEqual(1.0, muygps.scale())
        # elif gp_type == BenchmarkGP:
        #     self.assertFalse(muygps.scale.trained)
        #     self.assertEqual(mm.array([1.0]), muygps.scale())

    @parameterized.parameters(
        (kernel, e, gp, 100)
        for kernel in (
            Matern(
                smoothness=ScalarParam("sample", (1e-2, 5e4)),
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                ),
            ),
            RBF(
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                )
            ),
            Matern(
                smoothness=ScalarParam("log_sample", (1e-2, 5e4)),
                deformation=Isotropy(
                    l2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                ),
            ),
            RBF(
                deformation=Isotropy(
                    F2, length_scale=ScalarParam(2.0, (0.0, 3.0))
                )
            ),
            Matern(
                smoothness=ScalarParam("log_sample", (1e-2, 5e4)),
                deformation=Anisotropy(
                    l2,
                    length_scale0=ScalarParam(2.0, (0.0, 3.0)),
                    length_scale1=ScalarParam(2.0, (0.0, 3.0)),
                ),
            ),
            RBF(
                deformation=Anisotropy(
                    F2,
                    length_scale0=ScalarParam(2.0, (0.0, 3.0)),
                    length_scale1=ScalarParam(2.0, (0.0, 3.0)),
                )
            ),
        )
        for e in (
            (
                HomoscedasticNoise("sample", (1e-8, 1e-2)),
                HomoscedasticNoise("log_sample", (1e-8, 1e-2)),
            )
        )
        for gp in [MuyGPS]
        # for gp in (MuyGPS, BenchmarkGP)
    )
    def test_sample_init(self, kernel, noise, gp_type, its):
        for _ in range(its):
            muygps = gp_type(kernel=kernel, noise=noise)
            for name, param in kernel._hyperparameters.items():
                self._check_in_bounds(
                    param.get_bounds(),
                    muygps.kernel._hyperparameters[name],
                )
            self._check_in_bounds(noise.get_bounds(), muygps.noise)

    def _check_in_bounds(self, given_bounds, param):
        bounds = param.get_bounds()
        self.assertEqual(given_bounds, bounds)
        self.assertGreaterEqual(param(), bounds[0])
        self.assertLessEqual(param(), bounds[1])


class GPTestCase(parameterized.TestCase):
    def _prepare_tensors(
        self,
        muygps,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
    ):
        # prepare data
        train, test = _make_gaussian_data(
            train_count, test_count, feature_count, response_count
        )
        train_features = train["input"]
        train_responses = train["output"]
        test_features = test["input"]
        test_responses = test["output"]
        _check_ndarray(self.assertEqual, train_features, mm.ftype)
        _check_ndarray(self.assertEqual, train_responses, mm.ftype)
        _check_ndarray(self.assertEqual, test_features, mm.ftype)
        _check_ndarray(self.assertEqual, test_responses, mm.ftype)

        # neighbors and differences
        nbrs_lookup = NN_Wrapper(train_features, nn_count, **nn_kwargs)
        test_nn_indices, _ = nbrs_lookup.get_nns(test_features)
        indices = mm.arange(test_count)
        _check_ndarray(self.assertEqual, indices, mm.itype)

        (
            crosswise_diffs,
            pairwise_diffs,
            batch_nn_targets,
        ) = make_predict_tensors(
            indices,
            test_nn_indices,
            test_features,
            train_features,
            train_responses,
        )
        _check_ndarray(self.assertEqual, crosswise_diffs, mm.ftype)
        _check_ndarray(self.assertEqual, pairwise_diffs, mm.ftype)
        _check_ndarray(self.assertEqual, batch_nn_targets, mm.ftype)

        # make kernels
        K = _consistent_unchunk_tensor(muygps.kernel(pairwise_diffs))
        Kcross = _consistent_unchunk_tensor(muygps.kernel(crosswise_diffs))
        _check_ndarray(self.assertEqual, K, mm.ftype)
        _check_ndarray(self.assertEqual, Kcross, mm.ftype)
        # do validation
        self.assertEqual(K.shape, (test_count, nn_count, nn_count))
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        return (
            K,
            Kcross,
            batch_nn_targets,
            train_responses,
            test_nn_indices,
            pairwise_diffs,
        )


class GPTensorShapesTest(GPTestCase):
    @parameterized.parameters(
        (
            (1000, 100, f, 10, nn_kwargs, kwargs)
            for f in [100, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [2]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for kwargs in (
                {
                    "kernel": Matern(smoothness=ScalarParam(1.5)),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_tensor_shapes(
        self,
        train_count,
        test_count,
        feature_count,
        nn_count,
        nn_kwargs,
        kwargs,
    ):
        muygps = MuyGPS(**kwargs)

        K, Kcross, _, _, _, _ = self._prepare_tensors(
            muygps,
            train_count,
            test_count,
            feature_count,
            1,
            nn_count,
            nn_kwargs,
        )
        self.assertTrue(mm.all(K >= 0.0))
        self.assertTrue(mm.all(K <= 1.0))
        self.assertTrue(mm.all(Kcross >= 0.0))
        self.assertTrue(mm.all(Kcross <= 1.0))
        # # Check that kernels are positive semidefinite
        if config.state.low_precision() is True:
            _warn0(
                "numpy/jax/torch eigensolves are unstable using float32, "
                "skipping tests"
            )
            return
        for i in range(K.shape[0]):
            eigvals = mm.linalg.eigvals(K[i, :, :])
            # eigvals = mm.array(eigvals, dtype=mm.ftype)
            eigvals = eigvals.real
            _check_ndarray(self.assertEqual, eigvals, mm.ftype)
            self.assertTrue(
                mm.all(
                    mm.logical_or(
                        eigvals >= 0.0,
                        mm.isclose(eigvals, mm.zeros(eigvals.shape)),
                    )
                )
            )


class HomoscedasticNoiseTest(GPTestCase):
    @parameterized.parameters(
        (
            (1000, 100, f, r, 10, nn_kwargs, kwargs)
            for f in [100, 1]
            for r in [5, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [1]
            # for r in [1]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for kwargs in (
                {
                    "kernel": Matern(smoothness=ScalarParam(1.5)),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_homoscedastic_perturb(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kwargs,
    ):
        muygps = MuyGPS(**kwargs)

        K, Kcross, _, _, _, _ = self._prepare_tensors(
            muygps,
            train_count,
            test_count,
            feature_count,
            response_count,
            nn_count,
            nn_kwargs,
        )

        perturbed_K = _homoscedastic_perturb(K, muygps.noise())
        _check_ndarray(self.assertEqual, perturbed_K, mm.ftype)

        manual_K = K + muygps.noise() * mm.eye(nn_count)
        self.assertTrue(mm.allclose(perturbed_K, manual_K))
        return perturbed_K, Kcross


class HeteroscedasticNoiseTest(GPTestCase):
    @parameterized.parameters(
        (
            (1000, 100, f, r, 10, nn_kwargs, kernel)
            for f in [100, 1]
            for r in [5, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [1]
            # for r in [1]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for kernel in (
                Matern(smoothness=ScalarParam(1.5)),
                RBF(),
            )
        )
    )
    def test_heteroscedastic_perturb(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kernel,
    ):
        test_indices = (
            np.repeat(range(test_count), nn_count),
            np.tile(np.arange(nn_count), test_count),
            np.tile(np.arange(nn_count), test_count),
        )
        noise_tensor = mm.zeros((test_count, nn_count, nn_count))
        noise_matrix = _make_heteroscedastic_test_nugget(
            test_count, nn_count, 1e-5
        )
        noise_tensor = mm.assign(
            noise_tensor, noise_matrix.flatten(), *test_indices
        )
        muygps = MuyGPS(kernel, noise=HeteroscedasticNoise(noise_matrix))

        K, Kcross, _, _, _, _ = self._prepare_tensors(
            muygps,
            train_count,
            test_count,
            feature_count,
            response_count,
            nn_count,
            nn_kwargs,
        )

        perturbed_K = _heteroscedastic_perturb(K, muygps.noise())
        _check_ndarray(self.assertEqual, perturbed_K, mm.ftype)

        manual_K = K + noise_tensor
        self.assertTrue(mm.allclose(perturbed_K, manual_K))
        return perturbed_K, Kcross


class GPSolveTest(GPTestCase):
    @parameterized.parameters(
        (
            (1000, 100, f, r, 10, nn_kwargs, kwargs)
            for f in [100, 1]
            for r in [5, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [1]
            # for r in [1]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for kwargs in (
                {
                    "kernel": Matern(smoothness=ScalarParam(1.5)),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_tensor_solve(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kwargs,
    ):
        if config.state.low_precision() is True and feature_count < 10:
            _warn0(
                "test_tensor_solve() is unstable in low precision mode when "
                "distances are small. Skipping."
            )
            return
        muygps = MuyGPS(**kwargs)

        (
            K,
            Kcross,
            batch_nn_targets,
            train_responses,
            test_nn_indices,
            _,
        ) = self._prepare_tensors(
            muygps,
            train_count,
            test_count,
            feature_count,
            response_count,
            nn_count,
            nn_kwargs,
        )
        responses = _consistent_unchunk_tensor(
            muygps.posterior_mean(K, Kcross, batch_nn_targets)
        )
        _check_ndarray(self.assertEqual, responses, mm.ftype)

        # validate
        self.assertEqual(responses.shape, (test_count, response_count))

        for i in range(test_count):
            manual_responses = Kcross[i, :] @ mm.linalg.solve(
                K[i, :, :] + muygps.noise() * mm.eye(nn_count),
                train_responses[test_nn_indices[i], :],
            )
            _check_ndarray(self.assertEqual, manual_responses, mm.ftype)
            _consistent_assert(
                _precision_assert,
                self.assertSequenceAlmostEqual,
                np.array(responses[i, :]),
                np.array(manual_responses),
            )


class GPDiagonalVariance(GPTestCase):
    @parameterized.parameters(
        (
            (1000, 100, f, r, 10, nn_kwargs, kwargs)
            for f in [100, 1]
            for r in [10, 2, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [1]
            # for r in [10]
            for kwargs in (
                {
                    "kernel": Matern(smoothness=ScalarParam(1.5)),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_diagonal_variance(
        self,
        train_count,
        test_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        kwargs,
    ):
        muygps = MuyGPS(**kwargs)

        K, Kcross, _, _, _, _ = self._prepare_tensors(
            muygps,
            train_count,
            test_count,
            feature_count,
            response_count,
            nn_count,
            nn_kwargs,
        )
        diagonal_variance = _consistent_unchunk_tensor(
            muygps.posterior_variance(K, Kcross)
        )
        _check_ndarray(self.assertEqual, diagonal_variance, mm.ftype)

        # validate
        self.assertEqual(diagonal_variance.shape, (test_count, 1))
        diagonal_variance = diagonal_variance.reshape((test_count,))
        for i in range(test_count):
            manual_diagonal_variance = mm.array(1.0) - Kcross[
                i, :
            ] @ mm.linalg.solve(
                K[i, :, :] + muygps.noise() * mm.eye(nn_count),
                Kcross[i, :],
            )
            self.assertEqual(manual_diagonal_variance.dtype, mm.ftype)
            _precision_assert(
                _consistent_assert,
                self.assertAlmostEqual,
                np.array(diagonal_variance[i]),
                np.array(manual_diagonal_variance),
            )
            self.assertGreater(diagonal_variance[i], 0.0)


class MakeClassifierTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 1000, 10, b, n, nn_kwargs, lf, opt_fn_and_kwargs, kwargs)
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lf in [mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            for kwargs in (
                {
                    "kernel": Matern(
                        smoothness=ScalarParam("sample", (1e-1, 1e0))
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_make_multivariate_classifier(
        self,
        train_count,
        test_count,
        feature_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_fn,
        opt_fn_and_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch":
            _warn0(f"{self.__class__} does not support torch")
            return

        opt_fn, opt_kwargs = opt_fn_and_kwargs

        response_count = 2
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=True,
        )

        muygps, _ = make_classifier(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
            opt_fn=opt_fn,
            opt_kwargs=opt_kwargs,
            verbose=False,
        )

        self.assertEqual(k_kwargs["noise"](), muygps.noise())
        for name, param in k_kwargs["kernel"]._hyperparameters.items():
            if param.fixed() is False:
                print(
                    f"optimized to find value "
                    f"{muygps.kernel._hyperparameters[name]()}"
                )
            else:
                self.assertEqual(
                    param(),
                    muygps.kernel._hyperparameters[name](),
                )


class MakeRegressorTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, 1000, 10, b, n, nn_kwargs, lf, opt_fn_and_kwargs, k_kwargs)
            for b in [250]
            for n in [10]
            for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for lf in [mse_fn]
            for opt_fn_and_kwargs in _basic_opt_fn_and_kwarg_options
            # for ssm in ["analytic"]
            for k_kwargs in (
                {
                    "kernel": Matern(
                        smoothness=ScalarParam("sample", (1e-1, 1e0))
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                    "scale": FixedScale(),
                },
                {
                    "kernel": Matern(
                        smoothness=ScalarParam("sample", (1e-1, 1e0))
                    ),
                    "noise": HomoscedasticNoise(1e-5),
                    "scale": AnalyticScale(),
                },
            )
        )
    )
    def test_make_multivariate_regressor(
        self,
        train_count,
        test_count,
        feature_count,
        batch_count,
        nn_count,
        nn_kwargs,
        loss_fn,
        opt_fn_and_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch":
            _warn0(f"{self.__class__} does not support torch")
            return

        opt_fn, opt_kwargs = opt_fn_and_kwargs

        response_count = 1
        # construct the observation locations
        train, test = _make_gaussian_data(
            train_count,
            test_count,
            feature_count,
            response_count,
            categorical=False,
        )

        muygps, _ = make_regressor(
            train["input"],
            train["output"],
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            opt_kwargs=opt_kwargs,
            nn_kwargs=nn_kwargs,
            k_kwargs=k_kwargs,
        )

        self.assertEqual(k_kwargs["noise"](), muygps.noise())
        for name, param in k_kwargs["kernel"]._hyperparameters.items():
            if param.fixed() is False:
                print(
                    f"optimized to find value "
                    f"{muygps.kernel._hyperparameters[name]()}"
                )
            else:
                self.assertEqual(
                    param(),
                    muygps.kernel._hyperparameters[name](),
                )

        self.assertTrue(muygps.scale.trained)
        if isinstance(muygps.scale, AnalyticScale):
            print(f"\toptimized scale to find value " f"{muygps.scale()}")
        else:
            self.assertEqual(mm.array([1.0]), muygps.scale())


class GPScaleTest(GPTestCase):
    @parameterized.parameters(
        (
            (1000, f, r, 10, nn_kwargs, k_kwargs)
            for f in [50, 1]
            for r in [5, 1]
            for nn_kwargs in _basic_nn_kwarg_options
            # for f in [1]
            # for r in [10]
            # for nn_kwargs in [_basic_nn_kwarg_options[0]]
            for k_kwargs in (
                {
                    "kernel": Matern(smoothness=ScalarParam(1.5)),
                    "noise": HomoscedasticNoise(1e-5),
                },
                {
                    "kernel": RBF(),
                    "noise": HomoscedasticNoise(1e-5),
                },
            )
        )
    )
    def test_batch_scale_shapes(
        self,
        data_count,
        feature_count,
        response_count,
        nn_count,
        nn_kwargs,
        k_kwargs,
    ):
        muygps = MuyGPS(
            scale=AnalyticScale(response_count=response_count), **k_kwargs
        )

        # prepare data
        data = _make_gaussian_dict(data_count, feature_count, response_count)

        # neighbors and differences
        nbrs_lookup = NN_Wrapper(data["input"], nn_count, **nn_kwargs)
        indices = mm.arange(data_count)
        nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
        (_, pairwise_diffs, _, nn_targets) = make_train_tensors(
            indices,
            nn_indices,
            data["input"],
            data["output"],
        )

        K = muygps.kernel(pairwise_diffs)
        muygps = muygps.optimize_scale(pairwise_diffs, nn_targets)

        K = _consistent_unchunk_tensor(K)
        nn_targets = _consistent_unchunk_tensor(nn_targets)
        _check_ndarray(self.assertEqual, K, mm.ftype)
        _check_ndarray(self.assertEqual, nn_targets, mm.ftype)

        if response_count > 1:
            self.assertEqual(len(muygps.scale()), response_count)
            for i in range(response_count):
                scales = _get_scale_series(
                    K,
                    nn_targets[:, :, i].reshape(data_count, nn_count, 1),
                    muygps.noise(),
                )
                _check_ndarray(self.assertEqual, scales, mm.ftype)
                self.assertEqual(scales.shape, (data_count,))
                _precision_assert(
                    self.assertAlmostEqual,
                    np.array(muygps.scale()[i]),
                    np.mean(np.array(scales)),
                    low_bound=0,
                    high_bound=5,
                )
        else:
            scales = _get_scale_series(
                K,
                nn_targets[:, :, 0].reshape(data_count, nn_count, 1),
                muygps.noise(),
            )
            self.assertEqual(scales.shape, (data_count,))
            _check_ndarray(self.assertEqual, scales, mm.ftype)
            _precision_assert(
                self.assertAlmostEqual,
                np.array(muygps.scale()[0]),
                np.mean(np.array(scales)),
                low_bound=0,
                high_bound=5,
            )


if __name__ == "__main__":
    absltest.main()
