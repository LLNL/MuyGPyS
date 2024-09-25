# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

from sklearn.gaussian_process.kernels import Matern as Matern_sk
from sklearn.gaussian_process.kernels import RBF as RBF_sk

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS import config

from MuyGPyS._src.mpi_utils import _consistent_unchunk_tensor, _warn0
from MuyGPyS._test.utils import (
    _basic_nn_kwarg_options,
    _check_ndarray,
    _consistent_assert,
    _make_gaussian_matrix,
)
from MuyGPyS.gp.hyperparameter import ScalarParam, FixedScale, VectorParam
from MuyGPyS.gp.kernels import RBF, Matern
from MuyGPyS.gp.deformation import Anisotropy, Isotropy, F2, l2
from MuyGPyS.neighbors import NN_Wrapper


class DistancesTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs)
            # for f in [100]
            # for nn in [10]
            # for nn_kwargs in [_basic_nn_kwarg_options][0]
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
        )
    )
    def test_l2(
        self, train_count, feature_count, nn_count, test_count, nn_kwargs
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        points = train[nn_indices]
        self.assertEqual(points.shape, (test_count, nn_count, feature_count))
        dists = mm.array(
            np.array(
                [
                    np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1)
                    ** 2
                    for mat in points
                ]
            )
        )
        _check_ndarray(
            self.assertEqual,
            dists,
            mm.ftype,
            shape=(test_count, nn_count, nn_count),
        )
        ll_diffs = points[:, :, None, :] - points[:, None, :, :]
        _check_ndarray(
            self.assertEqual,
            ll_diffs,
            mm.ftype,
            shape=(test_count, nn_count, nn_count, feature_count),
        )
        ll_dists = ll_diffs**2
        ll_dists = mm.sum(ll_dists, axis=-1)
        _check_ndarray(
            self.assertEqual,
            ll_dists,
            mm.ftype,
            shape=(test_count, nn_count, nn_count),
        )
        rbf_iso = RBF(deformation=Isotropy(F2, length_scale=ScalarParam(1.0)))
        dists = _consistent_unchunk_tensor(
            rbf_iso.deformation.pairwise_tensor(train, nn_indices)
        )
        rbf_aniso = RBF(
            deformation=Anisotropy(
                F2, length_scale=VectorParam(ScalarParam(1.0), ScalarParam(1.0))
            )
        )
        diffs = _consistent_unchunk_tensor(
            rbf_aniso.deformation.pairwise_tensor(train, nn_indices)
        )
        _check_ndarray(
            self.assertEqual,
            dists,
            mm.ftype,
            shape=(test_count, nn_count, nn_count),
        )
        _check_ndarray(
            self.assertEqual,
            diffs,
            mm.ftype,
            shape=(test_count, nn_count, nn_count, feature_count),
        )
        l2_dists = l2(diffs)
        F2_dists = F2(diffs)
        _check_ndarray(
            self.assertEqual,
            l2_dists,
            mm.ftype,
            shape=(test_count, nn_count, nn_count),
        )
        _check_ndarray(
            self.assertEqual,
            F2_dists,
            mm.ftype,
            shape=(test_count, nn_count, nn_count),
        )
        _consistent_assert(self.assertTrue, mm.allclose(dists, F2_dists))
        _consistent_assert(self.assertTrue, mm.allclose(dists, l2_dists**2))
        _consistent_assert(self.assertTrue, mm.allclose(dists, ll_dists))


class BackendConfigUser:
    def __init__(self, _backend: str):
        self.backend = _backend

    def __enter__(self):
        self.state = config.state.backend
        config.update("muygpys_backend", self.backend)
        return self.state

    def __exit__(self, *args):
        config.update("muygpys_backend", self.state)


class ScaleTest(parameterized.TestCase):
    def _do_untrained(self, val):
        param = FixedScale()
        self.assertFalse(param.trained)
        self.assertEqual(1.0, param())
        param._set(val)
        self.assertEqual(val, param())

    @parameterized.parameters(v for v in [5.0])
    def test_untrained_good(self, val):
        self._do_untrained(val)

    def test_untrained_bad(self):
        with self.assertRaisesRegex(ValueError, "Scale parameter"):
            self._do_untrained([5.0])


class HyperparameterTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1.0, "bounds": (1e-5, 1e1)},
                {"val": 1e-5, "bounds": (1e-5, 1e1)},
                {"val": 1e1, "bounds": (1e-5, 1e1)},
            )
        )
    )
    def test_full_init(self, val, bounds):
        param = ScalarParam(val, bounds)
        self.assertEqual(val, param())
        self._check_in_bounds(bounds, param)

    def _check_in_bounds(self, given_bounds, param):
        bounds = param.get_bounds()
        self.assertEqual(given_bounds, bounds)
        self.assertGreaterEqual(param(), bounds[0])
        self.assertLessEqual(param(), bounds[1])

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": "fixed"},))
    )
    def test_fixed_init(self, val, bounds):
        param = ScalarParam(val, bounds)
        self.assertEqual(val, param())
        self.assertTrue(param.fixed())

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": "sample", "bounds": (1e-4, 1e2), "its": 100},
                {"val": "log_sample", "bounds": (1e-4, 1e2), "its": 100},
            )
        )
    )
    def test_sample(self, val, bounds, its):
        for _ in range(its):
            param = ScalarParam(val, bounds)
            self._check_in_bounds(bounds, param)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": "sample", "bounds": "fixed"},
                {"val": "log_sample", "bounds": "fixed"},
            )
        )
    )
    def test_fixed_sample(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "Fixed bounds do not support "):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1e-2, "bounds": (1e-1, 1e2)},
                {"val": 1e3, "bounds": (1e-1, 1e2)},
            )
        )
    )
    def test_oob(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "bound"):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": mm.array([1e-2, 1e1]), "bounds": "fixed"},
                {"val": mm.array([1e3]), "bounds": (1e-1, 1e2)},
            )
        )
    )
    def test_nonscalar(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "Nonscalar hyperparameter"):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": "wut", "bounds": (1e-1, 1e2)},))
    )
    def test_bad_val_string(self, val, bounds):
        with self.assertRaisesRegex(
            ValueError, "Unsupported string hyperparameter"
        ):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": "sample", "bounds": "fixed"},))
    )
    def test_string_on_fixed(self, val, bounds):
        with self.assertRaisesRegex(
            ValueError, "Fixed bounds do not support string"
        ):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": "badstring"},))
    )
    def test_bad_val_bounds(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": (1e2, 1e-1)},))
    )
    def test_bad_bounds(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "not lesser than upper bound"):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1.0, "bounds": (1e2, 1e-1, 1e2)},
                {"val": 1.0, "bounds": [1e2]},
            )
        )
    )
    def test_bad_bound_length(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "unsupported length"):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (
            kwargs
            for kwargs in (
                {"val": 1.0, "bounds": ("a", 1e-2)},
                {"val": 1.0, "bounds": (1e-1, "b")},
            )
        )
    )
    def test_bad_bound_vals(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "supported hyperparameter"):
            ScalarParam(val, bounds)

    @parameterized.parameters(
        (kwargs for kwargs in ({"val": 1.0, "bounds": 1e-2},))
    )
    def test_noniterable_bound(self, val, bounds):
        with self.assertRaisesRegex(ValueError, "non-iterable type"):
            ScalarParam(val, bounds)


class KernelTestCase(parameterized.TestCase):
    def _check_params_chassis(self, kernel_fn, **kwargs):
        for p in kernel_fn._hyperparameters:
            self._check_params(
                kernel_fn,
                p,
                kwargs.get(p),
            )

    def _check_params(self, kernel_fn, name, param):
        if param() is not None:
            self.assertEqual(param(), kernel_fn._hyperparameters[name]())
        if param.get_bounds() is not None:
            if param.fixed():
                self.assertTrue(kernel_fn._hyperparameters[name].fixed())
            else:
                self.assertFalse(kernel_fn._hyperparameters[name].fixed())
                self.assertEqual(
                    param.get_bounds(),
                    kernel_fn._hyperparameters[name].get_bounds(),
                )


class RBFTest(KernelTestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                # {"length_scale": ScalarParam(1.0, (1e-5, 1e1))},
                {"length_scale": ScalarParam(2.0)},
            ]
        )
    )
    def test_rbf(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, _ = nbrs_lookup.get_nns(test)
        deformation_model = Isotropy(F2, **k_kwargs)
        rbf = RBF(deformation=deformation_model)
        pairwise_dists = rbf.deformation.pairwise_tensor(train, nn_indices)
        self._check_params_chassis(rbf, **k_kwargs)
        Kin = _consistent_unchunk_tensor(rbf(pairwise_dists))
        self.assertEqual(Kin.shape, (test_count, nn_count, nn_count))
        points = train[nn_indices]
        rbf_sk = RBF_sk(length_scale=deformation_model.length_scale())
        Kin_sk = mm.array(np.array([rbf_sk(mat) for mat in points]))
        self.assertEqual(Kin_sk.shape, (test_count, nn_count, nn_count))
        _consistent_assert(self.assertTrue, mm.allclose(Kin, Kin_sk))
        crosswise_dists = rbf.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        Kcross = rbf(crosswise_dists)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        Kcross_sk = mm.array(
            np.array([rbf_sk(vec, mat) for vec, mat in zip(test, points)])
        ).reshape(test_count, nn_count)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertEqual(Kcross.dtype, Kcross_sk.dtype)
        _check_ndarray(self.assertEqual, Kcross, mm.ftype)
        self.assertTrue(mm.allclose(Kcross, Kcross_sk))


class ParamTest(KernelTestCase):
    @parameterized.parameters(
        (
            (k_kwargs, alt_kwargs)
            for k_kwargs in [{"length_scale": ScalarParam(10.0, (1e-5, 1e1))}]
            for alt_kwargs in [
                {"length_scale": ScalarParam(1.0, (1e-2, 1e4))},
                {"length_scale": ScalarParam("sample", (1e-3, 1e2))},
                {"length_scale": ScalarParam(2.0)},
            ]
        )
    )
    def test_rbf(self, k_kwargs, alt_kwargs):
        deformation_model = Isotropy(F2, length_scale=k_kwargs["length_scale"])
        self._test_chassis(RBF(deformation_model), k_kwargs, alt_kwargs)

    @parameterized.parameters(
        (
            (k_kwargs, alt_kwargs)
            for k_kwargs in [
                {
                    "smoothness": ScalarParam(0.42, (1e-4, 5e1)),
                    "length_scale": ScalarParam(1.0, (1e-5, 1e1)),
                }
            ]
            for alt_kwargs in [
                {
                    "smoothness": ScalarParam(1.0, (1e-2, 5e4)),
                    "length_scale": ScalarParam(7.2, (2e-5, 2e1)),
                },
                {
                    "smoothness": ScalarParam(1.0),
                    "length_scale": ScalarParam("sample", (2e-5, 2e1)),
                },
                {
                    "smoothness": ScalarParam("sample", (1e-2, 5e4)),
                },
            ]
        )
    )
    def test_matern(self, k_kwargs, alt_kwargs):
        deformation_model = Isotropy(l2, length_scale=k_kwargs["length_scale"])
        kernel_fn = Matern(
            deformation=deformation_model, smoothness=k_kwargs["smoothness"]
        )
        self._test_chassis(kernel_fn, k_kwargs, alt_kwargs)

    def _test_chassis(self, kernel_fn, k_kwargs, alt_kwargs):
        self._check_params_chassis(kernel_fn, **k_kwargs)
        # kernel_fn.set_params(**alt_kwargs)
        # self._check_params_chassis(kernel_fn, **alt_kwargs)


class MaternTest(KernelTestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "smoothness": ScalarParam(0.42, "fixed"),
                    "length_scale": ScalarParam(0.5),
                },
                {
                    "smoothness": ScalarParam(0.5),
                    "length_scale": ScalarParam(0.5),
                },
                {
                    "smoothness": ScalarParam(1.5),
                    "length_scale": ScalarParam(0.5),
                },
                {
                    "smoothness": ScalarParam(2.5),
                    "length_scale": ScalarParam(0.5),
                },
                {
                    "smoothness": ScalarParam(mm.inf),
                    "length_scale": ScalarParam(0.5),
                },
            ]
            # for f in [1]
            # for nn in [5]
            # for nn_kwargs in [_basic_nn_kwarg_options[1]]
            # for k_kwargs in [
            #     {
            #         "smoothness": {"val": mm.inf, "bounds": "fixed"},
            #         "length_scale": {"val": 1.0, "bounds": (1e-5, 1e1)},
            #     }
            # ]
        )
    )
    def test_matern(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [
            0.5,
            1.5,
            2.5,
            mm.inf,
        ]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        mtn = Matern(
            smoothness=k_kwargs["smoothness"],
            deformation=Isotropy(l2, length_scale=k_kwargs["length_scale"]),
        )
        pairwise_diffs = mtn.deformation.pairwise_tensor(train, nn_indices)
        # mtn = Matern(**k_kwargs)
        self._check_params_chassis(mtn, **k_kwargs)
        Kin = _consistent_unchunk_tensor(mtn(pairwise_diffs))
        self.assertEqual(Kin.shape, (test_count, nn_count, nn_count))
        points = train[nn_indices]
        mtn_sk = Matern_sk(
            nu=mtn.smoothness(),
            length_scale=mtn.deformation.length_scale(),
        )
        Kin_sk = mm.array(np.array([mtn_sk(mat) for mat in points]))
        self.assertEqual(Kin_sk.shape, (test_count, nn_count, nn_count))
        _consistent_assert(self.assertTrue, mm.allclose(Kin, Kin_sk))
        crosswise_diffs = mtn.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        Kcross = mtn(crosswise_diffs)
        Kcross_sk = mm.array(
            np.array(
                [mtn_sk(vec, mat) for vec, mat in zip(test, points)]
            ).reshape(test_count, nn_count)
        )
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertEqual(Kcross.dtype, Kcross_sk.dtype)
        _check_ndarray(self.assertEqual, Kcross, mm.ftype)
        self.assertTrue(mm.allclose(Kcross, Kcross_sk))


class AnisotropicShapesTest(KernelTestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [1, 3, 5, 10]
            for nn in [30]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "smoothness": ScalarParam(0.42, "fixed"),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
            ]
        )
    )
    def test_correct_number_length_scales(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [
            0.5,
            1.5,
            2.5,
            mm.inf,
        ]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        deformation_model = Anisotropy(
            metric=l2,
            length_scale=VectorParam(
                k_kwargs["length_scale0"], k_kwargs["length_scale1"]
            ),
        )
        mtn = Matern(
            smoothness=k_kwargs["smoothness"], deformation=deformation_model
        )
        pairwise_diffs = mtn.deformation.pairwise_tensor(train, nn_indices)
        with self.assertRaisesRegex(ValueError, "Difference tensor of shape "):
            _ = _consistent_unchunk_tensor(mtn(pairwise_diffs))


class AnisotropicMaternTest(KernelTestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "smoothness": ScalarParam(0.42, "fixed"),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "smoothness": ScalarParam(0.5),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "smoothness": ScalarParam(1.5),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "smoothness": ScalarParam(2.5),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "smoothness": ScalarParam(mm.inf),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
            ]
        )
    )
    def test_anisotropic_matern(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [0.5, 1.5, 2.5, mm.inf]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        deformation_model = Anisotropy(
            metric=l2,
            length_scale=VectorParam(
                k_kwargs["length_scale0"], k_kwargs["length_scale1"]
            ),
        )
        mtn = Matern(
            smoothness=k_kwargs["smoothness"], deformation=deformation_model
        )
        pairwise_diffs = mtn.deformation.pairwise_tensor(train, nn_indices)
        # mtn = Matern(**k_kwargs)
        self._check_params_chassis(
            mtn,
            **{
                "smoothness": k_kwargs["smoothness"],
                "length_scale0": k_kwargs["length_scale0"],
                "length_scale1": k_kwargs["length_scale1"],
            },
        )
        Kin = _consistent_unchunk_tensor(mtn(pairwise_diffs))
        self.assertEqual(Kin.shape, (test_count, nn_count, nn_count))
        points = train[nn_indices]
        length_scale0 = k_kwargs["length_scale0"]
        length_scale1 = k_kwargs["length_scale1"]
        mtn_sk = Matern_sk(
            nu=mtn.smoothness(),
            length_scale=mm.array([length_scale0(), length_scale1()]),
        )
        Kin_sk = mm.array(np.array([mtn_sk(mat) for mat in points]))
        self.assertEqual(Kin_sk.shape, (test_count, nn_count, nn_count))
        _consistent_assert(self.assertTrue, mm.allclose(Kin, Kin_sk))
        crosswise_diffs = mtn.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        Kcross = mtn(crosswise_diffs)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        Kcross_sk = mm.array(
            np.array(
                [mtn_sk(vec, mat) for vec, mat in zip(test, points)]
            ).reshape(test_count, nn_count)
        )
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertEqual(Kcross.dtype, Kcross_sk.dtype)
        _check_ndarray(self.assertEqual, Kcross, mm.ftype)
        self.assertTrue(mm.allclose(Kcross, Kcross_sk))


class AnisotropicRBFTest(KernelTestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "length_scale0": ScalarParam(0.1),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(10.0),
                },
                {
                    "length_scale0": ScalarParam(0.1),
                    "length_scale1": ScalarParam(10.0),
                },
                {
                    "length_scale0": ScalarParam(2.0),
                    "length_scale1": ScalarParam(0.01),
                },
            ]
        )
    )
    def test_anisotropic_rbf(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [0.5, 1.5, 2.5, mm.inf]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        deformation_model = Anisotropy(
            metric=F2,
            length_scale=VectorParam(
                k_kwargs["length_scale0"], k_kwargs["length_scale1"]
            ),
        )
        rbf = RBF(deformation=deformation_model)
        pairwise_diffs = rbf.deformation.pairwise_tensor(train, nn_indices)
        self._check_params_chassis(
            rbf,
            **{
                "length_scale0": k_kwargs["length_scale0"],
                "length_scale1": k_kwargs["length_scale1"],
            },
        )
        Kin = _consistent_unchunk_tensor(rbf(pairwise_diffs))
        self.assertEqual(Kin.shape, (test_count, nn_count, nn_count))
        points = train[nn_indices]
        length_scale0 = k_kwargs["length_scale0"]
        length_scale1 = k_kwargs["length_scale1"]
        sk_rbf = RBF_sk(
            length_scale=mm.array([length_scale0(), length_scale1()])
        )
        Kin_sk = mm.array(np.array([sk_rbf(mat) for mat in points]))
        self.assertEqual(Kin_sk.shape, (test_count, nn_count, nn_count))
        _consistent_assert(self.assertTrue, mm.allclose(Kin, Kin_sk))
        crosswise_diffs = rbf.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        Kcross = rbf(crosswise_diffs)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        Kcross_sk = mm.array(
            np.array([sk_rbf(vec, mat) for vec, mat in zip(test, points)])
        ).reshape(test_count, nn_count)
        self.assertEqual(Kcross.shape, (test_count, nn_count))
        self.assertEqual(Kcross.dtype, Kcross_sk.dtype)
        _check_ndarray(self.assertEqual, Kcross, mm.ftype)
        self.assertTrue(mm.allclose(Kcross, Kcross_sk))


class AnisotropicTest(KernelTestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "length_scale0": ScalarParam(0.1),
                    "length_scale1": ScalarParam(2.0),
                },
                {
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(10.0),
                },
                {
                    "length_scale0": ScalarParam(0.1),
                    "length_scale1": ScalarParam(10.0),
                },
                {
                    "length_scale0": ScalarParam(2.0),
                    "length_scale1": ScalarParam(0.01),
                },
            ]
        )
    )
    def test_anisotropic_1_length_scale_same_as_isotropic_rbf(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [0.5, 1.5, 2.5, mm.inf]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        deformation_model_aniso = Anisotropy(
            metric=F2,
            length_scale=VectorParam(
                k_kwargs["length_scale0"], k_kwargs["length_scale0"]
            ),
        )
        rbf_aniso = RBF(deformation=deformation_model_aniso)
        pairwise_diffs = rbf_aniso.deformation.pairwise_tensor(
            train, nn_indices
        )
        crosswise_diffs = rbf_aniso.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        self._check_params_chassis(
            rbf_aniso,
            **{
                "length_scale0": k_kwargs["length_scale0"],
                "length_scale1": k_kwargs["length_scale0"],
            },
        )

        deformation_model_iso = Isotropy(
            metric=F2,
            length_scale=k_kwargs["length_scale0"],
        )
        rbf_iso = RBF(deformation=deformation_model_iso)
        self._check_params_chassis(
            rbf_iso,
            **{
                "length_scale": k_kwargs["length_scale0"],
            },
        )
        pairwise_dists = rbf_iso.deformation.pairwise_tensor(train, nn_indices)
        crosswise_dists = rbf_iso.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )

        Kin_iso = _consistent_unchunk_tensor(rbf_iso(pairwise_dists))
        Kin_aniso = _consistent_unchunk_tensor(rbf_aniso(pairwise_diffs))
        Kcross_iso = rbf_iso(crosswise_dists)
        Kcross_aniso = rbf_aniso(crosswise_diffs)
        self.assertEqual(Kin_iso.shape, Kin_aniso.shape)
        self.assertEqual(Kcross_iso.shape, Kcross_aniso.shape)

    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "smoothness": ScalarParam(0.42, "fixed"),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(1.0),
                },
                {
                    "smoothness": ScalarParam(0.5),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(1.0),
                },
                {
                    "smoothness": ScalarParam(1.5),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(1.0),
                },
                {
                    "smoothness": ScalarParam(2.5),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(1.0),
                },
                {
                    "smoothness": ScalarParam(mm.inf),
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(1.0),
                },
            ]
        )
    )
    def test_anisotropic_same_length_scales_same_as_isotropic_matern(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [0.5, 1.5, 2.5, mm.inf]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        deformation_model_aniso = Anisotropy(
            metric=l2,
            length_scale=VectorParam(
                k_kwargs["length_scale0"], k_kwargs["length_scale1"]
            ),
        )
        mtn_aniso = Matern(
            smoothness=k_kwargs["smoothness"],
            deformation=deformation_model_aniso,
        )
        pairwise_diffs = mtn_aniso.deformation.pairwise_tensor(
            train, nn_indices
        )
        crosswise_diffs = mtn_aniso.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )

        self._check_params_chassis(
            mtn_aniso,
            **{
                "smoothness": k_kwargs["smoothness"],
                "length_scale0": k_kwargs["length_scale0"],
                "length_scale1": k_kwargs["length_scale1"],
            },
        )
        Kin_aniso = _consistent_unchunk_tensor(mtn_aniso(pairwise_diffs))

        deformation_model_iso = Isotropy(
            metric=l2,
            length_scale=k_kwargs["length_scale0"],
        )
        mtn_iso = Matern(
            smoothness=k_kwargs["smoothness"], deformation=deformation_model_iso
        )

        self._check_params_chassis(
            mtn_iso,
            **{
                "smoothness": k_kwargs["smoothness"],
                "length_scale": k_kwargs["length_scale0"],
            },
        )
        pairwise_dists = mtn_iso.deformation.pairwise_tensor(train, nn_indices)
        crosswise_dists = mtn_iso.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        Kin_iso = _consistent_unchunk_tensor(mtn_iso(pairwise_dists))
        self.assertTrue(mm.allclose(Kin_aniso, Kin_iso))
        Kcross_iso = mtn_iso(crosswise_dists)
        Kcross_aniso = mtn_aniso(crosswise_diffs)
        self.assertTrue(mm.allclose(Kcross_iso, Kcross_aniso))

    @parameterized.parameters(
        (
            (1000, f, nn, 10, nn_kwargs, k_kwargs)
            for f in [2]
            for nn in [5, 10, 100]
            for nn_kwargs in _basic_nn_kwarg_options
            for k_kwargs in [
                {
                    "length_scale0": ScalarParam(1.0),
                    "length_scale1": ScalarParam(1.0),
                },
            ]
        )
    )
    def test_anisotropic_same_length_scales_same_as_isotropic_rbf(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        nn_kwargs,
        k_kwargs,
    ):
        if config.state.backend == "torch" and k_kwargs["smoothness"][
            "val"
        ] not in [0.5, 1.5, 2.5, mm.inf]:
            bad_smoothness = k_kwargs["smoothness"]["val"]
            _warn0(
                "Skipping test because torch cannot handle Matern "
                f"smoothness={bad_smoothness}"
            )
            return
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, **nn_kwargs)
        nn_indices, nn_dists = nbrs_lookup.get_nns(test)
        nn_dists = mm.sqrt(nn_dists)
        deformation_model_aniso = Anisotropy(
            metric=F2,
            length_scale=VectorParam(
                k_kwargs["length_scale0"], k_kwargs["length_scale0"]
            ),
        )
        rbf_aniso = RBF(deformation=deformation_model_aniso)
        pairwise_diffs = rbf_aniso.deformation.pairwise_tensor(
            train, nn_indices
        )
        crosswise_diffs = rbf_aniso.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        self._check_params_chassis(
            rbf_aniso,
            **{
                "length_scale0": k_kwargs["length_scale0"],
                "length_scale1": k_kwargs["length_scale1"],
            },
        )
        Kin_aniso = _consistent_unchunk_tensor(rbf_aniso(pairwise_diffs))

        deformation_model_iso = Isotropy(
            metric=F2,
            length_scale=k_kwargs["length_scale0"],
        )
        rbf_iso = RBF(deformation=deformation_model_iso)
        pairwise_dists = rbf_iso.deformation.pairwise_tensor(train, nn_indices)
        self._check_params_chassis(
            rbf_iso,
            **{
                "length_scale": k_kwargs["length_scale0"],
            },
        )
        Kin_iso = _consistent_unchunk_tensor(rbf_iso(pairwise_dists))
        crosswise_dists = rbf_iso.deformation.crosswise_tensor(
            test, train, np.arange(test_count), nn_indices
        )
        Kcross_iso = rbf_iso(crosswise_dists)
        Kcross_aniso = rbf_aniso(crosswise_diffs)
        self.assertTrue(mm.allclose(Kin_iso, Kin_aniso))
        self.assertTrue(mm.allclose(Kcross_iso, Kcross_aniso))


if __name__ == "__main__":
    absltest.main()
