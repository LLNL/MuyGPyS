# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
import numpy as np
import MuyGPyS._src.math as mm

from absl.testing import parameterized

from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import DifferenceIsotropy, F2
from MuyGPyS.gp.hyperparameter import ScalarParam, Parameter, FixedScale
from MuyGPyS.gp.kernels.experimental import ShearKernel
from MuyGPyS.gp.noise import HomoscedasticNoise


def kk_f(x1, y1, x2, y2, a=1, b=1):
    return (
        1
        / 4
        * (
            a
            * (
                8 * b**2
                - 8 * b * ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                + (x1 - x2) ** 4
                + 2 * (x1 - x2) ** 2 * (y1 - y2) ** 2
                + (y1 - y2) ** 4
            )
            * mm.exp(-((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * b))
            / b**4
        )
    )


def kg1_f(x1, y1, x2, y2, a=1, b=1):
    return (
        1
        / 4
        * (
            a
            * (
                6 * b * (-((x1 - x2) ** 2) + (y1 - y2) ** 2)
                + (x1 - x2) ** 4
                - (y1 - y2) ** 4
            )
            * mm.exp(-((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * b))
            / b**4
        )
    )


def kg2_f(x1, y1, x2, y2, a=1, b=1):
    return (
        1
        / 4
        * (
            2
            * a
            * (x1 - x2)
            * (y1 - y2)
            * (-6 * b + (x1 - x2) ** 2 + (y1 - y2) ** 2)
            * mm.exp(-((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * b))
            / b**4
        )
    )


def g1g1_f(x1, y1, x2, y2, a=1, b=1):
    return (
        1
        / 4
        * (
            a
            * (
                4 * b**2
                - 4 * b * ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                + (x1 - x2) ** 4
                - 2 * (x1 - x2) ** 2 * (y1 - y2) ** 2
                + (y1 - y2) ** 4
            )
            * mm.exp(-((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * b))
            / b**4
        )
    )


def g1g2_f(x1, y1, x2, y2, a=1, b=1):
    return (
        1
        / 4
        * (
            2
            * a
            * (x1 - x2)
            * (y1 - y2)
            * ((x1 - x2) ** 2 - (y1 - y2) ** 2)
            * mm.exp(-((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * b))
            / b**4
        )
    )


def g2g2_f(x1, y1, x2, y2, a=1, b=1):
    return (
        1
        / 4
        * (
            4
            * a
            * (
                b**2
                - b * ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                + (x1 - x2) ** 2 * (y1 - y2) ** 2
            )
            * mm.exp(-((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * b))
            / b**4
        )
    )


# compute the full covariance matrix
def shear_kernel(x1, y1, x2, y2, a=1, b=1):
    full_m = np.zeros((3, 3))
    full_m[0, 0] = kk_f(x1, y1, x2, y2, a, b)
    full_m[0, 1] = kg1_f(x1, y1, x2, y2, a, b)
    full_m[0, 2] = kg2_f(x1, y1, x2, y2, a, b)
    full_m[1, 1] = g1g1_f(x1, y1, x2, y2, a, b)
    full_m[1, 2] = g1g2_f(x1, y1, x2, y2, a, b)
    full_m[2, 2] = g2g2_f(x1, y1, x2, y2, a, b)
    full_m[1, 0] = full_m[0, 1]
    full_m[2, 0] = full_m[0, 2]
    full_m[2, 1] = full_m[1, 2]

    return full_m


def original_shear(X1, X2=None, length_scale=1.0):
    if X2 is None:
        X2 = X1
    n1, _ = X1.shape
    n2, _ = X2.shape
    vals = np.zeros((3 * (n1), 3 * (n2)))
    vals[:] = np.nan
    for i, (ix, iy) in enumerate(X1):
        for j, (jx, jy) in enumerate(X2):
            tmp = shear_kernel(ix, iy, jx, jy, b=length_scale)
            for a in range(3):
                for b in range(3):
                    vals[(a * n1) + i, (b * n2) + j] = tmp[a, b]
    return vals


def targets_from_GP(features, n, ls):
    Kernel = ShearKernel(
        deformation=DifferenceIsotropy(
            F2,
            length_scale=Parameter(ls),
        ),
    )
    diffs = Kernel.deformation.pairwise_tensor(features, np.arange(features.shape[0]))
    Kin = 1.0 * Kernel(diffs, adjust=False)
    Kin_flat = Kin.reshape(3 * n**2, 3 * n**2) + 1e-10 * np.identity(3 * n**2)
    e = np.random.normal(0, 1, 3 * n**2)
    L = np.linalg.cholesky(Kin_flat)
    targets = np.dot(L, e).reshape(3, n**2).swapaxes(0, 1)
    return (targets)


def conventional_mean(Kin, Kcross, targets, noise):
    nugget_size = Kin.shape[0]
    test_count = int(Kcross.shape[0] / 3)
    return (
        Kcross @ np.linalg.solve(
            Kin + noise * np.eye(nugget_size),
            targets,
        )
    ).reshape(3, test_count).swapaxes(0, 1)


def conventional_variance(Kin, Kcross, Kin_test, noise):
    nugget_size = Kin.shape[0]
    return (
        Kin_test - Kcross @ np.linalg.solve(
            Kin + noise * np.eye(nugget_size),
            Kcross.T,
        )
    )


class BenchmarkTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTestCase, cls).setUpClass()
        cls.n = 25
        cls.length_scale = 0.05
        cls.noise_prior = 1e-4
        cls.nn_count = 50
        cls.features = np.vstack(
            (
                np.meshgrid(
                    np.linspace(0, 1, cls.n) ,
                    np.linspace(0, 1, cls.n)
                )[0].flatten(),
                np.meshgrid(
                    np.linspace(0, 1, cls.n) ,
                    np.linspace(0, 1, cls.n)
                )[1].flatten(),
            )
        ).T
        cls.dist_fn = DifferenceIsotropy(
            metric=F2,
            length_scale=ScalarParam(cls.length_scale),
        )
        cls.targets = targets_from_GP(cls.features, cls.n, cls.length_scale)
        cls.library_shear = MuyGPS(
            kernel=ShearKernel(
                deformation=DifferenceIsotropy(
                    F2,
                    length_scale=Parameter(cls.length_scale),
                ),
            ),
            noise=HomoscedasticNoise(cls.noise_prior),
        )
        cls.optimize_model = MuyGPS(
            kernel=ShearKernel(
                deformation=DifferenceIsotropy(
                    F2,
                    length_scale=Parameter(0.04, [0.01, 0.07]),
                ),
            ),
            noise=HomoscedasticNoise(cls.noise_prior),
            scale=FixedScale(),
        )
