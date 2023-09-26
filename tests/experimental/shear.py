# Copyright 2023-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from absl.testing import absltest
from absl.testing import parameterized

import MuyGPyS._src.math as mm
from MuyGPyS._src.math import numpy as np

from MuyGPyS._src.gp.tensors import _pairwise_differences
from MuyGPyS._test.utils import _check_ndarray
from MuyGPyS.gp.deformation import Isotropy, F2
from MuyGPyS.gp.hyperparameter import ScalarHyperparameter
from MuyGPyS.gp.kernels.experimental import ShearKernel


# This stuff depends on Bob's repo's being installed in the same directory as
# the MuyGPyS repo, and on this test's being run from `${MUYGPYS_ROOT}/tests`.


import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "analytic_kernel", "../../shear_kernel/analytic_kernel.py"
)
foo = importlib.util.module_from_spec(spec)
sys.modules["analytic_kernel"] = foo
spec.loader.exec_module(foo)
from analytic_kernel import shear_kernel


class ShearKernelTest(parameterized.TestCase):
    def test_shear_kernel(self):
        n = 25  # number of galaxies on a side
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1

        xx = np.linspace(xmin, xmax, n)
        yy = np.linspace(ymin, ymax, n)

        x, y = np.meshgrid(xx, yy)
        features = np.vstack((x.flatten(), y.flatten())).T
        diffs = _pairwise_differences(mm.array(features))
        length_scale = 1.0

        # distance functor to be used
        dist_fn = Isotropy(
            metric=F2,
            length_scale=ScalarHyperparameter(length_scale),
        )

        baseline_shears = np.zeros((3 * (n) ** 2, 3 * (n) ** 2))
        baseline_shears[:] = np.nan
        for i, (ix, iy) in enumerate(features):
            for j, (jx, jy) in enumerate(features):
                baseline_shears[
                    i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3
                ] = shear_kernel(ix, iy, jx, jy, b=length_scale)
        baseline_shears = mm.array(baseline_shears)

        library_shears = ShearKernel(deformation=dist_fn)(diffs)

        _check_ndarray(
            self.assertEqual,
            library_shears,
            mm.ftype,
            shape=baseline_shears.shape,
        )

        self.assertTrue(mm.allclose(library_shears, baseline_shears))


if __name__ == "__main__":
    absltest.main()
