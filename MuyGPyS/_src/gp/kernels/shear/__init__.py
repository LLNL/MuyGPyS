# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

# _collect_implementation returns a tuple, so need to subscript to get singleton
_shear_fn = _collect_implementation(
    "MuyGPyS._src.gp.kernels.shear", "_shear_fn"
)[0]
