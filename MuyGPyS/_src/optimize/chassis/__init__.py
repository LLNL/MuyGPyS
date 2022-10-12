# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

_scipy_optimize, _bayes_opt_optimize = _collect_implementation(
    "MuyGPyS._src.optimize.chassis",
    "_scipy_optimize",
    "_bayes_opt_optimize",
)
