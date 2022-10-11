# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

[_analytic_sigma_sq_optim] = _collect_implementation(
    "MuyGPyS._src.optimize.sigma_sq",
    "_analytic_sigma_sq_optim",
)
