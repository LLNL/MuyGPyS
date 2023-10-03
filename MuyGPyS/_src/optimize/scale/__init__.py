# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

[
    _analytic_scale_optim,
    _analytic_scale_optim_unnormalized,
] = _collect_implementation(
    "MuyGPyS._src.optimize.scale",
    "_analytic_scale_optim",
    "_analytic_scale_optim_unnormalized",
)
