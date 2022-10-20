# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

_mse_fn, _cross_entropy_fn, _lool_fn = _collect_implementation(
    "MuyGPyS._src.optimize.loss",
    "_mse_fn",
    "_cross_entropy_fn",
    "_lool_fn",
)
