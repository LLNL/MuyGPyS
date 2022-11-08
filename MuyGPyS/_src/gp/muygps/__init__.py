# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

(
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
    _muygps_fast_regress_solve,
    _muygps_fast_regress_precompute,
) = _collect_implementation(
    "MuyGPyS._src.gp.muygps",
    "_muygps_compute_solve",
    "_muygps_compute_diagonal_variance",
    "_muygps_fast_regress_solve",
    "_muygps_fast_regress_precompute",
)
