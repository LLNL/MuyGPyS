# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

(
    _muygps_posterior_mean,
    _muygps_diagonal_variance,
    _muygps_fast_posterior_mean,
    _muygps_fast_posterior_mean_precompute,
    _mmuygps_fast_posterior_mean,
) = _collect_implementation(
    "MuyGPyS._src.gp.muygps",
    "_muygps_posterior_mean",
    "_muygps_diagonal_variance",
    "_muygps_fast_posterior_mean",
    "_muygps_fast_posterior_mean_precompute",
    "_mmuygps_fast_posterior_mean",
)
