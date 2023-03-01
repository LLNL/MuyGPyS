# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

_ones, _zeros = _collect_implementation(
    "MuyGPyS._src.math",
    "_ones",
    "_zeros",
)
