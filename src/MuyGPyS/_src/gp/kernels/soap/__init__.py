# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS._src.util import _collect_implementation

(_soap_fn,) = _collect_implementation("MuyGPyS._src.gp.kernels.soap", "_soap_fn")
