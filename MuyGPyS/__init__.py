# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Public MuyGPyS modules and functions."""

__version__ = "0.4.1"

from MuyGPyS._src.config import (
    __jax_enabled__ as __jax_enabled__,
    __gpu_found__ as __gpu_found__,
)
