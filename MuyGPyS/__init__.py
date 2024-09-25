# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Public MuyGPyS modules and functions."""

import importlib.metadata

__version__ = importlib.metadata.version(__package__)

from MuyGPyS._src.config import (
    config as config,
    jax_config as jax_config,
    MPI as MPI,
)
