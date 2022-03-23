# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Public MuyGPyS modules and functions."""

__version__ = "0.5.1"

from MuyGPyS._src.config import config as config, jax_config as jax_config

if config.muygpys_jax_enabled is True and jax_config is not None:  # type: ignore
    jax_config.update("jax_enable_x64", True)
