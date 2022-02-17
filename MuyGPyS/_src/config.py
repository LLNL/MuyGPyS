# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

__jax_enabled__ = False
__gpu_found__ = False
try:
    from jax import default_backend as _default_backend

    __jax_enabled__ = True
    if _default_backend() in ["gpu", "tpu"]:
        __gpu_found__ = True
    del _default_backend

    # Currently needed because x32 Matern matrices are not covariances!
    from jax.config import config as _config

    _config.update("jax_enable_x64", True)
    del _config

except Exception:
    __jax_enabled__ = False
    __gpu_found__ = False


def disable_jax():
    __jax_enabled__ = True
