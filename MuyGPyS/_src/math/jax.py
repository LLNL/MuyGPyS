# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from jax import jit
from jax.numpy import ndarray as _ndarray


@jit
def _ones(*args, **kwargs) -> _ndarray:
    return jnp.ones(*args, **kwargs)


@jit
def _zeros(*args, **kwargs) -> _ndarray:
    return jnp.zeros(*args, **kwargs)
