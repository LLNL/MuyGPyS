# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax import jit

import MuyGPyS._src.math.jax as jnp


def _soap_fn(
    diffs,
    sensitivity: float
):

    return print("Jax backend not yet supported for SOAPKernels")
