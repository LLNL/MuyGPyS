# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp

from jax import jit


@jit
def _homoscedastic_perturb(K: jnp.ndarray, eps: float) -> jnp.ndarray:
    _, nn_count, _ = K.shape
    return K + eps * jnp.eye(nn_count)
