# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from jax import jit

import MuyGPyS._src.math.jax as jnp

@jit
def _omega(diffs) -> jnp.ndarray:
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 0

    qq_slice = diffs[tuple(slicer)]

    return qq_slice


@jit
def _T1(diffs) -> jnp.ndarray:
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 3

    dd_slice = diffs[tuple(slicer)]

    return dd_slice


@jit
def _T2(diffs) -> jnp.ndarray:
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 1

    diq_slice = diffs[tuple(slicer)]

    return diq_slice


@jit
def _T3(diffs) -> jnp.ndarray:
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 2

    djq_slice = diffs[tuple(slicer)]

    return djq_slice


@jit
def _Knm(omega, T1, T2, T3, sensitivity) -> jnp.ndarray:
    Knm = (sensitivity - 1.0) * (omega ** (sensitivity - 2.0)) * (T2 * T3) + (
        omega ** (sensitivity - 1.0)
    ) * T1
    return Knm


@jit
def _soap_fn(diffs: jnp.ndarray, sensitivity: float, **kwargs) -> jnp.ndarray:
    omega = _omega(diffs)
    T1 = _T1(diffs)
    T2 = _T2(diffs)
    T3 = _T3(diffs)

    Knm = _Knm(omega, T1, T2, T3, sensitivity)

    Kij = sensitivity * jnp.sum(Knm, axis=(-2, -1))

    return Kij