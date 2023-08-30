# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from jax.numpy import (
    abs,
    all,
    allclose,
    any,
    argmax,
    argmin,
    atleast_1d,
    atleast_2d,
    clip,
    concatenate,
    copy,
    corrcoef,
    cov,
    choose,
    divide,
    einsum,
    exp,
    expand_dims,
    float32,
    float64,
    histogram,
    inf,
    int32,
    int64,
    isclose,
    linalg,
    log,
    logical_or,
    max,
    mean,
    median,
    min,
    ndarray,
    newaxis,
    outer,
    prod,
    repeat,
    reshape,
    sqrt,
    sum,
    tile,
    unique,
    where,
    vstack,
)
from jax.numpy import (
    arange as _arange,
    array as _array,
    diagonal as _diagonal,
    eye as _eye,
    full as _full,
    linspace as _linspace,
    ones as _ones,
    zeros as _zeros,
)
from jax.lax.linalg import cholesky
from typing import Callable, Tuple, Type

from MuyGPyS._src.math.meta import (
    fix_function_type,
    fix_function_types,
    set_type,
)


def assign(x: ndarray, y: ndarray, *slices) -> ndarray:
    return x.at[slices].set(y)


ftype = set_type(float64, float32)
itype = set_type(int64, int32)

iarray = fix_function_type(itype, _array)
farray = fix_function_type(ftype, _array)
array = farray

ndarray = type(array(1))  # type: ignore

arange = fix_function_type(itype, _arange)
diagonal, eye, full, linspace, ones, zeros = fix_function_types(
    ftype, _diagonal, _eye, _full, _linspace, _ones, _zeros
)


def parameter(x):
    return x
