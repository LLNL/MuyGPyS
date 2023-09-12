# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from numpy import (
    all,
    allclose,
    any,
    argmax,
    argmin,
    atleast_1d,
    atleast_2d,
    concatenate,
    copy,
    corrcoef,
    cov,
    count_nonzero,
    choose,
    divide,
    einsum,
    exp,
    expand_dims,
    finfo,
    float32,
    float64,
    histogram,
    inf,
    int32,
    int64,
    isclose,
    isnan,
    linalg,
    log,
    logical_and,
    logical_or,
    max,
    mean,
    median,
    meshgrid,
    min,
    mod,
    nan,
    ndarray,
    number,
    outer,
    prod,
    random,
    repeat,
    reshape,
    sqrt,
    sum,
    tile,
    unique,
    where,
    vstack,
)
from numpy import (
    arange as _arange,
    array as _array,
    diagonal as _diagonal,
    eye as _eye,
    full as _full,
    linspace as _linspace,
    ones as _ones,
    zeros as _zeros,
)

# from numpy import *
from numpy.linalg import cholesky

from MuyGPyS._src.math.meta import (
    fix_function_type,
    fix_function_types,
    set_type,
)


def assign(x: ndarray, y: ndarray, *slices) -> ndarray:
    ret = copy(x)
    ret[slices] = y
    return ret


ftype = set_type(float64, float32)
# itype = set_type(int64, int32)
itype = int64

iarray = fix_function_type(itype, _array)
farray = fix_function_type(ftype, _array)
array = farray

arange = fix_function_type(itype, _arange)
diagonal, eye, full, linspace, ones, zeros = fix_function_types(
    ftype, _diagonal, _eye, _full, _linspace, _ones, _zeros
)


def parameter(x):
    return x
