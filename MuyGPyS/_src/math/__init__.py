# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config
from MuyGPyS._src.math.meta import fix_function_types
from MuyGPyS._src.math.numpy import (
    all as np_all,
    allclose as np_allclose,
    arange as np_arange,
    argmax as np_argmax,
    array as np_array,
    any as np_any,
    cholesky as np_cholesky,
    choose as np_choose,
    concatenate as np_concatenate,
    count_nonzero as np_count_nonzero,
    diagonal as np_diagonal,
    exp as np_exp,
    eye as np_eye,
    farray as np_farray,
    farray as np_farray,
    float32 as np_float32,
    float64 as np_float64,
    ftype as np_ftype,
    histogram as np_histogram,
    iarray as np_iarray,
    int32 as np_int32,
    int64 as np_int64,
    isclose as np_isclose,
    isnan as np_isnan,
    itype as np_itype,
    linalg as np_linalg,
    linspace as np_linspace,
    log as np_log,
    logical_and as np_logical_and,
    logical_or as np_logical_or,
    max as np_max,
    mean as np_mean,
    min as np_min,
    ndarray as np_ndarray,
    number as np_number,
    random as np_random,
    ones as np_ones,
    unique as np_unique,
    vstack as np_vstack,
    where as np_where,
    zeros as np_zeros,
)
from MuyGPyS._src.math.jax import (
    array as jnp_array,
    farray as jnp_farray,
    float32 as jnp_float32,
    float64 as jnp_float64,
    ftype as jnp_ftype,
    iarray as jnp_iarray,
    int32 as jnp_int32,
    int64 as jnp_int64,
    itype as jnp_itype,
    ndarray as jnp_ndarray,
)
from MuyGPyS._src.math.torch import (
    arange as torch_arange,
    array as torch_array,
    from_numpy as torch_from_numpy,
)
from MuyGPyS._src.util import _collect_implementation

(
    all,
    allclose,
    arange,
    argmax,
    array,
    atleast_1d,
    assign,
    corrcoef,
    cov,
    cholesky,
    eye,
    exp,
    diagonal,
    iarray,
    inf,
    int32,
    int64,
    itype,
    isclose,
    farray,
    float32,
    float64,
    ftype,
    full,
    linalg,
    linspace,
    log,
    logical_or,
    max,
    min,
    ndarray,
    ones,
    sqrt,
    sum,
    unique,
    vstack,
    where,
    zeros,
) = _collect_implementation(
    "MuyGPyS._src.math",
    "all",
    "allclose",
    "arange",
    "argmax",
    "array",
    "atleast_1d",
    "assign",
    "corrcoef",
    "cov",
    "cholesky",
    "eye",
    "exp",
    "diagonal",
    "iarray",
    "inf",
    "int32",
    "int64",
    "itype",
    "isclose",
    "farray",
    "float32",
    "float64",
    "ftype",
    "full",
    "linalg",
    "linspace",
    "log",
    "logical_or",
    "max",
    "min",
    "ndarray",
    "ones",
    "sqrt",
    "sum",
    "unique",
    "vstack",
    "where",
    "zeros",
)


# def np_iarray(*args, **kwargs):
#     kwargs.setdefault("dtype", np_itype)
#     return _np_array(*args, **kwargs)


# def np_farray(*args, **kwargs):
#     kwargs.setdefault("dtype", np_ftype)
#     return _np_array(*args, **kwargs)


# def np_array(*args, **kwargs):
#     return np_farray(*args, **kwargs)


# def np_ones(*args, **kwargs):
#     kwargs.setdefault("dtype", np_ftype)
#     return _np_ones(*args, **kwargs)


# def np_zeros(*args, **kwargs):
#     kwargs.setdefault("dtype", np_ftype)
#     return _np_zeros(*args, **kwargs)


# if config.state.ftype == "64":
#     ftype = float64
#     np_ftype = np_float64
# elif config.state.ftype == "32":
#     ftype = float32
#     np_ftype = np_float32  # type: ignore


# if config.state.backend == "jax":
#     np_itype = set_type(np_int64, np_int32)

# if config.state.backend != "jax":
#     itype = int64
#     np_itype = np_int64
# else:
#     # iarray = _array
#     # farray = _array
#     # array = _array

#     if config.state.ftype == "64":
#         np_itype = np_int64
#         itype = int64
#     else:
#         itype = int32
#         np_itype = np_int32


# def iarray(*args, **kwargs):
#     kwargs.setdefault("dtype", itype)
#     return _array(*args, **kwargs)


# def farray(*args, **kwargs):
#     kwargs.setdefault("dtype", ftype)
#     return _array(*args, **kwargs)


# def array(*args, **kwargs):
#     return farray(*args, **kwargs)


# def diagonal(*args, **kwargs):
#     kwargs.setdefault("dype", ftype)
#     return _diagonal(*args, **kwargs)


# def eye(*args, **kwargs):
#     kwargs.setdefault("dtype", ftype)
#     return _eye(*args, **kwargs)


# def ones(*args, **kwargs):
#     kwargs.setdefault("dtype", ftype)
#     return _ones(*args, **kwargs)


# def linspace(*args, **kwargs):
#     kwargs.setdefault("dtype", ftype)
#     return _linspace(*args, **kwargs)


# def zeros(*args, **kwargs):
#     kwargs.setdefault("dtype", ftype)
#     return _zeros(*args, **kwargs)
