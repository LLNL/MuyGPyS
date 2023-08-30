# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from torch import tensor as _array, Tensor as ndarray
from torch import (
    all,
    allclose,
    atleast_1d,
    atleast_2d,
    cat,
    clone,
    cov,
    corrcoef,
    cuda,
    einsum,
    exp,
    inf,
    int32,
    int64,
    isclose,
    float32,
    float64,
    from_numpy,
    linalg,
    log,
    logical_or,
    mean,
    median,
    nn,
    norm,
    numel,
    optim,
    outer,
    prod,
    rand,
    reshape,
    sqrt,
    tile,
    unique,
    unsqueeze,
    vstack,
    where,
)
from torch import (
    arange as _arange,
    diagonal as _diagonal,
    eye as _eye,
    full as _full,
    linspace as _linspace,
    repeat_interleave as repeat,
    ones as _ones,
    zeros as _zeros,
)
from torch import (
    argmax as torch_argmax,
    div as divide,
    max as torch_max,
    min as torch_min,
    sum as torch_sum,
)
from torch.linalg import cholesky

from MuyGPyS._src.math.meta import (
    fix_function_type,
    fix_function_types,
    set_type,
    wrap_torch_signatures,
)


def assign(x: ndarray, y: ndarray, *slices) -> ndarray:
    ret = clone(x)
    ret[slices] = y
    return ret


argmax, max, min, sum = wrap_torch_signatures(
    torch_argmax, torch_max, torch_min, torch_sum
)

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
    return nn.Parameter(_array(x))
