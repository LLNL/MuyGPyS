# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.numpy as np


def _omega(
    diffs
):
    
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 0

    qq_slice = diffs[tuple(slicer)]

    return qq_slice

def _T1(
    diffs
):
    
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 3

    dd_slice = diffs[tuple(slicer)]

    return dd_slice

def _T2(
    diffs
):
    
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 1

    diq_slice = diffs[tuple(slicer)]

    return diq_slice

def _T3(
    diffs
):
    
    ndim = diffs.ndim
    slicer = [slice(None)] * ndim
    slicer[-3] = 2

    djq_slice = diffs[tuple(slicer)]

    return djq_slice

def _soap_fn(
    diffs: np.ndarray,
    zeta=2.0
):

    omega = _omega(diffs)
    T1 = _T1(diffs)
    T2 = _T2(diffs)
    T3 = _T3(diffs)

    Knm = (zeta - 1) * (omega**(zeta - 2)) * T2 * T3 + (omega**(zeta - 1)) * T1

    return zeta * np.sum(Knm, axis=(-2, -1))