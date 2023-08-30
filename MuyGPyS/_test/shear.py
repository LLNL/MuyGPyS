# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Union

import MuyGPyS._src.math.numpy as np


def kernelf(diffs, a=1, b=1):
    return a * np.exp(-np.sum(diffs**2, axis=-1) / (2 * b))


def kk_f(diffs, a=1, b=1):
    sq_diffs = diffs**2
    sum_sq_diffs = np.sum(sq_diffs, axis=-1)
    prod_sq_diffs = np.prod(sq_diffs, axis=-1)
    sum_quad_diffs = np.sum(sq_diffs**2, axis=-1)
    return (
        1
        / 4
        * (
            a
            * (
                8 * b**2
                - 8 * b * sum_sq_diffs
                + 2 * prod_sq_diffs
                + sum_quad_diffs
            )
            * np.exp(-sum_sq_diffs / (2 * b))
            / b**4
        )
    )


def kg1_f(diffs, a=1, b=1):
    sq_diffs = diffs**2
    quad_diffs = sq_diffs**2
    sum_sq_diffs = np.sum(sq_diffs, axis=-1)
    diff_xy_quad_diffs = quad_diffs[..., 0] - quad_diffs[..., 1]
    diff_yx_sq_diffs = sq_diffs[..., 1] - sq_diffs[..., 0]
    return (
        1
        / 4
        * (
            a
            * (6 * b * diff_yx_sq_diffs + diff_xy_quad_diffs)
            * np.exp(-sum_sq_diffs / (2 * b))
            / b**4
        )
    )


def kg2_f(diffs, a=1, b=1):
    sq_diffs = diffs**2
    sum_sq_diffs = np.sum(sq_diffs, axis=-1)
    prod_diffs = np.prod(diffs, axis=-1)
    return (
        1
        / 4
        * (
            2
            * a
            * prod_diffs
            * (-6 * b + sum_sq_diffs)
            * np.exp(-sum_sq_diffs / (2 * b))
            / b**4
        )
    )


def g1g1_f(diffs, a=1, b=1):
    sq_diffs = diffs**2
    sum_sq_diffs = np.sum(sq_diffs, axis=-1)
    sum_quad_diffs = np.sum(sq_diffs**2, axis=-1)
    prod_sq_diffs = np.prod(sq_diffs, axis=-1)
    return (
        1
        / 4
        * (
            a
            * (
                4 * b**2
                - 4 * b * sum_sq_diffs
                - 2 * prod_sq_diffs
                + sum_quad_diffs
            )
            * np.exp(-sum_sq_diffs / (2 * b))
            / b**4
        )
    )


def g1g2_f(diffs, a=1, b=1):
    sq_diffs = diffs**2
    sum_sq_diffs = np.sum(sq_diffs, axis=-1)
    diff_xy_sq_diffs = sq_diffs[..., 0] - sq_diffs[..., 1]
    prod_diffs = np.prod(diffs, axis=-1)
    return (
        1
        / 4
        * (
            2
            * a
            * prod_diffs
            * diff_xy_sq_diffs
            * np.exp(-sum_sq_diffs / (2 * b))
            / b**4
        )
    )


def g2g2_f(diffs, a=1, b=1):
    sq_diffs = diffs**2
    sum_sq_diffs = np.sum(sq_diffs, axis=-1)
    prod_sq_diffs = np.prod(sq_diffs, axis=-1)
    return (
        1
        / 4
        * (
            4
            * a
            * (b**2 - b * sum_sq_diffs + prod_sq_diffs)
            * np.exp(-sum_sq_diffs / (2 * b))
            / b**4
        )
    )


# compute the full covariance matrix
def shear_kernel(diffs, a=1, b=1):
    full_m = np.zeros(diffs.shape[:-1] + (3, 3))
    full_m[..., 0, 0] = kk_f(diffs, a, b)
    full_m[..., 0, 1] = kg1_f(diffs, a, b)
    full_m[..., 0, 2] = kg2_f(diffs, a, b)
    full_m[..., 1, 1] = g1g1_f(diffs, a, b)
    full_m[..., 1, 2] = g1g2_f(diffs, a, b)
    full_m[..., 2, 2] = g2g2_f(diffs, a, b)
    full_m[..., 1, 0] = full_m[..., 0, 1]
    full_m[..., 2, 0] = full_m[..., 0, 2]
    full_m[..., 2, 1] = full_m[..., 1, 2]

    return full_m
