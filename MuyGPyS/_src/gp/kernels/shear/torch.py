# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


import MuyGPyS._src.math.numpy as np
import MuyGPyS._src.math.torch as torch


def _kk_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_sq_diffs,
    sum_quad_diffs,
    a=1.0,
    length_scale=1.0,
):
    return (
        1
        / 4
        * (
            a
            * (
                8 * length_scale**2
                - 8 * length_scale * sum_sq_diffs
                + 2 * prod_sq_diffs
                + sum_quad_diffs
            )
            * exp_inv_scaled_sum_sq_diffs
            / length_scale**4
        )
    )


def _kg1_fn(
    exp_inv_scaled_sum_sq_diffs,
    diff_xy_quad_diffs,
    diff_yx_sq_diffs,
    a=1.0,
    length_scale=1.0,
):
    return (
        1
        / 4
        * (
            a
            * (6 * length_scale * diff_yx_sq_diffs + diff_xy_quad_diffs)
            * exp_inv_scaled_sum_sq_diffs
            / length_scale**4
        )
    )


def _kg2_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_diffs,
    a=1.0,
    length_scale=1.0,
):
    return (
        1
        / 4
        * (
            2
            * a
            * prod_diffs
            * (-6 * length_scale + sum_sq_diffs)
            * exp_inv_scaled_sum_sq_diffs
            / length_scale**4
        )
    )


def _g1g1_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    sum_quad_diffs,
    prod_sq_diffs,
    a=1.0,
    length_scale=1.0,
):
    return (
        1
        / 4
        * (
            a
            * (
                4 * length_scale**2
                - 4 * length_scale * sum_sq_diffs
                - 2 * prod_sq_diffs
                + sum_quad_diffs
            )
            * exp_inv_scaled_sum_sq_diffs
            / length_scale**4
        )
    )


def _g1g2_fn(
    exp_inv_scaled_sum_sq_diffs,
    diff_xy_sq_diffs,
    prod_diffs,
    a=1.0,
    length_scale=1.0,
):
    return (
        1
        / 4
        * (
            2
            * a
            * prod_diffs
            * diff_xy_sq_diffs
            * exp_inv_scaled_sum_sq_diffs
            / length_scale**4
        )
    )


def _g2g2_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_sq_diffs,
    a=1.0,
    length_scale=1.0,
):
    return (
        1
        / 4
        * (
            4
            * a
            * (length_scale**2 - length_scale * sum_sq_diffs + prod_sq_diffs)
            * exp_inv_scaled_sum_sq_diffs
            / length_scale**4
        )
    )


# compute the full covariance matrix
def _shear_fn(diffs, a=1.0, length_scale=1.0):
    shape = torch.array(diffs.shape[:-1], dtype=int)
    n = torch.clone(shape[-2])
    n2 = 2 * n
    m = torch.clone(shape[-1])
    m2 = 2 * m
    shape[-1] *= 3
    shape[-2] *= 3
    print(n, m, shape)
    full_m = torch.zeros(tuple(shape))

    # compute intermediate difference tensors once here
    prod_diffs = torch.prod(diffs, axis=-1)
    sq_diffs = diffs**2
    quad_diffs = sq_diffs**2
    sum_sq_diffs = torch.sum(sq_diffs, axis=-1)
    prod_sq_diffs = torch.prod(sq_diffs, axis=-1)
    sum_quad_diffs = torch.sum(quad_diffs, axis=-1)
    diff_yx_sq_diffs = sq_diffs[..., 1] - sq_diffs[..., 0]
    diff_xy_sq_diffs = sq_diffs[..., 0] - sq_diffs[..., 1]
    diff_xy_quad_diffs = quad_diffs[..., 0] - quad_diffs[..., 1]
    exp_inv_scaled_sum_sq_diffs = torch.exp(-sum_sq_diffs / (2 * length_scale))

    full_m[..., :n, :m] = _kk_fn(
        exp_inv_scaled_sum_sq_diffs,
        sum_sq_diffs,
        prod_sq_diffs,
        sum_quad_diffs,
        a,
        length_scale,
    )
    full_m[..., :n, m:m2] = full_m[..., n:n2, :m] = _kg1_fn(
        exp_inv_scaled_sum_sq_diffs,
        diff_xy_quad_diffs,
        diff_yx_sq_diffs,
        a,
        length_scale,
    )
    full_m[..., :n, m2:] = full_m[..., n2:, :m] = _kg2_fn(
        exp_inv_scaled_sum_sq_diffs, sum_sq_diffs, prod_diffs, a, length_scale
    )
    full_m[..., n:n2, m:m2] = _g1g1_fn(
        exp_inv_scaled_sum_sq_diffs,
        sum_sq_diffs,
        sum_quad_diffs,
        prod_sq_diffs,
        a,
        length_scale,
    )
    full_m[..., n:n2, m2:] = full_m[..., n2:, m:m2] = _g1g2_fn(
        exp_inv_scaled_sum_sq_diffs,
        diff_xy_sq_diffs,
        prod_diffs,
        a,
        length_scale,
    )
    full_m[..., n2:, m2:] = _g2g2_fn(
        exp_inv_scaled_sum_sq_diffs,
        sum_sq_diffs,
        prod_sq_diffs,
        a,
        length_scale,
    )

    return full_m
