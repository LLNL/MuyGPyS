# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from jax import jit

import MuyGPyS._src.math.jax as jnp
import MuyGPyS._src.math.numpy as np


@jit
def _kk_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_sq_diffs,
    sum_quad_diffs,
    a=1,
    length_scale=1,
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


@jit
def _kg1_fn(
    exp_inv_scaled_sum_sq_diffs,
    diff_xy_quad_diffs,
    diff_yx_sq_diffs,
    a=1,
    length_scale=1,
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


@jit
def _kg2_fn(
    exp_inv_scaled_sum_sq_diffs, sum_sq_diffs, prod_diffs, a=1, length_scale=1
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


@jit
def _g1g1_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    sum_quad_diffs,
    prod_sq_diffs,
    a=1,
    length_scale=1,
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


@jit
def _g1g2_fn(
    exp_inv_scaled_sum_sq_diffs,
    diff_xy_sq_diffs,
    prod_diffs,
    a=1,
    length_scale=1,
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


@jit
def _g2g2_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_sq_diffs,
    a=1,
    length_scale=1,
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
@jit
def _shear_fn(diffs, a=1, length_scale=1):
    shape = np.array(diffs.shape[:-1], dtype=int)
    shape[-1] *= 3
    shape[-2] *= 3
    full_m = jnp.zeros(shape)

    # compute intermediate difference tensors once here
    prod_diffs = jnp.prod(diffs, axis=-1)
    sq_diffs = diffs**2
    quad_diffs = sq_diffs**2
    sum_sq_diffs = jnp.sum(sq_diffs, axis=-1)
    prod_sq_diffs = jnp.prod(sq_diffs, axis=-1)
    sum_quad_diffs = jnp.sum(quad_diffs, axis=-1)
    diff_yx_sq_diffs = sq_diffs[..., 1] - sq_diffs[..., 0]
    diff_xy_sq_diffs = sq_diffs[..., 0] - sq_diffs[..., 1]
    diff_xy_quad_diffs = quad_diffs[..., 0] - quad_diffs[..., 1]
    exp_inv_scaled_sum_sq_diffs = jnp.exp(-sum_sq_diffs / (2 * length_scale))

    full_m = full_m.at[..., 0::3, 0::3].set(
        _kk_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            prod_sq_diffs,
            sum_quad_diffs,
            a,
            length_scale,
        )
    )
    full_m = full_m.at[..., 0::3, 1::3].set(
        _kg1_fn(
            exp_inv_scaled_sum_sq_diffs,
            diff_xy_quad_diffs,
            diff_yx_sq_diffs,
            a,
            length_scale,
        )
    )
    full_m = full_m.at[..., 1::3, 0::3].set(full_m[..., 0::3, 1::3])
    full_m = full_m.at[..., 0::3, 2::3].set(
        _kg2_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            prod_diffs,
            a,
            length_scale,
        )
    )
    full_m = full_m.at[..., 2::3, 0::3].set(full_m[..., 0::3, 2::3])
    full_m = full_m.at[..., 1::3, 1::3].set(
        _g1g1_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            sum_quad_diffs,
            prod_sq_diffs,
            a,
            length_scale,
        )
    )
    full_m = full_m.at[..., 1::3, 2::3].set(
        _g1g2_fn(
            exp_inv_scaled_sum_sq_diffs,
            diff_xy_sq_diffs,
            prod_diffs,
            a,
            length_scale,
        )
    )
    full_m = full_m.at[..., 2::3, 1::3].set(full_m[..., 1::3, 2::3])
    full_m = full_m.at[..., 2::3, 2::3].set(
        _g2g2_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            prod_sq_diffs,
            a,
            length_scale,
        )
    )

    return full_m
