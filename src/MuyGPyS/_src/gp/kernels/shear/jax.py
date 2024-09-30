# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
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
    length_scale=1.0,
):
    return 0.25 * (
        (
            8 * length_scale**2
            - 8 * length_scale * sum_sq_diffs
            + 2 * prod_sq_diffs
            + sum_quad_diffs
        )
        * exp_inv_scaled_sum_sq_diffs
        / length_scale**4
    )


@jit
def _kg1_fn(
    exp_inv_scaled_sum_sq_diffs,
    diff_xy_quad_diffs,
    diff_yx_sq_diffs,
    length_scale=1.0,
):
    return 0.25 * (
        (6 * length_scale * diff_yx_sq_diffs + diff_xy_quad_diffs)
        * exp_inv_scaled_sum_sq_diffs
        / length_scale**4
    )


@jit
def _kg2_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_diffs,
    length_scale=1.0,
):
    return (
        0.5
        * prod_diffs
        * (-6 * length_scale + sum_sq_diffs)
        * exp_inv_scaled_sum_sq_diffs
        / length_scale**4
    )


@jit
def _g1g1_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    sum_quad_diffs,
    prod_sq_diffs,
    length_scale=1.0,
):
    return 0.25 * (
        (
            4 * length_scale**2
            - 4 * length_scale * sum_sq_diffs
            - 2 * prod_sq_diffs
            + sum_quad_diffs
        )
        * exp_inv_scaled_sum_sq_diffs
        / length_scale**4
    )


@jit
def _g1g2_fn(
    exp_inv_scaled_sum_sq_diffs,
    diff_xy_sq_diffs,
    prod_diffs,
    length_scale=1.0,
):
    return (
        0.5
        * prod_diffs
        * diff_xy_sq_diffs
        * exp_inv_scaled_sum_sq_diffs
        / length_scale**4
    )


@jit
def _g2g2_fn(
    exp_inv_scaled_sum_sq_diffs,
    sum_sq_diffs,
    prod_sq_diffs,
    length_scale=1.0,
):
    return (
        (length_scale**2 - length_scale * sum_sq_diffs + prod_sq_diffs)
        * exp_inv_scaled_sum_sq_diffs
        / length_scale**4
    )


# compute the full covariance matrix
@jit
def _shear_33_fn(diffs, length_scale=1.0, **kwargs):
    assert diffs.ndim >= 3
    shape = diffs.shape[:-1]
    n = shape[-2]
    m = shape[-1]
    prefix = shape[:-2]
    new_shape = prefix + (3, n, 3, m)
    full_m = jnp.zeros(new_shape)

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

    full_m = full_m.at[..., 0, :, 0, :].set(
        _kk_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            prod_sq_diffs,
            sum_quad_diffs,
            length_scale,
        )
    )  # (0, 0)
    full_m = full_m.at[..., 0, :, 1, :].set(
        _kg1_fn(
            exp_inv_scaled_sum_sq_diffs,
            diff_xy_quad_diffs,
            diff_yx_sq_diffs,
            length_scale,
        )
    )  # (0, 1), (1, 0)
    full_m = full_m.at[..., 1, :, 0, :].set(full_m[..., 0, :, 1, :])
    full_m = full_m.at[..., 0, :, 2, :].set(
        _kg2_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            prod_diffs,
            length_scale,
        )
    )  # (0, 2), (2, 0)
    full_m = full_m.at[..., 2, :, 0, :].set(full_m[..., 0, :, 2, :])
    full_m = full_m.at[..., 1, :, 1, :].set(
        _g1g1_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            sum_quad_diffs,
            prod_sq_diffs,
            length_scale,
        )
    )  # (1, 1)
    full_m = full_m.at[..., 1, :, 2, :].set(
        _g1g2_fn(
            exp_inv_scaled_sum_sq_diffs,
            diff_xy_sq_diffs,
            prod_diffs,
            length_scale,
        )
    )  # (1, 2), (2, 1)
    full_m = full_m.at[..., 2, :, 1, :].set(full_m[..., 1, :, 2, :])
    full_m = full_m.at[..., 2, :, 2, :].set(
        _g2g2_fn(
            exp_inv_scaled_sum_sq_diffs,
            sum_sq_diffs,
            prod_sq_diffs,
            length_scale,
        )
    )  # (2, 2)

    return full_m


def _shear_Kin23_fn(diffs, length_scale=1.0, **kwargs):
    raise NotImplementedError(
        "2in3out shear is not currently supported by the MuyGPs JAX backend."
    )


def _shear_Kcross23_fn(diffs, length_scale=1.0, **kwargs):
    raise NotImplementedError(
        "2in3out shear is not currently supported by the MuyGPs JAX backend."
    )
