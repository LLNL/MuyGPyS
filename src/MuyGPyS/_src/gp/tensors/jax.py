# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from functools import partial
from typing import Tuple

from jax import jit

import MuyGPyS._src.math.jax as jnp


@jit
def _make_heteroscedastic_tensor(
    measurement_noise: jnp.ndarray,
    batch_nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    return measurement_noise[batch_nn_indices]


@jit
def _make_fast_predict_tensors(
    batch_nn_indices: jnp.ndarray,
    train_features: jnp.ndarray,
    train_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    num_train = train_features.shape[0]
    batch_nn_indices_fast = jnp.concatenate(
        (
            jnp.expand_dims(jnp.arange(0, num_train), axis=1),
            batch_nn_indices[:, :-1],
        ),
        axis=1,
    )

    pairwise_diffs_fast = _pairwise_tensor(
        train_features, batch_nn_indices_fast
    )
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast]
    return pairwise_diffs_fast, batch_nn_targets_fast


@jit
def _batch_features_tensor(
    features: jnp.ndarray,
    batch_indices: jnp.ndarray,
) -> jnp.ndarray:
    return features[batch_indices]


@jit
def _crosswise_tensor(
    data: jnp.ndarray,
    nn_data: jnp.ndarray,
    data_indices: jnp.ndarray,
    nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    locations = data[data_indices]
    points = nn_data[nn_indices]
    return _crosswise_differences(locations, points)


@jit
def _crosswise_differences(
    locations: jnp.ndarray, points: jnp.ndarray
) -> jnp.ndarray:
    return locations[:, None, :] - points


@jit
def _pairwise_differences(points: jnp.ndarray) -> jnp.ndarray:
    if len(points.shape) == 3:
        return points[:, :, None, :] - points[:, None, :, :]
    elif len(points.shape) == 2:
        return points[:, None, :] - points[None, :, :]
    else:
        raise ValueError(f"points shape {points.shape} is not supported.")


@jit
def _pairwise_tensor(
    data: jnp.ndarray,
    nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    points = data[nn_indices]
    return _pairwise_differences(points)


@jit
def _crosswise_similarity(
    data: jnp.ndarray,
    nn_data: jnp.ndarray,
    data_indices: jnp.ndarray,
    nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    locations = data[data_indices]
    points = nn_data[nn_indices].swapaxes(2, 1)

    # working implementation without einsum
    # dot = np.sum(
    #     locations[:, None, :, None, :, None, :, None, :]
    #     * points[:, :, None, :, None, :, None, :, :],
    #     axis=-1,
    # ).swapaxes(1, 2)
    # shape = dot.shape

    # locations.shape = (i, x, d, a, q)
    # points.shape = (i, y, k, e, b, q)
    dot = jnp.einsum("ixdaq, iykebq -> iykxdeab", locations, points)

    crosswise_similarity = dot.reshape(*dot.shape[:-4], -1, *dot.shape[-2:])

    return crosswise_similarity


@jit
def _pairwise_similarity(
    data: jnp.ndarray,
    nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    points = data[nn_indices].swapaxes(2, 1)

    # working implementation without einsum
    # dot = np.sum(
    #     points[:, :, :, None, None, :, None, :, None, :]
    #     * points[:, None, None, :, :, None, :, None, :, :],
    #     axis=-1,
    # )

    # points.shape=(i, x, k, d, a, q) / (i, y, l, e, b, q)
    dot = jnp.einsum("ixkdaq,iylebq->ixkyldeab", points, points)

    pairwise_similarity = dot.reshape(*dot.shape[:5], -1, *dot.shape[-2:])

    return pairwise_similarity


@jit
def _out_similarity(
    data: jnp.ndarray,
    data_indices: jnp.ndarray
) -> np.ndarray:
    points = data[data_indices]

    dot = jnp.einsum("ixdaq, iyebq -> ixydeab", points, points)

    out_similarity = dot.reshape(*dot.shape[:3], -1, *dot.shape[-2:])

    return out_similarity


@jit
def _F2(diffs: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(diffs**2, axis=-1)


@jit
def _l2(diffs: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(_F2(diffs))


@jit
def _fast_nn_update(
    train_nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    train_count = train_nn_indices.shape[0]
    new_nn_indices = jnp.concatenate(
        (
            jnp.expand_dims(jnp.arange(0, train_count), axis=1),
            train_nn_indices[:, :-1],
        ),
        axis=1,
    )
    return new_nn_indices
