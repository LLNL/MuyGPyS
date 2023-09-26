# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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
    num_train, _ = train_features.shape
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
    batch_nn_targets_fast = train_targets[batch_nn_indices_fast, :]
    return pairwise_diffs_fast, batch_nn_targets_fast


@jit
def _make_predict_tensors(
    batch_indices: jnp.ndarray,
    batch_nn_indices: jnp.ndarray,
    test_features: jnp.ndarray,
    train_features: jnp.ndarray,
    train_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if test_features is None:
        test_features = train_features
    crosswise_diffs = _crosswise_tensor(
        test_features,
        train_features,
        batch_indices,
        batch_nn_indices,
    )
    pairwise_diffs = _pairwise_tensor(train_features, batch_nn_indices)
    batch_nn_targets = train_targets[batch_nn_indices, :]
    return crosswise_diffs, pairwise_diffs, batch_nn_targets


@jit
def _make_train_tensors(
    batch_indices: jnp.ndarray,
    batch_nn_indices: jnp.ndarray,
    train_features: jnp.ndarray,
    train_targets: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    crosswise_diffs, pairwise_diffs, batch_nn_targets = _make_predict_tensors(
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_targets,
    )
    batch_targets = train_targets[batch_indices, :]
    return crosswise_diffs, pairwise_diffs, batch_targets, batch_nn_targets


@jit
def _batch_features_tensor(
    features: jnp.ndarray,
    batch_indices: jnp.ndarray,
) -> jnp.ndarray:
    return features[batch_indices, :]


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
def _F2(diffs: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(diffs**2, axis=-1)


@jit
def _l2(diffs: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(_F2(diffs))


@jit
def _fast_nn_update(
    train_nn_indices: jnp.ndarray,
) -> jnp.ndarray:
    train_count, _ = train_nn_indices.shape
    new_nn_indices = jnp.concatenate(
        (
            jnp.expand_dims(jnp.arange(0, train_count), axis=1),
            train_nn_indices[:, :-1],
        ),
        axis=1,
    )
    return new_nn_indices
