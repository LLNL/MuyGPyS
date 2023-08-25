# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from typing import Tuple

import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.mpi_utils import (
    _chunk_function_tensor,
)
from MuyGPyS._src.gp.tensors.numpy import (
    _crosswise_tensor as _crosswise_tensor_n,
    _pairwise_tensor as _pairwise_tensor_n,
    _batch_features_tensor as _batch_features_tensor_n,
    _make_train_tensors as _make_train_tensors_n,
    _make_predict_tensors as _make_predict_tensors_n,
    _make_heteroscedastic_tensor as _make_heteroscedatic_tensor_n,
    _F2,
    _l2,
)


def _make_heteroscedastic_tensor(
    measurement_noise: np.ndarray,
    batch_nn_indices: np.ndarray,
) -> np.ndarray:
    return _chunk_function_tensor(
        _make_heteroscedatic_tensor_n,
        measurement_noise,
        batch_nn_indices,
    )


def _make_fast_predict_tensors(
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError(
        'Function "make_fast_predict_tensors" does not support mpi!'
    )


def _make_predict_tensors(
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _chunk_function_tensor(
        _make_predict_tensors_n,
        batch_indices,
        batch_nn_indices,
        test_features,
        train_features,
        train_targets,
        return_count=3,
    )


def _make_train_tensors(
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _chunk_function_tensor(
        _make_train_tensors_n,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_targets,
        return_count=4,
    )


def _batch_features_tensor(
    features: np.ndarray,
    batch_indices: np.ndarray,
) -> np.ndarray:
    _batch_features_tensor_n
    return _chunk_function_tensor(
        _batch_features_tensor_n,
        features,
        batch_indices,
    )


def _crosswise_tensor(
    data: np.ndarray,
    nn_data: np.ndarray,
    data_indices: np.ndarray,
    nn_indices: np.ndarray,
) -> np.ndarray:
    return _chunk_function_tensor(
        _crosswise_tensor_n,
        data,
        nn_data,
        data_indices,
        nn_indices,
    )


def _crosswise_differences(
    locations: np.ndarray, points: np.ndarray
) -> np.ndarray:
    raise NotImplementedError(
        'Function "muygps_crosswise_differences" does not support mpi!'
    )


def _pairwise_tensor(
    data: np.ndarray,
    nn_indices: np.ndarray,
) -> np.ndarray:
    return _chunk_function_tensor(
        _pairwise_tensor_n,
        data,
        nn_indices,
    )


def _pairwise_differences(points: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        'Function "muygps_pairwise_differences" does not support mpi!'
    )


def _fast_nn_update(
    train_nn_indices: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError(
        'Function "muygps_fast_nn_update" does not support mpi!'
    )
