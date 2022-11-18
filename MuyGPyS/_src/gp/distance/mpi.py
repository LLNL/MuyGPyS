# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from MuyGPyS._src.mpi_utils import (
    _chunk_function_tensor,
)
from MuyGPyS._src.gp.distance.numpy import (
    _crosswise_distances as _crosswise_distances_n,
    _pairwise_distances as _pairwise_distances_n,
    _make_train_tensors as _make_train_tensors_n,
    _make_regress_tensors as _make_regress_tensors_n,
)

from typing import Tuple

import numpy as np


def _make_fast_regress_tensors(
    metric: str,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raise NotImplementedError(
        f'Function "make_fast_regress_tensors" does not support mpi!'
    )


def _make_regress_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _chunk_function_tensor(
        _make_regress_tensors_n,
        metric,
        batch_indices,
        batch_nn_indices,
        test_features,
        train_features,
        train_targets,
        return_count=3,
    )


def _make_train_tensors(
    metric: str,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _chunk_function_tensor(
        _make_train_tensors_n,
        metric,
        batch_indices,
        batch_nn_indices,
        train_features,
        train_targets,
        return_count=4,
    )


def _crosswise_distances(
    data: np.ndarray,
    nn_data: np.ndarray,
    data_indices: np.ndarray,
    nn_indices: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
    return _chunk_function_tensor(
        _crosswise_distances_n,
        data,
        nn_data,
        data_indices,
        nn_indices,
        metric,
    )


def _pairwise_distances(
    data: np.ndarray,
    nn_indices: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
    return _chunk_function_tensor(
        _pairwise_distances_n,
        data,
        nn_indices,
        metric,
    )


def _fast_nn_update(
    nn_indices: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError(
        f'Function "muygps_fast_nn_update" does not support mpi!'
    )
