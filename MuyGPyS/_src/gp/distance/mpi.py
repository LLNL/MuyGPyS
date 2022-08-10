# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT


from MuyGPyS._src.gp.distance.numpy import (
    _crosswise_distances as _crosswise_distances_n,
    _pairwise_distances as _pairwise_distances_n,
    _make_train_tensors as _make_train_tensors_n,
    _make_regress_tensors as _make_regress_tensors_n,
)
from MuyGPyS import config

from mpi4py import MPI
from typing import Callable, List, Optional, Tuple

import numpy as np

world = config.mpi_state.comm_world
rank = config.mpi_state.comm_world.Get_rank()
size = config.mpi_state.comm_world.Get_size()


def _get_chunk_sizes(count: int, size: int):
    floor = int(count / size)
    remainder = count - floor * size
    return [
        floor + 1 if i >= (size - remainder) else floor for i in range(size)
    ]


def _prepare_parallel_data(size, chunk_sizes, *args):
    count = len(args)
    ret = [list() for _ in range(count)]
    offsets = np.array(
        [np.sum(chunk_sizes[:i]) for i in range(size)], dtype=int
    )
    for i in range(size):
        for j, arg in enumerate(args):
            ret[j].append(arg[offsets[i] : offsets[i] + chunk_sizes[i]])
    return ret


def _chunk_tensor(tensors, return_count=1):
    if rank == 0:
        if return_count == 1:
            tensors = [tensors]
        data_count = tensors[0].shape[0]
        chunk_sizes = _get_chunk_sizes(data_count, size)
        tensor_chunks = _prepare_parallel_data(size, chunk_sizes, *tensors)
    else:
        tensor_chunks = [None for _ in range(return_count)]
    local_chunks = list()
    for chunks in tensor_chunks:
        local_chunks.append(world.scatter(chunks, root=0))
    if return_count == 1:
        local_chunks = local_chunks[0]
    return local_chunks


def _chunk_function_tensor(func: Callable, *args, return_count=1):
    if rank == 0:
        tensors = func(*args)
    else:
        tensors = None
    return _chunk_tensor(tensors, return_count=return_count)


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
