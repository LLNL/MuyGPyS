# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config

from typing import Callable

import numpy as np

world = config.mpi_state.comm_world
if world is not None:
    rank = config.mpi_state.comm_world.Get_rank()
    size = config.mpi_state.comm_world.Get_size()
else:
    rank = 0
    size = 1


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


def _big_scatter(chunks, root=0):
    """
    Based upon this reply from a mpi4py dev
    https://github.com/mpi4py/mpi4py/issues/119#issuecomment-945390731
    """
    if size == 0:
        my_chunk = chunks
    elif rank == 0:
        for i, chunk in enumerate(chunks):
            if i == 0:
                my_chunk = chunk
            else:
                world.send(chunk, dest=i)
    else:
        my_chunk = world.recv(source=0)
    return my_chunk


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
        local_chunks.append(_big_scatter(chunks, root=0))
    if return_count == 1:
        local_chunks = local_chunks[0]
    return local_chunks


def _chunk_function_tensor(func: Callable, *args, return_count=1):
    if rank == 0:
        tensors = func(*args)
    else:
        tensors = None
    return _chunk_tensor(tensors, return_count=return_count)


def _consistent_unchunk_tensor(tensor) -> np.ndarray:
    """
    If we are using an MPI implementation, allgather the tensor across all
    cores. Otherwise NOOP.

    The purpose of this function is to allow the existing serial testing harness
    to also test the mpi implementations without the need for additional codes.

    Args:
        tensor:
            A tensor, which might be a simple serial tensors or distributed
            chunks if it is the product of the mpi implementation.

    Return:
        The same tensor if a serial implementation, else an allgathered tensor
        of the distributed chunks.
    """
    if tensor is None:
        return tensor
    if _is_mpi_mode() is True:
        if len(tensor.shape) > 1:
            return np.vstack(config.mpi_state.comm_world.allgather(tensor))
        else:
            return np.concatenate(config.mpi_state.comm_world.allgather(tensor))
    else:
        return tensor


def _consistent_chunk_tensor(tensor) -> np.ndarray:
    if _is_mpi_mode() is True:
        return _chunk_tensor(tensor)
    else:
        return tensor


def _consistent_reduce_scalar(scalar):
    if _is_mpi_mode() is True:
        from mpi4py import MPI

        return world.allreduce(scalar, op=MPI.SUM)
    else:
        return scalar


def _is_mpi_mode() -> bool:
    return (
        config.muygpys_mpi_enabled is True  # type: ignore
        and config.muygpys_jax_enabled is False  # type: ignore
    )
