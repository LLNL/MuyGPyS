# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np


_basic_nn_kwarg_options = (
    {"nn_method": "exact", "algorithm": "ball_tree"},
    {
        "nn_method": "hnsw",
        "space": "l2",
        "ef_construction": 100,
        "M": 16,
    },
)

_exact_nn_kwarg_options = (
    {"nn_method": "exact", "algorithm": "ball_tree"},
    # {
    #     "nn_method": "hnsw",
    #     "space": "l2",
    #     "ef_construction": 100,
    #     "M": 16,
    # },
)

_fast_nn_kwarg_options = (
    # {"nn_method": "exact", "algorithm": "ball_tree"},
    {
        "nn_method": "hnsw",
        "space": "l2",
        "ef_construction": 100,
        "M": 16,
    },
)


def _make_gaussian_matrix(data_count, feature_count):
    """
    Create a matrix of i.i.d. Gaussian datapoints.

    Parameters
    ----------
    data_count : int
        The number of data rows.
    feature_count : int
        The number of data columns.

    Returns
    -------
    np.ndarray(float), shape = ``(data_count, feature_count)''
        An i.i.d. Gaussian matrix.
    """
    return np.random.randn(data_count, feature_count)


def _make_gaussian_dict(
    data_count, feature_count, response_count, categorical=False
):
    """
    Create a data dict including "input", "output", and "labels" keys mapping to
    i.i.d. Gaussian matrices.

    Parameters
    ----------
    data_count : int
        The number of data rows.
    feature_count : int
        The number of data columns in the "input" matrix.
    resonse_count : int
        The number of data columns in the "output" matrix.
    categorical : Boolean
        If true, convert the "output" matrix to a one-hot encoding matrix.

    Returns
    -------
    dict
        A dict with keys "input" mapping to a ``(data_count, feature_count)''
        matrix, "output" mapping to a ``(data_count, response_count)'' matrix,
        and "labels" mapping to a ``(data_count)'' vector.
    """
    locations = _make_gaussian_matrix(data_count, feature_count)
    observations = _make_gaussian_matrix(data_count, response_count)
    labels = np.argmax(observations, axis=1)
    if categorical is True:
        observations = np.eye(response_count)[labels] - (1 / response_count)
    return {
        "input": locations,
        "output": observations,
        "labels": labels,
    }


def _make_gaussian_data(
    train_count, test_count, feature_count, response_count, categorical=False
):
    """
    Create train and test dicts including "input", "output", and "labels" keys
    mapping to i.i.d. Gaussian matrices.

    Parameters
    ----------
    data_count : int
        The number of data rows.
    feature_count : int
        The number of data columns in the "input" matrix.
    resonse_count : int
        The number of data columns in the "output" matrix.
    categorical : Boolean
        If true, convert the "output" matrix to a one-hot encoding matrix.

    Returns
    -------
    dict
        A dict with keys "input" mapping to a ``(data_count, feature_count)''
        matrix, "output" mapping to a ``(data_count, response_count)'' matrix,
        and "labels" mapping to a ``(data_count)'' vector.
    dict
        A dict with keys "input" mapping to a ``(data_count, feature_count)''
        matrix, "output" mapping to a ``(data_count, response_count)'' matrix,
        and "labels" mapping to a ``(data_count)'' vector.
    """
    return (
        _make_gaussian_dict(
            train_count, feature_count, response_count, categorical=categorical
        ),
        _make_gaussian_dict(
            test_count, feature_count, response_count, categorical=categorical
        ),
    )
