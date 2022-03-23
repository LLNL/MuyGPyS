# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np

from typing import Dict, Generator, Tuple, Union

from MuyGPyS import config

if config.muygpys_hnswlib_enabled is True:  # type: ignore
    _basic_nn_kwarg_options = [
        {"nn_method": "exact", "algorithm": "ball_tree"},
        {
            "nn_method": "hnsw",
            "space": "l2",
            "ef_construction": 100,
            "M": 16,
        },
    ]
else:
    _basic_nn_kwarg_options = [
        {"nn_method": "exact", "algorithm": "ball_tree"},
    ]

_exact_nn_kwarg_options = ({"nn_method": "exact", "algorithm": "ball_tree"},)

_basic_opt_method_and_kwarg_options = [
    ["scipy", dict()],
    ["bayesian", {"random_state": 1, "init_points": 3, "n_iter": 10}],
]

_advanced_opt_method_and_kwarg_options = [
    ["scipy", dict()],
    ["bayesian", {"random_state": 1, "init_points": 5, "n_iter": 20}],
]


def _sq_rel_err(
    tru: Union[float, np.ndarray], est: Union[float, np.ndarray]
) -> float:
    """
    Compute the relative squared error between two arguments.

    Args:
        tru:
            The approximated quantity.
        est:
            The estimate to be evaluated.

    Returns:
        An i.i.d. Gaussian matrix of shape `(data_count, feature_count)`.
    """
    return ((tru - est) / tru) ** 2


def _make_gaussian_matrix(
    data_count: int,
    feature_count: int,
) -> np.ndarray:
    """
    Create a matrix of i.i.d. Gaussian datapoints.

    Args:
        data_count:
            The number of data rows.
        feature_count:
            The number of data columns.

    Returns:
        An i.i.d. Gaussian matrix of shape `(data_count, feature_count)`.
    """
    return np.random.randn(data_count, feature_count)


def _make_gaussian_dict(
    data_count: int,
    feature_count: int,
    response_count: int,
    categorical: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Create a data dict including "input", "output", and "labels" keys mapping to
    i.i.d. Gaussian matrices.

    Args:
        data_count:
            The number of data rows.
        feature_count:
            The number of data columns in the `data["input"]` matrix.
        resonse_count:
            The number of data columns in the `data["output"]` matrix.
        categorical:
            If `True`, convert the `data["output"]` matrix to a one-hot encoding
            matrix.

    Returns:
        A dict with keys `"input"` mapping to a `(data_count, feature_count)`
        matrix, `"output"` mapping to a `(data_count, response_count)` matrix,
        and `"labels"` mapping to a `(data_count)` vector.
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
    train_count: int,
    test_count: int,
    feature_count: int,
    response_count: int,
    categorical: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Create train and test dicts including `"input"`, `"output"`, and `"labels"`
    keys mapping to i.i.d. Gaussian matrices.

    Args:
        train_count:
            The number of train rows.
        test_count:
            The number of test rows.
        feature_count:
            The number of data columns in the `test["input"]` and
            `train["input"]` matrices.
        resonse_count:
            The number of data columns in the `test["output"]` and
            `train["input"]` matrices.
        categorical:
            If `True`, convert the `data["output"]` matrix to a one-hot encoding
            matrix.

    Returns
    -------
    train:
        A dict with keys `"input"` mapping to a matrix of shape
        `(train_count, feature_count)`, `"output`" mapping to a matrix of shape
        `(train_count, response_count)`, and `"labels"` mapping to a vector of
        shape `(train_count,)`.
    test:
        A dict with keys `"input"` mapping to a matrix of shape
        `(test_count, feature_count)`, `"output`" mapping to a matrix of shape
        `(test_count, response_count)`, and `"labels"` mapping to a vector of
        shape `(test_count,)`.
    """
    return (
        _make_gaussian_dict(
            train_count, feature_count, response_count, categorical=categorical
        ),
        _make_gaussian_dict(
            test_count, feature_count, response_count, categorical=categorical
        ),
    )


def _subsample(
    data: Dict[str, np.ndarray],
    sample_count: int,
) -> Dict[str, np.ndarray]:
    """
    Randomly sample row indices without replacement from data dict.

    NOTICE: This function and its Dict data format are intended for testing.

    Args:
        data:
            A dict with keys `"input"` and `"output"`. `data["input"]` maps to a
            matrix of shape `(data_count, feature_count)` whose rows consist of
            observation vectors. `data["output"]` maps to a matrix of shape
            `(data_count, response_count)` whose rows consist of response
            vectors.
        sample_count:
            The desired number of samples.

    Returns:
        A dict of the same form as `data`, but containing only the sampled
        indices.
    """
    count = data["input"].shape[0]
    samples = np.random.choice(count, sample_count, replace=False)
    return {
        "input": data["input"][samples, :],
        "output": data["output"][samples, :],
    }


def _balanced_subsample(
    data: Dict[str, np.ndarray],
    sample_count: int,
) -> Dict[str, np.ndarray]:
    """
    Randomly sample row indices without replacement from data dict, ensuring
    that classes receive as close to equal representation as possible.

    Partitions the data based upon their true classes, and attempts to randomly
    sample without replacement a balanced quantity within each partition. May
    not work well on heavily skewed data except with very small sample sizes.

    NOTICE: This function and its Dict data format are intended for testing.

    Args:
        data:
            A dict with keys `"input"` and `"output"`. `data["input"]` maps to a
            matrix of shape `(data_count, feature_count)` whose rows consist of
            observation vectors. `data["output"]` maps to a matrix of shape
            `(data_count, response_count)` whose rows consist of response
            vectors.
        sample_count:
            The desired number of samples.

    Returns:
        A dict of the same form as `data`, but containing only the sampled
        indices, who have as close to parity in class representation as
        possible.
    """
    labels = np.argmax(data["output"], axis=1)
    classes = np.unique(labels)
    class_count = len(classes)
    each_sample_count = int(sample_count / class_count)

    class_indices = np.array([np.where(labels == i)[0] for i in classes])
    sample_sizes = np.array(
        [np.min((len(arr), each_sample_count)) for arr in class_indices]
    )
    balanced_samples = np.concatenate(
        [
            np.random.choice(class_indices[i], sample_sizes[i], replace=False)
            for i in range(class_count)
        ]
    )
    return {
        "input": data["input"][balanced_samples, :],
        "output": data["output"][balanced_samples, :],
    }


def _normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalizes data matrix to have row l2-norms of 1

    Args:
        X:
            A matrix of shape `(data_count, feature_count)`.

    Returns:
        A row-normalized matrix of shape `(data-count, feature_count)`.
    """
    # return X * np.sqrt(1 / np.sum(X ** 2, axis=1))[:, None]
    return X * np.sqrt(X.shape[1] / np.sum(X**2, axis=1))[:, None]


def _get_sigma_sq_series(
    K: np.ndarray,
    nn_indices: np.ndarray,
    target_col: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Return the series of sigma^2 scale parameters for each neighborhood
    solve.

    NOTE[bwp]: This function is only for testing purposes.

    Args:
        K:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing
            the `(nn_count, nn_count` -shaped kernel matrices corresponding
            to each of the batch elements.
        nn_indices:
            An integral matrix of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the test batch.
        target_col:
            A vector of shape `(batch_count)` consisting of the target for
            each nearest neighbor.

    Returns:
        A vector of shape `(response_count)` listing the value of sigma^2
        for the given response dimension.
    """
    batch_count, nn_count = nn_indices.shape

    sigmas = np.zeros((batch_count,))
    for i, el in enumerate(_get_sigma_sq(K, target_col, nn_indices, eps)):
        sigmas[i] = el
    return sigmas / nn_count


def _get_sigma_sq(
    K: np.ndarray,
    target_col: np.ndarray,
    nn_indices: np.ndarray,
    eps: float,
) -> Generator[float, None, None]:
    """
    Generate series of :math:`\\sigma^2` scale parameters for each
    individual solve along a single dimension:

    .. math::
        \\sigma^2 = \\frac{1}{k} * Y_{nn}^T K_{nn}^{-1} Y_{nn}

    Here :math:`Y_{nn}` and :math:`K_{nn}` are the target and kernel
    matrices with respect to the nearest neighbor set in scope, where
    :math:`k` is the number of nearest neighbors.

    Args:
        K:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing
            the `(nn_count, nn_count` -shaped kernel matrices corresponding
            to each of the batch elements.
        target_col:
            A vector of shape `(batch_count)` consisting of the target for
            each nearest neighbor.
        nn_indices:
            An integral matrix of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the test batch.

    Return:
        A generator producing `batch_count` optimal values of
        :math:`\\sigma^2` for each neighborhood for the given response
        dimension.
    """
    batch_count, nn_count = nn_indices.shape
    for j in range(batch_count):
        Y_0 = target_col[nn_indices[j, :]]
        yield Y_0 @ np.linalg.solve(K[j, :, :] + eps * np.eye(nn_count), Y_0)
