# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, Generator, Optional, Tuple, Type, Union

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS import config
from MuyGPyS._src.mpi_utils import _is_mpi_mode
from MuyGPyS.optimize import Bayes_optimize, L_BFGS_B_optimize

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

_basic_opt_fn_and_kwarg_options = [
    [L_BFGS_B_optimize, dict()],
    [
        Bayes_optimize,
        {
            "random_state": 1,
            "init_points": 3,
            "n_iter": 10,
            "allow_duplicate_points": True,
        },
    ],
]

_advanced_opt_fn_and_kwarg_options = [
    [L_BFGS_B_optimize, dict()],
    [
        Bayes_optimize,
        {
            "random_state": 1,
            "init_points": 5,
            "n_iter": 20,
            "allow_duplicate_points": True,
        },
    ],
]


def _sq_rel_err(
    tru: Union[float, mm.ndarray], est: Union[float, mm.ndarray]
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
) -> mm.ndarray:
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
    return mm.array(np.random.randn(data_count, feature_count))


def _make_uniform_matrix(
    data_count: int,
    feature_count: int,
) -> mm.ndarray:
    """
    Create a matrix of i.i.d. Uniform(0,1) datapoints.

    Args:
        data_count:
            The number of data rows.
        feature_count:
            The number of data columns.

    Returns:
        An i.i.d. Gaussian matrix of shape `(data_count, feature_count)`.
    """
    return mm.array(np.random.rand(data_count, feature_count))


def _make_gaussian_dict(
    data_count: int,
    feature_count: int,
    response_count: int,
    categorical: bool = False,
) -> Dict[str, mm.ndarray]:
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
    labels = mm.argmax(observations, axis=1)
    if categorical is True:
        observations = mm.eye(response_count)[labels] - (1 / response_count)
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
) -> Tuple[Dict[str, mm.ndarray], Dict[str, mm.ndarray]]:
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
    data: Dict[str, mm.ndarray],
    sample_count: int,
) -> Dict[str, mm.ndarray]:
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
    samples = mm.array(
        np.random.choice(count, sample_count, replace=False), dtype=mm.itype
    )
    return {
        "input": data["input"][samples, :],
        "output": data["output"][samples, :],
    }


def _balanced_subsample(
    data: Dict[str, mm.ndarray],
    sample_count: int,
) -> Dict[str, mm.ndarray]:
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
    labels = mm.argmax(data["output"], axis=1)
    classes = mm.unique(labels)
    class_count = len(classes)
    each_sample_count = int(sample_count / class_count)

    class_indices = [np.where(labels == i)[0] for i in classes]
    sample_sizes = [
        np.min((len(arr), each_sample_count)) for arr in class_indices
    ]

    balanced_samples = np.concatenate(
        [
            np.random.choice(class_indices[i], sample_sizes[i], replace=False)
            for i in range(class_count)
        ]
    )
    return {
        "input": mm.array(data["input"][balanced_samples, :]),
        "output": mm.array(data["output"][balanced_samples, :]),
    }


def _normalize(X: mm.ndarray) -> mm.ndarray:
    """
    Normalizes data matrix to have row l2-norms of 1

    Args:
        X:
            A matrix of shape `(data_count, feature_count)`.

    Returns:
        A row-normalized matrix of shape `(data-count, feature_count)`.
    """
    return X * mm.sqrt(X.shape[1] / mm.sum(X**2, axis=1))[:, None]


def _get_scale_series(
    K: mm.ndarray,
    nn_targets_column: mm.ndarray,
    noise_variance: float,
) -> mm.ndarray:
    """
    Return the series of :math:`sigma^2` scale parameters for each neighborhood
    solve.

    NOTE[bwp]: This function is only for testing purposes.

    Args:
        K:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing
            the `(nn_count, nn_count` -shaped kernel matrices corresponding
            to each of the batch elements.
        nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, 1)` containing
            one dimension of the expected response for each nearest neighbor of
            each batch element.

    Returns:
        A vector of shape `(response_count)` listing the value of the scale
        parameter for the given response dimension.
    """
    batch_count, nn_count, _ = nn_targets_column.shape

    scales = np.zeros((batch_count,))
    for i, el in enumerate(_get_scale(K, nn_targets_column, noise_variance)):
        scales[i] = el
    return mm.array(scales / nn_count)


def _get_scale(
    K: mm.ndarray,
    nn_targets_column: mm.ndarray,
    noise_variance: float,
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
        nn_targets_column:
            Tensor of floats of shape `(batch_count, nn_count, 1)` containing
            one dimension of the expected response for each nearest neighbor of
            each batch element.

    Return:
        A generator producing `batch_count` optimal values of the
        :math:`\\sigma^2` variance scale parameter for each neighborhood for the
        given response dimension.
    """
    batch_count, nn_count, _ = nn_targets_column.shape
    for j in range(batch_count):
        Y_0 = nn_targets_column[j, :, 0]
        yield Y_0 @ mm.linalg.solve(
            K[j, :, :] + noise_variance * mm.eye(nn_count), Y_0
        )


def _check_ndarray(
    assert_fn: Callable,
    array: mm.ndarray,
    dtype: Type,
    ctype: Type = mm.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
):
    assert_fn(type(array), ctype)
    assert_fn(array.dtype, dtype)
    if shape is not None:
        assert_fn(array.shape, shape)


def _precision_assert(assert_fn, *args, low_bound=4, high_bound=7):
    if config.state.ftype == "32":
        return assert_fn(*args, low_bound)
    else:
        return assert_fn(*args, high_bound)


def _consistent_assert(assert_fn, *args):
    """
    Performs an assert on the root core if in mpi, otherwise performs the assert
    as normal.

    The purpose of this function is to allow the existing serial testing harness
    to also test the mpi implementations without the need for additional codes.

    Args:
        assert_fn:
            An absl assert function.
        args:
            Arguments to the assert function
    """
    if _is_mpi_mode() is True:
        if config.mpi_state.comm_world.Get_rank() == 0:
            assert_fn(*args)
    else:
        assert_fn(*args)


def _make_heteroscedastic_test_nugget(
    batch_count: int, nn_count: int, magnitude: float
):
    """
    Produces a test heteroscedastic 3D tensor parameter of shape
    `(batch_count, nn_count, nn_count)`.

    NOTE[amd]: This function is only for testing purposes.

    Args:
        batch_count:
            Number of points to be predicted.
        nn_count:
            Number of nearest neighbors in the kernel.
        magnitude:
            Maximum noise magnitude.


    Return:
        A `(batch_count, nn_count)` shaped tensor for heteroscedastic
        noise modeling.
    """
    return magnitude * mm.ones((batch_count, nn_count))
