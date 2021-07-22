# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Convenience functions for prediction workflows.
"""

import numpy as np

from time import perf_counter
from typing import Dict, Optional, Tuple, Union

from MuyGPyS.gp.distance import (
    crosswise_distances,
    pairwise_distances,
)
from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper


def classify_any(
    surrogate: Union[MuyGPS, MMuyGPS],
    test: np.ndarray,
    train: np.ndarray,
    train_nbrs_lookup: NN_Wrapper,
    train_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulatneously predicts the surrogate regression means for each test item.

    Args:
        surrogate:
            Surrogate regressor.
        test:
            Test observations of shape `(test_count, feature_count)`.
        train:
            Train observations of shape `(train_count, feature_count)`.
        train_nbrs_lookup:
            Trained nearest neighbor query data structure.
        train_labels:
            One-hot encoding of class labels for all training data of shape
            `(train_count, class_count)`.

    Returns
    -------
    predictions:
        The surrogate predictions of shape `(test_count, class_count)` for each
        test observation.
    timing:
        Timing for the subroutines of this function.
    """
    test_count = test.shape[0]
    class_count = train_labels.shape[1]

    # detect one hot encoding, e.g. {0,1}, {-0.1, 0.9}, {-1,1}, ...
    one_hot_false = float(np.min(train_labels[0, :]))
    predictions = np.full((test_count, class_count), one_hot_false)

    time_start = perf_counter()
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test)
    time_nn = perf_counter()

    nn_labels = train_labels[test_nn_indices, :]
    nonconstant_mask = np.max(nn_labels[:, :, 0], axis=-1) != np.min(
        nn_labels[:, :, 0], axis=-1
    )

    predictions[np.invert(nonconstant_mask), :] = nn_labels[
        np.invert(nonconstant_mask), 0, :
    ]
    time_agree = perf_counter()

    if np.sum(nonconstant_mask) > 0:
        predictions[nonconstant_mask] = surrogate.regress_from_indices(
            np.where(nonconstant_mask == True)[0],
            test_nn_indices[nonconstant_mask, :],
            test,
            train,
            train_labels,
        )
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }
    return predictions, timing


def classify_two_class_uq(
    surrogate: Union[MuyGPS, MMuyGPS],
    test: np.ndarray,
    train: np.ndarray,
    train_nbrs_lookup: NN_Wrapper,
    train_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simultaneously predicts the surrogate means and variances for each test item
    under the assumption of binary classification.

    Args:
        surrogate:
            Surrogate regressor.
        test:
            Test observations of shape `(test_count, feature_count)`.
        train:
            Train observations of shape `(train_count, feature_count)`.
        train_nbrs_lookup:
            Trained nearest neighbor query data structure.
        train_labels:
            One-hot encoding of class labels for all training data of shape
            `(train_count, class_count)`.

    Returns
    -------
    means:
        The surrogate predictions for each test observation of shape
        `(test_count, 2)`.
    variances:
        The posterior variances for each test observation of shape
        `(test_count,)`
    timing:
        Timing for the subroutines of this function.
    """
    test_count = test.shape[0]
    train_count = train.shape[0]

    means = np.zeros((test_count, 2))
    variances = np.zeros(test_count)

    time_start = perf_counter()
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test)
    time_nn = perf_counter()

    nn_labels = train_labels[test_nn_indices, :]
    nonconstant_mask = np.max(nn_labels[:, :, 0], axis=-1) != np.min(
        nn_labels[:, :, 0], axis=-1
    )
    means[np.invert(nonconstant_mask)] = nn_labels[
        np.invert(nonconstant_mask), 0
    ]
    variances[np.invert(nonconstant_mask)] = 0.0
    time_agree = perf_counter()

    if np.sum(nonconstant_mask) > 0:
        (
            means[nonconstant_mask, :],
            variances[nonconstant_mask],
        ) = surrogate.regress_from_indices(
            np.where(nonconstant_mask == True)[0],
            test_nn_indices[nonconstant_mask, :],
            test,
            train,
            train_labels,
            variance_mode="diagonal",
        )

        # means[nonconstant_mask, :] = mu
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }

    return means, variances, timing


def regress_any(
    regressor: Union[MuyGPS, MMuyGPS],
    test: np.ndarray,
    train: np.ndarray,
    train_nbrs_lookup: NN_Wrapper,
    train_targets: np.ndarray,
    variance_mode: Optional[str] = None,
) -> Union[
    Tuple[np.ndarray, Dict[str, float]],
    Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, float]],
]:
    """
    Simultaneously predicts the response for each test item.

    Args:
        regressor:
            Regressor object.
        test:
            Test observations of shape `(test_count, feature_count)`.
        train:
            Train observations of shape `(train_count, feature_count)`.
        train_nbrs_lookup:
            Trained nearest neighbor query data structure.
        train_targets:
            Observed response for all training data of shape
            `(train_count, class_count)`.
        variance_mode : str or None
            Specifies the type of variance to return. Currently supports
            `diagonal` and None. If None, report no variance term.

    Returns
    -------
    means:
        The predicted response of shape `(test_count, response_count,)` for
        each of the test examples.
    variances:
        The independent posterior variances for each of the test examples. Of
        shape `(test_count,)` if the argument `regressor` is an instance of
        :class:`MuyGPyS.gp.muygps.MuyGPS`, and of shape
        `(test_count, response_count)` if `regressor` is an instance of
        :class:`MuyGPyS.gp.muygps.MultivariateMuyGPS`. Returned only when
        `variance_mode == "diagonal"`.
    timing : dict
        Timing for the subroutines of this function.
    """
    test_count = test.shape[0]
    train_count = train.shape[0]

    time_start = perf_counter()
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test)
    time_nn = perf_counter()

    time_agree = perf_counter()

    predictions = regressor.regress_from_indices(
        np.arange(test_count),
        test_nn_indices,
        test,
        train,
        train_targets,
        variance_mode=variance_mode,
    )
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }
    return predictions, timing
