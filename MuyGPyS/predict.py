# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from MuyGPyS.gp.distance import (
    crosswise_distances,
    pairwise_distances,
)
import numpy as np

from time import perf_counter


def classify_any(
    muygps,
    test,
    train,
    train_nbrs_lookup,
    train_labels,
):
    """
    Simulatneously predicts the surrogate regression means for each test item.

    Parameters
    ----------
    muygps : MuyGPyS.GP.MuyGPS
        Local kriging approximate GP.
    test : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        Testing data.
    train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        Training data.
    train_nbrs_lookup : MuyGPyS.neighbors.NN_Wrapper
        Trained nearest neighbor query data structure.
    train_labels : numpy.ndarray(int), shape = ``(train_count, class_count)''
        One-hot encoding of class labels for all training data.

    Returns
    -------
    predictions : numpy.ndarray(float), shape = ``(test_count,)''
        The predicted labels associated with each test observation.
    timing : dict
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
        predictions[nonconstant_mask] = muygps.regress_from_indices(
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
    muygps,
    test,
    train,
    train_nbrs_lookup,
    train_labels,
):
    """
    Simultaneously predicts the surrogate means and variances for each test item
    under the assumption of binary classification.

    Parameters
    ----------
    muygps : MuyGPyS.GP.MuyGPS
        Local kriging approximate GP.
    test : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        Testing data.
    train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        Training data.
    train_nbrs_lookup : `MuyGPyS.neighbors.NN_Wrapper'
        Trained nearest neighbor query data structure.
    train_labels : numpy.ndarray(int), shape = ``(train_count, class_count)''
        One-hot encoding of class labels for all training data.

    Returns
    -------
    means : numpy.ndarray(float), shape = ``(test_count,)''
        The predicted labels associated with each test observation.
    timing : dict
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
        ) = muygps.regress_from_indices(
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
    muygps,
    test,
    train,
    train_nbrs_lookup,
    train_targets,
    nn_count,
    variance_mode=None,
):
    """
    Simultaneously predicts the response for each test item.

    Parameters
    ----------
    muygps : MuyGPyS.GP.MuyGPS
        Local kriging approximate GP.
    test : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        Testing data.
    train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        Training raining data.
    train_nbrs_lookup : `MuyGPyS.neighbors.NN_Wrapper'
        Trained nearest neighbor query data structure.
    train_targets : numpy.ndarray(float), shape = ``(train_count, class_count)''
        Observed outputs for all training data.
    nn_count : int
        The number of nearest neighbors used for inference
    variance_mode : str or None
        Specifies the type of variance to return. Currently supports
        ``diagonal'' and None. If None, report no variance term.

    Returns
    -------
    predictions : numpy.ndarray(float), shape = ``(batch_count, class_count,)''
        The predicted response for each of the given indices. The form returned
        when ``variance_mode == None''.
    predictions : tuple
        Form returned when ``variance_mode is not None''. A pair of
        numpy.ndarrays, where the first element is the prediction matrix and the
        second is the variance object. This object is a vector of independent
        variances if ``variance_mode == "diagonal"''.
    timing : dict
        Timing for the subroutines of this function.
    """
    test_count = test.shape[0]
    train_count = train.shape[0]

    time_start = perf_counter()
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test)
    time_nn = perf_counter()

    time_agree = perf_counter()

    predictions = muygps.regress(
        np.array([*range(test_count)]),
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
