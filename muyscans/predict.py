#!/usr/bin/env python
# encoding: utf-8
"""
@file prediction.py

Created by priest2 on 2020-10-19

Leave-one-out hyperparameter optimization logic and testing.
"""

import numpy as np

from time import perf_counter


def classify_any(
    lkgp,
    embedded_test,
    embedded_train,
    train_nbrs_lookup,
    train_labels,
    nn_count,
):
    """
    Obtains the predicted class labels for each test. Predicts on all of the
    testing data at once.

    Parameters
    ----------
    lkgp : muyscans.GP.LKGP
        Local kriging approximate GP.
    test : numpy.ndarray, type = float, shape = ``(test_count, input_dim)''
        Embedded training data.
    train : numpy.ndarray, type = float, shape = ``(train_count, input_dim)''
        Embedded training data.
    train_nbrs_lookup : `muyscans.ML.NN_Wrapper'
        Trained nearest neighbor query data structure.
    train_labels : numpy.ndarray, type = int,
                   shape = ``(train_count, class_count)''
        One-hot encoding of class labels for all embedded data.
    nn_count : int
        The number of nearest neighbors used for inference

    Returns
    -------
    predictions : numpy.ndarray, type = int, shape = ``(test_count,)''
        The predicted labels associated with each test observation.
    timing : dict
        Timing for the subroutines of this function.
    """
    test_count = embedded_test.shape[0]
    class_count = train_labels.shape[1]

    # detect one hot encoding, e.g. {0,1}, {-0.1, 0.9}, {-1,1}, ...
    one_hot_false = float(np.min(train_labels[0, :]))
    predictions = np.full((test_count, class_count), one_hot_false)

    time_start = perf_counter()
    test_nn_indices = train_nbrs_lookup.get_nns(embedded_test)
    time_nn = perf_counter()

    nn_labels = train_labels[test_nn_indices, :]
    nonconstant_mask = np.max(nn_labels[:, :, 0], axis=-1) != np.min(
        nn_labels[:, :, 0], axis=-1
    )

    predictions[np.invert(nonconstant_mask), :] = nn_labels[
        np.invert(nonconstant_mask), 0, :
    ]
    time_agree = perf_counter()

    predictions[nonconstant_mask] = lkgp.regress(
        np.where(nonconstant_mask == True)[0],
        test_nn_indices[nonconstant_mask, :],
        embedded_test,
        embedded_train,
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
    lkgp,
    embedded_test,
    embedded_train,
    train_nbrs_lookup,
    train_labels,
    nn_count,
):
    """
    Obtains the predicted means and variances each test for binary
    classification. Predicts on all of the testing data at once.

    Parameters
    ----------
    lkgp : muyscans.GP.LKGP
        Local kriging approximate GP.
    test : numpy.ndarray, type = float, shape = ``(test_count, input_dim)''
        Embedded training data.
    train : numpy.ndarray, type = float, shape = ``(train_count, input_dim)''
        Embedded training data.
    train_nbrs_lookup : `muyscans.ML.NN_Wrapper'
        Trained nearest neighbor query data structure.
    train_labels : numpy.ndarray, type = int,
                   shape = ``(train_count, class_count)''
        One-hot encoding of class labels for all embedded data.
    nn_count : int
        The number of nearest neighbors used for inference

    Returns
    -------
    means : numpy.ndarray, type = int, shape = ``(test_count,)''
        The predicted labels associated with each test observation.
    timing : dict
        Timing for the subroutines of this function.
    """
    test_count = embedded_test.shape[0]
    train_count = embedded_train.shape[0]

    means = np.zeros((test_count, 2))
    variances = np.zeros(test_count)

    time_start = perf_counter()
    test_nn_indices = train_nbrs_lookup.get_nns(embedded_test)
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

    mu, variances[nonconstant_mask] = lkgp.regress(
        np.where(nonconstant_mask == True)[0],
        test_nn_indices[nonconstant_mask, :],
        embedded_test,
        embedded_train,
        train_labels,
        variance_mode="diagonal",
    )
    # NOTE[bwp] there is probably a way to write this directly into `means`
    # without the extra copy...
    means[nonconstant_mask, :] = mu
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }

    return means, variances, timing


def regress_any(
    lkgp,
    embedded_test,
    embedded_train,
    train_nbrs_lookup,
    train_targets,
    nn_count,
    variance_mode=None,
):
    """
    Obtains the predicted response for each test. Predicts on all of the
    testing data at once.

    Parameters
    ----------
    lkgp : muyscans.GP.LKGP
        Local kriging approximate GP.
    test : numpy.ndarray, type = float, shape = ``(test_count, input_dim)''
        Embedded training data.
    train : numpy.ndarray, type = float, shape = ``(train_count, input_dim)''
        Embedded training data.
    train_nbrs_lookup : `muyscans.ML.NN_Wrapper'
        Trained nearest neighbor query data structure.
    train_targets : numpy.ndarray, type = float,
                   shape = ``(train_count, output_dim)''
        Observed outputs for all training data.
    nn_count : int
        The number of nearest neighbors used for inference
    variance_mode : str or None
        Specifies the type of variance to return. Currently supports
        ``diagonal'' and None. If None, report no variance term.

    Returns
    -------
    predictions : numpy.ndarray, type = float,
                shape = ``(batch_count, output_dim,)''
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
    test_count = embedded_test.shape[0]
    train_count = embedded_train.shape[0]

    predicted_labels = np.zeros((test_count,))

    time_start = perf_counter()
    test_nn_indices = train_nbrs_lookup.get_nns(embedded_test)
    time_nn = perf_counter()

    time_agree = perf_counter()

    predictions = lkgp.regress(
        np.array([*range(test_count)]),
        test_nn_indices,
        embedded_test,
        embedded_train,
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
