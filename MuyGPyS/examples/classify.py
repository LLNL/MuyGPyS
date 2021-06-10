# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from MuyGPyS.optimize.chassis import scipy_optimize_from_tensors
from MuyGPyS.gp.distance import crosswise_distances, pairwise_distances

from MuyGPyS.optimize.batch import get_balanced_batch
from MuyGPyS.predict import (
    classify_two_class_uq,
    classify_any,
)
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.uq import train_two_class_interval
from MuyGPyS.gp.muygps import MuyGPS

from time import perf_counter

example_lambdas = [
    lambda alpha, beta, correct_count, incorrect_count: np.argmin(alpha + beta),
    lambda alpha, beta, correct_count, incorrect_count: np.argmin(
        2 * alpha + beta
    ),
    lambda alpha, beta, correct_count, incorrect_count: np.argmin(
        4 * alpha + beta
    ),
    lambda alpha, beta, correct_count, incorrect_count: np.argmin(
        10 * alpha + beta
    ),
    lambda alpha, beta, correct_count, incorrect_count: np.argmin(
        incorrect_count * alpha + correct_count * beta
    ),
]


def make_classifier(
    train_data,
    train_labels,
    nn_count=30,
    batch_size=200,
    loss_method="log",
    k_kwargs=dict(),
    nn_kwargs=dict(),
    verbose=False,
):
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Parameters
    ----------
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors.
    train_labels = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row label vectors for the training data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_size : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only ``log'' (also known as
        ``cross_entropy'') and ``mse'' for regression.
    k_kwargs : dict
        Parameters for the kernel, possibly including kernel type, distance
        metric, epsilon and sigma hyperparameter specifications, and
        specifications for kernel hyperparameters. If all of the hyperparameters
        are fixed or are not given optimization bounds, no optimization will
        occur.
    nn_kwargs : dict
        Parameters for the nearest neighbors wrapper. See
        `MuyGPyS.neighbors.NN_Wrapper` for the supported methods and their
        parameters.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    muygps : MuyGPyS.gp.MuyGPS
        A (possibly trained) MuyGPs object.
    nbrs_lookup : MuyGPyS.neighbors.NN_Wrapper
        A data structure supporting nearest neighbor queries into
        ``train_data''.

    Examples
    --------
    >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
    >>> from MuyGPyS.examples.regress import make_regressor
    >>> train = _make_gaussian_dict(10000, 100, 10, categorial=True)
    >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
    >>> k_kwargs = {
    ...         "kern": "rbf",
    ...         "metric": "F2",
    ...         "eps": {"val": 1e-5},
    ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)},
    ... }
    >>> muygps, nbrs_lookup = make_classifier(
    ...         train['input'],
    ...         train['output'],
    ...         nn_count=30,
    ...         batch_size=200,
    ...         loss_method="log",
    ...         k_kwargs=k_kwargs,
    ...         nn_kwargs=nn_kwargs,
    ...         verbose=False,
    ... )
    """
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_data,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()

    muygps = MuyGPS(**k_kwargs)
    if muygps.fixed_nosigmasq() is False:
        # collect batch
        batch_indices, batch_nn_indices = get_balanced_batch(
            nbrs_lookup,
            np.argmax(train_labels, axis=1),
            batch_size,
        )
        time_batch = perf_counter()

        crosswise_dists = crosswise_distances(
            train_data,
            train_data,
            batch_indices,
            batch_nn_indices,
            metric=muygps.kernel.metric,
        )
        pairwise_dists = pairwise_distances(
            train_data, batch_nn_indices, metric=muygps.kernel.metric
        )
        time_tensor = perf_counter()

        # maybe do something with these estimates?
        estimates = scipy_optimize_from_tensors(
            muygps,
            batch_indices,
            batch_nn_indices,
            crosswise_dists,
            pairwise_dists,
            train_labels,
            loss_method=loss_method,
            verbose=verbose,
        )
        time_opt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")

    return muygps, nbrs_lookup


def do_classify(
    test_data,
    train_data,
    train_labels,
    nn_count=30,
    batch_size=200,
    loss_method="log",
    k_kwargs=dict(),
    nn_kwargs=dict(),
    verbose=False,
):
    """
    Convenience function for initializing a model and performing surrogate
    classification.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Parameters
    ----------
    test_data : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        A matrix of row observation vectors of the test data.
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the train data.
    train_labels = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row label vectors for the training data.
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the testing data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_size : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only ``log'' (also known as
        ``cross_entropy'') and ``mse'' for regression.
    uq_objectives : list(Callable)
        List of functions taking four arguments: bit masks alpha and beta - the
        type 1 and type 2 error counts at each grid location, respectively - and
        the numbers of correctly and incorrectly classified training examples.
        Used to determine scale parameter for confidence intervals. See
        `MuyGPyS.examples.classify.example_lambdas` for examples. If None, do
        not perform uncertainty quantifification.
        NOTE[bwp]: Supports only 2-class classification at the moment.
    k_kwargs : dict
        Parameters for the kernel, possibly including kernel type, distance
        metric, epsilon and sigma hyperparameter specifications, and
        specifications for kernel hyperparameters. If all of the hyperparameters
        are fixed or are not given optimization bounds, no optimization will
        occur.
    nn_kwargs : dict
        Parameters for the nearest neighbors wrapper. See
        `MuyGPyS.neighbors.NN_Wrapper` for the supported methods and their
        parameters.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    muygps : MuyGPyS.gp.MuyGPS
        A (possibly trained) MuyGPs object.
    nbrs_lookup : MuyGPyS.neighbors.NN_Wrapper
        A data structure supporting nearest neighbor queries into
        ``train_data''.

    Examples
    --------
    >>> import numpy as np
    >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
    >>> from MuyGPyS.examples.regress import do_classify
    >>> train, test  = _make_gaussian_dict(10000, 100, 100, 10, categorial=True)
    >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
    >>> k_kwargs = {
    ...         "kern": "rbf",
    ...         "metric": "F2",
    ...         "eps": {"val": 1e-5},
    ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)},
    ... }
    >>> muygps, nbrs_lookup, surrogate_predictions = do_classify(
    ...         test['input'],
    ...         train['input'],
    ...         train['output'],
    ...         nn_count=30,
    ...         batch_size=200,
    ...         loss_method="log",
    ...         k_kwargs=k_kwargs,
    ...         nn_kwargs=nn_kwargs,
    ...         verbose=False,
    ... )
    >>> predicted_labels = np.argmax(surrogate_predictions, axis=1)
    >>> true_labels = np.argmax(test['output'], axis=1)
    >>> acc = np.mean(predicted_labels == true_labels)
    >>> print(f"obtained accuracy {acc}")
    obtained accuracy: 0.973...
    """
    muygps, nbrs_lookup = make_classifier(
        train_data,
        train_labels,
        nn_count=nn_count,
        batch_size=batch_size,
        loss_method=loss_method,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        verbose=verbose,
    )
    surrogate_predictions, pred_timing = classify_any(
        muygps,
        test_data,
        train_data,
        nbrs_lookup,
        train_labels,
    )
    if verbose is True:
        print(f"prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
    return muygps, nbrs_lookup, surrogate_predictions


def do_classify_uq(
    test_data,
    train_data,
    train_labels,
    nn_count=30,
    opt_batch_size=200,
    uq_batch_size=500,
    loss_method="log",
    uq_objectives=example_lambdas,
    k_kwargs=dict(),
    nn_kwargs=dict(),
    verbose=False,
):
    """
    Convenience function for initializing a model and performing surrogate
    classification.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Parameters
    ----------
    test_data : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        A matrix of row observation vectors of the test data.
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the train data.
    train_labels = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row label vectors for the training data.
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the testing data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_size : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only ``mse'' for regression.
    k_kwargs : dict
        Parameters for the kernel, possibly including kernel type, distance
        metric, epsilon and sigma hyperparameter specifications, and
        specifications for kernel hyperparameters. If all of the hyperparameters
        are fixed or are not given optimization bounds, no optimization will
        occur.
    nn_kwargs : dict
        Parameters for the nearest neighbors wrapper. See
        `MuyGPyS.neighbors.NN_Wrapper` for the supported methods and their
        parameters.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    muygps : MuyGPyS.gp.MuyGPS
        A (possibly trained) MuyGPs object.
    nbrs_lookup : MuyGPyS.neighbors.NN_Wrapper
        A data structure supporting nearest neighbor queries into
        ``train_data''.

    Examples
    --------
    >>> import numpy as np
    >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
    >>> from MuyGPyS.examples.regress import do_classify_uq, do_uq
    >>> train, test  = _make_gaussian_dict(10000, 100, 100, 10, categorial=True)
    >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
    >>> k_kwargs = {
    ...         "kern": "rbf",
    ...         "metric": "F2",
    ...         "eps": {"val": 1e-5},
    ...         "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)},
    ... }
    >>> muygps, nbrs_lookup, surrogate_predictions = do_classify(
    ...         test['input'],
    ...         train['input'],
    ...         train['output'],
    ...         nn_count=30,
    ...         batch_size=200,
    ...         loss_method="log",
    ...         k_kwargs=k_kwargs,
    ...         nn_kwargs=nn_kwargs,
    ...         verbose=False,
    ... )
    >>> accuracy, uq = do_uq(surrogate_predictions, test['output'], masks)
    >>> print(f"obtained accuracy {accuracy}")
    obtained accuracy: 0.973...
    >>> print(f"obtained mask uq: \n{uq}")
    obtained mask uq :
        [[8.21000000e+02 8.53836784e-01 9.87144569e-01]
        [8.59000000e+02 8.55646100e-01 9.87528717e-01]
        [1.03500000e+03 8.66666667e-01 9.88845510e-01]
        [1.03500000e+03 8.66666667e-01 9.88845510e-01]
        [5.80000000e+01 6.72413793e-01 9.77972239e-01]]
    """
    muygps, nbrs_lookup = make_classifier(
        train_data,
        train_labels,
        nn_count=nn_count,
        batch_size=opt_batch_size,
        loss_method=loss_method,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        verbose=verbose,
    )

    surrogate_predictions, variances, pred_timing = classify_two_class_uq(
        muygps,
        test_data,
        train_data,
        nbrs_lookup,
        train_labels,
    )

    min_label = np.min(train_labels[0, :])
    max_label = np.max(train_labels[0, :])
    # if min_label == 0.0 and max_label == 1.0:
    #     predicted_labels = np.argmax(predictions, axis=1)
    # elif min_label == -1.0 and max_label == 1.0:
    #     predicted_labels = 2 * np.argmax(predictions, axis=1) - 1
    # else:
    #     raise ("Unhandled label encoding min ({min_label}, {max_label})!")
    mid_value = (min_label + max_label) / 2
    time_pred = perf_counter()

    one_hot_labels = 2 * np.argmax(train_labels, axis=1) - 1

    batch_indices, batch_nn_indices = get_balanced_batch(
        nbrs_lookup,
        one_hot_labels,
        uq_batch_size,
    )
    time_uq_batch = perf_counter()

    # Training of confidence interval scaling using different objectives.
    cutoffs = train_two_class_interval(
        muygps,
        batch_indices,
        batch_nn_indices,
        train_data,
        train_labels,
        one_hot_labels,
        uq_objectives,
    )

    # Compute index masks indicating the predictions that include `0` in the
    # confidence interval for each of the training objectives.
    masks = make_masks(surrogate_predictions, cutoffs, variances, mid_value)
    time_cutoff = perf_counter()

    if verbose is True:
        print(f"uq batching time: {time_cutoff - time_pred}")
        print(f"cutoff time: {time_cutoff - time_uq_batch}s")
        print(f"prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
    return muygps, nbrs_lookup, surrogate_predictions, masks


def make_masks(predictions, cutoffs, variances, mid_value):
    """
    Compute boolean masks over all of the test data indicating which test
    indices are considered ambiguous

    Parameters
    ----------
    predictions : np.ndarray(float), shape = ``(test_count, class_count)''
        The surrogate predictions.
    cutoffs : np.ndarray(float), shape = ``(objective_count,)''
        The confidence interval scale parameter that minimizes each
        considered objective function.
    variances : np.ndarray(float), shape = ``(test_count)''
        The diagonal posterior variance of each test item.
    mid_value : float
        The discriminating value determining absolute uncertainty. Likely ``0''
        or ``0.5''.

    Returns
    -------
    np.ndarray(Boolean), shape = ``(objective_count, test_count)''
        A number of index masks indexing into the training set. Each ``True''
        index includes 0.0 within the associated prediction's confidence
        interval.
    """
    return np.array(
        [
            np.logical_and(
                predictions[:, 1] - cut * variances < mid_value,
                predictions[:, 1] + cut * variances > mid_value,
            )
            for cut in cutoffs
        ]
    )


def do_uq(surrogate_predictions, test_labels, masks):
    """
    Convenience function performing uncertainty quantification given predicted
    labels and ground truth for a given set of confidence interval scales.

    Parameters
    ----------
    surrogate_predictions : np.ndarray(float),
                            shape = ``(test_count, class_count)''
        The surrogate predictions, based e.g. on an invocation of
        ``do_classify_uq''.
    test_labels : np.ndarray(float), shape = ``(test_count, class_count)''
        A matrix listing the one-hot encodings of each observation's class.
    masks : np.ndarray(Boolean), shape = ``(objective_count, test_count)''
        A number of index masks indexing into the training set. Each ``True''
        index includes 0.0 within the associated prediction's confidence
        interval.

    Returns
    -------
    accuracy : float
        The accuracy over all of the test data.
    uq : numpy.ndarray(float), shape = ``(objective_count, 3)''
        The uncertainty quantification associated with each input mask. The
        first column is the total number of ambiguous samples. The second column
        is the accuracy of the ambiguous samples. The third column is the
        accuracy of the unambiguous samples.
    """
    correct = np.argmax(surrogate_predictions, axis=1) == np.argmax(
        test_labels, axis=1
    )
    uq = np.array(
        [
            [
                np.sum(mask),
                np.mean(correct[mask]),
                np.mean(correct[np.invert(mask)]),
            ]
            for mask in masks
        ]
    )
    for i in range(uq.shape[0]):
        if uq[i, 0] == 0:
            uq[i, 1] = 0.0
    return np.mean(correct), uq
