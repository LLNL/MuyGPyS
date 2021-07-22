# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""Resources and high-level API for some classification workflows.
"""

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
from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS

from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "log",
    k_kwargs: Dict = dict(),
    nn_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[MuyGPS, NN_Wrapper]:
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

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
    ...         batch_count=200,
    ...         loss_method="log",
    ...         k_kwargs=k_kwargs,
    ...         nn_kwargs=nn_kwargs,
    ...         verbose=False,
    ... )

    Args:
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_labels
            A matrix of shape `(train_count, class_count)` whose rows consist
            of one-hot class label vectors of the train data.
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The number of elements to sample batch for hyperparameter
            optimization.
        loss_method:
            The loss method to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
            Currently supports only `"log"` (or `"cross-entropy"`) and `"mse"`
            for classification.
        k_kwargs:
            Parameters for the kernel, possibly including kernel type, distance
            metric, epsilon and sigma hyperparameter specifications, and
            specifications for kernel hyperparameters. See
            :ref:`MuyGPyS-gp-kernels` for examples and requirements. If all of
            the hyperparameters are fixed or are not given optimization bounds,
            no optimization will occur.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        verbose : Boolean
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        ``train_features''.
    """
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_features,
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
            batch_count,
        )
        time_batch = perf_counter()

        crosswise_dists = crosswise_distances(
            train_features,
            train_features,
            batch_indices,
            batch_nn_indices,
            metric=muygps.kernel.metric,
        )
        pairwise_dists = pairwise_distances(
            train_features, batch_nn_indices, metric=muygps.kernel.metric
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


def make_multivariate_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "mse",
    kern: str = "matern",
    k_args: Union[List[Dict], Tuple[Dict, ...]] = list(),
    nn_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[MMuyGPS, NN_Wrapper]:
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
        >>> from MuyGPyS.testing.test_utils import _make_gaussian_data
        >>> from MuyGPyS.examples.regress import make_regressor
        >>> train = _make_gaussian_dict(10000, 100, 10, categorial=True)
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_args = [
        ...         {
        ...             "length_scale": {"val": 1.0, "bounds": (1e-2, 1e2)}
        ...             "eps": {"val": 1e-5},
        ...         },
        ...         {
        ...             "length_scale": {"val": 1.5, "bounds": (1e-2, 1e2)}
        ...             "eps": {"val": 1e-5},
        ...         },
        ... ]
        >>> mmuygps, nbrs_lookup = make_multivariate_classifier(
        ...         train['input'],
        ...         train['output'],
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         kern="rbf",
        ...         k_args=k_args,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )

    Args:
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_labels:
            A matrix of shape `(train_count, class_count)` whose rows consist
            of one-hot encoded label vectors of the train data.
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The number of elements to sample batch for hyperparameter
            optimization.
        loss_method:
            The loss method to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
            Currently supports only `"mse"` for regression.
        kern:
            The kernel function to be used. See :ref:`MuyGPyS-gp-kernels` for
            details.
        k_args:
            A list of `response_count` dicts containing kernel initialization
            keyword arguments. Each dict specifies parameters for the kernel,
            possibly including epsilon and sigma hyperparameter specifications
            and specifications for specific kernel hyperparameters. If all of
            the hyperparameters are fixed or are not given optimization bounds,
            no optimization will occur.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        verbose:
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    """
    train_count, response_count = train_labels.shape
    if response_count != len(k_args):
        raise ValueError(
            f"supplied arguments for {len(k_args)} kernels, which does not "
            f"match expected {response_count} responses!"
        )
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_features,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()

    mmuygps = MMuyGPS(kern, *k_args)
    if mmuygps.fixed_nosigmasq() is False:
        # collect batch
        batch_indices, batch_nn_indices = get_balanced_batch(
            nbrs_lookup,
            np.argmax(train_labels, axis=1),
            batch_count,
        )
        time_batch = perf_counter()

        crosswise_dists = crosswise_distances(
            train_features,
            train_features,
            batch_indices,
            batch_nn_indices,
            metric=mmuygps.metric,
        )
        pairwise_dists = pairwise_distances(
            train_features, batch_nn_indices, metric=mmuygps.metric
        )
        time_tensor = perf_counter()

        # maybe do something with these estimates?
        for i, muygps in enumerate(mmuygps.models):
            if muygps.fixed_nosigmasq() is False:
                estimates = scipy_optimize_from_tensors(
                    muygps,
                    batch_indices,
                    batch_nn_indices,
                    crosswise_dists,
                    pairwise_dists,
                    train_labels[:, i].reshape(train_count, 1),
                    loss_method=loss_method,
                    verbose=verbose,
                )
        time_opt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")

    return mmuygps, nbrs_lookup


def _decide_and_make_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "log",
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper]:
    if kern is not None and isinstance(k_kwargs, list):
        return make_multivariate_classifier(
            train_features,
            train_labels,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            kern=kern,
            k_args=k_kwargs,
            nn_kwargs=nn_kwargs,
            verbose=verbose,
        )
    else:
        if isinstance(k_kwargs, dict):
            return make_classifier(
                train_features,
                train_labels,
                nn_count=nn_count,
                batch_count=batch_count,
                loss_method=loss_method,
                k_kwargs=k_kwargs,
                nn_kwargs=nn_kwargs,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Expected k_kwargs to be a dict!")


def do_classify(
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "log",
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Union[Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray]]:
    """
    Convenience function for initializing a model and performing surrogate
    classification.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Parameters
    ----------
    test_features: numpy.ndarray(float), shape = ``(test_count, feature_count)''
        A matrix of row observation vectors of the test data.
    train_features : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the train data.
    train_labels = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row label vectors for the training data.
    train_features : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the testing data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_count : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only ``log'' (also known as
        ``cross_entropy'') and `"mse"` for regression.
    uq_objectives : list(Callable)
        List of functions taking four arguments: bit masks alpha and beta - the
        type 1 and type 2 error counts at each grid location, respectively - and
        the numbers of correctly and incorrectly classified training examples.
        Used to determine scale parameter for confidence intervals. See
        `MuyGPyS.examples.classify.example_lambdas` for examples. If None, do
        not perform uncertainty quantifification.
        NOTE[bwp]: Supports only 2-class classification at the moment.
    kern : str
        The kernel function to be used. Only relevant for multivariate case
        where ``k_kwargs'' is a list of hyperparameter dicts.
        NOTE[bwp]: Currently supports only ``matern'' and ``rbf''.
    k_kwargs : dict or list(dict)
        Parameters for the kernel, possibly including kernel type, distance
        metric, epsilon and sigma hyperparameter specifications, and
        specifications for kernel hyperparameters. If all of the hyperparameters
        are fixed or are not given optimization bounds, no optimization will
        occur. If ``kern'' is specified and ``k_kwargs'' is a list of such
        dicts, will create a multivariate regressor model.
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
        ``train_features''.

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
    ...         batch_count=200,
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
    classifier, nbrs_lookup = _decide_and_make_classifier(
        train_features,
        train_labels,
        nn_count=nn_count,
        batch_count=batch_count,
        loss_method=loss_method,
        kern=kern,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        verbose=verbose,
    )

    surrogate_predictions, pred_timing = classify_any(
        classifier,
        test_features,
        train_features,
        nbrs_lookup,
        train_labels,
    )
    if verbose is True:
        print(f"prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
    return classifier, nbrs_lookup, surrogate_predictions


def do_classify_uq(
    test_features,
    train_features,
    train_labels,
    nn_count=30,
    opt_batch_count=200,
    uq_batch_count=500,
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
    test_features : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        A matrix of row observation vectors of the test data.
    train_features : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the train data.
    train_labels = numpy.ndarray(float),
                    shape = ``(train_count, response_count)''
        A matrix of row label vectors for the training data.
    train_features : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        A matrix of row observation vectors of the testing data.
    nn_count : int
        The number of nearest neighbors to employ.
    batch_count : int
        The batch size for hyperparameter optimization.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if all of
        the parameters specified by ``k_kwargs'' are fixed.
        NOTE[bwp]: Currently supports only `"mse"` for regression.
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
        ``train_features''.

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
    ...         batch_count=200,
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
        train_features,
        train_labels,
        nn_count=nn_count,
        batch_count=opt_batch_count,
        loss_method=loss_method,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        verbose=verbose,
    )

    surrogate_predictions, variances, pred_timing = classify_two_class_uq(
        muygps,
        test_features,
        train_features,
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
        uq_batch_count,
    )
    time_uq_batch = perf_counter()

    # Training of confidence interval scaling using different objectives.
    cutoffs = train_two_class_interval(
        muygps,
        batch_indices,
        batch_nn_indices,
        train_features,
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
