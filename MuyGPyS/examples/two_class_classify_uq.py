# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a two-class classification with UQ workflow.

Implements a two-class classification workflow with a bespoke uncertainty
quantification tuning method. [muyskens2021star]_ describes this method and its
application to a star-galaxy image separation problem.

:func:`~MuyGPyS.examples.classify.do_classify_uq` is a high-level api for
executing a two-class classification workflow with the uncertainty
quantification. It calls the maker APIs
:func:`MuyGPyS.examples.classify.make_classifier` and
:func:`MuyGPyS.examples.classify.make_multivariate_classifier` to create and
train models, and performs the inference using the functions
:func:`~MuyGPyS.examples.classify.classify_two_class_uq`,
:func:`~MuyGPyS.examples.classify.make_masks`, and
:func:`~MuyGPyS.examples.classify.train_two_class_interval`.
:func:`~MuyGPyS.examples.classify.do_uq` takes the true labels of the test data
and the `surrgoate_prediction` and `masks` outputs to report the statistics of
the confidence intervals associated with each supplied objective function.
"""

import numpy as np

from time import perf_counter
from typing import Callable, Dict, List, Tuple, Union

from MuyGPyS.examples.classify import (
    make_classifier,
)
from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import get_balanced_batch
from MuyGPyS._src.mpi_utils import (
    _is_mpi_mode,
    _consistent_chunk_tensor,
    _consistent_reduce_scalar,
)


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


def do_classify_uq(
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    opt_batch_count: int = 200,
    uq_batch_count: int = 500,
    loss_method: str = "log",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    uq_objectives: Union[
        List[Callable], Tuple[Callable, ...]
    ] = example_lambdas,
    k_kwargs: Dict = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[MuyGPS, NN_Wrapper, np.ndarray, np.ndarray]:
    """
    Convenience function for initializing a model and performing two-class
    surrogate classification, while tuning uncertainty quantification.

    Performs the classification workflow with uncertainty quantification tuning
    as described in [muyskens2021star]_.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
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
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> accuracy, uq = do_uq(surrogate_predictions, test['output'], masks)
        >>> print(f"obtained accuracy {accuracy}")
        obtained accuracy: 0.973...
        >>> print(f"obtained mask uq : \\n{uq}")
        obtained mask uq :
        [[8.21000000e+02 8.53836784e-01 9.87144569e-01]
        [8.59000000e+02 8.55646100e-01 9.87528717e-01]
        [1.03500000e+03 8.66666667e-01 9.88845510e-01]
        [1.03500000e+03 8.66666667e-01 9.88845510e-01]
        [5.80000000e+01 6.72413793e-01 9.77972239e-01]]

    Args:
        test_features:
            A matrix of shape `(test_count, feature_count)` whose rows consist
            of observation vectors of the test data.
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the train data.
        train_labels:
            A matrix of shape `(train_count, response_count)` whose rows consist
            of label vectors for the training data.
        nn_count:
            The number of nearest neighbors to employ.
        opt_batch_count:
            The batch size for hyperparameter optimization.
        uq_batch_count:
            The batch size for uncertainty quantification calibration.
        loss_method:
            The loss method to use in hyperparameter optimization. Ignored if
            all of the parameters specified by `k_kwargs` are fixed. Currently
            supports only `"log"` (also known as `"cross_entropy"`) and `"mse"`
            for classification.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` and `"scipy"`.
        uq_objectives : list(Callable)
            List of `objective_count`functions taking four arguments: bit masks
            `alpha` and `beta` - the type 1 and type 2 error counts at each grid
            location, respectively - and the numbers of correctly and
            incorrectly classified training examples. Used to tune the scale
            parameter :math:`\\sigma^2` for setting confidence intervals. See
            `MuyGPyS.examples.classify.example_lambdas` for examples.
        k_kwargs:
            Parameters for the kernel, possibly including kernel type, distance
            metric, epsilon and sigma hyperparameter specifications, and
            specifications for kernel hyperparameters. If all of the
            hyperparameters are fixed or are not given optimization bounds, no
            optimization will occur.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        verbose:
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    surrogate_predictions:
        A matrix of shape `(test_count, response_count)` whose rows indicate
        the surrogate predictions of the model. The predicted classes are given
        by the indices of the largest elements of each row.
    masks:
        A matrix of shape `(objective_count, test_count)` whose rows consist of
        index masks into the training set. Each `True` index includes 0.0 within
        the associated prediction's confidence interval.
    """
    muygps, nbrs_lookup = make_classifier(
        train_features,
        train_labels,
        nn_count=nn_count,
        batch_count=opt_batch_count,
        loss_method=loss_method,
        obj_method=obj_method,
        opt_method=opt_method,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        opt_kwargs=opt_kwargs,
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
        print("prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
    return muygps, nbrs_lookup, surrogate_predictions, masks


def make_masks(
    predictions: np.ndarray,
    cutoffs: np.ndarray,
    variances: np.ndarray,
    mid_value: float,
) -> np.ndarray:
    """
    Compute boolean masks over all of the test data indicating which test
    indices are considered ambiguous

    Args:
        predictions:
            A matrix of shape `(test_count, class_count)` whose rows consist of
            the surrogate predictions.
        cutoffs:
            A vector of shape `(objective_count,)` indicating the confidence
            interval scale parameter :math:`\\sigma^2` that minimizes each of
            the considered objective function.
        variances:
            A vector of shape `(test_count,)` indicating the diagonal
            posterior variance of each test item.
        mid_value:
            The discriminating value determining absolute uncertainty. Usually
            `0.0` or `0.5`.

    Returns:
        A matrix of shape `(objective_count, test_count)` whose rows consist of
        index masks into the training set. Each `True` index includes
        `mid_value` within the associated prediction's confidence interval.
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


def do_uq(
    surrogate_predictions: np.ndarray,
    test_labels: np.ndarray,
    masks: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Convenience function performing uncertainty quantification given predicted
    labels and ground truth for a given set of confidence interval scales.

    Args:
        predictions:
            A matrix of shape `(test_count, class_count)` whose rows consist of
            the surrogate predictions.
        test_labels:
            A matrix of shape `(test_count, class_count)` listing the true
            one-hot encodings of each test observation's class.
        masks:
            A matrix of shape `(objective_count, test_count)` whose rows consist
            of index masks into the training set. Each `True` index includes
            `0.0` within the associated prediction's confidence interval.

    Returns
    -------
    accuracy:
        The accuracy over all of the test data.
    uq:
        A matrix of shape `(objective_count, 3)` listing the uncertainty
        quantification associated with each input mask (i.e. each objective
        function). The first column is the total number of ambiguous samples,
        i.e. those whose confidence interval contains the `mid_value`, usually
        `0.0`. The second column is the accuracy of the ambiguous samples. The
        third column is the accuracy of the unambiguous samples.
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


def classify_two_class_uq(
    surrogate: Union[MuyGPS, MMuyGPS],
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_nbrs_lookup: NN_Wrapper,
    train_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simultaneously predicts the surrogate means and variances for each test item
    under the assumption of binary classification.

    Args:
        surrogate:
            Surrogate regressor.
        test_features:
            Test observations of shape `(test_count, feature_count)`.
        train_features:
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
    test_count, _ = test_features.shape
    # train_count, _ = train_features.shape

    time_start = perf_counter()
    test_feature = _consistent_chunk_tensor(test_features)
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test_features)
    time_nn = perf_counter()

    nn_labels = train_labels[test_nn_indices, :]

    means = np.zeros((nn_labels.shape[0], 2))
    variances = np.zeros(nn_labels.shape[0])
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
            test_features,
            train_features,
            train_labels,
            variance_mode="diagonal",
            apply_sigma_sq=False,
            indices_by_rank=_is_mpi_mode(),
        )

    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }

    return means, variances, timing


def train_two_class_interval(
    surrogate: MuyGPS,
    batch_indices: np.ndarray,
    batch_nn_indices: np.ndarray,
    train_features: np.ndarray,
    train_responses: np.ndarray,
    train_labels: np.ndarray,
    objective_fns: Union[List[Callable], Tuple[Callable, ...]],
) -> np.ndarray:
    """
    For 2-class classification problems, get estimate of the confidence
    interval scaling parameter.

    Args:
        surrogate:
            Surrogate regressor.
        batch_indices:
            Batch observation indices of shape `(batch_count)`.
        batch_nn_indices:
            Indices of the nearest neighbors of shape `(batch_count, nn_count)`.
        train:
            The full training data matrix of shape
            `(train_count, feature_count)`.
        train_responses:
            One-hot encoding of class labels for all training data of shape
            `(train_count, class_count)`.
        train_labels:
            List of class labels for all training data of shape
            `(train_count,)`.
        objective_fns:
            A collection of `objective_count` functions taking the four
            arguments bit masks alpha and beta - the type 1 and type 2 error
            counts at each grid location, respectively - and the numbers of
            correctly and incorrectly classified training examples. Each
            objective function effervesces a cutoff value to calibrate UQ
            for class decision-making.

    Returns:
        A vector of shape `(objective_count)` indicating the confidence interval
        scale parameter that minimizes each considered objective function.
    """
    targets = train_labels[batch_indices]
    targets = _consistent_chunk_tensor(targets)
    batch_indices = _consistent_chunk_tensor(batch_indices)
    batch_nn_indices = _consistent_chunk_tensor(batch_nn_indices)

    mean, variance = surrogate.regress_from_indices(
        batch_indices,
        batch_nn_indices,
        train_features,
        train_features,
        train_responses,
        variance_mode="diagonal",
        apply_sigma_sq=False,
        indices_by_rank=_is_mpi_mode(),
    )
    predicted_labels = 2 * np.argmax(mean, axis=1) - 1

    correct_mask = predicted_labels == targets
    incorrect_mask = np.invert(correct_mask)

    # NOTE[bwp]: might want to make this range configurable by the user as well.
    cutv = np.linspace(0.01, 20, 1999)
    _alpha = np.zeros((len(cutv)))
    _beta = np.zeros((len(cutv)))
    for i in range(len(cutv)):
        _alpha[i] = 1 - np.mean(
            np.logical_and(
                (
                    mean[incorrect_mask, 1]
                    - cutv[i] * np.sqrt(variance[incorrect_mask])
                )
                < 0.0,
                (
                    mean[incorrect_mask, 1]
                    + cutv[i] * np.sqrt(variance[incorrect_mask])
                )
                > 0.0,
            )
        )
        _alpha[i] = _consistent_reduce_scalar(_alpha[i])
        _beta[i] = np.mean(
            np.logical_and(
                (
                    mean[correct_mask, 1]
                    - cutv[i] * np.sqrt(variance[correct_mask])
                )
                < 0.0,
                (
                    mean[correct_mask, 1]
                    + cutv[i] * np.sqrt(variance[correct_mask])
                )
                > 0.0,
            )
        )
        _beta[i] = _consistent_reduce_scalar(_beta[i])

    correct_count = np.sum(correct_mask)
    correct_count = _consistent_reduce_scalar(correct_count)
    incorrect_count = np.sum(incorrect_mask)
    incorrect_count = _consistent_reduce_scalar(incorrect_count)
    cutoffs = np.array(
        [
            cutv[obj_f(_alpha, _beta, correct_count, incorrect_count)]
            for obj_f in objective_fns
        ]
    )
    return cutoffs
