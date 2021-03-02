#!/usr/bin/env python
# encoding: utf-8
"""
@file classify.py

Created by priest2 on 2020-10-27

End-to-end application of LKGP.
"""

import numpy as np

from muyscans.embed import apply_embedding
from muyscans.data.utils import normalize
from muyscans.optimize.batch import get_classify_batch, sample_balanced_batch
from muyscans.optimize.objective import (
    loo_crossval,
    get_loss_func,
)
from muyscans.predict import (
    classify_two_class_uq,
    classify_any,
)
from muyscans.neighbors import NN_Wrapper
from muyscans.uq import train_two_class_interval
from muyscans.gp.lkgp import LKGP

from scipy import optimize as opt
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
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


def do_classify(
    train,
    test,
    nn_count=30,
    embed_dim=30,
    opt_batch_size=200,
    uq_batch_size=200,
    kern="matern",
    embed_method="pca",
    loss_method="log",
    hyper_dict=None,
    uq_objectives=None,
    do_normalize=False,
    exact=True,
    verbose=False,
):
    """
    Performs classification using LKGP.

    Parameters
    ----------
    train : dict
        A dict with keys "input", "output", and "lookup". "input" maps to a
        matrix of row observation vectors, e.g. flattened images. "output" maps
        to a matrix listing the one-hot encodings of each observation's class.
        "lookup" is effectively the argmax over this matrix's columns.
    test : dict
        A dict with keys "input", "output", and "lookup". "input" maps to a
        matrix of row observation vectors, e.g. flattened images. "output" maps
        to a matrix listing the one-hot encodings of each observation's class.
        "lookup" is effectively the argmax over this matrix's columns.
    nn_count : int
        The number of nearest neighbors to employ.
    embed_dim : int
        The PCA dimension onto which data will be embedded.
    opt_batch_size : int
        The batch size for hyperparameter optimization. Unused if ``nu'' is
        None.
    uq_batch_size : int
        The batch size for uncertainty quantification optimization. Unused if
        ``uq_objectives'' is None.
    kern : str
        The kernel to use. Supports ``matern'', ``rbf'', and ``nngp''.
    embed_method : str
        The embedding method to use.
        NOTE[bwp]: Currently supports only ``pca'' and None.
    loss_method : str
        The loss method to use in hyperparameter optimization. Ignored if
        ``hyper_dict'' fully specifies the kernel in question.
        NOTE[bwp]: Currently supports only ``cross_entropy'' for classification.
    hyper_dict : dict or None
        If specified, use the given parameters for the kernel. If None, perform
        hyperparamter optimization.
    uq_objectives : list(Callable)
        List of functions taking four arguments: bit masks alpha and beta - the
        type 1 and type 2 error counts at each grid location, respectively - and
        the numbers of correctly and incorrectly classified training examples.
        Used to determine scale parameter for confidence intervals. See
        `muyscans.examples.classify.example_lambdas` for examples. If None, do
        not perform uncertainty quantifification.
        NOTE[bwp]: Supports only 2-class classification at the moment.
    do_normalize : Boolean
        Flag indicating whether to normalize. Currently redundant, but might
        want to keep this for now.
    exact : Boolean
        If true, use sklearn.neighbors.NearestNeighbors for exact KNN. Else, use
        hnswlib.Index for approximate KNN.
    verbose : Boolean
        If ``True'', print summary statistics.

    Returns
    -------
    predictions : numpy.ndarray, type = int, shape = ``(test_count,)''
        The predicted labels associated with each test observation.
    masks : numpy.ndarray, type = Boolean, shape = ``(obj_count, test_count)''
        A list of index masks into the test set that indicates which test
        elements are ambiguous based upon the confidence intervals derived from
        each objective function.
    """
    num_class = len(np.unique(train["lookup"]))
    test_count = test["lookup"].shape[0]
    train_count = train["lookup"].shape[0]
    time_start = perf_counter()

    # Perform embedding
    embedded_train, embedded_test = apply_embedding(
        train["input"], test["input"], embed_dim, embed_method, do_normalize
    )
    time_embed = perf_counter()

    # Construct NN lookup datastructure.
    train_nbrs_lookup = NN_Wrapper(
        embedded_train,
        nn_count,
        exact,
    )
    time_nn = perf_counter()
    time_batch = perf_counter()

    # Make LKGP object
    lkgp = LKGP(kern=kern)
    if hyper_dict is None:
        hyper_dict = dict()
    unset_params = lkgp.set_params(**hyper_dict)
    if "sigma_sq" in unset_params:
        unset_params.remove("sigma_sq")

    # Train hyperparameters by maximizing LOO predictions for batched
    # observations if `nu` unspecified.
    if len(unset_params) > 0:
        # collect balanced batch
        batch_indices, batch_nn_indices = get_classify_batch(
            train_nbrs_lookup,
            train["lookup"],
            opt_batch_size,
            train_count,
        )
        time_batch = perf_counter()

        # set loss function
        loss_fn = get_loss_func(loss_method)

        # collect optimization settings
        bounds = lkgp.optim_bounds(unset_params)
        x0 = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds])
        if verbose is True:
            print(f"parameters to be optimized: {unset_params}")
            print(f"bounds: {bounds}")
            print(f"sampled x0: {x0}")

        # perform optimization
        optres = opt.minimize(
            loo_crossval,
            x0,
            args=(
                loss_fn,
                lkgp,
                unset_params,
                batch_indices,
                batch_nn_indices,
                embedded_train,
                train["output"],
            ),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if verbose is True:
            print(f"optimizer results: \n{optres}")
        lkgp.set_param_array(unset_params, optres.x)
    time_hyperopt = perf_counter()

    # record timing
    timing = {
        "embed": time_embed - time_start,
        "nn": time_nn - time_embed,
        "batch": time_batch - time_nn,
        "hyperopt": time_hyperopt - time_batch,
    }

    # Prediction on test data.
    if uq_objectives is not None:
        # Posterior inference on two class problem.
        predictions, variances, pred_timing = classify_two_class_uq(
            lkgp,
            embedded_test,
            embedded_train,
            train_nbrs_lookup,
            train["output"],
            nn_count,
        )
        min_label = np.min(train["output"][0, :])
        max_label = np.max(train["output"][0, :])
        if min_label == 0.0 and max_label == 1.0:
            predicted_labels = np.argmax(predictions, axis=1)
        elif min_label == -1.0 and max_label == 1.0:
            predicted_labels = 2 * np.argmax(predictions, axis=1) - 1
        else:
            raise ("Unhandled label encoding min ({min_label}, {max_label})!")
        mid_value = (min_label + max_label) / 2
        time_pred = perf_counter()

        batch_indices, batch_nn_indices = get_classify_batch(
            train_nbrs_lookup,
            train["lookup"],
            uq_batch_size,
            train_count,
        )
        time_uq_batch = perf_counter()

        # Training of confidence interval scaling using different objectives.
        # NOTE[bwp]: Currently hard-coded, but will make objectives
        # user-specifiable in the near future.
        cutoffs = train_two_class_interval(
            lkgp,
            batch_indices,
            batch_nn_indices,
            embedded_train,
            train["output"],
            train["lookup"],
            uq_objectives,
        )

        # Compute index masks indicating the predictions that include `0` in the
        # confidence interval for each of the training objectives.
        masks = np.array(
            [
                np.logical_and(
                    predictions[:, 1] - cut * variances < mid_value,
                    predictions[:, 1] + cut * variances > mid_value,
                )
                for cut in cutoffs
            ]
        )
        time_cutoff = perf_counter()

        # Profiling printouts if requested.
        if verbose is True:
            timing["pred"] = time_pred - time_hyperopt
            timing["pred_full"] = (pred_timing,)
            timing["uq_batch"] = time_uq_batch - time_pred
            timing["cutoff"] = time_cutoff - time_uq_batch

            print(f"lkgp params : {lkgp.params}")
            print(f"cutoffs : {cutoffs}")
            print(f"timing : {timing}")
        return predictions, masks
    else:
        # Posterior mean surrogate classification with no UQ.
        predictions, pred_timing = classify_any(
            lkgp,
            embedded_test,
            embedded_train,
            train_nbrs_lookup,
            train["output"],
            nn_count,
        )
        time_pred = perf_counter()

        # Profiling printouts if requested.
        if verbose is True:
            timing["pred"] = time_pred - time_hyperopt
            timing["pred_full"] = (pred_timing,)

            print(f"lkgp params : {lkgp.params}")
            print(f"timing : {timing}")
        return predictions


def do_uq(predicted_labels, test, masks):
    """
    Convenience function performing uncertainty quantification given predicted
    labels and ground truth for a given set of confidence interval scales.

    Parameters
    ----------
    predicted_labels : np.ndarray, type = int, shape = ``(test_count)''
        The list of predicted labels, based on e.g. an invocation of
        ``do_classify''.
    test : dict
        A dict with keys "input", "output", and "lookup". "input" maps to a
        matrix of row observation vectors, e.g. flattened images. "output" maps
        to a matrix listing the one-hot encodings of each observation's class.
        "lookup" is effectively the argmax over this matrix's columns.
    masks : np.ndarray, type = Boolean, shape = ``(mask_count, test_count)''
        A number of index masks indexing into the training set. Each ``True''
        index includes 0.0 within the associated prediction's confidence
        interval.

    Returns
    -------
    accuracy : float
        The accuracy over all of the test data.
    uq : numpy.ndarray, type = float, shape = ``(mask_count, 3)''
        The uncertainty quantification associated with each input mask. The
        first column is the total number of ambiguous samples. The second column
        is the accuracy of the ambiguous samples. The third column is the
        accuracy of the unambiguous samples.
    """
    correct = predicted_labels == test["lookup"]
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
    return np.mean(correct), uq
