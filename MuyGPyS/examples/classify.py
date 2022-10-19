# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Resources and high-level API for a simple classification workflow.

:func:`~MuyGPyS.examples.classify.make_classifier` is a high-level API for
creating and training :class:`MuyGPyS.gp.muygps.MuyGPS` objects for
classification. :func:`~MuyGPyS.examples.classify.make_multivariate_classifier`
is a high-level API for creating and training
:class:`MuyGPyS.gp.muygps.MultivariateMuyGPS` objects for classification.

:func:`~MuyGPyS.examples.classify.do_classify` is a high-level api for executing
a simple, generic classification workflow given data. It calls the maker APIs
above and :func:`~MuyGPyS.examples.classify.classify_any`.
"""

import numpy as np

from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from MuyGPyS.gp.distance import make_train_tensors
from MuyGPyS.optimize.chassis import optimize_from_tensors

from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize.batch import get_balanced_batch
from MuyGPyS._src.mpi_utils import (
    _is_mpi_mode,
    _consistent_chunk_tensor,
)


def make_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "log",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    k_kwargs: Dict = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[MuyGPS, NN_Wrapper], Tuple[MuyGPS, NN_Wrapper, np.ndarray, np.ndarray]
]:
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
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> # Can alternately return distance tensors for reuse
        >>> muygps, nbrs_lookup = make_classifier(
        ...         train['input'],
        ...         train['output'],
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="log",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         return_distances=True,
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
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` and `"scipy"`.
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
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
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        return_distances:
            If `True` and any training occurs, returns a
            `(batch_count, nn_count)` matrix containing the crosswise distances
            between the batch's elements and their nearest neighbor sets and a
            `(batch_count, nn_count, nn_count)` matrix containing the pairwise
            distances between the batch's nearest neighbor sets.
        verbose : Boolean
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    crosswise_dists:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
        Only returned if `return_distances is True`.
    pairwise_dists:
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements. Only returned if
        `return_distances is True`.
    """
    time_start = perf_counter()

    nbrs_lookup = NN_Wrapper(
        train_features,
        nn_count,
        **nn_kwargs,
    )
    time_nn = perf_counter()

    muygps = MuyGPS(**k_kwargs)
    if muygps.fixed() is False:
        # collect batch
        batch_indices, batch_nn_indices = get_balanced_batch(
            nbrs_lookup,
            np.argmax(train_labels, axis=1),
            batch_count,
        )
        time_batch = perf_counter()

        (
            crosswise_dists,
            pairwise_dists,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            muygps.kernel.metric,
            batch_indices,
            batch_nn_indices,
            train_features,
            train_labels,
        )
        time_tensor = perf_counter()

        # maybe do something with these estimates?
        muygps = optimize_from_tensors(
            muygps,
            batch_targets,
            batch_nn_targets,
            crosswise_dists,
            pairwise_dists,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            sigma_method=None,
            verbose=verbose,
            **opt_kwargs,
        )
        time_opt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")

        if return_distances is True:
            return muygps, nbrs_lookup, crosswise_dists, pairwise_dists

    return muygps, nbrs_lookup


def make_multivariate_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "mse",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    kern: str = "matern",
    k_args: Union[List[Dict], Tuple[Dict, ...]] = list(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[MMuyGPS, NN_Wrapper],
    Tuple[MMuyGPS, NN_Wrapper, np.ndarray, np.ndarray],
]:
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
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         kern="rbf",
        ...         k_args=k_args,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> # Can alternately return distance tensors for reuse
        >>> mmuygps, nbrs_lookup = make_multivariate_classifier(
        ...         train['input'],
        ...         train['output'],
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_method="mse",
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         kern="rbf",
        ...         k_args=k_args,
        ...         nn_kwargs=nn_kwargs,
        ...         return_distances=return_distances,
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
        obj_method:
            Indicates the objective function to be minimized. Currently
            restricted to `"loo_crossval"`.
        opt_method:
            Indicates the optimization method to be used. Currently restricted
            to `"bayesian"` and `"scipy"`.
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
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        return_distances:
            If `True` and any training occurs, returns a
            `(batch_count, nn_count)` matrix containing the crosswise distances
            between the batch's elements and their nearest neighbor sets and a
            `(batch_count, nn_count, nn_count)` matrix containing the pairwise
            distances between the batch's nearest neighbor sets.
        verbose:
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
    crosswise_dists:
        A matrix of shape `(batch_count, nn_count)` whose rows list the distance
        of the corresponding batch element to each of its nearest neighbors.
        Only returned if `return_distances is True`.
    pairwise_dists:
        A tensor of shape `(batch_count, nn_count, nn_count,)` whose latter two
        dimensions contain square matrices containing the pairwise distances
        between the nearest neighbors of the batch elements. Only returned if
        `return_distances is True`.
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
    if mmuygps.fixed() is False:
        # collect batch
        batch_indices, batch_nn_indices = get_balanced_batch(
            nbrs_lookup,
            np.argmax(train_labels, axis=1),
            batch_count,
        )
        time_batch = perf_counter()

        (
            crosswise_dists,
            pairwise_dists,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            mmuygps.metric,
            batch_indices,
            batch_nn_indices,
            train_features,
            train_labels,
        )
        time_tensor = perf_counter()

        # maybe do something with these estimates?
        for i, muygps in enumerate(mmuygps.models):
            if muygps.fixed() is False:
                mmuygps.models[i] = optimize_from_tensors(
                    muygps,
                    batch_targets[:, i].reshape(batch_targets.shape[0], 1),
                    batch_nn_targets[:, :, i].reshape(
                        batch_nn_targets.shape[0], nn_count, 1
                    ),
                    crosswise_dists,
                    pairwise_dists,
                    loss_method=loss_method,
                    obj_method=obj_method,
                    opt_method=opt_method,
                    sigma_method=None,
                    verbose=verbose,
                    **opt_kwargs,
                )
        time_opt = perf_counter()

        if verbose is True:
            print(f"NN lookup creation time: {time_nn - time_start}s")
            print(f"batch sampling time: {time_batch - time_nn}s")
            print(f"tensor creation time: {time_tensor - time_batch}s")
            print(f"hyper opt time: {time_opt - time_tensor}s")

        if return_distances is True:
            return mmuygps, nbrs_lookup, crosswise_dists, pairwise_dists

    return mmuygps, nbrs_lookup


def _decide_and_make_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "log",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper],
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray, np.ndarray],
]:
    if kern is not None and isinstance(k_kwargs, list):
        return make_multivariate_classifier(
            train_features,
            train_labels,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_method=loss_method,
            obj_method=obj_method,
            opt_method=opt_method,
            kern=kern,
            k_args=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            return_distances=return_distances,
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
                obj_method=obj_method,
                opt_method=opt_method,
                k_kwargs=k_kwargs,
                nn_kwargs=nn_kwargs,
                opt_kwargs=opt_kwargs,
                return_distances=return_distances,
                verbose=verbose,
            )
        else:
            raise ValueError("Expected k_kwargs to be a dict!")


def _unpack(first, *rest):
    return first, rest


def do_classify(
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_method: str = "log",
    obj_method: str = "loo_crossval",
    opt_method: str = "bayes",
    kern: Optional[str] = None,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    return_distances: bool = False,
    verbose: bool = False,
) -> Union[
    Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray],
    Tuple[
        Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray, np.ndarray, np.ndarray
    ],
]:
    """
    Convenience function for initializing a model and performing surrogate
    classification.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
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
        ...         obj_method="loo_crossval",
        ...         opt_method="bayes",
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> # Can alternately return distance tensors for reuse
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
        ...         return_distances=return_distances,
        ...         verbose=False,
        ... )
        >>> predicted_labels = np.argmax(surrogate_predictions, axis=1)
        >>> true_labels = np.argmax(test['output'], axis=1)
        >>> acc = np.mean(predicted_labels == true_labels)
        >>> print(f"obtained accuracy {acc}")
        obtained accuracy: 0.973...

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
        train_features:
            A matrix of shape `(train_count, feature_count)` whose rows consist
            of observation vectors of the testing data.
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The batch size for hyperparameter optimization.
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
        kern:
            The kernel function to be used. Only relevant for multivariate case
            where `k_kwargs` is a list of hyperparameter dicts. Currently
            supports only `"matern"` and `"rbf"`.
        k_kwargs:
            Parameters for the kernel, possibly including kernel type, distance
            metric, epsilon and sigma hyperparameter specifications, and
            specifications for kernel hyperparameters. If all of the
            hyperparameters are fixed or are not given optimization bounds, no
            optimization will occur. If `"kern"` is specified and `"k_kwargs"`
            is a list of such dicts, will create a multivariate regressor model.
        nn_kwargs:
            Parameters for the nearest neighbors wrapper. See
            :class:`MuyGPyS.neighbors.NN_Wrapper` for the supported methods and
            their parameters.
        opt_kwargs:
            Parameters for the wrapped optimizer. See the docs of the
            corresponding library for supported parameters.
        return_distances:
            If `True` and any training occurs, returns a
            `(batch_count, nn_count)` matrix containing the crosswise distances
            between the batch's elements and their nearest neighbor sets and a
            `(batch_count, nn_count, nn_count)` matrix containing the pairwise
            distances between the batch's nearest neighbor sets.
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
    """
    classifier_args = _decide_and_make_classifier(
        train_features,
        train_labels,
        nn_count=nn_count,
        batch_count=batch_count,
        loss_method=loss_method,
        obj_method=obj_method,
        opt_method=opt_method,
        kern=kern,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        opt_kwargs=opt_kwargs,
        return_distances=return_distances,
        verbose=verbose,
    )
    classifier, classifier_args_less1 = _unpack(*classifier_args)
    nbrs_lookup, classifier_args_less2 = _unpack(*classifier_args_less1)
    if len(classifier_args_less2) > 0:
        crosswise_dists, pairwise_dists = classifier_args_less2

    surrogate_predictions, pred_timing = classify_any(
        classifier,
        test_features,
        train_features,
        nbrs_lookup,
        train_labels,
    )
    if verbose is True:
        print("prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
    if len(classifier_args_less2) > 0:
        return (
            classifier,
            nbrs_lookup,
            surrogate_predictions,
            crosswise_dists,
            pairwise_dists,
        )
    else:
        return classifier, nbrs_lookup, surrogate_predictions


def classify_any(
    surrogate: Union[MuyGPS, MMuyGPS],
    test_features: np.ndarray,
    train_features: np.ndarray,
    train_nbrs_lookup: NN_Wrapper,
    train_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulatneously predicts the surrogate regression means for each test item.

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
    predictions:
        The surrogate predictions of shape `(test_count, class_count)` for each
        test observation.
    timing:
        Timing for the subroutines of this function.
    """
    test_count = test_features.shape[0]
    class_count = train_labels.shape[1]

    # detect one hot encoding, e.g. {0,1}, {-0.1, 0.9}, {-1,1}, ...
    one_hot_false = float(np.min(train_labels[0, :]))

    time_start = perf_counter()
    test_features = _consistent_chunk_tensor(test_features)
    test_nn_indices, _ = train_nbrs_lookup.get_nns(test_features)
    time_nn = perf_counter()

    nn_labels = train_labels[test_nn_indices, :]

    predictions = np.full((nn_labels.shape[0], class_count), one_hot_false)
    nonconstant_mask = np.max(nn_labels[:, :, 0], axis=-1) != np.min(
        nn_labels[:, :, 0], axis=-1
    )

    predictions[np.invert(nonconstant_mask), :] = nn_labels[
        np.invert(nonconstant_mask), 0, :
    ]
    time_agree = perf_counter()

    if np.sum(nonconstant_mask) > 0:
        nonconstant_indices = np.where(nonconstant_mask == True)[0]
        nonconstant_nn_indices = test_nn_indices[nonconstant_mask, :]
        predictions[nonconstant_mask] = surrogate.regress_from_indices(
            nonconstant_indices,
            nonconstant_nn_indices,
            test_features,
            train_features,
            train_labels,
            apply_sigma_sq=False,
            indices_by_rank=_is_mpi_mode(),
        )
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }
    return predictions, timing
