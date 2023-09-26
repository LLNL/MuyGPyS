# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
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
from typing import Dict, List, Tuple, Union

from MuyGPyS.examples.from_indices import posterior_mean_from_indices
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS.gp.tensors import make_train_tensors
from MuyGPyS.neighbors import NN_Wrapper
from MuyGPyS.optimize import Bayes_optimize, OptimizeFn
from MuyGPyS.optimize.batch import get_balanced_batch
from MuyGPyS.optimize.loss import LossFn, cross_entropy_fn


def make_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    nn_count: int = 30,
    batch_count: int = 200,
    loss_fn: LossFn = cross_entropy_fn,
    opt_fn: OptimizeFn = Bayes_optimize,
    k_kwargs: Dict = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[MuyGPS, NN_Wrapper]:
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
        >>> from MuyGPyS.examples.regress import make_classifier
        >>> from MuyGPyS.gp.deformation import F2, Isotropy
        >>> from MuyGPyS.gp.hyperparameter import Parameter
        >>> from MuyGPyS.gp.kernels import RBF
        >>> from MuyGPyS.gp.noise import HomoscedasticNoise
        >>> from MuyGPyS.optimize import Bayes_optimize
        >>> from MuyGPyS.examples.classify import make_classifier
        >>> train_features, train_responses = make_train()  # stand-in function
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_kwargs = {
        ...     "kernel": RBF(
        ...         deformation=Isotropy(
        ...             metric=F2,
        ...             length_scale=Parameter(1.0, (1e-2, 1e2))
        ...         )
        ...     ),
        ...     "noise": HomoscedasticNoise(1e-5),
        ... }
        >>> muygps, nbrs_lookup = make_classifier(
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_fn=cross_entropy_fn,
        ...         opt_fn=Bayes_optimize,
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
        loss_fn:
            The loss functor to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
        opt_fn:
            The optimization functor to use in hyperparameter optimization.
            Ignored if all of the parameters specified by argument `k_kwargs`
            are fixed.
        k_kwargs:
            Parameters for the kernel, possibly including kernel type,
            deformation function, noise and scale hyperparameter specifications,
            and specifications for kernel hyperparameters. See
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
        verbose : Boolean
            If `True`, print summary statistics.

    Returns
    -------
    muygps:
        A (possibly trained) MuyGPs object.
    nbrs_lookup:
        A data structure supporting nearest neighbor queries into
        `train_features`.
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
            crosswise_diffs,
            pairwise_diffs,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            batch_indices,
            batch_nn_indices,
            train_features,
            train_labels,
        )
        time_tensor = perf_counter()

        # maybe do something with these estimates?
        muygps = opt_fn(
            muygps,
            batch_targets,
            batch_nn_targets,
            crosswise_diffs,
            pairwise_diffs,
            loss_fn=loss_fn,
            verbose=verbose,
            **opt_kwargs,
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
    loss_fn: LossFn = cross_entropy_fn,
    opt_fn: OptimizeFn = Bayes_optimize,
    k_args: Union[List[Dict], Tuple[Dict, ...]] = list(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[MMuyGPS, NN_Wrapper]:
    """
    Convenience function for creating MuyGPyS functor and neighbor lookup data
    structure.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
        >>> from MuyGPyS.examples.classify import make_multivariate_classifier
        >>> from MuyGPyS.gp.deformation import F2, Isotropy
        >>> from MuyGPyS.gp.hyperparameter import Parameter
        >>> from MuyGPyS.gp.kernels import RBF
        >>> from MuyGPyS.gp.noise import HomoscedasticNoise
        >>> from MuyGPyS.optimize import Bayes_optimize
        >>> train_features, train_responses = make_train()  # stand-in function
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_args = [
        ...     {
        ...         "kernel": RBF(
        ...             deformation=Isotropy(
        ...                 metric=F2,
        ...                 length_scale=Parameter(0.5, (0.01, 1)),
        ...             ),
        ...         )
        ...         "noise": HomoscedasticNoise(1e-5),
        ...     },
        ...     {
        ...         "kernel": RBF(
        ...             deformation=Isotropy(
        ...                 metric=F2,
        ...                 length_scale=Parameter(0.5, (0.01, 1)),
        ...             ),
        ...         )
        ...         "noise": HomoscedasticNoise(1e-5),
        ...     },
        ... ]
        >>> mmuygps, nbrs_lookup = make_multivariate_classifier(
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_fn=cross_entropy_fn,
        ...         opt_fn=Bayes_optimize,
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
        loss_fn:
            The loss functor to use in hyperparameter optimization. Ignored if
            all of the parameters specified by argument `k_kwargs` are fixed.
        opt_fn:
            The optimization functor to use in hyperparameter optimization.
            Ignored if all of the parameters specified by argument `k_kwargs`
            are fixed.
        k_args:
            A list of `response_count` dicts containing kernel initialization
            keyword arguments. Each dict specifies parameters for the kernel,
            possibly including noise and scale hyperparameter specifications
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

    mmuygps = MMuyGPS(*k_args)
    if mmuygps.fixed() is False:
        # collect batch
        batch_indices, batch_nn_indices = get_balanced_batch(
            nbrs_lookup,
            np.argmax(train_labels, axis=1),
            batch_count,
        )
        time_batch = perf_counter()

        (
            crosswise_diffs,
            pairwise_diffs,
            batch_targets,
            batch_nn_targets,
        ) = make_train_tensors(
            batch_indices,
            batch_nn_indices,
            train_features,
            train_labels,
        )
        time_tensor = perf_counter()

        # maybe do something with these estimates?
        for i, muygps in enumerate(mmuygps.models):
            if muygps.fixed() is False:
                mmuygps.models[i] = opt_fn(
                    muygps,
                    batch_targets[:, i].reshape(batch_targets.shape[0], 1),
                    batch_nn_targets[:, :, i].reshape(
                        batch_nn_targets.shape[0], nn_count, 1
                    ),
                    crosswise_diffs,
                    pairwise_diffs,
                    loss_fn=loss_fn,
                    verbose=verbose,
                    **opt_kwargs,
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
    loss_fn: LossFn = cross_entropy_fn,
    opt_fn: OptimizeFn = Bayes_optimize,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper]:
    if isinstance(k_kwargs, list):
        return make_multivariate_classifier(
            train_features,
            train_labels,
            nn_count=nn_count,
            batch_count=batch_count,
            loss_fn=loss_fn,
            opt_fn=opt_fn,
            k_args=k_kwargs,
            nn_kwargs=nn_kwargs,
            opt_kwargs=opt_kwargs,
            verbose=verbose,
        )
    else:
        if isinstance(k_kwargs, dict):
            return make_classifier(
                train_features,
                train_labels,
                nn_count=nn_count,
                batch_count=batch_count,
                loss_fn=loss_fn,
                opt_fn=opt_fn,
                k_kwargs=k_kwargs,
                nn_kwargs=nn_kwargs,
                opt_kwargs=opt_kwargs,
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
    loss_fn: LossFn = cross_entropy_fn,
    opt_fn: OptimizeFn = Bayes_optimize,
    k_kwargs: Union[Dict, Union[List[Dict], Tuple[Dict, ...]]] = dict(),
    nn_kwargs: Dict = dict(),
    opt_kwargs: Dict = dict(),
    verbose: bool = False,
) -> Tuple[Union[MuyGPS, MMuyGPS], NN_Wrapper, np.ndarray]:
    """
    Convenience function for initializing a model and performing surrogate
    classification.

    Expected parameters include keyword argument dicts specifying kernel
    parameters and nearest neighbor parameters. See the docstrings of the
    appropriate functions for specifics.

    Example:
        >>> import numpy as np
        >>> from MuyGPyS.examples.classify import do_classify
        >>> from MuyGPyS.gp.deformation import F2, Isotropy
        >>> from MuyGPyS.gp.hyperparameter import Parameter
        >>> from MuyGPyS.gp.kernels import RBF
        >>> from MuyGPyS.gp.noise import HomoscedasticNoise
        >>> from MuyGPyS.optimize import Bayes_optimize
        >>> train_features, train_responses = make_train()  # stand-in function
        >>> test_features, test_responses = make_test()  # stand-in function
        >>> nn_kwargs = {"nn_method": "exact", "algorithm": "ball_tree"}
        >>> k_kwargs = {
        ...     "kernel": RBF(
        ...         deformation=Isotropy(
        ...             metric=F2,
        ...             length_scale=Parameter(0.5, (0.01, 1)),
        ...         ),
        ...     )
        ...     "noise": HomoscedasticNoise(1e-5),
        ... }
        >>> muygps, nbrs_lookup, surrogate_predictions = do_classify(
        ...         test_features,
        ...         train_features,
        ...         train_responses,
        ...         nn_count=30,
        ...         batch_count=200,
        ...         loss_fn=cross_entropy_fn,
        ...         opt_fn=Bayes_optimize,
        ...         k_kwargs=k_kwargs,
        ...         nn_kwargs=nn_kwargs,
        ...         verbose=False,
        ... )
        >>> predicted_labels = np.argmax(surrogate_predictions, axis=1)
        >>> true_labels = np.argmax(test_features, axis=1)
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
        nn_count:
            The number of nearest neighbors to employ.
        batch_count:
            The batch size for hyperparameter optimization.
        loss_fn:
            The loss functor to use in hyperparameter optimization. Ignored if
            all of the parameters specified by `k_kwargs` are fixed.
        opt_fn:
            The optimization functor to use in hyperparameter optimization.
            Ignored if all of the parameters specified by argument `k_kwargs`
            are fixed.
        k_kwargs:
            Parameters for the kernel, possibly including kernel type,
            deformation function, noise and scale hyperparameter specifications,
            and specifications for kernel hyperparameters. If all of the
            hyperparameters are fixed or are not given optimization bounds, no
            optimization will occur. If `"k_kwargs"` is a list of such dicts,
            will create a multivariate classifier model.
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
    """
    classifier, nbrs_lookup = _decide_and_make_classifier(
        train_features,
        train_labels,
        nn_count=nn_count,
        batch_count=batch_count,
        loss_fn=loss_fn,
        opt_fn=opt_fn,
        k_kwargs=k_kwargs,
        nn_kwargs=nn_kwargs,
        opt_kwargs=opt_kwargs,
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
        print("prediction time breakdown:")
        for k in pred_timing:
            print(f"\t{k} time:{pred_timing[k]}s")
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
    _, class_count = train_labels.shape

    # detect one hot encoding, e.g. {0,1}, {-0.1, 0.9}, {-1,1}, ...
    one_hot_false = float(np.min(train_labels[0, :]))

    time_start = perf_counter()
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
        nonconstant_indices = np.where(nonconstant_mask)[0]
        nonconstant_nn_indices = test_nn_indices[nonconstant_mask, :]
        predictions[nonconstant_mask] = posterior_mean_from_indices(
            surrogate,
            nonconstant_indices,
            nonconstant_nn_indices,
            test_features,
            train_features,
            train_labels,
        )
    time_pred = perf_counter()

    timing = {
        "nn": time_nn - time_start,
        "agree": time_agree - time_nn,
        "pred": time_pred - time_agree,
    }
    return predictions, timing
