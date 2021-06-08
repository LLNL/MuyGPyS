# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from scipy.special import softmax
from sklearn.metrics import log_loss


def get_loss_func(loss_method):
    loss_method = loss_method.lower()
    if loss_method == "cross-entropy" or loss_method == "log":
        return cross_entropy_fn
    elif loss_method == "mse":
        return mse_fn
    else:
        raise NotImplementedError(
            f"Loss function {loss_method} is not implemented."
        )


def cross_entropy_fn(predictions, targets):
    """
    Computes the cross entropy loss the predicted versus known response.
    Transforms `predictions' to be row-stochastic, and ensures that `targets'
    contains no negative elements.

    Parameters
    ----------
    predictions : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The predicted response.
    targets : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The expected response.

    Returns
    -------
    float
        The cross-entropy loss of the prediction.
    """
    one_hot_targets = np.zeros(targets.shape)
    one_hot_targets[targets > 0.0] = 1.0

    return log_loss(
        one_hot_targets, softmax(predictions, axis=1), eps=1e-6, normalize=False
    )


def mse_fn(predictions, targets):
    """
    Computes mean squared error loss of the predicted versus known response.
    Treats multivariate outputs as interchangeable in terms of loss penalty.

    Parameters
    ----------
    predictions : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The predicted response.
    targets : numpy.ndarray(int), shape = ``(batch_count, response_count)''
        The expected response.

    Returns
    -------
    float
        The mse loss of the prediction.
    """
    batch_count = predictions.shape[0]
    response_count = predictions.shape[1]
    squared_errors = np.sum((predictions - targets) ** 2)
    return squared_errors / (batch_count * response_count)


def loo_crossval(
    x0,
    objective_fn,
    muygps,
    optim_params,
    pairwise_dists,
    crosswise_dists,
    batch_nn_targets,
    batch_targets,
):
    """
    Returns leave-one-out cross validation performance for a `MuyGPS` object.
    Predicts on all of the training data at once.

    Parameters
    ----------
    x0 : numpy.ndarray(float), shape = ``(opt_count,)''
        Current guess for hyperparameter values.
    objective_fn : callable
        The function to be optimized.
    muygps : MuyGPyS.GP.MuyGPS
        The MuyGPS object.
    optim_params : dict(MuyGPyS.gp.kernels.Hyperparameter),
                   shape = ``(opt_count,)''
        Dictionary of references of unfixed hyperparameters belonging to the
        MuyGPS object.
    pairwise_dists : numpy.ndarray(float),
                     shape = ``(batch_size, nn_count, nn_count)''
        Distance tensor whose second two dimensions give the pairwise distances
        between the nearest neighbors of each batch element.
    crosswise_dists : numpy.ndarray(float),
                     shape = ``(batch_size, nn_count)''
        Distance matrix whose rows give the distances between each batch
        element and its nearest neighbors.
    batch_nn_targets : numpy.ndarray(float),
                       shape = ``(batch_size, nn_count, response_count)''
        Tensor listing the expected response for each nearest neighbor of each
        batch element.
    batch_targets : numpy.ndarray(float),
                    shape = ``(batch_size, response_count)''
        Matrix whose rows give the expected response for each  batch element.

    Returns
    -------
    float
        The evaluation of ``objective_fn'' on the predicted versus expected
        response.
    """
    for i, key in enumerate(optim_params):
        optim_params[key]._set_val(x0[i])

    K = muygps.kernel(pairwise_dists)
    Kcross = muygps.kernel(crosswise_dists)

    predictions = muygps.regress(K, Kcross, batch_nn_targets)

    return objective_fn(predictions, batch_targets)


def old_loo_crossval(
    x0,
    objective_fn,
    muygps,
    params,
    batch_indices,
    batch_nn_indices,
    embedded_train,
    train_targets,
):
    """
    Returns leave-one-out cross validation performance for a `MuyGPS` object.
    Predicts on all of the training data at once.

    Parameters
    ----------
    x0 : float
        Hyperparameter values.
    objective_fn : callable
        The function to be used to optimize ``nu''.
    muygps : MuyGPyS.GP.MuyGPS
        Local kriging approximate MuyGPS.
    params : set
        Set of parameter names to optimize.
    batch_indices : numpy.ndarray(int), shape = ``(batch_size,)''
        Batch observation indices.
    batch_nn_indices : numpy.ndarray(int), shape = ``(n_batch, nn_count)''
        Indices of the nearest neighbors
    batch_nn_distances : numpy.ndarray(float),
                         shape = ``(batch_size, nn_count)''
        Distances from each batch observation to its nearest neighbors.
    embedded_train : numpy.ndarray(float),
                     shape = ``(train_count, embedding_dim)''
        The full embedded training data matrix.
    train_targets : numpy.ndarray(float),
                   shape = ``(train_count, response_count)''
        List of output response for all embedded data, e.g. one-hot class
        encodings for classification.

    Returns
    -------
    float
        The evaluation of ``objective_fn'' on the predicted versus expected
        response.
    """
    muygps.set_param_array(params, x0)

    predictions = muygps.regress(
        batch_indices,
        batch_nn_indices,
        embedded_train,
        embedded_train,
        train_targets,
    )

    targets = train_targets[batch_indices, :]

    return objective_fn(predictions, targets)


# def scipy_optimize(
#     muygps,
#     pairwise_dists,
#     crosswise_dists,
#     batch_nn_indices,
#     batch_indices,
#     train_targets,
#     loss_method="mse",
#     verbose=False,
# ):
#     # get the loss function
#     loss_fn = get_loss_func(loss_method)

#     # construct target tensors
#     batch_nn_targets = train_targets[batch_nn_indices, :]
#     batch_targets = train_targets[batch_indices, :]

#     # get optimization parameters
#     optim_params = muygps.get_optim_params()
#     x0 = np.array([optim_params[p]() for p in optim_params])
#     bounds = np.array([optim_params[p].get_bounds() for p in optim_params])
#     if verbose is True:
#         print(f"parameters to be optimized: {[p for p in optim_params]}")
#         print(f"bounds: {bounds}")
#         print(f"sampled x0: {x0}")

#     # do optimization
#     optres = opt.minimize(
#         loo_crossval,
#         x0,
#         args=(
#             loss_fn,
#             muygps,
#             optim_params,
#             pairwise_dists,
#             crosswise_dists,
#             batch_nn_targets,
#             batch_targets,
#         ),
#         method="L-BFGS-B",
#         bounds=bounds,
#     )
#     if verbose is True:
#         print(f"optimizer results: \n{optres}")

#     # set final values
#     for i, key in enumerate(optim_params):
#         optim_params[key]._set_val(x0[i])

#     return x0
