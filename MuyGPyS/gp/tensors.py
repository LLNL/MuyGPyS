# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Tensor functions

Compute special tensors for the purposes of kernel construction.
"""

from typing import Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import (
    _make_fast_predict_tensors,
    _batch_features_tensor,
    _fast_nn_update,
    _make_heteroscedastic_tensor,
)


def make_heteroscedastic_tensor(
    measurement_noise: mm.ndarray,
    batch_nn_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Create the heteroscedastic noise tensor for nonuniform noise values.

    Used to produce the noise tensor needed during batched training and
    prediction. Creates the `noise_tensor` tensor required by heteroscedastic
    MuyGPs models.

    Args:
        measurement_noise:
            A matrix of floats of shape `(batch_count,)` providing the noise
            corresponding to the response variable at each input value in the
            data.
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count, nn_count)`
            listing the measurement noise for the nearest neighbors for all
            observations in the batch.

    Returns:
        A matrix of floats of shape `(batch_count, nn_count)` providing the
        noise corresponding to the nearest neighbor responses for all
        observations in the batch.
    """
    return _make_heteroscedastic_tensor(measurement_noise, batch_nn_indices)


def fast_nn_update(
    train_nn_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Modify the nearest neighbor indices of the training data to include self.

    This function is only intended for use in concert with
    :func:~MuyGPyS.gp.tensors.make_fast_predict_tensors` and
    :func:`MuyGPyS.gp.muygps.MuyGPS.fast_coefficients`.

    Example:
        >>> train_nn_indices, _ = nbrs_lookup.get_nns(train_features)
        >>> train_nn_indices = fast_nn_update(train_nn_indices)
        >>> pairwise_diffs, nn_targets = make_fast_predict_tensors(
        ...     train_nn_indices,
        ...     train_features,
        ...     train_responses,
        ... )
        >>> Kin = muygps_fast.kernel(pairwise_diffs)
        >>> precomputed_coefficients_matrix = muygps_fast.fast_coefficients(
        ...     Kin, nn_targets
        ... )
        >>> # Late on, once test data is encountered
        >>> test_indices = np.arange(test_count)
        >>> test_nn_indices, _ = nbrs_lookup.get_nns(test_features)
        >>> closest_neighbor = test_nn_indices[:, 0]
        >>> closest_set = train_nn_indices[closest_neighbor, :]

    Args:
        train_nn_indices:
            A matrix of integers of shape `(train_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.

    Returns:
        An integral matrix of shape `(train_count, nn_count)` that is similar
        to the input, but the most distant neighbor index is removed and the
        index reference to self has been inserted.
    """
    return _fast_nn_update(train_nn_indices)


def make_fast_predict_tensors(
    batch_nn_indices: mm.ndarray,
    train_features: mm.ndarray,
    train_targets: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Create the difference and target tensors for fast posterior mean inference.

    Creates `pairwise_diffs` and `batch_nn_targets` tensors required by
    :func:`MuyGPyS.gp.muygps.MuyGPS.fast_posterior_mean`.

    Args:
        batch_nn_indices:
            A matrix of integers of shape `(batch_count, nn_count)` listing the
            nearest neighbor indices for all observations in the batch.
        train_features:
            The full floating point training data matrix of shape
            `(train_count, feature_count)`.
        train_targets:
            A matrix of shape `(train_count, response_count)` whose rows are
            vector-valued responses for each training element.

    Returns
    -------
    pairwise_diffs:
        A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
        containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
        nearest neighbor difference tensors corresponding to each of the
        batch elements.
    batch_nn_targets:
        Tensor of floats of shape `(batch_count, nn_count, response_count)`
        containing the expected response for each nearest neighbor of each batch
        element.
    """
    return _make_fast_predict_tensors(
        batch_nn_indices,
        train_features,
        train_targets,
    )


def batch_features_tensor(
    features: mm.ndarray,
    batch_indices: mm.ndarray,
) -> mm.ndarray:
    """
    Compute a tensor of feature vectors for each batch element.

    Args:
        features:
            The full floating point training or testing data matrix of shape
            `(train_count, feature_count)` or `(test_count, feature_count)`.
        batch_indices:
            A vector of integers of shape `(batch_count,)` identifying the
            training batch of observations to be approximated.

    Returns:
        A tensor of shape `(batch_count, feature_count)` containing
        the feature vectors for each batch element.
    """
    return _batch_features_tensor(features, batch_indices)
