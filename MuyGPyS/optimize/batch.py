# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Sampling elements with their nearest neighbors from data

MuyGPyS includes convenience functions for sampling batches of data from
existing datasets.
These batches are returned in the form of row indices, both of the sampled data
as well as their nearest neighbors.
Also included is the ability to sample "balanced" batches, where the data is
partitioned by class and we attempt to sample as close to an equal number of
items from each class as is possible.
"""

from typing import Tuple

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS.neighbors import NN_Wrapper


def get_balanced_batch(
    nbrs_lookup: NN_Wrapper,
    labels: mm.ndarray,
    batch_count: int,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Decide whether to sample a balanced batch or return the full filtered batch.

    This method is the go-to method for sampling from classification datasets
    when one desires a sample with equal representation of every class. The
    function simply calls :func:`MuyGPyS.optimize.batch.full_filtered_batch` if
    the supplied list of training data class labels is smaller than the batch
    count, otherwise calling
    :func:`MuyGPyS.optimize.batch_sample_balanced_batch`.

    Example:
        >>> import numpy as np
        >>> From MuyGPyS.optimize.batch import get_balanced_batch
        >>> train_features, train_responses = get_train()
        >>> nn_count = 10
        >>> nbrs_lookup = NN_Wrapper(train_features, nn_count)
        >>> batch_count = 200
        >>> train_labels = np.argmax(train_responses, axis=1)
        >>> balanced_indices, balanced_nn_indices = get_balanced_batch(
        ...         nbrs_lookup, train_labels, batch_count
        >>> )

    Args:
        nbrs_lookup:
            Trained nearest neighbor query data structure.
        labels:
            List of class labels of shape `(train_count,)` for all training
            data.
        batch_count: int
            The number of batch elements to sample.

    Returns
    -------
    indices:
        The indices of the sampled training points of shape
        `(batch_count,)`.
    nn_indices:
        The indices of the nearest neighbors of the sampled training points
        of shape `(batch_count, nn_count)`.
    """
    if len(labels) > batch_count:
        return sample_balanced_batch(nbrs_lookup, labels, batch_count)
    else:
        return full_filtered_batch(nbrs_lookup, labels)


def full_filtered_batch(
    nbrs_lookup: NN_Wrapper,
    labels: mm.ndarray,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Return a batch composed of the entire training set, filtering out elements
    with constant nearest neighbor sets.

    Args:
        nbrs_lookup:
            Trained nearest neighbor query data structure.
        labels:
            List of class labels of shape `(train_count,)` for all train data.

    Returns
    -------
    indices:
        The indices of the sampled training points of shape `(batch_count,)`.
    nn_indices:
        The indices of the nearest neighbors of the sampled training points of
        shape `(batch_count, nn_count)`.
    """
    indices = mm.arange(len(labels))
    nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
    nn_labels = labels[nn_indices]

    # filter out indices whose neighors all belong to one class
    # What if the index is mislabeled? Currently assuming that constant nn
    # labels -> correctly classified.
    nonconstant_mask = np.max(np.iarray(nn_labels), axis=1) != np.min(
        np.iarray(nn_labels),
        axis=1,
    )

    batch_indices = indices[nonconstant_mask]
    batch_nn_indices = nn_indices[nonconstant_mask, :]
    return batch_indices, batch_nn_indices


def sample_balanced_batch(
    nbrs_lookup: NN_Wrapper,
    labels: mm.ndarray,
    batch_count: int,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Collect a class-balanced batch of training indices.

    The returned batch is filtered to remove samples whose nearest neighbors
    share the same class label, and is balanced so that each class is equally
    represented (where possible.)

    Args:
        nbrs_lookup:
            Trained nearest neighbor query data structure.
        labels:
            List of class labels of shape `(train_count,)` for all train data.
        batch_count:
            The number of batch elements to sample.

    Returns
    -------
    nonconstant_balanced_indices:
        The indices of the sampled training points of shape `(batch_count,)`.
        These indices are guaranteed to have nearest neighbors with differing
        class labels.
    batch_nn_indices:
        The indices of the nearest neighbors of the sampled training points of
        shape `(batch_count, nn_count)`.
    """
    indices = mm.arange(len(labels))
    nn_indices, _ = nbrs_lookup.get_batch_nns(indices)
    nn_labels = labels[nn_indices]
    # filter out indices whose neighors all belong to one class
    # What if the index is mislabeled? Currently assuming that constant nn
    # labels -> correctly classified.
    nonconstant_mask = np.max(np.iarray(nn_labels), axis=1) != np.min(
        np.iarray(nn_labels),
        axis=1,
    )
    classes = np.unique(labels)
    class_count = len(classes)
    each_batch_count = int(batch_count / class_count)

    nonconstant_indices = [
        np.where(np.logical_and(nonconstant_mask, labels == i))[0]
        for i in classes
    ]

    batch_counts = np.iarray(
        [np.min((len(arr), each_batch_count)) for arr in nonconstant_indices]
    )

    nonconstant_balanced_indices = mm.iarray(
        np.concatenate(
            [
                np.random.choice(
                    nonconstant_indices[i], batch_counts[i], replace=False
                )
                for i in range(class_count)
            ]
        )
    )

    batch_nn_indices = nn_indices[nonconstant_balanced_indices, :]
    return nonconstant_balanced_indices, batch_nn_indices


def sample_batch(
    nbrs_lookup: NN_Wrapper,
    batch_count: int,
    train_count: int,
) -> Tuple[mm.ndarray, mm.ndarray]:
    """
    Collect a batch of training indices.

    This is a simple sampling method where training examples are selected
    uniformly at random, without replacement.

    Example:
        >>> From MuyGPyS.optimize.batch import sample_batch
        >>> train_features, train_responses = get_train()
        >>> train_count, _ = train_features.shape
        >>> nn_count = 10
        >>> nbrs_lookup = NN_Wrapper(train_features, nn_count)
        >>> batch_count = 200
        >>> batch_indices, batch_nn_indices = sample_batch(
        ...         nbrs_lookup, batch_count, train_count
        >>> )

    Args:
        nbrs_lookup:
            Trained nearest neighbor query data structure.
        batch_count:
            The number of batch elements to sample.
        train_count : int
            The total number of training examples.

    Returns
    -------
    batch_indices:
        The indices of the sampled training points of shape `(batch_count,)`.
    batch_nn_indices:
        The indices of the nearest neighbors of the sampled training points of
        shape `(batch_count, nn_count)`.
    """
    if train_count > batch_count:
        batch_indices = mm.iarray(
            np.random.choice(train_count, batch_count, replace=False)
        )
    else:
        batch_indices = mm.arange(train_count, dtype=mm.itype)
    batch_nn_indices, _ = nbrs_lookup.get_batch_nns(batch_indices)
    return batch_indices, batch_nn_indices
