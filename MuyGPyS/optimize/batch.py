#!/usr/bin/env python
# encoding: utf-8
"""
@file batch.py

Created by priest2 on 2020-11-23

Training batch sampling.
"""

import numpy as np


def get_balanced_batch(
    train_nbrs_lookup,
    lookup,
    batch_size,
):
    """
    Decide whether to sample a balanced batch or return the full filtered batch.

    Parameters
    ----------
    train_nbrs_lookup : `MuyGPyS.neighbors.NN_Wrapper'
        Trained nearest neighbor query data structure.
    lookup : numpy.ndarray(int), shape = ``(train_count,)''
        List of class labels for all training data.
    batch_size : int
        The number of batch elements to sample.

    Returns
    -------
    numpy.ndarray(int), shape = ``(batch_count,)''
        The indices of the sampled training points.
    numpy.ndarray(int), shape = ``(batch_count, nn_count)''
        The indices of the nearest neighbors of the sampled training points.
    """
    if len(lookup) > batch_size:
        return sample_balanced_batch(train_nbrs_lookup, lookup, batch_size)
    else:
        return full_filtered_batch(train_nbrs_lookup, lookup)


def full_filtered_batch(
    train_nbrs_lookup,
    lookup,
):
    """
    Return a batch composed of the entire training set, filtering out elements
    with constant nearest neighbor sets.

    Parameters
    ----------
    train_nbrs_lookup : `MuyGPyS.neighbors.NN_Wrapper'
        Trained nearest neighbor query data structure.
    lookup : numpy.ndarray(int), shape = ``(train_count,)''
        List of class labels for all embedded data.

    Returns
    -------
    batch_indices : numpy.ndarray(int), shape = ``(batch_count,)''
        The indices of the sampled training points.
    batch_nn_indices : numpy.ndarray(int), shape = ``(batch_count, nn_count)''
        The indices of the nearest neighbors of the sampled training points.
    """
    train_indices = np.array([*range(len(lookup))])
    train_nn_indices = train_nbrs_lookup.get_batch_nns(train_indices)
    train_nn_labels = lookup[train_nn_indices]

    # filter out indices whose neighors all belong to one class
    # What if the index is mislabeled? Currently assuming that constant nn
    # labels -> correctly classified.
    train_nonconstant_mask = np.max(train_nn_labels, axis=1) != np.min(
        train_nn_labels,
        axis=1,
    )

    batch_indices = train_indices[train_nonconstant_mask]
    batch_nn_indices = train_nn_indices[train_nonconstant_mask, :]
    return batch_indices, batch_nn_indices


def sample_balanced_batch(
    train_nbrs_lookup,
    lookup,
    batch_size,
):
    """
    Collect a class-balanced batch of training indices.

    The returned batch is filtered to remove samples whose nearest neighbors
    share the same class label, and is balanced so that each class is equally
    represented (where possible.)

    Parameters
    ----------
    train_nbrs_lookup : `MuyGPyS.neighbors.NN_Wrapper'
        Trained nearest neighbor query data structure.
    lookup : numpy.ndarray(int), shape = ``(train_count,)''
        List of class labels for all embedded data.
    batch_size : int
        The number of batch elements to sample.

    Returns
    -------
    nonconstant_balanced_indices : numpy.ndarray(int),
                                   shape = ``(batch_count,)''
        The indices of the sampled training points.
    batch_nn_indices : numpy.ndarray(int), shape = ``(batch_count, nn_count)''
        The indices of the nearest neighbors of the sampled training points.
    """
    train_indices = np.array([*range(len(lookup))])
    train_nn_indices = train_nbrs_lookup.get_batch_nns(train_indices)
    train_nn_labels = lookup[train_nn_indices]
    # filter out indices whose neighors all belong to one class
    # What if the index is mislabeled? Currently assuming that constant nn
    # labels -> correctly classified.
    train_nonconstant_mask = np.max(train_nn_labels, axis=1) != np.min(
        train_nn_labels,
        axis=1,
    )
    classes = np.unique(lookup)
    class_count = len(classes)
    each_batch_size = int(batch_size / class_count)

    nonconstant_indices = [
        np.where(np.logical_and(train_nonconstant_mask, lookup == i))[0]
        for i in classes
    ]

    batch_sizes = np.array(
        [np.min((len(arr), each_batch_size)) for arr in nonconstant_indices]
    )

    nonconstant_balanced_indices = np.concatenate(
        [
            np.random.choice(
                nonconstant_indices[i], batch_sizes[i], replace=False
            )
            for i in range(class_count)
        ]
    )

    batch_nn_indices = train_nn_indices[nonconstant_balanced_indices, :]
    return nonconstant_balanced_indices, batch_nn_indices


def sample_batch(
    train_nbrs_lookup,
    batch_count,
    train_count,
):
    """
    Collect a batch of training indices.

    Parameters
    ----------
    train_nbrs_lookup : `MuyGPyS.neighbors.NN_Wrapper'
        Trained nearest neighbor query data structure.
    batch_count : int
        The number of batch elements to sample.
    train_count : int
        The total number of training examples.

    Returns
    -------
    batch_indices : numpy.ndarray(int), shape = ``(batch_count,)''
        The indices of the sampled training points.
    batch_nn_indices : numpy.ndarray(int), shape = ``(batch_count, nn_count)''
        The indices of the nearest neighbors of the sampled training points.
    """
    if train_count > batch_count:
        batch_indices = np.random.choice(
            train_count, batch_count, replace=False
        )
    else:
        batch_indices = np.array([*range(train_count)])
    batch_nn_indices = train_nbrs_lookup.get_batch_nns(batch_indices)
    return batch_indices, batch_nn_indices
