#!/usr/bin/env python
# encoding: utf-8
"""
@file utils.py

Created by priest2 on 2020-11-23

Utility functions for manipulating data.
"""

import numpy as np


def subsample(data, sample_count):
    count = data["input"].shape[0]
    samples = np.random.choice(count, sample_count, replace=False)
    return {
        "input": data["input"][samples, :],
        "output": data["output"][samples, :],
        "lookup": data["lookup"][samples],
    }


def balanced_subsample(data, sample_count):
    classes = np.unique(data["lookup"])
    class_count = len(classes)
    each_sample_count = int(sample_count / class_count)

    class_indices = np.array(
        [np.where(data["lookup"] == i)[0] for i in classes]
    )
    sample_sizes = np.array(
        [np.min((len(arr), each_sample_count)) for arr in class_indices]
    )
    balanced_samples = np.concatenate(
        [
            np.random.choice(class_indices[i], sample_sizes[i], replace=False)
            for i in range(class_count)
        ]
    )
    return {
        "input": data["input"][balanced_samples, :],
        "output": data["output"][balanced_samples, :],
        "lookup": data["lookup"][balanced_samples],
    }


def normalize(X):
    """
    Normalizes data matrix to have row l2-norms of 1

    Parameters
    ----------
    X : numpy.ndarray, shape = (n_rows, n_cols)
        Observation locations. The first dimenion assumed to be the number of
        observations.

    Returns
    -------
    numpy.ndarray, shape = (n_rows, n_cols)
        Normalized X.
    """
    # return X * np.sqrt(1 / np.sum(X ** 2, axis=1))[:, None]
    return X * np.sqrt(X.shape[1] / np.sum(X ** 2, axis=1))[:, None]
