#!/usr/bin/env python
# encoding: utf-8
"""
@file embed.py

Created by priest2 on 2020-10-27

End-to-end application of LKGP.
"""

import numpy as np

from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

from muyscans.data.utils import normalize


def embed_all(
    train,
    test,
    embed_dim,
    embed_method="pca",
    do_normalize=False,
    in_place=False,
):
    embedded_train, embedded_test = apply_embedding(
        train["input"], test["input"], embed_dim, embed_method, do_normalize
    )
    if in_place is True:
        train["input"] = embedded_train
        test["input"] = embedded_test
    else:
        etrain = {k: train[k] for k in train}
        etest = {k: test[k] for k in test}
        etrain["input"] = embedded_train
        etest["input"] = embedded_test
        return etrain, etest


def apply_embedding(
    train_data, test_data, embed_dim, embed_method, do_normalize
):
    if embed_method is None:
        return train_data, test_data
    else:
        embed_method = embed_method.lower()
        if embed_method == "pca":
            return pca_embed(train_data, test_data, embed_dim, do_normalize)
        else:
            raise NotImplementedError(
                f"Embedding {embed_method} is not implemented."
            )


def pca_embed(
    train_data,
    test_data,
    embed_dim,
    do_normalize,
):
    """
    Embed matrix data using PCA.

    From this answer: https://stackoverflow.com/a/12168892

    Parameters
    ----------
    train_data : numpy.ndarray, type = float, shape = ``(train_count, data_dim)''
        The train data matrix, e.g. rows of flattened images.
    test_data : numpy.ndarray, type = float, shape = ``(test_count, data_dim)''
        The test data matrix, e.g. rows of flattened images.
    embed_dim : int
        The PCA dimension onto which data will be embedded.
    do_normalize : Boolean
        Indicates whether to apply normalization to the function.

    Returns
    -------
    embedded_train : numpy.ndarray, type = float,
            shape = ``(train_count, embed_dim)''
        The training data embedded into the specified dimension.
    embedded_test : numpy.ndarray, type = float,
            shape = ``(test_count, embed_dim)''
        The testing data embedded into the specified dimension.
    """
    test_count = test_data.shape[0]
    data = np.vstack((test_data, train_data))
    if do_normalize is True:
        data = normalize(data)
    _, evecs = largest_eigsh(data.T @ data, embed_dim, which="LM")
    embedded_data = data @ evecs

    embedded_test = embedded_data[:test_count]
    embedded_train = embedded_data[test_count:]

    return embedded_train, embedded_test
