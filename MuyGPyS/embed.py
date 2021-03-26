#!/usr/bin/env python
# encoding: utf-8
"""
@file embed.py

Created by priest2 on 2020-10-27

End-to-end application of LKGP.
"""

import numpy as np

from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

from MuyGPyS.data.utils import normalize


def embed_all(
    train,
    test,
    embed_dim,
    embed_method="pca",
    do_normalize=False,
    in_place=False,
):
    """
    Embed a dataset into a lower dimensionsal feature space.

    Parameters
    ----------
    train : dict
        A dict with keys "input" and "output". "input" maps to a matrix of row
        observation vectors. "output" maps to a matrix listing the observed
        responses of the phenomenon under study.
    test : dict
        A dict with keys "input" and "output". "input" maps to a matrix of row
        observation vectors. "output" maps to a matrix listing the observed
        responses of the phenomenon under study.
    embed_dim : int
        The dimension onto which the data vectors will be embedded.
    embed_method : str
        The embedding method to be used.
        NOTE[bwp] Current supporing only "pca".
    do_normalize : Boolean
        Indicates whether to normalize prior to embedding.
    in_place : Boolean
        If true, overwrite the original dicts. Else, return new dicts.

    Return
    ------
    etrain : dict
        The same data dict, but with embedded input. Only return if
        ``in_place == True''.
    etest : dict
        The same data dict, but with embedded input. Only return if
        ``in_place == True''.
    """
    # embedded_train, embedded_test = apply_embedding(
    #     train["input"], test["input"], embed_dim, embed_method, do_normalize
    # )
    if in_place is True:
        train["input"], test["input"] = apply_embedding(
            train["input"], test["input"], embed_dim, embed_method, do_normalize
        )
        # train["input"] = embedded_train
        # test["input"] = embedded_test
    else:
        etrain = dict()
        etest = dict()
        etrain["output"] = train["output"]
        etest["output"] = test["output"]
        etrain["lookup"] = train["lookup"]
        etest["lookup"] = test["lookup"]

        etrain["input"], etest["input"] = apply_embedding(
            train["input"], test["input"], embed_dim, embed_method, do_normalize
        )
        # etrain = {k: train[k] for k in train}
        # etest = {k: test[k] for k in test}
        # etrain["input"] = embedded_train
        # etest["input"] = embedded_test
        return etrain, etest


def apply_embedding(
    train_data, test_data, embed_dim, embed_method, do_normalize
):
    """
    Select and compute the appropriate embedding function.

    Parameters
    ----------
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        The train data matrix.
    test_data : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        The test data matrix.
    embed_dim : int
        The dimension onto which the data vectors will be embedded.
    embed_method : str
        The embedding method to be used.
        NOTE[bwp] Current supporing only "pca".
    do_normalize : Boolean
        Indicates whether to normalize prior to embedding.
    in_place : Boolean
        If true, overwrite the original dicts. Else, return new dicts.

    Return
    ------
    etrain : dict
        The same data dict, but with embedded input. Only return if
        ``in_place == True''.
    etest : dict
        The same data dict, but with embedded input. Only return if
        ``in_place == True''.
    """
    if embed_method is None:
        return train_data, test_data
    else:
        embed_method = embed_method.lower()
        if embed_method == "pca":
            return _pca_embed(train_data, test_data, embed_dim, do_normalize)
        else:
            raise NotImplementedError(
                f"Embedding {embed_method} is not implemented."
            )


def _pca_embed(
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
    train_data : numpy.ndarray(float), shape = ``(train_count, feature_count)''
        The train data matrix.
    test_data : numpy.ndarray(float), shape = ``(test_count, feature_count)''
        The test data matrix.
    embed_dim : int
        The PCA dimension onto which data will be embedded.
    do_normalize : Boolean
        Indicates whether to apply normalization prior to embedding.

    Returns
    -------
    embedded_train : numpy.ndarray(float), shape = ``(train_count, embed_dim)''
        The training data embedded into the specified dimension.
    embedded_test : numpy.ndarray(float), shape = ``(test_count, embed_dim)''
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
