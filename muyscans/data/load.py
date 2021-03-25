#!/usr/bin/env python
# encoding: utf-8
"""
@file load.py

Created by priest2 on 2020-10-19

Convenience functions for loading data sets into expected dict format.

Currently MNIST support requires tensorflow-datasets==2.0.0.
"""

import numpy as np
import pandas as pd

from functools import reduce

from muyscans.data.utils import normalize


def process_mnist(data_chunk, flatten=True):
    """
    Process and normalize a dict of tensors.

    Parameters
    ----------
    data_check : dict
        Expects keys "image" and "label", each of which maps to tensors arraying
        observation indices against observed values (or labels).
    flatten : Boolean
        If true, images tensors will be flattened into matrices so that each
        observation is associated with a vector.

    Returns
    -------
    dict
        A dict with keys "input", "output", and "lookup". "input" maps to a
        matrix of row observation vectors, e.g. flattened images. "output" maps
        to a matrix listing the one-hot encodings of each observation's class.
        "lookup" is effectively the argmax over this matrix's columns.
    """
    image, lookup = data_chunk["image"], data_chunk["label"]

    image = np.array(image, dtype=np.float32)
    #     image = (image - np.mean(image)) / np.std(image)
    shape = image.shape
    count = shape[0]
    pixels = reduce(lambda x, y: x * y, shape[1:])
    image = image.reshape(count, pixels)
    image = image / np.linalg.norm(image, axis=-1)[:, None]
    image = image.reshape(shape)
    if flatten is True:
        image = image.reshape(count, np.prod(shape[1:]))
    label = np.eye(10)[lookup] - 0.1

    return {"input": image, "output": label, "lookup": lookup}


def make_mnist(flatten=True):
    """
    Load the mnist datasets as dicts of matrices.

    Parameters
    ----------
    flatten : Boolean
        If true, images tensors will be flattened into matrices.

    Returns
    -------
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
    """
    import tensorflow_datasets as tfds

    ds_train, ds_test = tfds.as_numpy(
        tfds.load("mnist:3.*.*", split=["train", "test"], batch_size=-1)
    )
    return (
        process_mnist(ds_train, flatten=flatten),
        process_mnist(ds_test, flatten=flatten),
    )


def process_stargal(sg_samples):
    """
    Construct star-galaxy data dict from np.ndarray.

    Parameters
    ----------
    sg_samples : np.ndarray(float), shape = ``(count, dim + 2)''


    Returns
    -------
    ret : dict
        A dict with keys "input", "output", and "lookup". "input" maps to a
        matrix of row observation vectors, e.g. flattened images. "output" maps
        to a matrix listing the one-hot encodings of each observation's class.
        "lookup" is effectively the argmax over this matrix's columns.
    """
    image = sg_samples[:, 2:]
    image = normalize(image)
    lookup = sg_samples[:, 1].astype(int)
    # label = np.eye(2)[lookup] - 0.1
    label = 2 * np.eye(2)[lookup] - 1.0
    lookup = 2 * lookup - 1
    return {"input": image, "output": label, "lookup": lookup}


def make_stargal(fname="data/star-gal/galstar.csv", test_count=10000):
    """
    Load the mnist datasets as dicts of matrices.

    Parameters
    ----------
    fname : str
        Relative path to the local copy of the ``galstar.csv'' file.
        NOTE[bwp]: Currently using file obtained from Amanda.

    Returns
    -------
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
    """
    stargal = pd.read_csv(fname, sep=",").to_numpy()
    np.random.shuffle(stargal)
    return (
        process_stargal(stargal[test_count:, :]),
        process_stargal(stargal[:test_count, :]),
    )


# def make_satfit(dirname="data/satfit/satfit.npz"):
#     """
#     Load the satellite data.

#     Parameters
#     ----------
#     fname : str
#         Relative path to the local copy of the ``galstar.csv'' file.
#         NOTE[bwp]: Currently using file obtained from Amanda.

#     Returns
#     -------
#     train : dict
#         A dict with keys "input", "output", and "lookup". "input" maps to a
#         tensor of (possibly flattened) images. "output" maps to a matrix listing
#         the one-hot encodings of each observation's class. "lookup" is
#         effectively the argmax over this matrix's columns.
#     test : dict
#         A dict with keys "input", "output", and "lookup". "input" maps to a
#         tensor of (possibly flattened) images. "output" maps to a matrix listing
#         the one-hot encodings of each observation's class. "lookup" is
#         effectively the argmax over this matrix's columns.
#     """
#     stargal = pd.read_csv(fname, sep=",").to_numpy()
#     np.random.shuffle(stargal)
#     return (
#         process_stargal(stargal[test_count:, :]),
#         process_stargal(stargal[:test_count, :]),
#     )