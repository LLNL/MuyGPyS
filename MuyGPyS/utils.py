# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS 
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import inspect

from numbers import Number
from numpy.random import randint
from sys import maxsize


def _val_to_list(val, L):
    """
    Takes a value val and returns a list containing L copies if val is any
    scalar number, a callable (e.g. an activation function), or a length-2 tuple
    (e.g. a convolutional stride). Otherwise just returns val.

    Parameters
    ----------
    val : array_like or scalar
        Input value.
    L : int
        The length of the return list.

    Returns
    -------
    array_like, shape = (L,)
        Either L copies of val or val unchanged.

    """
    if (
        isinstance(val, Number)
        or callable(val)
        or isinstance(val, tuple)
        or isinstance(val, str)
        or val is None
    ):
        return [val for l in range(L)]
    else:
        return val


def _vals_to_lists(L, *vals):
    """
    Applies vals_to_lists_ to ever element of *vals. Non-list parameters will
    be converted to lists of copies, signifying that a hyperparameter is
    constant across all layers.
    """
    return (_val_to_list(val, L) for val in vals)


def randint64(**kwargs):
    """
    Convenience function for obtaining a random 64bit int.
    """
    return randint(-maxsize + 1, maxsize, **kwargs)


def safe_apply(fn, *args, **kwargs):
    """
    Accepts a callable and applies the parameters specified by args and kwargs,
    while ignoring any kwargs that are not present in the prior_fn's signature.
    """
    the_kwargs = {p.name for p in inspect.signature(fn).parameters.values()}
    relevant_kwargs = {k: kwargs[k] for k in (kwargs.keys() & the_kwargs)}
    return fn(*args, **relevant_kwargs)
