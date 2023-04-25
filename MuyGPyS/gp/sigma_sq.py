# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Sigma Square hyperparameter
"""

from typing import Callable, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import _fullname


class SigmaSq:
    """
    A :math:`\\sigma^2` covariance scale parameter.

    :math:`\\sigma^2` is a scaling parameter that one multiplies with the
    found diagonal variances of a :class:`MuyGPyS.gp.muygps.MuyGPS` or
    :class:`MuyGPyS.gp.muygps.MultivariateMuyGPS` regression in order to obtain
    the predicted posterior variance. Trained values assume a number of
    dimensions equal to the number of response dimensions, and correspond to
    scalar scaling parameters along the corresponding dimensions.

    Args:
        response_count:
            The integer number of response dimensions.
    """

    def __init__(self, response_count: int = 1):
        self.val = mm.ones(response_count)
        self._trained = False

    def __str__(self, **kwargs):
        return f"{type(self).__name__}({self.val})"

    def _set(self, val: mm.ndarray) -> None:
        """
        Value setter.

        Args:
            val:
                The new value of the hyperparameter.
        """
        if not isinstance(val, mm.ndarray):
            raise ValueError(
                f"Expected {_fullname(mm.ndarray)} for SigmaSq value update, "
                f"not {_fullname(val.__class__)}"
            )
        if self.val.shape != val.shape:
            raise ValueError(
                f"Bad attempt to assign SigmaSq of shape {self.val.shape} a "
                f"value of shape {val.shape}"
            )
        if val.dtype != mm.ftype:
            val = mm.farray(val)
        self.val = val
        self._trained = True

    def __call__(self) -> mm.ndarray:
        """
        Value accessor.

        Returns:
            The current value of the hyperparameter.
        """
        return self.val

    @property
    def trained(self) -> bool:
        """
        Report whether the value has been set.

        Returns:
            `True` if trained, `False` otherwise.
        """
        return self._trained

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Report the shape of the SigmaSq value.

        Returns:
            The shape of the SigmaSq value.
        """
        return self.val.shape


def sigma_sq_scale(fn: Callable) -> Callable:
    def scaled_fn(*args, sigma_sq=[1.0], **kwargs):
        return mm.outer(fn(*args, **kwargs), sigma_sq)

    return scaled_fn


def sigma_sq_apply(fn: Callable, sigma_sq: SigmaSq) -> Callable:
    def scaled_fn(*args, **kwargs):
        return fn(*args, sigma_sq=sigma_sq(), **kwargs)

    return scaled_fn
