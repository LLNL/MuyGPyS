# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Sigma Square hyperparameter
"""

from typing import Callable, Tuple, Type

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import _fullname
from MuyGPyS._src.optimize.sigma_sq import _analytic_sigma_sq_optim


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

    def __init__(
        self,
        response_count: int = 1,
        _backend_ones: Callable = mm.ones,
        _backend_ndarray: Type = mm.ndarray,
        _backend_ftype: Type = mm.ftype,
        _backend_farray: Callable = mm.farray,
        _backend_outer: Callable = mm.outer,
        **kwargs,
    ):
        self.val = _backend_ones(response_count)
        self._trained = False

        self._backend_ndarray = _backend_ndarray
        self._backend_ftype = _backend_ftype
        self._backend_farray = _backend_farray
        self._backend_outer = _backend_outer

    def __str__(self, **kwargs):
        return f"{type(self).__name__}({self.val})"

    def _set(self, val: mm.ndarray) -> None:
        """
        Value setter.

        Args:
            val:
                The new value of the hyperparameter.
        """
        if not isinstance(val, self._backend_ndarray):
            raise ValueError(
                f"Expected {_fullname(self._backend_ndarray)} for SigmaSq "
                f"value update, not {_fullname(val.__class__)}"
            )
        if self.val.shape != val.shape:
            raise ValueError(
                f"Bad attempt to assign SigmaSq of shape {self.val.shape} a "
                f"value of shape {val.shape}"
            )
        if val.dtype != self._backend_ftype:
            val = self._backend_farray(val)
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

    def scale_fn(self, fn: Callable) -> Callable:
        """
        Modify a function to outer product its output with `sigma_sq`.

        Args:
            fn:
                A function.

        Returns:
            A function that returns the outer product of the output of `fn`
        """

        def scaled_fn(*args, sigma_sq=self(), **kwargs):
            return self._backend_outer(fn(*args, **kwargs), sigma_sq)

        return scaled_fn

    def get_opt_fn(self, muygps) -> Callable:
        def noop_sigma_sq_opt_fn(K, nn_targets, *args, **kwargs):
            return muygps.sigma_sq()

        return noop_sigma_sq_opt_fn


class AnalyticSigmaSq(SigmaSq):
    """
    An optimizable :math:`\\sigma^2` covariance scale parameter.

    Identical to :class:`~MuyGPyS.gp.sigma_sq.SigmaSq`, save that its
    `get_opt_fn` method performs an analytic optimization.

    Args:
        response_count:
            The integer number of response dimensions.
    """

    def get_opt_fn(self, muygps) -> Callable:
        """
        Get a function to optimize the value of the :math:`\\sigma^2` scale
        parameter for each response dimension.

        We approximate :math:`\\sigma^2` by way of averaging over the analytic
        solution from each local kernel.

        .. math::
            \\sigma^2 = \\frac{1}{bk} * \\sum_{i \\in B}
                        Y_{nn_i}^T K_{nn_i}^{-1} Y_{nn_i}

        Here :math:`Y_{nn_i}` and :math:`K_{nn_i}` are the target and kernel
        matrices with respect to the nearest neighbor set in scope, where
        :math:`k` is the number of nearest neighbors and :math:`b = |B|` is the
        number of batch elements considered.

        Args:
            muygps:
                The model to used to create and perturb the kernel.

        Returns:
            A function with signature
            `(K, nn_targets, *args, **kwargs) -> mm.ndarray` that perturbs the
            `(batch_count, nn_count, nn_count)` tensor `K` with `muygps`'s noise
            model before solving it against the
            `(batch_count, nn_count, response_count)` tensor `nn_targets`.
        """

        def analytic_sigma_sq_opt_fn(K, nn_targets, *args, **kwargs):
            return _analytic_sigma_sq_optim(muygps.noise.perturb(K), nn_targets)

        return analytic_sigma_sq_opt_fn
