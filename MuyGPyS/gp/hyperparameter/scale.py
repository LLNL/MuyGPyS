# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Variance scale hyperparameter
"""

from typing import Callable, Tuple, Type

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.util import _fullname
from MuyGPyS._src.optimize.scale import (
    _analytic_scale_optim,
    _analytic_scale_optim_unnormalized,
)


class ScaleFn:
    """
    A :math:`\\sigma^2` covariance scale parameter base functor.

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
        self.val = _backend_ones(
            self._check_positive_integer(response_count, "response")
        )
        self._trained = False

        self._backend_ndarray = _backend_ndarray
        self._backend_ftype = _backend_ftype
        self._backend_farray = _backend_farray
        self._backend_outer = _backend_outer

    def _check_positive_integer(self, count, name) -> int:
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                f"{name} count must be a positive integer, not {count}"
            )
        return count

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
                f"Expected {_fullname(self._backend_ndarray)} for variance "
                f"scale value update, not {_fullname(val.__class__)}"
            )
        if self.val.shape != val.shape:
            raise ValueError(
                "Bad attempt to assign variance scale of shape "
                f"{self.val.shape} a value of shape {val.shape}"
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
        Report the shape of the scale parameter.

        Returns:
            The shape of the scale parameter.
        """
        return self.val.shape

    def scale_fn(self, fn: Callable) -> Callable:
        """
        Modify a function to outer product its output with `scale`.

        Args:
            fn:
                A function.

        Returns:
            A function that returns the outer product of the output of `fn`
        """

        def scaled_fn(*args, scale=self(), **kwargs):
            return self._backend_outer(fn(*args, **kwargs), scale)

        return scaled_fn

    def get_opt_fn(self, muygps) -> Callable:
        def noop_scale_opt_fn(K, nn_targets, *args, **kwargs):
            return muygps.scale()

        return noop_scale_opt_fn


class FixedScale(ScaleFn):
    """
    A :math:`\\sigma^2` covariance scale parameter.

    A `Scale` parameter with a null optimization method. This parameter is
    therefore insensitive to optimization.

    Args:
        response_count:
            The integer number of response dimensions.
    """

    def get_opt_fn(self, muygps) -> Callable:
        """
        Return a function that optimizes the value of the variance scale.

        Args:
            muygps:
                A model to be ignored.

        Returns:
            A function that always returns the value of this scale parameter.
        """

        def noop_scale_opt_fn(K, nn_targets, *args, **kwargs):
            return muygps.scale()

        return noop_scale_opt_fn


class AnalyticScale(ScaleFn):
    """
    An optimizable :math:`\\sigma^2` covariance scale parameter.

    Identical to :class:`~MuyGPyS.gp.scale.FixedScale`, save that its
    `get_opt_fn` method performs an analytic optimization.

    Args:
        response_count:
            The integer number of response dimensions.
    """

    def __init__(self, _backend_fn: Callable = _analytic_scale_optim, **kwargs):
        super().__init__(**kwargs)
        self._fn = _backend_fn

    def get_opt_fn(self, muygps) -> Callable:
        """
        Get a function to optimize the value of the :math:`\\sigma^2` scale
        parameter for each response dimension.

        We approximate a scalar :math:`\\sigma^2` by way of averaging over the
        analytic solution from each local kernel. Given observations :math:`X`
        with responses :math:`Y`, noise model :math:`\\varepsilon`, and kernel
        function :math:`K_\\theta(\\cdot, \\cdot)`, computes:

        .. math::
            \\sigma^2 = \\frac{1}{bk} * \\sum_{i \\in B} Y(X_{N_i})^T
                \\left (
                    K_\\theta(X_{N_i}, X_{N_i}) + \\varepsilon_{N_i}
                \\right )^{-1}
                Y(X_{N_i}).

        Here :math:`N_i` is the set of nearest neighbor indices of the
        :math:`i`th batch element, :math:`k` is the number of nearest neighbors
        and :math:`b = |B|` is the number of batch elements considered.

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

        def analytic_scale_opt_fn(K, nn_targets, *args, **kwargs):
            return self._fn(muygps.noise.perturb(K), nn_targets)

        return analytic_scale_opt_fn


class DownSampleScale(ScaleFn):
    """
    An optimizable :math:`\\sigma^2` covariance scale parameter.

    Identical to :class:`~MuyGPyS.gp.scale.FixedScale`, save that its
    `get_opt_fn` method performs an analytic optimization.

    Args:
        response_count:
            The integer number of response dimensions.
        down_count:
            The integer number of neighbors to sample, without replacement.
            Must be less than `nn_count`.
        iteration_count:
            The number of iterations to
    """

    def __init__(
        self,
        down_count: int = 10,
        iteration_count: int = 10,
        _backend_fn: Callable = _analytic_scale_optim_unnormalized,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._down_count = self._check_positive_integer(
            down_count, "down sample"
        )
        self._iteration_count = self._check_positive_integer(
            iteration_count, "down sample iteration"
        )
        self._fn = _backend_fn

    def get_opt_fn(self, muygps) -> Callable:
        """
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

        def downsample_analytic_scale_opt_fn(K, nn_targets, *args, **kwargs):
            batch_count, nn_count, _ = K.shape
            if nn_count <= self._down_count:
                raise ValueError(
                    f"bad attempt to downsample {self._down_count} elements "
                    f"from a set of only {nn_count} options"
                )
            pK = muygps.noise.perturb(K)
            scales = []
            for _ in range(self._iteration_count):
                sampled_indices = np.random.choice(
                    np.arange(nn_count),
                    size=self._down_count,
                    replace=False,
                )
                sampled_indices.sort()

                pK_down = pK[:, sampled_indices, :]
                pK_down = pK_down[:, :, sampled_indices]
                nn_targets_down = nn_targets[:, sampled_indices, :]
                scales.append(self._fn(pK_down, nn_targets_down))

            return mm.atleast_1d(np.median(scales, axis=0)) / (
                self._down_count * batch_count
            )

        return downsample_analytic_scale_opt_fn
