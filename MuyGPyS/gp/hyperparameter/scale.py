# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Variance scale hyperparameter
"""

from collections.abc import Sequence
from typing import Callable

import MuyGPyS._src.math as mm
import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.optimize.scale import (
    _analytic_scale_optim,
    _analytic_scale_optim_unnormalized,
)


class ScaleFn:
    """
    A :math:`\\sigma^2` covariance scale parameter base functor.

    :math:`\\sigma^2` is a scaling parameter that one multiplies with the
    found diagonal variances of a :class:`MuyGPyS.gp.muygps.MuyGPS` regression
    in order to obtain the predicted posterior variance. Trained values assume a
    number of dimensions equal to the number of response dimensions, and
    correspond to scalar scaling parameters along the corresponding dimensions.

    Args:
        val:
            A floating point value, if intended to be set manually. Defaults to
            1.0.
    """

    def __init__(self, val: float = 1.0, **kwargs):
        self.val = self._check_positive_float(val)
        self._trained = False

    def _check_positive_integer(self, count, name) -> int:
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                f"{name} count must be a positive integer, not {count}"
            )
        return count

    def _check_positive_float(self, val) -> float:
        if (
            isinstance(val, Sequence)
            or hasattr(val, "__len__")
            and len(val) != 1
        ):
            raise ValueError(f"Scale parameter must be scalar, not {val}.")
        if not isinstance(val, mm.ndarray):
            val = float(val)
        if val <= 0.0:
            raise ValueError(f"Scale parameter must be positive, not {val}.")
        return val

    def __str__(self, **kwargs):
        return f"{type(self).__name__}({self.val})"

    def _set(self, val: float) -> None:
        """
        Value setter.

        Args:
            val:
                The new value of the hyperparameter.
        """
        self.val = self._check_positive_float(val)
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
            return scale * fn(*args, **kwargs)

        return scaled_fn

    def get_opt_fn(self, muygps) -> Callable:
        def noop_scale_opt_fn(Kin, nn_targets, *args, **kwargs):
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

        def noop_scale_opt_fn(Kin, nn_targets, *args, **kwargs):
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
        iteration_count:
            The number of iterations to run during optimization.
    """

    def __init__(
        self,
        iteration_count: int = 1,
        _backend_fn: Callable = _analytic_scale_optim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.iteration_count = iteration_count
        self._fn = _backend_fn

    def get_opt_fn(self, muygps) -> Callable:
        """
        Get a function to optimize the value of the :math:`\\sigma^2` scale
        parameter for each response dimension.

        We approximate a scalar :math:`\\sigma^2` by way of averaging over the
        analytic solution from each local kernel. Given observations :math:`X`
        with responses :math:`Y`, noise model :math:`\\varepsilon`, and kernel
        function :math:`Kin_\\theta(\\cdot, \\cdot)`, computes:

        .. math::
            \\sigma^2 = \\frac{1}{bk} * \\sum_{i \\in B} Y(X_{N_i})^T
                \\left (
                    Kin_\\theta(X_{N_i}, X_{N_i}) + \\varepsilon_{N_i}
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
            `(Kin, nn_targets, *args, **kwargs) -> mm.ndarray` that perturbs the
            `(batch_count, nn_count, nn_count)` tensor `Kin` with `muygps`'s
            noise model before solving it against the
            `(batch_count, nn_count, response_count)` tensor `nn_targets`.
        """

        def analytic_scale_opt_fn(Kin, nn_targets, *args, **kwargs):
            scale = self._fn(muygps.noise.perturb(Kin), nn_targets, **kwargs)
            if np.array(self.val).size != 1:
                # iterative process only works for scalar responses
                return scale
            for _ in range(1, self.iteration_count):
                scale = 0.5 * (
                    scale
                    + self._fn(
                        scale * muygps.noise.perturb(Kin), nn_targets, **kwargs
                    )
                )
            return scale

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
            `(Kin, nn_targets, *args, **kwargs) -> mm.ndarray` that perturbs the
            `(batch_count, nn_count, nn_count)` tensor `Kin` with `muygps`'s
            noise model before solving it against the
            `(batch_count, nn_count, response_count)` tensor `nn_targets`.
        """

        def downsample_analytic_scale_opt_fn(Kin, nn_targets, *args, **kwargs):
            batch_count, nn_count, _ = Kin.shape
            if nn_count <= self._down_count:
                raise ValueError(
                    f"bad attempt to downsample {self._down_count} elements "
                    f"from a set of only {nn_count} options"
                )
            pK = muygps.noise.perturb(Kin)
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
                nn_targets_down = nn_targets[:, sampled_indices]
                scales.append(self._fn(pK_down, nn_targets_down))

            return np.median(scales, axis=0) / (self._down_count * batch_count)

        return downsample_analytic_scale_opt_fn
