# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Hyperparameters and kernel functors

Defines kernel functors (inheriting
:class:`~MuyGPyS.gp.kernels.kernel_fn.KernelFn`) that transform crosswise
difference tensors into cross-covariance matrices and pairwise difference
matrices into covariance or kernel tensors.

See the following example to initialize an :class:`MuyGPyS.gp.kernels.Matern`
object. Other kernel functors are similar, but require different
hyperparameters.

Example:
    >>> from MuyGPyS.gp.kernels import Matern
    >>> kern = Matern(
    ...         nu = {"val": "log_sample", "bounds": (0.1, 2.5)},
    ...         length_scale = {"val": 7.2},
    ...         metric = "l2",
    ... }

One uses a previously computed `pairwise_diffs` tensor (see
:func:`MuyGPyS.gp.tensor.pairwise_tensor`) to compute a kernel tensor whose
second two dimensions contain square kernel matrices. Similarly, one uses a
previously computed `crosswise_diffs` matrix (see
:func:`MuyGPyS.gp.tensor.crosswise_diffs`) to compute a cross-covariance
matrix. See the following example, which assumes that you have already
constructed the difference `numpy.nparrays` and the kernel `kern` as shown
above.

Example:
    >>> K = kern(pairwise_diffs)
    >>> Kcross = kern(crosswise_diffs)
"""

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _F2, _l2


class IsotropicDistortion:
    def __init__(self, metric):
        if metric == "l2":
            self._dist_fn = _l2
        elif metric == "F2":
            self._dist_fn = _F2
        else:
            raise ValueError(f"Metric {metric} is not supported!")

    def __call__(self, diffs):
        return self._dist_fn(diffs)


class NullDistortion:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("NullDistortion cannot be called!")


def apply_distortion(distortion_fn):
    def distortion_appier(fn):
        def distorted_fn(diffs, *args, **kwargs):
            return fn(distortion_fn(diffs), *args, **kwargs)

        return distorted_fn

    return distortion_appier


def embed_with_distortion_model(fn, distortion_fn):
    if isinstance(distortion_fn, IsotropicDistortion):
        return apply_distortion(distortion_fn)(fn)
    elif isinstance(distortion_fn, NullDistortion):
        return fn
    else:
        raise ValueError(f"Noise model {type(distortion_fn)} is not supported!")


class KernelFn:
    """
    A kernel functor.

    Base class for kernel functors that include a hyperparameter Dict and a
    call mechanism.

    Args:
        kwargs:
            Ignored (by this base class) keyword arguments.
    """

    def __init__(self, metric="l2", **kwargs):
        """
        Initialize dict holding hyperparameters.
        """
        self.hyperparameters = dict()
        self.metric = metric
        if self.metric is None:
            self._distortion_fn = NullDistortion()
        else:
            self._distortion_fn = IsotropicDistortion(self.metric)

    def set_params(self, **kwargs) -> None:
        """
        Reset hyperparameters using hyperparameter dict(s).

        Args:
            kwargs:
                Hyperparameter kwargs.
        """
        for name in kwargs:
            self.hyperparameters[name]._set(**kwargs[name])

    def __call__(self, diffs: mm.ndarray) -> mm.ndarray:
        raise NotImplementedError(
            f"__call__ is not implemented for base KernelFn"
        )

    def get_optim_params(
        self,
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        raise NotImplementedError(
            f"get_optim_params is not implemented for base KernelFn"
        )

    def get_opt_fn(self) -> Callable:
        raise NotImplementedError(
            f"get_opt_fn is not implemented for base KernelFn"
        )

    def __str__(self) -> str:
        """
        Print state of hyperparameter dict.

        Intended only for testing purposes.
        """
        ret = ""
        for p in self.hyperparameters:
            param = self.hyperparameters[p]
            ret += f"{p} : {param()} - {param.get_bounds()}\n"
        return ret[:-1]
