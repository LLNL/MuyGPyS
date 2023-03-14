# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Hyperparameters and kernel functors

Defines kernel functors (inheriting
:class:`~MuyGPyS.gp.kernels.kernel_fn.KernelFn`) that transform crosswise
distance matrices into cross-covariance matrices and pairwise distance matrices
into covariance or kernel matrices.

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

One uses a previously computed `pairwise_dists` tensor (see
:func:`MuyGPyS.gp.distance.pairwise_distance`) to compute a kernel tensor whose
second two dimensions contain square kernel matrices. Similarly, one uses a
previously computed `crosswise_dists` matrix (see
:func:`MuyGPyS.gp.distance.crosswise_distance`) to compute a cross-covariance
matrix. See the following example, which assumes that you have already
constructed the distance `numpy.nparrays` and the kernel `kern` as shown above.

Example:
    >>> K = kern(pairwise_dists)
    >>> Kcross = kern(crosswise_dists)
"""

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm


class KernelFn:
    """
    A kernel functor.

    Base class for kernel functors that include a hyperparameter Dict and a
    call mechanism.

    Args:
        kwargs:
            Ignored (by this base class) keyword arguments.
    """

    def __init__(self, **kwargs):
        """
        Initialize dict holding hyperparameters.
        """
        self.hyperparameters = dict()
        self.metric = ""

    def set_params(self, **kwargs) -> None:
        """
        Reset hyperparameters using hyperparameter dict(s).

        Args:
            kwargs:
                Hyperparameter kwargs.
        """
        for name in kwargs:
            self.hyperparameters[name]._set(**kwargs[name])

    def __call__(self, dists: mm.ndarray) -> mm.ndarray:
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
