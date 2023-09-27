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
    ...     smoothness=Parameter("log_sample", (0.1, 2.5)),
    ...     deformation=Isotropy(
    ...         metric=l2,
    ...         length_scale=Parameter(1.0),
    ...     ),
    ... )

One uses a previously computed `pairwise_diffs` tensor (see
:func:`MuyGPyS.gp.tensor.pairwise_tensor`) to compute a kernel tensor whose
second two dimensions contain square kernel matrices. Similarly, one uses a
previously computed `crosswise_diffs` matrix (see
:func:`MuyGPyS.gp.tensor.crosswise_diffs`) to compute a cross-covariance
matrix. See the following example, which assumes that you have already
constructed the difference `numpy.ndarrays` and the kernel `kern` as shown
above.

Example:
    >>> K = kern(pairwise_diffs)
    >>> Kcross = kern(crosswise_diffs)
"""

from typing import Callable, Dict, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS.gp.deformation import DeformationFn


class KernelFn:
    """
    A kernel functor.

    Base class for kernel functors that include a hyperparameter Dict and a
    call mechanism.

    Args:
        kwargs:
            Ignored (by this base class) keyword arguments.
    """

    def __init__(
        self,
        deformation: DeformationFn,
    ):
        """
        Initialize dict holding hyperparameters.
        """
        self._hyperparameters: Dict = dict()
        self.deformation = deformation
        self._make_base()

    def _make_base(self):
        self.deformation.populate_length_scale(self._hyperparameters)

    def _make(self):
        raise NotImplementedError("_make is not implemented for base KernelFn")

    def set_params(self, **kwargs) -> None:
        """
        Reset hyperparameters using hyperparameter dict(s).

        Args:
            kwargs:
                Hyperparameter kwargs.
        """
        for name in kwargs:
            self._hyperparameters[name]._set(kwargs[name])

    def __call__(self, diffs: mm.ndarray, **kwargs) -> mm.ndarray:
        raise NotImplementedError(
            "__call__ is not implemented for base KernelFn"
        )

    def get_opt_fn(self) -> Callable:
        raise NotImplementedError(
            "get_opt_fn is not implemented for base KernelFn"
        )

    def get_opt_params(
        self,
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        """
        Report lists of unfixed hyperparameter names, values, and bounds.

        Returns
        -------
            names:
                A list of unfixed hyperparameter names.
            params:
                A list of unfixed hyperparameter values.
            bounds:
                A list of unfixed hyperparameter bound tuples.
        """
        return self.deformation.get_opt_params()

    def __str__(self) -> str:
        """
        Print state of hyperparameter dict.

        Intended only for testing purposes.
        """
        ret = ""
        for p in self._hyperparameters:
            param = self._hyperparameters[p]
            ret += f"{p} : {param()} - {param.get_bounds()}\n"
        return ret[:-1]
