# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Tensor-valued Hyperparameters

`TensorParam` specifications are expected to be provided as an `mm.ndarray`
value.
"""

from typing import Optional, Tuple

import MuyGPyS._src.math as mm

from MuyGPyS._src.math.numpy import ndarray as numpy_ndarray

try:
    from MuyGPyS._src.math.jax import ndarray as jax_ndarray
except Exception:
    from MuyGPyS._src.math.numpy import ndarray as jax_ndarray  # type: ignore
try:
    from MuyGPyS._src.math.torch import ndarray as torch_ndarray
except Exception:
    from MuyGPyS._src.math.numpy import ndarray as torch_ndarray  # type: ignore


class TensorParam:
    """
    A MuyGPs kernel or model Tensor Hyperparameter.

    TensorParam are defined solely by a value, which must be numeric arrays.
    Currently only used for heteroscedastic noise.

    Args:
        val:
            A mm.ndarray containing the value of the tensor hyperparameter
    """

    def __init__(
        self,
        val: mm.ndarray,
    ):
        """
        Initialize a tensor hyperparameter.
        """
        self._set_val(val)

    def _set(
        self,
        val: Optional[mm.ndarray] = None,
    ) -> None:
        """
        Reset hyperparameter value using keyword arguments.

        Args:
            val:
                A valid value.
        """
        if val is not None:
            self._set_val(val)

    def _set_val(self, val: mm.ndarray) -> None:
        """
        Set tensor hyperparameter value.

        Throws on out-of-range and other badness.

        Args:
            val:
                A valid mm.ndarray value.

        Raises:
            ValueError:
                A non-numeric, non-fixed, or string val
                will produce an error.
        """
        if isinstance(val, str):
            raise ValueError("TensorParam class does not support strings.")
        if not isinstance(val, mm.ndarray):
            if type(val) not in [numpy_ndarray, torch_ndarray, jax_ndarray]:
                raise ValueError(
                    f"Non-array tensor hyperparameter type {type(val)} is not "
                    f"allowed. Expected {mm.ndarray}"
                )
            else:
                import warnings

                warnings.warn(
                    f"Expected tensor hyperparameter type {mm.ndarray}, not "
                    f"{type(val)}. This is most likely not intended except in "
                    "backend tests"
                )
        if self.fixed() is False:
            raise ValueError("TensorParam objects do not support optimization.")
        self._val = val

    def __call__(self) -> mm.ndarray:
        """
        Value accessor.

        Returns:
            The current value of the tensor hyperparameter.
        """
        return self._val

    def fixed(self) -> bool:
        """
        Report whether the parameter is fixed, and is to be ignored during
        optimization. Always returns True for tensor hyperparameters.

        Returns:
            `True`.
        """
        return True

    def get_bounds(self) -> Tuple[float, float]:
        raise NotImplementedError(
            "TensorParam does not support optimization bounds!"
        )

    def append_lists(self, name, names, params, bounds):
        pass
