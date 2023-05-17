# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
TensorHyperparameters

TensorHyperparameters specifications are expected to be provided in `Dict` form with
the key `"val"`. `"val"` is either an mm.ndarray value.
"""

from typing import Callable, Optional, Tuple, Type

import MuyGPyS._src.math as mm


class TensorHyperparameter:
    """
    A MuyGPs kernel or model Tensor Hyperparameter.

    TensorHyperparameters are defined solely by a value. Values must
    be scalar numeric arrays. Currently only used for heteroscedastic noise.

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
            raise ValueError(
                "TensorHyperparameter class does not support strings."
            )
        if not isinstance(val, mm.ndarray):
            raise ValueError(
                f"Non-array tensor hyperparameter value {val} is not allowed."
            )
        if self.fixed() is False:
            raise ValueError(
                "TensorHyperparameters objects do not support optimization."
            )
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
            "TensorHyperparameter does not support optimization bounds!"
        )

    def apply(self, fn: Callable, name: str) -> Callable:
        if self.fixed():

            def applied_fn(*args, **kwargs):
                kwargs.setdefault(name, self())
                return fn(*args, **kwargs)

            return applied_fn

        return fn

    def append_lists(self, name, names, params, bounds):
        pass


def _init_tensor_hyperparameter(
    val_def: mm.ndarray,
    type: Type = TensorHyperparameter,
    **kwargs,
) -> TensorHyperparameter:
    """
    Initialize a tensor hyperparameter given default value.

    Args:
        val:
            A valid value.
        kwargs:
            A hyperparameter dict including as subset of the key `val`.
    """
    val = kwargs.get("val", val_def)
    return type(val)
