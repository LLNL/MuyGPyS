# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Hyperparameters

Hyperparameters specifications are expected to be provided in `Dict` form with
the keys `"val"` and `"bounds"`. `"bounds"` is either a 2-tuple indicating a
lower and upper optimization bound or the string `"fixed"`, which exempts the
hyperparameter from optimization (`"fixed"` is the default behavior if `bounds`
is unspecified). `"val"` is either a floating point value (within the range
between the upper and lower bounds if specified).
"""

from collections.abc import Sequence
from numbers import Number
from typing import Callable, cast, List, Optional, Tuple, Type, Union

from MuyGPyS import config

import MuyGPyS._src.math.numpy as np
from MuyGPyS._src.mpi_utils import _is_mpi_mode


class Hyperparameter:
    """
    A MuyGPs kernel or model Hyperparameter.

    Hyperparameters are defined by a value and optimization bounds. Values must
    be scalar numeric types, and bounds are either a len == 2 iterable container
    whose elements are numeric scalars in increasing order, or the string
    `fixed`. If `bounds == "fixed"` (the default behavior), the hyperparameter
    value will remain fixed during optimization. `val` must remain within the
    range of the upper and lower bounds, if not `fixed`.

    Args:
        val:
            A scalar within the range of the upper and lower bounds (if given).
            val can also be the strings `"sample"` or `"log_sample"`, which will
            result in randomly sampling a value within the range given by the
            bounds.
        bounds:
            Iterable container of len 2 containing lower and upper bounds (in
            that order), or the string `"fixed"`.

    Raises:
        ValueError:
            Any `bounds` string other than `"fixed"` will produce an error.
        ValueError:
            A non-iterable non-string type for `bounds` will produce an
            error.
        ValueError:
            A `bounds` iterable of len other than 2 will produce an error.
        ValueError:
            Iterable `bounds` values of non-numeric types will produce an error.
        ValueError:
            A lower bound that is not less than an upper bound will produce
            an error.
        ValueError:
            `val == "sample" or val == "log_sample"` will produce an error
            if `self._bounds == "fixed"`.
        ValueError:
            Any string other than `"sample"` or `"log_sample"` will produce
            an error.
        ValueError:
            A `val` outside of the range specified by `self._bounds` will
            produce an error.
    """

    def __init__(
        self,
        val: Union[str, float],
        bounds: Union[str, Tuple[float, float]],
    ):
        """
        Initialize a hyperparameter.
        """
        self._set_bounds(bounds)
        self._set_val(val)

    def _set(
        self,
        val: Optional[Union[str, float]] = None,
        bounds: Optional[Union[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Reset hyperparameter value and/or bounds using keyword arguments.

        Args:
            val:
                A valid value or `"sample"` or `"log_sample"`.
            bounds:
                Iterable container of len 2 containing lower and upper bounds
                (in that order), or the string `"fixed"`.
        Raises:
            ValueError:
                Any `bounds` string other than `"fixed"` will produce an error.
            ValueError:
                A non-numeric, non-numeric, or non-string type for `bounds` will
                produce an error.
            ValueError:
                A `bounds` iterable of len other than 2 will produce an error.
            ValueError:
                Iterable `bounds` values of non-numeric types will produce an
                error.
            ValueError:
                A lower bound that is not less than an upper bound will produce
                an error.
            ValueError:
                `val == "sample" or val == "log_sample"` will produce an error
                if `self._bounds == "fixed"`.
            ValueError:
                Any string other than `"sample"` or `"log_sample"` will produce
                an error.
            ValueError:
                A `val` outside of the range specified by `self._bounds` will
                produce an error.
        """
        if bounds is not None:
            self._set_bounds(bounds)
        if val is not None:
            self._set_val(val)

    def _set_val(self, val: Union[str, float]) -> None:
        """
        Set hyperparameter value; sample if appropriate.

        Throws on out-of-range and other badness.

        Args:
            val:
                A valid scalar value or the strings `"sample"` or
                `"log_sample"`.

        Raises:
            ValueError:
                A non-scalar, non-numeric and non-string val will produce an
                error.
            ValueError:
                `val == "sample" or val == "log_sample"` will produce an error
                if `bounds == "fixed"`.
            ValueError:
                Any `val` string other than `"sample"` or `"log_sample"` will
                produce an error.
            ValueError:
                A `val` outside of the range specified by `bounds` will
                produce an error.
        """
        if isinstance(val, str):
            if self.fixed() is True:
                if isinstance(val, str):
                    raise ValueError(
                        f"Fixed bounds do not support string value ({val}) prompts."
                    )
            if val == "sample":
                val = float(
                    np.random.uniform(low=self._bounds[0], high=self._bounds[1])
                )
                if _is_mpi_mode() is True:
                    val = config.mpi_state.comm_world.bcast(val, root=0)
            elif val == "log_sample":
                val = float(
                    np.exp(
                        np.random.uniform(
                            low=np.log(self._bounds[0]),
                            high=np.log(self._bounds[1]),
                        )
                    )
                )
                if _is_mpi_mode() is True:
                    val = config.mpi_state.comm_world.bcast(val, root=0)
            else:
                raise ValueError(
                    f"Unsupported string hyperparameter value {val}."
                )
        if isinstance(val, Sequence) or hasattr(val, "__len__"):
            raise ValueError(
                f"Nonscalar hyperparameter value {val} is not allowed."
            )
        val = float(val)
        if self.fixed() is False:
            any_below = np.any(
                np.choose(
                    cast(float, val) < cast(float, self._bounds[0]) - 1e-5,
                    [False, True],
                )
            )
            any_above = np.any(
                np.choose(
                    cast(float, val) > cast(float, self._bounds[1]) + 1e-5,
                    [False, True],
                )
            )
            if any_below == True:
                raise ValueError(
                    f"Hyperparameter value {val} is lesser than the "
                    f"optimization lower bound {self._bounds[0]}"
                )
            if any_above == True:
                raise ValueError(
                    f"Hyperparameter value {val} is greater than the "
                    f"optimization upper bound {self._bounds[1]}"
                )
        self._val = val

    def _set_bounds(
        self,
        bounds: Union[str, Tuple[float, float]],
    ) -> None:
        """
        Set hyperparameter bounds.

        Args:
            bounds:
                Iterable container of len 2 containing lower and upper bounds
                (in that order), or the string `"fixed"`.

        Raises:
            ValueError:
                Any string other than `"fixed"` will produce an error.
            ValueError:
                A non-iterable type will produce an error.
            ValueError:
                An iterable of len other than 2 will produce an error.
            ValueError:
                Iterable values of non-numeric types will produce an error.
            ValueError:
                A lower bound that is not less than an upper bound will produce
                an error.
        """
        if isinstance(bounds, str) is True:
            if bounds == "fixed":
                self._bounds = (0.0, 0.0)  # default value
                self._fixed = True
            else:
                raise ValueError(f"Unknown bound option {bounds}.")
        else:
            if hasattr(bounds, "__iter__") is not True:
                raise ValueError(
                    f"Unknown bound optiom {bounds} of a non-iterable type "
                    f"{type(bounds)}."
                )
            if len(bounds) != 2:
                raise ValueError(
                    f"Provided hyperparameter optimization bounds have "
                    f"unsupported length {len(bounds)}."
                )
            if isinstance(bounds[0], Number) is not True:
                raise ValueError(
                    f"Nonscalar {bounds[0]} of type {type(bounds[0])} is not a "
                    f"supported hyperparameter bound type."
                )
            if isinstance(bounds[1], Number) is not True:
                raise ValueError(
                    f"Nonscalar {bounds[1]} of type {type(bounds[1])} is not a "
                    f"supported hyperparameter bound type."
                )
            bounds = (float(bounds[0]), float(bounds[1]))
            if bounds[0] > bounds[1]:
                raise ValueError(
                    f"Lower bound {bounds[0]} is not lesser than upper bound "
                    f"{bounds[1]}."
                )
            self._bounds = bounds
            self._fixed = False

    def __call__(self) -> float:
        """
        Value accessor.

        Returns:
            The current value of the hyperparameter.
        """
        return self._val

    def get_bounds(self) -> Tuple[float, float]:
        """
        Bounds accessor.

        Returns:
            The lower and upper bound tuple.
        """
        return self._bounds

    def fixed(self) -> bool:
        """
        Report whether the parameter is fixed, and is to be ignored during
        optimization.

        Returns:
            `True` if fixed, `False` otherwise.
        """
        return self._fixed


def _init_hyperparameter(
    val_def: Union[str, float],
    bounds_def: Union[str, Tuple[float, float]],
    type: Type = Hyperparameter,
    **kwargs,
) -> Hyperparameter:
    """
    Initialize a hyperparameter given default values.

    Args:
        val:
            A valid value or `"sample"` or `"log_sample"`.
        bounds:
            Iterable container of len 2 containing lower and upper bounds (in
            that order), or the string `"fixed"`.
        kwargs:
            A hyperparameter dict including as subset of the keys `val` and
            `bounds`.
    """
    val = kwargs.get("val", val_def)
    bounds = kwargs.get("bounds", bounds_def)
    return type(val, bounds)


def apply_hyperparameter(fn: Callable, param: Hyperparameter, name: str):
    if param.fixed():

        def applied_fn(*args, **kwargs):
            kwargs.setdefault(name, param())
            return fn(*args, **kwargs)

        return applied_fn

    return fn


def append_optim_params_lists(
    param: Hyperparameter,
    name: str,
    names: List[str],
    params: List[float],
    bounds: List[Tuple[float, float]],
):
    if not param.fixed():
        names.append(name)
        params.append(param())
        bounds.append(param.get_bounds())
