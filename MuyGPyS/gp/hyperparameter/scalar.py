# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Scalar Hyperparameter

Hyperparameters specifications are expected to provide a numeric scalar or
string `val` a 2-tuple or string `bounds`. `bounds` is either a 2-tuple
indicating a lower and upper optimization bound or the string `"fixed"`, which
exempts the hyperparameter from optimization (`"fixed"` is the default behavior
if `bounds` is unspecified). `val` is either a floating point value (within the
range between the upper and lower bounds if specified) or the strings "sample"
or "log_sample", which sample a guess between the provided lower and upper
bounds.
"""

from collections.abc import Sequence
from numbers import Number
from typing import Callable, cast, List, Tuple, Union

import MuyGPyS._src.math.numpy as np
import MuyGPyS._src.math as mm
from MuyGPyS import config
from MuyGPyS._src.mpi_utils import _is_mpi_mode


class Parameter:
    """
    A MuyGPs kernel or model Hyperparameter. Also called `ScalarParam`.

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
        bounds: Union[str, Tuple[float, float]] = "fixed",
    ):
        """
        Initialize a hyperparameter.
        """
        self._set_bounds(bounds)
        self._set_val(val)

    def __str__(self, **kwargs):
        bstring = "fixed" if self._fixed is True else self._bounds
        return f"{type(self).__name__}({self._val}, {bstring})"

    def _set(self, rhs) -> None:
        """
        Reset hyperparameter value and/or bounds using keyword arguments.

        Args:
            rhs:
                Another hyperparameter.
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
        self._val = rhs._val
        self._bounds = rhs._bounds
        self._fixed = rhs._fixed

    def _sample_val(self, val: str) -> float:
        if self.fixed() is True:
            if isinstance(val, str):
                raise ValueError(
                    f"Fixed bounds do not support string value ({val}) prompts."
                )
        if val == "sample":
            newval = float(
                np.random.uniform(low=self._bounds[0], high=self._bounds[1])
            )
        elif val == "log_sample":
            newval = float(
                np.exp(
                    np.random.uniform(
                        low=np.log(self._bounds[0]),
                        high=np.log(self._bounds[1]),
                    )
                )
            )
        else:
            raise ValueError(f"Unsupported string hyperparameter value {val}.")
        if _is_mpi_mode() is True:
            newval = config.mpi_state.comm_world.bcast(newval, root=0)
        return newval

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
            val = self._sample_val(val)
        if isinstance(val, Sequence) or hasattr(val, "__len__"):
            raise ValueError(
                f"Nonscalar hyperparameter value {val} is not allowed."
            )
        if not isinstance(val, mm.ndarray):
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
            if any_below:
                raise ValueError(
                    f"Hyperparameter value {val} is lesser than the "
                    f"optimization lower bound {self._bounds[0]}"
                )
            if any_above:
                raise ValueError(
                    f"Hyperparameter value {val} is greater than the "
                    f"optimization upper bound {self._bounds[1]}"
                )
        self._val = mm.parameter(val)

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

    def apply_fn(self, fn: Callable, name: str) -> Callable:
        def applied_fn(*args, **kwargs):
            kwargs.setdefault(name, self())
            return fn(*args, **kwargs)

        return applied_fn

    def append_lists(
        self,
        name: str,
        names: List[str],
        params: List[float],
        bounds: List[Tuple[float, float]],
    ):
        if not self.fixed():
            names.append(name)
            params.append(self())
            bounds.append(self.get_bounds())
