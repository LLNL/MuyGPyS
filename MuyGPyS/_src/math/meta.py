# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from MuyGPyS import config


def fix_type(dtype):
    def typed_fn(fn):
        def fn_wrapper(*args, **kwargs):
            kwargs.setdefault("dtype", dtype)
            return fn(*args, **kwargs)

        return fn_wrapper

    return typed_fn


def fix_function_type(dtype, fn):
    return fix_type(dtype)(fn)


def fix_function_types(dtype, *fns):
    return tuple(fix_function_type(dtype, fn) for fn in fns)


def wrap_torch_signature(fn):
    def wrapped_fn(x, axis=0):
        return fn(x, dim=axis)

    return wrapped_fn


def wrap_torch_signatures(*fns):
    return tuple(wrap_torch_signature(fn) for fn in fns)


def set_type(t64, t32):
    if config.state.ftype == "64":
        return t64
    elif config.state.ftype == "32":
        return t32
