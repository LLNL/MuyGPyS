# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

from typing import Callable


def _make_backend_loss(loss_fn: Callable, make_predict_and_loss_fn: Callable):
    def new_loss_fn(*args, **kwargs):
        return loss_fn(*args, **kwargs)

    def new_make_predict_and_loss_fn(*args, **kwargs):
        return make_predict_and_loss_fn(new_loss_fn, *args, **kwargs)

    new_loss_fn.make_predict_and_loss_fn = new_make_predict_and_loss_fn  # type: ignore
    return new_loss_fn
