#!/usr/bin/env python
# encoding: utf-8
"""
@file kernels.py

Created by priest2 on 2020-11-10

Implements various kernels.
"""

import numpy as np


def cosine(x_locs, z_locs=None):
    x_diags = np.sum(x_locs ** 2, axis=1)
    if z_locs is None:
        z_locs = x_locs
        z_diags = x_diags
    else:
        z_diags = np.sum(z_locs ** 2, axis=1)
        if x_locs.shape[1] != z_locs.shape[1]:
            raise ValueError(
                f"x_locs shape {x_locs.shape} is incompatible with y_locs shape"
                f" {y_locs.shape}"
            )
    cross_diff_tensor = 1 - np.einsum(
        "bj, bij -> bi",
        x_locs[batch_indices],
        z_locs[batch_nn_indices],
    )
    cross_diag_tensor = np.einsum(
        "b, bi -> bi",
        np.sqrt(train_diags[batch_indices]),
        np.sqrt(train_diags[batch_nn_indices]),
    )
    cross_dist_tensor = cross_diff_tensor / cross_diag_tensor
    return dist_tensor
