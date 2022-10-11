# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience functions for optimizing the `sigma_sq` parameter of
:class:`MuyGPyS.gp.muygps.MuyGPS` objects.

Currently only supports an analytic approximation, but will support other
methods in the future.
"""

import numpy as np

from copy import deepcopy
from typing import Optional, Union

from MuyGPyS.gp.muygps import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS._src.optimize.sigma_sq import _analytic_sigma_sq_optim


def muygps_sigma_sq_optim(
    muygps: MuyGPS,
    pairwise_dists: np.ndarray,
    nn_targets: np.ndarray,
    sigma_method: Optional[str] = "analytic",
) -> MuyGPS:
    """
    Optimize the value of the :math:`\\sigma^2` scale parameter for each
    response dimension.

    The optimization to be applied depends upon the value of `sigma_method`.

    Args:
        muygps:
            The model to be optimized.
        pairwise_dists:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing the
            `(nn_count, nn_count)`-shaped pairwise nearest neighbor distance
            matrices corresponding to each of the batch elements.
        nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.
        sigma_method:
            The optimization method to apply. Currently only supports
            `"analytic"` and `None`.

    Returns:
        A new MuyGPs model whose sigma_sq parameter has been optimized.
    """
    if sigma_method is None:
        return muygps

    sigma_method = sigma_method.lower()
    if sigma_method == "analytic":
        return muygps_analytic_sigma_sq_optim(
            muygps, pairwise_dists, nn_targets
        )
    else:
        raise ValueError(f"Unrecognized sigma_method {sigma_method}")


def mmuygps_sigma_sq_optim(
    mmuygps: MMuyGPS,
    pairwise_dists: np.ndarray,
    nn_targets: np.ndarray,
    sigma_method: Optional[str] = "analytic",
) -> MMuyGPS:
    """
    Optimize the value of the :math:`\\sigma^2` scale parameter for each
    response dimension of a MultivariateMuyGPS object.

    The optimization to be applied depends upon the value of `sigma_method`.

    Args:
        mmuygps:
            The model to be optimized.
        pairwise_dists:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing the
            `(nn_count, nn_count)`-shaped pairwise nearest neighbor distance
            matrices corresponding to each of the batch elements.
        nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.
        sigma_method:
            The optimization method to apply. Currently only supports
            `"analytic"` and `None`.

    Returns:
        A new MultivariateMuyGPs model whose sigma_sq parameter has been
        optimized.
    """
    if sigma_method is None:
        return mmuygps

    sigma_method = sigma_method.lower()
    if sigma_method == "analytic":
        return mmuygps_analytic_sigma_sq_optim(
            mmuygps, pairwise_dists, nn_targets
        )
    else:
        raise ValueError(f"Unrecognized sigma_method {sigma_method}")


def muygps_analytic_sigma_sq_optim(
    muygps: MuyGPS, pairwise_dists: np.ndarray, nn_targets: np.ndarray
) -> MuyGPS:
    """
    Optimize the value of the :math:`\\sigma^2` scale parameter for each
    response dimension.

    We approximate :math:`\\sigma^2` by way of averaging over the analytic
    solution from each local kernel.

    .. math::
        \\sigma^2 = \\frac{1}{bk} * \\sum_{i \\in B}
                    Y_{nn_i}^T K_{nn_i}^{-1} Y_{nn_i}

    Here :math:`Y_{nn_i}` and :math:`K_{nn_i}` are the target and kernel
    matrices with respect to the nearest neighbor set in scope, where
    :math:`k` is the number of nearest neighbors and :math:`b = |B|` is the
    number of batch elements considered.

    Args:
        muygps:
            The model to be optimized.
        pairwise_dists:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing the
            `(nn_count, nn_count)`-shaped pairwise nearest neighbor distance
            matrices corresponding to each of the batch elements.
        nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.

    Returns:
        A new MuyGPs model whose sigma_sq parameter has been optimized.
    """
    K = muygps.kernel(pairwise_dists)
    ret = deepcopy(muygps)
    ret.sigma_sq._set(_analytic_sigma_sq_optim(K, nn_targets, ret.eps()))
    return ret


def mmuygps_analytic_sigma_sq_optim(
    mmuygps: MMuyGPS, pairwise_dists: np.ndarray, nn_targets: np.ndarray
) -> MMuyGPS:
    """
    Optimize the value of the :math:`\\sigma^2` scale parameter for each
    response dimension.

    We approximate :math:`\\sigma^2` by way of averaging over the analytic
    solution from each local kernel.

    .. math::
        \\sigma^2 = \\frac{1}{bk} * \\sum_{i \\in B}
                    Y_{nn_i}^T K_{nn_i}^{-1} Y_{nn_i}

    Here :math:`Y_{nn_i}` and :math:`K_{nn_i}` are the target and kernel
    matrices with respect to the nearest neighbor set in scope, where
    :math:`k` is the number of nearest neighbors and :math:`b = |B|` is the
    number of batch elements considered.

    Args:
        muygps:
            The model to be optimized.
        pairwise_dists:
            A tensor of shape `(batch_count, nn_count, nn_count)` containing the
            `(nn_count, nn_count)`-shaped pairwise nearest neighbor distance
            matrices corresponding to each of the batch elements.
        nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.

    Returns:
        A new MuyGPs model whose sigma_sq parameter has been optimized.
    """
    ret = deepcopy(mmuygps)
    batch_count, nn_count, response_count = nn_targets.shape
    if response_count != len(ret.models):
        raise ValueError(
            f"Response count ({response_count}) does not match the number "
            f"of models ({len(ret.models)})."
        )

    K = np.zeros((batch_count, nn_count, nn_count))
    sigma_sqs = np.zeros((response_count,))
    for i, model in enumerate(ret.models):
        K = model.kernel(pairwise_dists)
        sigma_sqs[i] = _analytic_sigma_sq_optim(
            K,
            nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            model.eps(),
        )
        model.sigma_sq._set(np.atleast_1d(sigma_sqs[i]))
    ret.sigma_sq._set(sigma_sqs)
    return ret
