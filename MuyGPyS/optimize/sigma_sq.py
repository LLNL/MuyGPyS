# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Convenience functions for optimizing the `sigma_sq` parameter of
:class:`~MuyGPyS.gp.muygps.MuyGPS` objects.

Currently only supports an analytic approximation, but will support other
methods in the future.
"""

from copy import deepcopy
from typing import Callable, Optional

import MuyGPyS._src.math as mm
from MuyGPyS.gp import MuyGPS, MultivariateMuyGPS as MMuyGPS
from MuyGPyS._src.optimize.sigma_sq import _analytic_sigma_sq_optim
from MuyGPyS.optimize.utils import _switch_on_sigma_method
from MuyGPyS.gp.noise.perturbation import select_perturb_fn


def muygps_sigma_sq_optim(
    muygps: MuyGPS,
    pairwise_diffs: mm.ndarray,
    nn_targets: mm.ndarray,
    sigma_method: Optional[str] = "analytic",
) -> MuyGPS:
    """
    Optimize the value of the :math:`\\sigma^2` scale parameter for each
    response dimension.

    The optimization to be applied depends upon the value of `sigma_method`.

    Args:
        muygps:
            The model to be optimized.
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
            containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
            nearest neighbor difference tensors corresponding to each of the
            batch elements.
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
    return _switch_on_sigma_method(
        sigma_method,
        muygps_analytic_sigma_sq_optim,
        lambda muygps, pairwise_diffs, nn_targets: muygps,
        muygps,
        pairwise_diffs,
        nn_targets,
    )


def mmuygps_sigma_sq_optim(
    mmuygps: MMuyGPS,
    pairwise_diffs: mm.ndarray,
    nn_targets: mm.ndarray,
    sigma_method: Optional[str] = "analytic",
) -> MMuyGPS:
    """
    Optimize the value of the :math:`\\sigma^2` scale parameter for each
    response dimension of a MultivariateMuyGPS object.

    The optimization to be applied depends upon the value of `sigma_method`.

    Args:
        mmuygps:
            The model to be optimized.
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
            containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
            nearest neighbor difference tensors corresponding to each of the
            batch elements.
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
    return _switch_on_sigma_method(
        sigma_method,
        mmuygps_analytic_sigma_sq_optim,
        lambda mmuygps, pairwise_diffs, nn_targets: mmuygps,
        mmuygps,
        pairwise_diffs,
        nn_targets,
    )


def make_sigma_sq_optim(
    sigma_method: Optional[str], muygps: MuyGPS
) -> Callable:
    return _switch_on_sigma_method(
        sigma_method,
        make_analytic_sigma_sq_optim,
        make_none_sigma_sq_optim,
        muygps,
        _analytic_sigma_sq_optim,
        select_perturb_fn(muygps.eps),
    )


def make_none_sigma_sq_optim(muygps: MuyGPS, *args) -> Callable:
    return lambda: lambda K, nn_targets, **kwargs: muygps.sigma_sq()


def make_analytic_sigma_sq_optim(
    muygps: MuyGPS, analytic_optim_fn, perturb_fn: Callable
) -> Callable:
    if not muygps.eps.fixed():

        def ss_opt_fn(K, nn_targets, **kwargs):
            return analytic_optim_fn(perturb_fn(K, kwargs["eps"]), nn_targets)

    else:

        def ss_opt_fn(K, nn_targets, **kwargs):
            return analytic_optim_fn(perturb_fn(K, muygps.eps()), nn_targets)

    return ss_opt_fn


def muygps_analytic_sigma_sq_optim(
    muygps: MuyGPS, pairwise_diffs: mm.ndarray, nn_targets: mm.ndarray
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
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
            containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
            nearest neighbor difference tensors corresponding to each of the
            batch elements.
        nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, response_count)`
            containing the expected response for each nearest neighbor of each
            batch element.

    Returns:
        A new MuyGPs model whose sigma_sq parameter has been optimized.
    """
    ret = deepcopy(muygps)
    perturb_fn = select_perturb_fn(ret.eps)
    K = ret.kernel(pairwise_diffs)
    ss = _analytic_sigma_sq_optim(perturb_fn(K, ret.eps()), nn_targets)
    ret.sigma_sq._set(ss)
    return ret


def mmuygps_analytic_sigma_sq_optim(
    mmuygps: MMuyGPS, pairwise_diffs: mm.ndarray, nn_targets: mm.ndarray
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
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, feature_count)`
            containing the `(nn_count, nn_count, feature_count)`-shaped pairwise
            nearest neighbor difference tensors corresponding to each of the
            batch elements.
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

    sigma_sqs = mm.zeros((response_count,))
    for i, model in enumerate(ret.models):
        perturb_fn = select_perturb_fn(model.eps)
        K = model.kernel(pairwise_diffs)
        new_sigma_val = _analytic_sigma_sq_optim(
            perturb_fn(K, model.eps()),
            nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
        )[0]
        sigma_sqs = mm.assign(sigma_sqs, new_sigma_val, i)
        model.sigma_sq._set(mm.atleast_1d(sigma_sqs[i]))
    ret.sigma_sq._set(sigma_sqs)
    return ret
