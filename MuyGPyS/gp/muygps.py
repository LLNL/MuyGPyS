# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable, List, Tuple

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.fast_precompute import (
    _muygps_fast_posterior_mean_precompute,
    FastPrecomputeCoefficients,
)
from MuyGPyS.gp.fast_mean import _muygps_fast_posterior_mean, FastPosteriorMean
from MuyGPyS.gp.hyperparameter import FixedScale, ScaleFn
from MuyGPyS.gp.kernels import KernelFn
from MuyGPyS.gp.mean import _muygps_posterior_mean, PosteriorMean
from MuyGPyS.gp.noise import HomoscedasticNoise, NoiseFn
from MuyGPyS.gp.variance import _muygps_diagonal_variance, PosteriorVariance


@auto_str
class MuyGPS:
    """
    Local Kriging Gaussian Process.

    Performs approximate GP inference by locally approximating an observation's
    response using its nearest neighbors. Implements the MuyGPs algorithm as
    articulated in [muyskens2021muygps]_.

    Kernels accept different hyperparameter dictionaries specifying
    hyperparameter settings. Keys can include `val` and `bounds`. `bounds` must
    be either a len == 2  iterable container whose elements are scalars in
    increasing order, or the string `fixed`. If `bounds == fixed` (the default
    behavior), the hyperparameter value will remain fixed during optimization.
    `val` must be either a scalar (within the range of the upper and lower
    bounds if given) or the strings `"sample"` or `log_sample"`, which will
    randomly sample a value within the range given by the bounds.

    In addition to individual kernel hyperparamters, each MuyGPS object also
    possesses a noise model, possibly with parameters, and a vector of
    :math:`\\sigma^2` indicating the scale parameter associated with the
    posterior variance of each dimension of the response.

    Example:
        >>> from MuyGPyS.gp import MuyGPS
        >>> muygps = MuyGPS(
        ...    kernel=Matern(
        ...        smoothness=Parameter(0.38, (0.1, 2.5)),
        ...        deformation=Isotropy(
        ...            metric=F2,
        ...            length_scale=Parameter(0.2),
        ...        ),
        ...    ),
        ...    noise=HomoscedasticNoise(1e-5),
        ...    scale=AnalyticScale(),
        ... )

    MuyGPyS depends upon linear operations on specially-constructed tensors in
    order to efficiently estimate GP realizations. One can use (see their
    documentation for details) :func:`MuyGPyS.gp.tensors.pairwise_tensor` to
    construct pairwise difference tensors and
    :func:`MuyGPyS.gp.tensors.crosswise_tensor` to produce crosswise diff
    tensors that `MuyGPS` can then use to construct kernel tensors and
    cross-covariance matrices, respectively.

    We can easily realize kernel tensors using a `MuyGPS` object's `kernel`
    functor once we have computed a `pairwise_diffs` tensor and a
    `crosswise_diffs` matrix.

    Example:
        >>> K = muygps.kernel(pairwise_diffs)
        >>> Kcross = muygps.kernel(crosswise_diffs)


    Args:
        kernel:
            The kernel to be used. Defines :math:`K_\\theta(\\cdot, \\cdot)` as
            referenced in `MuyGPS` functions.
        noise:
            A noise model. Defines :math:`\\varepsilon` as referenced in
            `MuyGPS` functions.
        scale:
            A variance scale parameter. Defines :math:`\\sigma^2` as referenced
            in `MuyGPS` functions.
    """

    def __init__(
        self,
        kernel: KernelFn,
        noise: NoiseFn = HomoscedasticNoise(0.0, "fixed"),
        scale: ScaleFn = FixedScale(response_count=1),
        _backend_mean_fn: Callable = _muygps_posterior_mean,
        _backend_var_fn: Callable = _muygps_diagonal_variance,
        _backend_fast_mean_fn: Callable = _muygps_fast_posterior_mean,
        _backend_fast_precompute_fn: Callable = _muygps_fast_posterior_mean_precompute,
    ):
        self.kernel = kernel
        self.scale = scale
        self.noise = noise
        self._backend_mean_fn = _backend_mean_fn
        self._backend_var_fn = _backend_var_fn
        self._backend_fast_mean_fn = _backend_fast_mean_fn
        self._backend_fast_precompute_fn = _backend_fast_precompute_fn
        self._make()

    def _make(self) -> None:
        self.kernel._make()
        self._mean_fn = PosteriorMean(
            self.noise, _backend_fn=self._backend_mean_fn
        )
        self._var_fn = PosteriorVariance(
            self.noise, self.scale, _backend_fn=self._backend_var_fn
        )
        self._fast_posterior_mean_fn = FastPosteriorMean(
            _backend_fn=self._backend_fast_mean_fn
        )
        self._fast_precompute_fn = FastPrecomputeCoefficients(
            self.noise, _backend_fn=self._backend_fast_precompute_fn
        )

    def fixed(self) -> bool:
        """
        Checks whether all kernel and model parameters are fixed.

        This is a convenience utility to determine whether optimization is
        required.

        Returns:
            Returns `True` if all parameters are fixed, and `False` otherwise.
        """
        for p in self.kernel._hyperparameters:
            if not self.kernel._hyperparameters[p].fixed():
                return False
        if not self.noise.fixed():
            return False
        return True

    def get_opt_params(
        self,
    ) -> Tuple[List[str], mm.ndarray, mm.ndarray]:
        """
        Return lists of unfixed hyperparameter names, values, and bounds.

        Returns
        -------
            names:
                A list of unfixed hyperparameter names.
            params:
                A list of unfixed hyperparameter values.
            bounds:
                A list of unfixed hyperparameter bound tuples.
        """
        names, params, bounds = self.kernel.get_opt_params()
        self.noise.append_lists("noise", names, params, bounds)
        return names, mm.array(params), mm.array(bounds)

    def posterior_mean(
        self, K: mm.ndarray, Kcross: mm.ndarray, batch_nn_targets: mm.ndarray
    ) -> mm.ndarray:
        """
        Returns the posterior mean from the provided covariance,
        cross-covariance, and target tensors.

        Computes parallelized local solves of systems of linear equations using
        the last two dimensions of `K` along with `Kcross` and
        `batch_nn_targets` to predict responses in terms of the posterior mean.
        Assumes that kernel tensor `K` and cross-covariance
        matrix `Kcross` are already computed and given as arguments.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations, as computed in
        Equation (3.4) of [muyskens2021muygps]_. Given observation set
        :math:`X` with responses :math:`Y`, noise prior set
        :math:`\\varepsilon`, and kernel function
        :math:`K_\\theta(\\cdot, \\cdot)`, computes the following for each
        prediction element :math:`\\mathbf{z}_i` with nearest neighbors index
        set :math:`N_i`:

        .. math::
            \\widehat{Y} (\\mathbf{z}_i \\mid X_{N_i}) =
                \\sigma^2 K_\\theta (\\mathbf{z}_i, X_{N_i})
                (K_\\theta (X_{N_i}, X_{N_i}) + \\varepsilon_{N_i})^{-1}
                Y(X_{N_i}).

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)`-shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A matrix of shape `(batch_count, nn_count)` whose rows consist
                of `(1, nn_count)`-shaped cross-covariance vector corresponding
                to each of the batch elements and its nearest neighbors.
            batch_nn_targets:
                A tensor of shape `(batch_count, nn_count, response_count)`
                whose last dimension lists the vector-valued responses for the
                nearest neighbors of each batch element.

        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        return self._mean_fn(K, Kcross, batch_nn_targets)

    def posterior_variance(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
    ) -> mm.ndarray:
        """
        Returns the posterior variance from the provided covariance and
        cross-covariance tensors.

        Return the local posterior variances of each prediction, corresponding
        to the diagonal elements of a covariance matrix. Given observation set
        :math:`X` with responses :math:`Y`, noise prior set
        :math:`\\varepsilon`, and kernel function
        :math:`K_\\theta(\\cdot, \\cdot)`, computes the following for each
        prediction element :math:`\\mathbf{z}_i` with nearest neighbors index
        set :math:`N_i`:

        .. math::
            Var \\left (
                \\widehat{Y} (\\mathbf{z}_i \\mid X_{N_i})
            \\right) =
                \\sigma^2 \\left (
                    K_\\theta (\\mathbf{z}_i, \\mathbf{z}_i) -
                    K_\\theta (\\mathbf{z}_i, X_{N_i})
                    \\left (
                        K_\\theta (X_{N_i}, X_{N_i}
                    \\right ) + \\varepsilon_{N_i})^{-1}
                    K_\\theta (X_{N_i}, \\mathbf{z}_i)
                \\right ).

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)`-shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A matrix of shape `(batch_count, nn_count)` whose rows consist
                of `(1, nn_count)`-shaped cross-covariance vector corresponding
                to each of the batch elements and its nearest neighbors.

        Returns:
            A vector of shape `(batch_count, response_count)` consisting of the
            diagonal elements of the posterior variance.
        """
        return self._var_fn(K, Kcross)

    def fast_coefficients(
        self,
        K: mm.ndarray,
        train_nn_targets_fast: mm.ndarray,
    ) -> mm.ndarray:
        """
        Produces coefficient matrix for the fast posterior mean given in
        Equation (8) of [dunton2022fast]_.

        Given observation set :math:`X` with responses :math:`Y`, noise prior
        set :math:`\\varepsilon`, and kernel function
        :math:`K_\\theta(\\cdot, \\cdot)`, computes the following for
        each observation element :math:`\\mathbf{x}_i` with nearest neighbors
        index set :math:`N^*_i`, containing `i` and the indices of the
        `nn_count - 1` nearest neighbors of :math:`\\mathbf{x}_i`:

        .. math::
            C_i =
                \\left ( K_\\theta (X_{N_i}, X_{N_i} \\right )
                + \\varepsilon_{N_i})^{-1} Y(X_{N_i}).

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)`-shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A matrix of shape `(batch_count, nn_count)` whose rows consist
                of `(1, nn_count)`-shaped cross-covariance vector corresponding
                to each of the batch elements and its nearest neighbors.

        Returns:
            A matrix :math:`C` of shape `(train_count, nn_count)` whose rows are
            the precomputed coefficients for fast posterior mean inference.

        """

        return self._fast_precompute_fn(K, train_nn_targets_fast)

    def fast_posterior_mean(
        self,
        Kcross: mm.ndarray,
        coeffs_tensor: mm.ndarray,
    ) -> mm.ndarray:
        """
        Performs fast posterior mean inference using provided cross-covariance
        and precomputed coefficient matrix.

        Assumes that cross-covariance matrix `Kcross` is already computed and
        given as an argument.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations, as computed in
        Equation (9) of [dunton2022fast]_. Given the coefficients :math:`C`
        created by :func:`~MuyGPyS.gp.muygps.MuyGPS.fast_coefficients` and
        Equation (8) of [dunton2022fast]_, observation set
        :math:`X`, noise prior set :math:`\\varepsilon`, and kernel function
        :math:`K_\\theta(\\cdot, \\cdot)`, computes the following for each test
        point :math:`\\mathbf{z}` and index set :math:`N^*_i` containing the
        union of the index :math:`i` of the nearest neighbor
        :math:`\\mathbf{x}_i` of :math:`\\mathbf{z}` and the `nn_count - 1`
        nearest neighbors of :math:`\\mathbf{x}_i`:

        .. math::
            \\widehat{Y} \\left ( \\mathbf{z} \\mid X \\right ) =
                \\sigma^2 K_\\theta(\\mathbf{z}, X_{N^*_i}) C_i.

        Args:
            Kcross:
                A matrix of shape `(batch_count, nn_count)` whose rows consist
                of `(1, nn_count)`-shaped cross-covariance vector corresponding
                to each of the batch elements and its nearest neighbors.
            coeffs_tensor:
                A matrix of shape `(batch_count, nn_count, response_count)`
                whose rows are given by precomputed coefficients.


        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        return self._fast_posterior_mean_fn(Kcross, coeffs_tensor)

    def get_opt_mean_fn(self) -> Callable:
        """
        Return a posterior mean function for use in optimization.

        Assumes that optimization parameter literals will be passed as keyword
        arguments.

        Returns:
            A function implementing the posterior mean, where `noise` is either
            fixed or takes updating values during optimization. The function
            expects keyword arguments corresponding to current hyperparameter
            values for unfixed parameters.
        """
        return self._mean_fn.get_opt_fn()

    def get_opt_var_fn(self) -> Callable:
        """
        Return a posterior variance function for use in optimization.

        Assumes that optimization parameter literals will be passed as keyword
        arguments.

        Returns:
            A function implementing posterior variance, where `noise` is either
            fixed or takes updating values during optimization. The function
            expects keyword arguments corresponding to current hyperparameter
            values for unfixed parameters.
        """
        return self._var_fn.get_opt_fn()

    def optimize_scale(
        self, pairwise_diffs: mm.ndarray, nn_targets: mm.ndarray
    ):
        """
        Optimize the value of the :math:`sigma^2` scale parameter.

        Uses the optimization method specified by the type of the `scale`
        parameter to optimize its value.

        Args:
            pairwise_diffs:
                A tensor of shape
                `(batch_count, nn_count, nn_count, feature_count)` containing
                the `(nn_count, nn_count, feature_count)`-shaped pairwise
                nearest neighbor difference tensors corresponding to each of the
                batch elements.
            nn_targets:
                Tensor of floats of shape
                `(batch_count, nn_count, response_count)` containing the
                expected response for each nearest neighbor of each batch
                element.

        Returns:
            A reference to this model with a freshly-optimized `scale`
            parameter.
        """
        K = self.kernel(pairwise_diffs)
        opt_fn = self.scale.get_opt_fn(self)
        self.scale._set(opt_fn(K, nn_targets))
        self._make()
        return self

    def __eq__(self, rhs) -> bool:
        if isinstance(rhs, self.__class__):
            return all(
                (
                    all(
                        self.kernel._hyperparameters[h]()
                        == rhs.kernel._hyperparameters[h]()
                        for h in self.kernel._hyperparameters
                    ),
                    self.noise() == rhs.noise(),
                    self.scale() == rhs.scale(),
                )
            )
        else:
            return False
