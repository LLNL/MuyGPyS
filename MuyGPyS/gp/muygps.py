# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable, List, Tuple, Union
from copy import deepcopy

import MuyGPyS._src.math as mm
from MuyGPyS._src.util import auto_str
from MuyGPyS.gp.kernels import (
    Matern,
    RBF,
)
from MuyGPyS.gp.mean import PosteriorMean
from MuyGPyS.gp.sigma_sq import SigmaSq
from MuyGPyS.gp.variance import PosteriorVariance
from MuyGPyS.gp.noise import HeteroscedasticNoise, HomoscedasticNoise, NullNoise
from MuyGPyS.gp.fast_mean import FastPosteriorMean
from MuyGPyS.gp.fast_precompute import FastPrecomputeCoefficients


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
    possesses a homoscedastic :math:`\\varepsilon` noise parameter and a
    vector of :math:`\\sigma^2` indicating the scale parameter associated
    with the posterior variance of each dimension of the response.

    :math:`\\sigma^2` is the only parameter assumed to be a training target by
    default, and is treated differently from all other hyperparameters. All
    other training targets must be manually specified in `k_kwargs`.

    Example:
        >>> from MuyGPyS.gp import MuyGPS
        >>> k_kwargs = {
        ...         "kern": "rbf",
        ...         "metric": "F2",
        ...         "eps": {"val": 1e-5},
        ...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
        ...         "length_scale": {"val": 7.2},
        ... }
        >>> muygps = MuyGPS(**k_kwarg)

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
            The kernel to be used.
        eps:
            A noise model.
        response_count:
            The number of response dimensions.
    """

    def __init__(
        self,
        kernel: Union[Matern, RBF],
        eps: Union[
            NullNoise, HomoscedasticNoise, HeteroscedasticNoise
        ] = HomoscedasticNoise(0.0, "fixed"),
        response_count: int = 1,
    ):
        self.kernel = kernel
        self.sigma_sq = SigmaSq(response_count)
        self.eps = eps
        self._make()

    def _make(self) -> None:
        self.kernel._make()
        self._mean_fn = PosteriorMean(self.eps)
        self._var_fn = PosteriorVariance(self.eps, self.sigma_sq)
        self._fast_posterior_mean_fn = FastPosteriorMean()
        self._fast_precompute_fn = FastPrecomputeCoefficients(self.eps)

    def set_eps(self, **eps) -> None:
        """
        Reset :math:`\\varepsilon` value or bounds.

        Uses existing value and bounds as defaults.

        Args:
            eps:
                A hyperparameter dict.
        """
        self.eps._set(**eps)

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
        if not self.eps.fixed():
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
        self.eps.append_lists("eps", names, params, bounds)
        return names, mm.array(params), mm.array(bounds)

    def fast_coefficients(
        self,
        K: mm.ndarray,
        train_nn_targets_fast: mm.ndarray,
    ) -> mm.ndarray:
        """
        Produces coefficient matrix for the fast posterior mean given in
        Equation (8) of [dunton2022fast]_.

        To form each row of this matrix, we compute

        .. math::
            \\mathbf{C}_{N^*}(i, :) =
                (K_{\\hat{\\theta}} (X_{N^*}, X_{N^*})
                + \\varepsilon I_k)^{-1} Y(X_{N^*}).

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the ith
        test point and the `nn_count - 1` nearest neighbors of this nearest
        neighbor, :math:`K_{\\hat{\\theta}}` is the trained kernel functor
        specified by `self.kernel`, :math:`\\varepsilon I_k` is a diagonal
        noise matrix whose diagonal is the value of the
        `self.eps` hyperparameter, and :math:`Y(X_{N^*})` is the
        `(train_count,)` vector of responses corresponding to the
        training features indexed by $N^*$.

        Args:
            K:
                The full pairwise kernel tensor of shape
                `(train_count, nn_count, nn_count)`.
            train_nn_targets_fast:
                The nearest neighbor response of each
                training points of shape
                `(train_count, nn_count, response_count)`.
        Returns:
            A matrix of shape `(train_count, nn_count)` whose rows are
            the precomputed coefficients for fast posterior mean inference.

        """

        return self._fast_precompute_fn(
            K,
            train_nn_targets_fast,
        )

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
        Equation (3.4) of [muyskens2021muygps]_. For each batch element
        :math:`\\mathbf{x}_i`, we compute

        .. math::
            \\widehat{Y}_{NN} (\\mathbf{x}_i \\mid X_{N_i}) =
                K_\\theta (\\mathbf{x}_i, X_{N_i})
                (K_\\theta (X_{N_i}, X_{N_i}) + \\varepsilon I_k)^{-1}
                Y(X_{N_i}).

        Here :math:`X_{N_i}` is the set of nearest neighbors of
        :math:`\\mathbf{x}_i` in the training data, :math:`K_\\theta` is the
        kernel functor specified by `self.kernel`, :math:`\\varepsilon I_k` is a
        diagonal homoscedastic noise matrix whose diagonal is the value of the
        `self.eps` hyperparameter, and :math:`Y(X_{N_i})` is the
        `(nn_count, response_count)` matrix of responses of the nearest
        neighbors given by the second two dimensions of the `batch_nn_targets`
        argument.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A matrix of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance matrix corresponding
                to each of the batch elements.
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
        Returns the posterior mean from the provided covariance and
        cross-covariance tensors.

        Return the local posterior variances of each prediction, corresponding
        to the diagonal elements of a covariance matrix. For each batch element
        :math:`\\mathbf{x}_i`, we compute

        .. math::
            Var(\\widehat{Y}_{NN} (\\mathbf{x}_i \\mid X_{N_i})) =
                K_\\theta (\\mathbf{x}_i, \\mathbf{x}_i) -
                K_\\theta (\\mathbf{x}_i, X_{N_i})
                (K_\\theta (X_{N_i}, X_{N_i}) + \\varepsilon I_k)^{-1}
                K_\\theta (X_{N_i}, \\mathbf{x}_i).

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A matrix of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance matrix corresponding
                to each of the batch elements.

        Returns:
            A vector of shape `(batch_count, response_count)` consisting of the
            diagonal elements of the posterior variance.
        """
        return self._var_fn(K, Kcross)

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
        Equation (9) of [dunton2022fast]_. For each test point
        :math:`\\mathbf{z}`, we compute

        .. math::
            \\widehat{Y} (\\mathbf{z} \\mid X) =
                K_\\theta (\\mathbf{z}, X_{N^*}) \\mathbf{C}_{N^*}.

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the queried
        test point :math:`\\mathbf{z}` and the nearest neighbors of that
        training point, :math:`K_\\theta` is the kernel functor
        specified by `self.kernel`, and :math:`\\mathbf{C}_{N^*}` is
        the matrix of precomputed coefficients given in Equation (8)
        of [dunton2022fast]_.

        Args:
            Kcross:
                A matrix of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance vector corresponding
                to each of the batch elements.
            coeffs_tensor:
                A matrix of shape `(batch_count, nn_count, response_count)`
                whose rows are given by precomputed coefficients.


        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        return self._fast_posterior_mean_fn(Kcross, coeffs_tensor)

    def apply_new_noise(
        self,
        new_noise: Union[HeteroscedasticNoise, HomoscedasticNoise, NullNoise],
    ):
        """
        Updates the homo/heteroscedastic noise parameter(s) of a MuyGPs model.
        To be used when the MuyGPs model has been trained and needs to be
        used for prediction, or if multiple batches are needed during training
        of a heteroscedastic model.

        Args:
            new_noise:
                If homoscedastic, a float to update the nugget parameter.
                If heteroscedastic, a matrix of shape
                `(test_count, nn_count)` containing the measurement
                noise corresponding to the nearest neighbors of each test point.
        Returns:
            A MuyGPs model with updated noise parameter(s).
        """
        ret = deepcopy(self)
        ret.eps = new_noise
        ret._mean_fn = PosteriorMean(ret.eps)
        ret._var_fn = PosteriorVariance(ret.eps, ret.sigma_sq)
        return ret

    def get_opt_mean_fn(self) -> Callable:
        """
        Return a posterior mean function for use in optimization.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` and assumes
        that either `eps` will be passed via a keyword argument or not at all.

        Returns:
            A function implementing the posterior mean, where `eps` is either
            fixed or takes updating values during optimization. The function
            expects keyword arguments corresponding to current hyperparameter
            values for unfixed parameters.
        """
        return self._mean_fn.get_opt_fn()

    def get_opt_var_fn(self) -> Callable:
        """
        Return a posterior variance function for use in optimization.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` and assumes
        that either `eps` will be passed via a keyword argument or not at all.

        Returns:
            A function implementing posterior variance, where `eps` is either
            fixed or takes updating values during optimization. The function
            expects keyword arguments corresponding to current hyperparameter
            values for unfixed parameters.
        """
        return self._var_fn.get_opt_fn()

    def __eq__(self, rhs) -> bool:
        if isinstance(rhs, self.__class__):
            return all(
                (
                    all(
                        self.kernel._hyperparameters[h]()
                        == rhs.kernel._hyperparameters[h]()
                        for h in self.kernel._hyperparameters
                    ),
                    self.eps() == rhs.eps(),
                    self.sigma_sq() == rhs.sigma_sq(),
                )
            )
        else:
            return False
