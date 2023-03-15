# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Multivariate MuyGPs implementation
"""

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.tensors import _make_fast_predict_tensors
from MuyGPyS._src.gp.muygps import (
    _muygps_fast_posterior_mean_precompute,
    _mmuygps_fast_posterior_mean,
)
from MuyGPyS._src.gp.noise import _homoscedastic_perturb
from MuyGPyS.gp.kernels import SigmaSq
from MuyGPyS.gp.muygps import MuyGPS


class MultivariateMuyGPS:
    """
    Multivariate Local Kriging Gaussian Process.

    Performs approximate GP inference by locally approximating an observation's
    response using its nearest neighbors with a separate kernel allocated for
    each response dimension, implemented as individual
    :class:`MuyGPyS.gp.muygps.MuyGPS` objects.

    This class is similar in interface to :class:`MuyGPyS.gp.muygps.MuyGPS`, but
    requires a list of hyperparameter dicts at initialization.

    Example:
        >>> from MuyGPyS.gp import MultivariateMuyGPS as MMuyGPS
        >>> k_kwargs1 = {
        ...         "eps": {"val": 1e-5},
        ...         "nu": {"val": 0.67, "bounds": (0.1, 2.5)},
        ...         "length_scale": {"val": 7.2},
        ... }
        >>> k_kwargs2 = {
        ...         "eps": {"val": 1e-5},
        ...         "nu": {"val": 0.38, "bounds": (0.1, 2.5)},
        ...         "length_scale": {"val": 7.2},
        ... }
        >>> k_args = [k_kwargs1, k_kwargs2]
        >>> mmuygps = MMuyGPS("matern", *k_args)

    We can realize kernel tensors for each of the models contained within a
    `MultivariateMuyGPS` object by iterating over its `models` member. Once we
    have computed a `pairwise_dists` tensor and a `crosswise_dists` matrix, it
    is straightforward to perform each of these realizations.

    Example:
        >>> for model in MuyGPyS.models:
        >>>         K = model.kernel(pairwise_dists)
        >>>         Kcross = model.kernel(crosswise_dists)
        >>>         # do something with K and Kcross...

    Args
        kern:
            The kernel to be used. Each kernel supports different
            hyperparameters that can be specified in kwargs. Currently supports
            only `matern` and `rbf`.
        model_args:
            Dictionaries defining each internal
            :class:`MuyGPyS.gp.muygps.MuyGPS` instance.
    """

    def __init__(
        self,
        kern: str,
        *model_args,
    ):
        self.kern = kern.lower()
        self.models = [MuyGPS(kern, **args) for args in model_args]
        self.metric = self.models[0].kernel.metric  # this is brittle
        self.sigma_sq = SigmaSq(len(self.models))

    def fixed(self) -> bool:
        """
        Checks whether all kernel and model parameters are fixed for each model,
        excluding :math:`\\sigma^2`.

        Returns:
            Returns `True` if all parameters in all models are fixed, and
            `False` otherwise.
        """
        return bool(all([model.fixed() for model in self.models]))

    def posterior_mean(
        self,
        pairwise_dists: mm.ndarray,
        crosswise_dists: mm.ndarray,
        batch_nn_targets: mm.ndarray,
    ) -> mm.ndarray:
        """
        Performs simultaneous posterior mean inference on provided distance
        tensors and the target matrix.

        Computes parallelized local solves of systems of linear equations using
        the kernel realizations, one for each internal model, of the last two
        dimensions of `pairwise_dists` along with `crosswise_dists` and
        `batch_nn_targets` to predict responses in terms of the posterior mean.
        Assumes that distance tensor `pairwise_dists` and
        crosswise distance matrix `crosswise_dists` are already computed and
        given as arguments.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations by solving a system
        of linear equations induced by each kernel functor, one per response
        dimension, in a generalization of Equation (3.4) of
        [muyskens2021muygps]_. For each batch element :math:`\\mathbf{x}_i` we
        compute

        .. math::
            \\widehat{Y}_{NN} (\\mathbf{x}_i \\mid X_{N_i})_{:,j} =
                K^{(j)}_\\theta (\\mathbf{x}_i, X_{N_i})
                (K^{(j)}_\\theta (X_{N_i}, X_{N_i}) + \\varepsilon_j I_k)^{-1}
                Y(X_{N_i})_{:,j}.

        Here :math:`X_{N_i}` is the set of nearest neighbors of
        :math:`\\mathbf{x}_i` in the training data, :math:`K^{(j)}_\\theta` is
        the kernel functor associated with the jth internal model, corresponding
        to the jth response dimension, :math:`\\varepsilon_j I_k` is a diagonal
        homoscedastic noise matrix whose diagonal is the value of the
        `self.models[j].eps` hyperparameter, and :math:`Y(X_{N_i})_{:,j}` is the
        `(batch_count,)` vector of the jth responses of the nearest neighbors
        given by a slice of the `batch_nn_targets` argument.

        Args:
            pairwise_dists:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)` -shaped pairwise nearest neighbor
                distance matrices corresponding to each of the batch elements.
            crosswise_dists:
                A matrix of shape `(batch_count, nn_count)` whose rows list the
                distance between each batch element element and its nearest
                neighbors.
            batch_nn_targets:
                A tensor of shape `(batch_count, nn_count, response_count)`
                listing the vector-valued responses for the nearest neighbors
                of each batch element.

        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        batch_count, nn_count, response_count = batch_nn_targets.shape
        responses = mm.zeros((batch_count, response_count))
        for i, model in enumerate(self.models):
            K = model.kernel(pairwise_dists)
            Kcross = model.kernel(crosswise_dists)
            responses = mm.assign(
                responses,
                model.posterior_mean(
                    K,
                    Kcross,
                    batch_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
                ).reshape(batch_count),
                slice(None),
                i,
            )
        return responses

    def posterior_variance(
        self,
        pairwise_dists: mm.ndarray,
        crosswise_dists: mm.ndarray,
    ) -> mm.ndarray:
        """
        Performs simultaneous posterior variance inference on provided distance
        tensors.

        Return the local posterior variances of each prediction, corresponding
        to the diagonal elements of a covariance matrix. For each batch element
        :math:`\\mathbf{x}_i`, we compute

        .. math::
            Var(\\widehat{Y}_{NN} (\\mathbf{x}_i \\mid X_{N_i}))_j =
                K^{(j)}_\\theta (\\mathbf{x}_i, \\mathbf{x}_i) -
                K^{(j)}_\\theta (\\mathbf{x}_i, X_{N_i})
                (K^{(j)}_\\theta (X_{N_i}, X_{N_i}) + \\varepsilon I_k)^{-1}
                K^{(j)}_\\theta (X_{N_i}, \\mathbf{x}_i).

        Args:
            pairwise_dists:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)` -shaped pairwise nearest neighbor
                distance matrices corresponding to each of the batch elements.
            crosswise_dists:
                A matrix of shape `(batch_count, nn_count)` whose rows list the
                distance between each batch element element and its nearest
                neighbors.

        Returns:
            A vector of shape `(batch_count, response_count)` consisting of the
            diagonal elements of the posterior variance for each model.
        """
        batch_count, _ = crosswise_dists.shape
        response_count = len(self.models)
        diagonal_variance = mm.zeros((batch_count, response_count))
        for i, model in enumerate(self.models):
            K = model.kernel(pairwise_dists)
            Kcross = model.kernel(crosswise_dists)
            ss = self.sigma_sq()[i]
            diagonal_variance = mm.assign(
                diagonal_variance,
                model.posterior_variance(K, Kcross).reshape(batch_count) * ss,
                slice(None),
                i,
            )
        return diagonal_variance

    def build_fast_posterior_mean_coeffs(
        self,
        train: mm.ndarray,
        nn_indices: mm.ndarray,
        targets: mm.ndarray,
    ) -> mm.ndarray:
        """
        Produces coefficient tensor for fast posterior mean inference given in
        Equation (8) of [dunton2022fast]_.

        To form the tensor, we compute

        .. math::
            \\mathbf{C}_{N^*}(i, :, j) =
                (K_{\\hat{\\theta_j}} (X_{N^*}, X_{N^*}) +
                \\varepsilon I_k)^{-1} Y(X_{N^*}).

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the ith
        test point and the `nn_count - 1` nearest neighbors of this nearest
        neighbor, :math:`K_{\\hat{\\theta_j}}` is the trained kernel functor
        corresponding the jth response and specified by `self.models`,
        :math:`\\varepsilon I_k` is a diagonal homoscedastic noise matrix whose
        diagonal  is the value of the `self.eps` hyperparameter,
        and :math:`Y(X_{N^*})` is the `(train_count, response_count)`
        matrix of responses corresponding to the training features indexed
        by $N^*$.

        Args:
            train:
                The full training data matrix of shape
                `(train_count, feature_count)`.
            nn_indices:
                The nearest neighbors indices of each
                training points of shape `(train_count, nn_count)`.
            targets:
                A matrix of shape `(train_count, response_count)` whose rows are
                vector-valued responses for each training element.
        Returns:
            A tensor of shape `(batch_count, nn_count, response_count)`
            whose entries comprise the precomputed coefficients for fast
            posterior mean inference.
        """
        (
            pairwise_dists_fast,
            train_nn_targets_fast,
        ) = _make_fast_predict_tensors(self.metric, nn_indices, train, targets)

        train_count, nn_count, response_count = train_nn_targets_fast.shape
        coeffs_tensor = mm.zeros((train_count, nn_count, response_count))
        for i, model in enumerate(self.models):
            K = model.kernel(pairwise_dists_fast)
            mm.assign(
                coeffs_tensor,
                _muygps_fast_posterior_mean_precompute(
                    _homoscedastic_perturb(K, model.eps()),
                    train_nn_targets_fast[:, :, i],
                ),
                slice(None),
                slice(None),
                i,
            )

        return coeffs_tensor

    def fast_posterior_mean(
        self,
        crosswise_dists: mm.ndarray,
        coeffs_tensor: mm.ndarray,
    ) -> mm.ndarray:
        """
        Performs fast posterior mean inference using provided crosswise
        distances and precomputed coefficient matrix.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations, as computed in
        Equation (9) of [dunton2022fast]_. For each test point
        :math:`\\mathbf{z}`, we compute

        .. math::
            \\widehat{Y} (\\mathbf{z} \\mid X) =
                K_\\theta (\\mathbf{z}, X_{N^*}) \mathbf{C}_{N^*}.

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the queried
        test point :math:`\\mathbf{z}` and the nearest neighbors of that
        training point, :math:`K_\\theta` is the kernel functor specified by
        `self.kernel`, and :math:`\mathbf{C}_{N^*}` is the matrix of
        precomputed coefficients given in Equation (8) of [dunton2022fast]_.

        Args:
            crosswise_dists:
                A matrix of shape `(batch_count, nn_count)` whose rows list the
                distance of the corresponding test element to each of its
                nearest neighbors.
            coeffs_tensor:
                A tensor of shape `(batch_count, nn_count, response_count)`
                providing the precomputed coefficients.

        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        Kcross = mm.zeros(coeffs_tensor.shape)
        for i, model in enumerate(self.models):
            mm.assign(
                Kcross,
                model.kernel(crosswise_dists),
                slice(None),
                slice(None),
                i,
            )
        return _mmuygps_fast_posterior_mean(Kcross, coeffs_tensor)
