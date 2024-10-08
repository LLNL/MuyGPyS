# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
Multivariate MuyGPs implementation

The separate Multivariate MuyGPs model is deprecated and will be removed in
future versions.
"""
from typing import Optional, Tuple
from warnings import warn

import MuyGPyS._src.math as mm
from MuyGPyS._src.gp.muygps import _mmuygps_fast_posterior_mean
from MuyGPyS._src.mpi_utils import mpi_chunk
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

    **This class is deprecated and will be removed in future versions.**

    Example:
        >>> from MuyGPyS.gp import MultivariateMuyGPS as MMuyGPS
        >>> k_kwargs1 = {
        ...     "noise": Parameter(1e-5),
        ...     "kernel": Matern(
        ...         smoothness=Parameter(0.67, (0.1, 2.5)),
        ...         deformation=Isotropy(
        ...             metric=l2,
        ...             length_scale=Parameter(0.2),
        ...         scale=AnalyticScale(),
        ...     ),
        ... }
        >>> k_kwargs2 = {
        ...     "noise": Parameter(1e-5),
        ...     "kernel": Matern(
        ...         smoothness=Parameter(0.67, (0.1, 2.5)),
        ...         deformation=Isotropy(
        ...             metric=l2,
        ...             length_scale=Parameter(0.2),
        ...         scale=AnalyticScale(),
        ...     ),
        ... }
        >>> k_args = [k_kwargs1, k_kwargs2]
        >>> mmuygps = MMuyGPS(*k_args)

    We can realize kernel tensors for each of the models contained within a
    `MultivariateMuyGPS` object by iterating over its `models` member. Once we
    have computed `pairwise_diffs` and `crosswise_diffs` tensors, it
    is straightforward to perform each of these realizations.

    Example:
        >>> for model in MuyGPyS.models:
        >>>     Kin = model.kernel(pairwise_diffs)
        >>>     Kcross = model.kernel(crosswise_diffs)
        >>>     # do something with Kin and Kcross...

    Args
        model_args:
            Dictionaries defining each internal
            :class:`MuyGPyS.gp.muygps.MuyGPS` instance.
    """

    def __init__(
        self,
        *model_args,
    ):
        warn(
            f"{self.__class__.__name__} is deprecated and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.models = [MuyGPS(**args) for args in model_args]

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
        pairwise_diffs: mm.ndarray,
        crosswise_diffs: mm.ndarray,
        batch_nn_targets: mm.ndarray,
    ) -> mm.ndarray:
        """
        Performs simultaneous posterior mean inference on provided difference
        tensors and the target matrix.

        Computes parallelized local solves of systems of linear equations using
        the kernel realizations, one for each internal model, of the last two
        dimensions of `pairwise_diffs` along with `crosswise_diffs` and
        `batch_nn_targets` to predict responses in terms of the posterior mean.
        Assumes that difference tensors `pairwise_diffs` and `crosswise_diffs`
        are already computed and given as arguments.

        Returns the predicted response in the form of a posterior mean for each
        element of the batch of observations by solving a system of linear
        equations induced by each kernel functor, one per response dimension, in
        a generalization of Equation (3.4) of [muyskens2021muygps]_. For each
        response dimension :math:`j`, given observation set :math:`X` with
        responses :math:`Y`, noise prior set :math:`\\varepsilon^{(j)}`, and
        kernel function :math:`Kin_{\\theta^{(j)}}(\\cdot, \\cdot)`, computes the
        following for each prediction element :math:`\\mathbf{z}_i` with nearest
        neighbors index set :math:`N_i`:

        .. math::
            \\widehat{Y} (\\mathbf{z}_i \\mid X_{N_i})_j =
                \\sigma^2_j Kin_{\\theta^{(j)}} (\\mathbf{z}_i, X_{N_i})
                \\left (
                    Kin_{\\theta^{(j)}} (X_{N_i}, X_{N_i})
                    + \\varepsilon^{(j)}_{N_i}
                \\right )^{-1}
                Y(X_{N_i})_{:,j}.

        Args:
            pairwise_diffs:
                A tensor of shape
                `(batch_count, nn_count, nn_count, feature_count)` containing
                the `(nn_count, nn_count, feature_count)`-shaped pairwise
                nearest neighbor difference tensors corresponding to each of the
                batch elements.
            crosswise_diffs:
                A matrix of shape `(batch_count, nn_count, feature_count)` whose
                rows list the difference between each feature of each batch
                element element and its nearest neighbors.
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
            Kin = model.kernel(pairwise_diffs)
            Kcross = model.kernel(crosswise_diffs)
            responses = mm.assign(
                responses,
                model.posterior_mean(
                    Kin,
                    Kcross,
                    batch_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
                ).reshape(batch_count),
                slice(None),
                i,
            )
        return responses

    def posterior_variance(
        self,
        pairwise_diffs: mm.ndarray,
        crosswise_diffs: mm.ndarray,
    ) -> mm.ndarray:
        """
        Returns the posterior variance from the provided difference tensors.

        Return the local posterior variances of each prediction, corresponding
        to the diagonal elements of a covariance matrix. For each response
        dimension, given observation set :math:`X` with responses :math:`Y`,
        noise prior set :math:`\\varepsilon^{(j)}`, and kernel function
        :math:`Kin_{\\theta^{(j)}}(\\cdot, \\cdot)`, computes the following for
        each prediction element :math:`\\mathbf{z}_i` with nearest neighbors
        index set :math:`N_i`:

        .. math::
            Var \\left (
                \\widehat{Y} (\\mathbf{z}_i \\mid X_{N_i})
            \\right)_j =
                \\sigma_j^2 \\left (
                    Kin_{\\theta^{(j)}} (\\mathbf{z}_i, \\mathbf{z}_i) -
                    Kin_{\\theta^{(j)}} (\\mathbf{z}_i, X_{N_i})
                    \\left (
                        Kin_{\\theta^{(j)}} (X_{N_i}, X_{N_i}
                    \\right ) + \\varepsilon^{(j)}_{N_i})^{-1}
                    Kin_{\\theta^{(j)}} (X_{N_i}, \\mathbf{z}_i)
                \\right ).

        Args:
            pairwise_diffs:
                A tensor of shape
                `(batch_count, nn_count, nn_count, feature_count)` containing
                the `(nn_count, nn_count, feature_count)`-shaped pairwise
                nearest neighbor difference tensors corresponding to each of the
                batch elements.
            crosswise_diffs:
                A matrix of shape `(batch_count, nn_count, feature_count)` whose
                rows list the difference between each feature of each batch
                element element and its nearest neighbors.

        Returns:
            A vector of shape `(batch_count, response_count)` consisting of the
            diagonal elements of the posterior variance for each model.
        """
        batch_count = crosswise_diffs.shape[0]
        response_count = len(self.models)
        diagonal_variance = mm.zeros((batch_count, response_count))
        for i, model in enumerate(self.models):
            Kin = model.kernel(pairwise_diffs)
            Kcross = model.kernel(crosswise_diffs)
            ss = model.scale()
            diagonal_variance = mm.assign(
                diagonal_variance,
                model.posterior_variance(Kin, Kcross).reshape(batch_count) * ss,
                slice(None),
                i,
            )
        return diagonal_variance

    def fast_coefficients(
        self,
        pairwise_diffs_fast: mm.ndarray,
        train_nn_targets_fast: mm.ndarray,
    ) -> mm.ndarray:
        """
        Produces coefficient matrix for the fast posterior mean given in
        Equation (8) of [dunton2022fast]_ for each response dimenion.

        Fro each response dimension :math:`j`, given observation set :math:`X`
        with responses :math:`Y`, noise prior set :math:`\\varepsilon^{(j)}`, and
        kernel function :math:`Kin_{\\theta^{(j)}}(\\cdot, \\cdot)`, computes the
        following for each observation element :math:`\\mathbf{x}_i` with
        nearest neighbors index set :math:`N^*_i`, containing `i` and the
        indices of the `nn_count - 1` nearest neighbors of
        :math:`\\mathbf{x}_i`:

        .. math::
            C^{(j)}_i =
                \\left (
                    Kin_{\\theta^{(j)}}(X_{N_i}, X_{N_i})
                    + \\varepsilon^{(j)}_{N_i}
                \\right )^{-1}
                Y(X_{N_i})_{:, j}.

        Args:
            pairwise_diffs:
                A tensor of shape
                `(train_count, nn_count, nn_count, feature_count)` containing
                the `(nn_count, nn_count, feature_count)`-shaped pairwise
                nearest neighbor difference tensors corresponding to each of the
                batch elements.
            batch_nn_targets:
                A tensor of shape `(train_count, nn_count, response_count)`
                listing the vector-valued responses for the nearest neighbors
                of each batch element.
        Returns:
            A tensor of shape `(batch_count, nn_count, response_count)`
            whose entries comprise the precomputed coefficients for fast
            posterior mean inference.
        """

        train_count, nn_count, response_count = train_nn_targets_fast.shape
        coeffs_tensor = mm.zeros((train_count, nn_count, response_count))

        for i, model in enumerate(self.models):
            Kin = model.kernel(pairwise_diffs_fast)
            mm.assign(
                coeffs_tensor,
                model.fast_coefficients(
                    model.noise.perturb(Kin),
                    train_nn_targets_fast[:, :, i],
                ),
                slice(None),
                slice(None),
                i,
            )

        return coeffs_tensor

    def fast_posterior_mean(
        self,
        crosswise_diffs: mm.ndarray,
        coeffs_tensor: mm.ndarray,
    ) -> mm.ndarray:
        """
        Performs fast posterior mean inference using provided cross-covariance
        and precomputed coefficient matrix for each response dimension.

        Returns the predicted response across each response dimension in the
        form of a posterior mean for each element of the batch of observations,
        as computed in Equation (9) of [dunton2022fast]_. For each response
        dimension :math:`j`, given the coefficients :math:`C^{(j)}` created by
        :func:`~MuyGPyS.gp.muygps.MultivariateMuyGPS.fast_coefficients` and
        Equation (8) of [dunton2022fast]_, observation set :math:`X`, noise
        prior set :math:`\\varepsilon^{(j)}`, and kernel function
        :math:`Kin_{\\theta^{(j)}}(\\cdot, \\cdot)`, computes the following for each
        test point :math:`\\mathbf{z}` and index set :math:`N^*_i` containing
        the union of the index :math:`i` of the nearest neighbor
        :math:`\\mathbf{x}_i` of :math:`\\mathbf{z}` and the `nn_count - 1`
        nearest neighbors of :math:`\\mathbf{x}_i`:

        .. math::
            \\widehat{Y} \\left ( \\mathbf{z} \\mid X \\right )_j =
                \\sigma^2 Kin_{\\theta^{(j)}}(\\mathbf{z}, X_{N^*_i}) C^{(j)}_i.

        Args:
            crosswise_diffs:
                A matrix of shape `(batch_count, nn_count, feature_count)` whose
                rows list the difference between each feature of each batch
                element element and its nearest neighbors.
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
                model.kernel(crosswise_diffs),
                slice(None),
                slice(None),
                i,
            )
        return _mmuygps_fast_posterior_mean(Kcross, coeffs_tensor)

    def optimize_scale(
        self, pairwise_diffs: mm.ndarray, nn_targets: mm.ndarray
    ):
        """
        Optimize the value of the :math:`sigma^2` scale parameter for each
        response dimension.

        Uses the optimization method specified by the types of the `scale`
        parameters to optimize their value.

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
            A reference to this model whose global scale parameter (and those
            of its submodels) has been optimized.
        """
        batch_count, nn_count, response_count = nn_targets.shape
        if response_count != len(self.models):
            raise ValueError(
                f"Response count ({response_count}) does not match the number "
                f"of models ({len(self.models)})."
            )
        for i, model in enumerate(self.models):
            Kin = model.kernel(pairwise_diffs)
            opt_fn = model.scale.get_opt_fn(model)
            new_scale_val = opt_fn(
                Kin,
                nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            )
            model.scale._set(new_scale_val)
        return self

    @mpi_chunk(return_count=3)
    def make_predict_tensors(
        self,
        batch_indices: mm.ndarray,
        batch_nn_indices: mm.ndarray,
        test_features: Optional[mm.ndarray],
        train_features: mm.ndarray,
        train_targets: mm.ndarray,
        **kwargs,
    ) -> Tuple[mm.ndarray, mm.ndarray, mm.ndarray]:
        """
        Create the metric and target tensors for prediction using the model's
        deformation.

        @NOTE[mwp] uses the first model's deformation, and expects all model
        deformations to agree in tensor shapes.

        Creates the `crosswise_tensor`, `pairwise_tensor` and `batch_nn_targets`
        tensors required by :func:`~MuyGPyS.gp.MuyGPS.posterior_mean` and
        :func:`~MuyGPyS.gp.MuyGPS.posterior_variance`.

        Args:
            batch_indices:
                A vector of integers of shape `(batch_count,)` identifying the
                training batch of observations to be approximated.
            batch_nn_indices:
                A matrix of integers of shape `(batch_count, nn_count)` listing
                the nearest neighbor indices for all observations in the batch.
            test_features:
                The full floating point testing data matrix of shape
                `(test_count, feature_count)`.
            train_features:
                The full floating point training data matrix of shape
                `(train_count, ...)`.
            train_targets:
                A matrix of shape `(train_count, ...)` whose rows are
                vector-valued responses for each training element.

        Returns
        -------
        crosswise_tensor:
            A tensor of shape `(batch_count, nn_count, ...)` whose second and
            subsequent dimensions list the metric comparison between each batch
            element element and its nearest neighbors.
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, ...)`
            containing the `(nn_count, nn_count, ...)`-shaped pairwise nearest
            neighbor metrics tensors corresponding to each of the batch
            elements.
        batch_nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, ...)` containing
            the expected response for each nearest neighbor of each batch
            element.
        """
        return self.models[0].make_predict_tensors(
            batch_indices,
            batch_nn_indices,
            test_features,
            train_features,
            train_targets,
            disable_mpi=True,
        )

    @mpi_chunk(return_count=4)
    def make_train_tensors(
        self,
        batch_indices: mm.ndarray,
        batch_nn_indices: mm.ndarray,
        train_features: mm.ndarray,
        train_targets: mm.ndarray,
        **kwargs,
    ) -> Tuple[mm.ndarray, mm.ndarray, mm.ndarray, mm.ndarray]:
        """
        Create the metric and target tensors needed for training.

        @NOTE[mwp] uses the first model's deformation, and expects all model
        deformations to agree in tensor shapes.

        Similar to :func:`~MuyGPyS.gp.muygps.MuyGPS.make_predict_tensors` but
        returns the additional `batch_targets` matrix, which is only defined for
        a batch of training data.

        Args:
            batch_indices:
                A vector of integers of shape `(batch_count,)` identifying the
                training batch of observations to be approximated.
            batch_nn_indices:
                A matrix of integers of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the batch.
            train_features:
                The full floating point training data matrix of shape
                `(train_count, ...)`.
            train_targets:
                A matrix of shape `(train_count, ...)` whose rows are
                vector-valued responses for each training element.

        Returns
        -------
        crosswise_tensor:
            A tensor of shape `(batch_count, nn_count, ...)` whose second and
            subsequent dimensions list the metric comparison between each batch
            element element and its nearest neighbors.
        pairwise_diffs:
            A tensor of shape `(batch_count, nn_count, nn_count, ...)`
            containing the `(nn_count, nn_count, ...)`-shaped pairwise nearest
            neighbor metrics tensors corresponding to each of the batch
            elements.
        batch_targets:
            Matrix of floats of shape `(batch_count, ...)` whose rows
            give the expected response for each batch element.
        batch_nn_targets:
            Tensor of floats of shape `(batch_count, nn_count, ...)` containing
            the expected response for each nearest neighbor of each batch
            element.
        """
        return self.models[0].make_train_tensors(
            batch_indices,
            batch_nn_indices,
            train_features,
            train_targets,
            disable_mpi=True,
        )
