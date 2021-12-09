# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""MuyGPs implementation
"""

import numpy as np

from typing import Dict, Generator, Optional, Tuple, Union

from MuyGPyS.gp.distance import make_regress_tensors
from MuyGPyS.gp.kernels import (
    _get_kernel,
    _init_hyperparameter,
    Hyperparameter,
    SigmaSq,
)


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
        >>> from MuyGPyS.gp.muygps import MuyGPS
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
    documentation for details) :func:`MuyGPyS.gp.distance.pairwise_distances` to
    construct pairwise distance tensors and
    :func:`MuyGPyS.gp.distance.crosswise_distances` to produce crosswise distance
    matrices that `MuyGPS` can then use to construct kernel tensors and
    cross-covariance matrices, respectively.

    We can easily realize kernel tensors using a `MuyGPS` object's `kernel`
    functor once we have computed a `pairwise_dists` tensor and a
    `crosswise_dists` matrix.

    Example:
        >>> K = muygps.kernel(pairwise_dists)
        >>> Kcross = muygps.kernel(crosswise_dists)


    Args:
        kern:
            The kernel to be used. Each kernel supports different
            hyperparameters that can be specified in kwargs. Currently supports
            only `matern` and `rbf`.
        eps:
            A hyperparameter dict.
        kwargs:
            Addition parameters to be passed to the kernel, possibly including
            additional hyperparameter dicts and a metric keyword.
    """

    def __init__(
        self,
        kern: str = "matern",
        eps: Dict[str, Union[float, Tuple[float, float]]] = {"val": 0.0},
        **kwargs,
    ):
        self.kern = kern.lower()
        self.kernel = _get_kernel(self.kern, **kwargs)
        self.eps = _init_hyperparameter(1e-14, "fixed", **eps)
        self.sigma_sq = SigmaSq()

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
        for p in self.kernel.hyperparameters:
            if self.kernel.hyperparameters[p].get_bounds() != "fixed":
                return False
        if self.eps.get_bounds() != "fixed":
            return False
        return True

    def get_optim_params(self) -> Dict[str, Hyperparameter]:
        """
        Return a dictionary of references to the unfixed kernel hyperparameters.

        This is a convenience function for obtaining all of the information
        necessary to optimize hyperparameters. It is important to note that the
        values of the dictionary are references to the actual hyperparameter
        objects underying the kernel functor - changing these references will
        change the kernel.

        Returns:
            A dict mapping hyperparameter names to references to their objects.
            Only returns hyperparameters whose bounds are not set as `fixed`.
            Returned hyperparameters can include `eps`, but not `sigma_sq`,
            as it is currently optimized via a separate closed-form method.
        """
        optim_params = {
            p: self.kernel.hyperparameters[p]
            for p in self.kernel.hyperparameters
            if self.kernel.hyperparameters[p].get_bounds() != "fixed"
        }
        if self.eps.get_bounds() != "fixed":
            optim_params["eps"] = self.eps
        return optim_params

    def _compute_solve(
        self,
        K: np.ndarray,
        Kcross: np.ndarray,
        batch_nn_targets: np.ndarray,
    ) -> np.ndarray:
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A tensor of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance matrix corresponding
                to each of the batch elements.
            batch_nn_targets:
                A tensor of shape `(batch_count, nn_count, response_count)`
                whose last dimension lists the vector-valued responses for the
                nearest neighbors of each batch element.

        Returns:
            A matrix of shape `(batch_count, response_count)` listing the
            predicted response for each of the batch elements.
        """
        batch_count, nn_count, response_count = batch_nn_targets.shape
        responses = Kcross.reshape(batch_count, 1, nn_count) @ np.linalg.solve(
            K + self.eps() * np.eye(nn_count), batch_nn_targets
        )
        return responses.reshape(batch_count, response_count)

    def _compute_diagonal_variance(
        self,
        K: np.ndarray,
        Kcross: np.ndarray,
    ) -> np.ndarray:
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A tensor of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance matrix corresponding
                to each of the batch elements.

        Returns:
            A vector of shape `(batch_count)` listing the diagonal variances for
            each of the batch elements.
        """
        batch_count, nn_count = Kcross.shape
        return np.array(
            [
                1.0
                - Kcross[i, :]
                @ np.linalg.solve(
                    K[i, :, :] + self.eps() * np.eye(nn_count), Kcross[i, :]
                )
                for i in range(batch_count)
            ]
        )

    def regress_from_indices(
        self,
        indices: np.ndarray,
        nn_indices: np.ndarray,
        test: np.ndarray,
        train: np.ndarray,
        targets: np.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
        return_distances: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Performs simultaneous regression on a list of observations.

        This is similar to the old regress API in that it implicitly creates and
        discards the distance and kernel tensors and matrices. If these data
        structures are needed for later reference, instead use
        :func:`~MuyGPyS.gp.muygps.MuyGPS.regress`.

        Args:
            indices:
                An integral vector of shape `(batch_count,)` indices of the
                observations to be approximated.
            nn_indices:
                An integral matrix of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the test batch.
            test:
                The full testing data matrix of shape
                `(test_count, feature_count)`.
            train:
                The full training data matrix of shape
                `(train_count, feature_count)`.
            targets:
                A matrix of shape `(train_count, response_count)` whose rows are
                vector-valued responses for each training element.
            variance_mode:
                Specifies the type of variance to return. Currently supports
                `"diagonal"` and None. If None, report no variance term.
            apply_sigma_sq:
                Indicates whether to scale the posterior variance by `sigma_sq`.
                Unused if `variance_mode is None` or `sigma_sq == "unlearned"`.
            return_distances:
                If `True`, returns a `(test_count, nn_count)` matrix containing
                the crosswise distances between the test elements and their
                nearest neighbor sets and a `(test_count, nn_count, nn_count)`
                tensor containing the pairwise distances between the test data's
                nearest neighbor sets.

        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count,)` whose rows are
            the predicted response for each of the given indices.
        diagonal_variance:
            A vector of shape `(batch_count,)` consisting of the diagonal
            elements of the posterior variance, or a matrix of shape
            `(batch_count, response_count)` for a multidimensional response.
            Only returned where `variance_mode == "diagonal"`.
        crosswise_dists:
            A matrix of shape `(test_count, nn_count)` whose rows list the
            distance of the corresponding test element to each of its nearest
            neighbors. Only returned if `return_distances is True`.
        pairwise_dists:
            A tensor of shape `(test_count, nn_count, nn_count,)` whose latter
            two dimensions contain square matrices containing the pairwise
            distances between the nearest neighbors of the test elements. Only
            returned if `return_distances is True`.
        """
        (
            crosswise_dists,
            pairwise_dists,
            batch_nn_targets,
        ) = make_regress_tensors(
            self.kernel.metric, indices, nn_indices, test, train, targets
        )
        K = self.kernel(pairwise_dists)
        Kcross = self.kernel(crosswise_dists)
        responses = self.regress(
            K,
            Kcross,
            batch_nn_targets,
            variance_mode=variance_mode,
            apply_sigma_sq=apply_sigma_sq,
        )
        if return_distances is False:
            return responses
        else:
            if variance_mode is None:
                return responses, crosswise_dists, pairwise_dists
            else:
                responses, variances = responses
                return responses, variances, crosswise_dists, pairwise_dists

    def regress(
        self,
        K: np.array,
        Kcross: np.array,
        batch_nn_targets: np.array,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs simultaneous regression on provided covariance,
        cross-covariance, and target.

        Computes parallelized local solves of systems of linear equations using
        the last two dimensions of `K` along with `Kcross` and
        `batch_nn_targets` to predict responses in terms of the posterior mean.
        Also computes the posterior variance if `variance_mode` is set
        appropriately. Assumes that kernel tensor `K` and cross-covariance
        matrix `Kcross` are already computed and given as arguments. To
        implicitly construct these values from indices (useful if the kernel or
        distance tensors and matrices are not needed for later reference)
        instead use :func:`~MuyGPyS.gp.muygps.MuyGPS.regress_from_indices`.

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
        `(nn_count, respones_count)` matrix of responses of the nearest
        neighbors given by the second two dimensions of the `batch_nn_targets`
        argument.

        If `variance_mode == "diagonal"`, also return the local posterior
        variances of each prediction, corresponding to the diagonal elements of
        a covariance matrix. For each batch element :math:`\\mathbf{x}_i`, we
        compute

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
                A tensor of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance matrix corresponding
                to each of the batch elements.
            batch_nn_targets:
                A tensor of shape `(batch_count, nn_count, response_count)` whose
                last dimension lists the vector-valued responses for the
                nearest neighbors of each batch element.
            variance_mode:
                Specifies the type of variance to return. Currently supports
                `"diagonal"` and None. If None, report no variance term.
            apply_sigma_sq:
                Indicates whether to scale the posterior variance by `sigma_sq`.
                Unused if `variance_mode is None` or `sigma_sq == "unlearned"`.

        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count,)` whose rows are
            the predicted response for each of the given indices.
        diagonal_variance:
            A vector of shape `(batch_count,)` consisting of the diagonal
            elements of the posterior variance, or a matrix of shape
            `(batch_count, response_count)` for a multidimensional response.
            Only returned where `variance_mode == "diagonal"`.
        """
        responses = self._compute_solve(K, Kcross, batch_nn_targets)
        if variance_mode is None:
            return responses
        elif variance_mode == "diagonal":
            diagonal_variance = self._compute_diagonal_variance(K, Kcross)
            if apply_sigma_sq is True and isinstance(
                self.sigma_sq(), np.ndarray
            ):
                sigmas = self.sigma_sq()
                if len(sigmas) == 1:
                    diagonal_variance *= sigmas
                else:
                    diagonal_variance = np.array(
                        [ss * diagonal_variance for ss in sigmas]
                    ).T
            return responses, diagonal_variance
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )

    def sigma_sq_optim(
        self,
        K: np.ndarray,
        nn_indices: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Optimize the value of the :math:`\\sigma^2` scale parameter for each
        response dimension.

        We approximate :math:`\\sigma^2` by way of averaging over the analytic
        solution from each local kernel.

        .. math::
            \\sigma^2 = \\frac{1}{n} * Y^T  K^{-1}  Y

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            nn_indices:
                An integral matrix of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the test batch.
            targets:
                A matrix of shape `(batch_count, response_count)` whose rows list
                the vector-valued responses for all of the training targets.

        Returns:
            A vector of shape `(response_count)` listing the value of sigma^2
            for each dimension.
        """
        batch_count, nn_count = nn_indices.shape
        _, response_count = targets.shape

        sigma_sq = np.zeros((response_count,))
        for i in range(response_count):
            sigma_sq[i] = sum(
                self._get_sigma_sq(K, targets[:, i], nn_indices)
            ) / (nn_count * batch_count)

        self.sigma_sq._set(sigma_sq)
        return self.sigma_sq()

    def _get_sigma_sq_series(
        self,
        K: np.ndarray,
        nn_indices: np.ndarray,
        target_col: np.ndarray,
    ) -> np.ndarray:
        """
        Return the series of sigma^2 scale parameters for each neighborhood
        solve.

        NOTE[bwp]: This function is only for testing purposes.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            nn_indices:
                An integral matrix of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the test batch.
            target_col:
                A vector of shape `(batch_count)` consisting of the target for
                each nearest neighbor.

        Returns:
            A vector of shape `(response_count)` listing the value of sigma^2
            for the given response dimension.
        """
        batch_count, nn_count = nn_indices.shape

        sigmas = np.zeros((batch_count,))
        for i, el in enumerate(self._get_sigma_sq(K, target_col, nn_indices)):
            sigmas[i] = el
        return sigmas / nn_count

    def _get_sigma_sq(
        self,
        K: np.ndarray,
        target_col: np.ndarray,
        nn_indices: np.ndarray,
    ) -> Generator[float, None, None]:
        """
        Generate series of :math:`\\sigma^2` scale parameters for each
        individual solve along a single dimension:

        .. math::
            \\sigma^2 = \\frac{1}{k} * Y_{nn}^T K_{nn}^{-1} Y_{nn}

        Here :math:`Y_{nn}` and :math:`K_{nn}` are the target and kernel
        matrices with respect to the nearest neighbor set in scope, where
        :math:`k` is the number of nearest neighbors.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            target_col:
                A vector of shape `(batch_count)` consisting of the target for
                each nearest neighbor.
            nn_indices:
                An integral matrix of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the test batch.

        Return:
            A generator producing `batch_count` optimal values of
            :math:`\\sigma^2` for each neighborhood for the given response
            dimension.
        """
        batch_count, nn_count = nn_indices.shape
        for j in range(batch_count):
            Y_0 = target_col[nn_indices[j, :]]
            yield Y_0 @ np.linalg.solve(
                K[j, :, :] + self.eps() * np.eye(nn_count), Y_0
            )


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
        >>> from MuyGPyS.gp.muygps import MultivariateMuyGPS as MMuyGPS
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
        self.sigma_sq = SigmaSq()

    def fixed(self) -> bool:
        """
        Checks whether all kernel and model parameters are fixed for each model,
        excluding :math:`\\sigma^2`.

        Returns:
            Returns `True` if all parameters in all models are fixed, and
            `False` otherwise.
        """
        return bool(np.all([model.fixed() for model in self.models]))

    def sigma_sq_optim(
        self,
        pairwise_dists: np.ndarray,
        nn_indices: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Optimize the value of the :math:`\\sigma^2` scale parameter for each
        response dimension.

        We approximate :math:`\\sigma^2` by way of averaging over the analytic
        solution from each local kernel.

        .. math::
            \\sigma^2 = \\frac{1}{n} * Y^T  K^{-1}  Y

        Args:
            pairwise_dists:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count)` -shaped pairwise nearest neighbor
                distance matrices corresponding to each of the batch elements.
            nn_indices:
                An integral matrix of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the testing
                batch.
            targets:
                A matrix of shape `(train_count, response_count)` whose rows
                are the responses for each training element.

        Returns:
            A vector of shape `(response_count,)` listing the found value of
            :math:`\\sigma^2` for each response dimension.
        """
        batch_count, nn_count = nn_indices.shape
        _, response_count = targets.shape
        if response_count != len(self.models):
            raise ValueError(
                f"Response count ({response_count}) does not match the number "
                f"of models ({len(self.models)})."
            )

        K = np.zeros((batch_count, nn_count, nn_count))
        sigma_sqs = np.zeros((response_count,))
        for i, muygps in enumerate(self.models):
            K = muygps.kernel(pairwise_dists)
            sigma_sq = np.zeros(1)
            sigma_sq[0] = np.array(
                sum(muygps._get_sigma_sq(K, targets[:, i], nn_indices))
                / (nn_count * batch_count)
            )
            muygps.sigma_sq._set(val=sigma_sq)
            sigma_sqs[i] = sigma_sq[0]
        self.sigma_sq._set(sigma_sqs)
        return self.sigma_sq()

    def regress_from_indices(
        self,
        indices: np.ndarray,
        nn_indices: np.ndarray,
        test: np.ndarray,
        train: np.ndarray,
        targets: np.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
        return_distances: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Performs simultaneous regression on a list of observations.

        Implicitly creates and discards the distance tensors and matrices. If
        these data structures are needed for later reference, instead use
        :func:`~MuyGPyS.gp.muygps.MultivariateMuyGPS.regress`.

        Args:
            indices:
                An integral vector of shape `(batch_count,)` indices of the
                observations to be approximated.
            nn_indices:
                An integral matrix of shape `(batch_count, nn_count)` listing the
                nearest neighbor indices for all observations in the test batch.
            test:
                The full testing data matrix of shape
                `(test_count, feature_count)`.
            train:
                The full training data matrix of shape
                `(train_count, feature_count)`.
            targets:
                A matrix of shape `(train_count, response_count)` whose rows are
                vector-valued responses for each training element.
            variance_mode:
                Specifies the type of variance to return. Currently supports
                `"diagonal"` and None. If None, report no variance term.
            apply_sigma_sq:
                Indicates whether to scale the posterior variance by `sigma_sq`.
                Unused if `variance_mode is None` or `sigma_sq == "unlearned"`.
            return_distances:
                If `True`, returns a `(test_count, nn_count)` matrix containing
                the crosswise distances between the test elements and their
                nearest neighbor sets and a `(test_count, nn_count, nn_count)`
                tensor containing the pairwise distances between the test data's
                nearest neighbor sets.
        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count,)` whose rows are
            the predicted response for each of the given indices.
        variance:
            A vector of shape `(batch_count,)` consisting of the diagonal
            elements of the posterior variance. Only returned where
            `variance_mode == "diagonal"`.
        crosswise_dists:
            A matrix of shape `(test_count, nn_count)` whose rows list the
            distance of the corresponding test element to each of its nearest
            neighbors. Only returned if `return_distances is True`.
        pairwise_dists:
            A tensor of shape `(test_count, nn_count, nn_count,)` whose latter
            two dimensions contain square matrices containing the pairwise
            distances between the nearest neighbors of the test elements. Only
            returned if `return_distances is True`.
        """
        (
            crosswise_dists,
            pairwise_dists,
            batch_nn_targets,
        ) = make_regress_tensors(
            self.metric,
            indices,
            nn_indices,
            test,
            train,
            targets,
        )
        responses = self.regress(
            pairwise_dists,
            crosswise_dists,
            batch_nn_targets,
            variance_mode=variance_mode,
            apply_sigma_sq=apply_sigma_sq,
        )
        if return_distances is False:
            return responses
        else:
            if variance_mode is None:
                return responses, crosswise_dists, pairwise_dists
            else:
                responses, variances = responses
                return responses, variances, crosswise_dists, pairwise_dists

    def regress(
        self,
        pairwise_dists: np.ndarray,
        crosswise_dists: np.ndarray,
        batch_nn_targets: np.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs simultaneous regression on provided distance tensors and
        the target matrix.

        Computes parallelized local solves of systems of linear equations using
        the kernel realizations, one for each internal model, of the last two
        dimensions of `pairwise_dists` along with `crosswise_dists` and
        `batch_nn_targets` to predict responses in terms of the posterior mean.
        Also computes the posterior variance if `variance_mode` is set
        appropriately. Assumes that distance tensor `pairwise_dists` and
        crosswise distance matrix `crosswise_dists` are already computed and
        given as arguments. To implicitly construct these values from indices
        (useful if the distance tensors and matrices are not needed for later
        reference) instead use
        :func:`~MuyGPyS.gp.muygps.MultivariateMuyGPS.regress_from_indices`.

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
        `(batch_count,)` vector of the jth responses of the neartest neighbors
        given by a slice of the `batch_nn_targets` argument.

        If `variance_mode == "diagonal"`, also return the local posterior
        variances of each prediction, corresponding to the diagonal elements of
        a covariance matrix. For each batch element :math:`\\mathbf{x}_i`, we
        compute

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
            batch_nn_targets:
                A tensor of shape `(batch_count, nn_count, response_count)`
                listing the vector-valued responses for the nearest neighbors
                of each batch element.
            variance_mode:
                Specifies the type of variance to return. Currently supports
                `"diagonal"` and None. If None, report no variance term.
            apply_sigma_sq:
                Indicates whether to scale the posterior variance by `sigma_sq`.
                Unused if `variance_mode is None` or `sigma_sq == "unlearned"`.


        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count,)` whose rows are
            the predicted response for each of the given indices.
        diagonal_variance:
            A vector of shape `(batch_count, response_count)` consisting of the
            diagonal elements of the posterior variance for each model. Only
            returned where `variance_mode == "diagonal"`.
        """
        batch_count, nn_count, response_count = batch_nn_targets.shape
        responses = np.zeros((batch_count, response_count))
        if variance_mode is None:
            pass
        elif variance_mode == "diagonal":
            diagonal_variance = np.zeros((batch_count, response_count))
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )
        for i, model in enumerate(self.models):
            K = model.kernel(pairwise_dists)
            Kcross = model.kernel(crosswise_dists)
            responses[:, i] = model._compute_solve(
                K,
                Kcross,
                batch_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
            ).reshape(batch_count)
            if variance_mode == "diagonal":
                diagonal_variance[:, i] = model._compute_diagonal_variance(
                    K, Kcross
                ).reshape(batch_count)
                if apply_sigma_sq and isinstance(self.sigma_sq(), np.ndarray):
                    diagonal_variance[:, i] *= self.sigma_sq()[i]
        if variance_mode == "diagonal":
            return responses, diagonal_variance
        return responses
