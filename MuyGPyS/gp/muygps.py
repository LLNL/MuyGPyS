# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, Union
from MuyGPyS.gp.distance import crosswise_distances

from MuyGPyS.gp.kernels import (
    _get_kernel,
    _init_hyperparameter,
    Hyperparameter,
    SigmaSq,
)

from MuyGPyS import config

from MuyGPyS._src.gp.distance import (
    _make_regress_tensors,
    _make_fast_regress_tensors,
)


from MuyGPyS._src.gp.distance.numpy import (
    _make_regress_tensors as _make_regress_tensors_n,
)
from MuyGPyS._src.gp.muygps import (
    _muygps_compute_solve,
    _muygps_compute_diagonal_variance,
    _muygps_fast_regress_solve,
    _muygps_fast_regress_precompute,
)
from MuyGPyS._src.mpi_utils import _is_mpi_mode
from MuyGPyS.optimize.utils import _switch_on_opt_method


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
            if not self.kernel.hyperparameters[p].fixed():
                return False
        if not self.eps.fixed():
            return False
        return True

    def get_optim_params(
        self,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
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
        names, params, bounds = self.kernel.get_optim_params()
        if not self.eps.fixed():
            names.append("eps")
            params.append(self.eps())
            bounds.append(self.eps.get_bounds())
        return names, np.array(params), np.array(bounds)

    @staticmethod
    def _compute_solve(
        K: np.ndarray,
        Kcross: np.ndarray,
        batch_nn_targets: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        @NOTE[bwp] We might want to get rid of these static methods.

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
            eps:
                The value of the homoscedastic nugget parameter.

        Returns:
            A matrix of shape `(batch_count, response_count)` listing the
            predicted response for each of the batch elements.
        """
        return _muygps_compute_solve(K, Kcross, batch_nn_targets, eps)

    @staticmethod
    def _compute_diagonal_variance(
        K: np.ndarray,
        Kcross: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """
        Simultaneously solve all of the GP inference systems of linear
        equations.

        @NOTE[bwp] We might want to get rid of these static methods.

        Args:
            K:
                A tensor of shape `(batch_count, nn_count, nn_count)` containing
                the `(nn_count, nn_count` -shaped kernel matrices corresponding
                to each of the batch elements.
            Kcross:
                A matrix of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance vector corresponding
                to each of the batch elements.
            eps:
                The value of the homoscedastic nugget parameter.

        Returns:
            A vector of shape `(batch_count)` listing the diagonal variances for
            each of the batch elements.
        """
        return _muygps_compute_diagonal_variance(K, Kcross, eps)

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
        indices_by_rank: bool = False,
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
                Unused if `variance_mode is None` or
                `sigma_sq.trained() is False`.
            return_distances:
                If `True`, returns a `(test_count, nn_count)` matrix containing
                the crosswise distances between the test elements and their
                nearest neighbor sets and a `(test_count, nn_count, nn_count)`
                tensor containing the pairwise distances between the test data's
                nearest neighbor sets.
            indices_by_rank:
                If `True`, construct the tensors using local indices with no
                communication. Only for use in MPI mode.

        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count)` whose rows are
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
            A tensor of shape `(test_count, nn_count, nn_count)` whose latter
            two dimensions contain square matrices containing the pairwise
            distances between the nearest neighbors of the test elements. Only
            returned if `return_distances is True`.
        """
        tensor_fn = (
            _make_regress_tensors_n
            if _is_mpi_mode() is True and indices_by_rank is True
            else _make_regress_tensors
        )
        (crosswise_dists, pairwise_dists, batch_nn_targets,) = tensor_fn(
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

    def build_fast_regress_coeffs(
        self,
        train: np.ndarray,
        nn_indices: np.ndarray,
        targets: np.ndarray,
        indices_by_rank: bool = False,
    ) -> np.ndarray:
        """
        Produces coefficient matrix for fast regression given in Equation
        (8) of [dunton2022fast]_. To form each row of this matrix, we compute

        .. math::
            \\mathbf{C}_{N^*}(i, :) =
                (K_{\\hat{\\theta}} (X_{N^*}, X_{N^*})
                + \\varepsilon I_k)^{-1} Y(X_{N^*}).

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the ith
        test point and the `nn_count - 1` nearest neighbors of this nearest
        neighbor, :math:`K_{\\hat{\\theta}}` is the trained kernel functor
        specified by `self.kernel`, :math:`\\varepsilon I_k` is a diagonal
        homoscedastic noise matrix whose diagonal is the value of the
        `self.eps` hyperparameter, and :math:`Y(X_{N^*})` is the
        `(train_count,)` vector of responses corresponding to the
        training features indexed by $N^*$.

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
            A matrix of shape `(train_count, nn_count)` whose rows are
            the precomputed coefficients for fast regression.

        """
        (
            pairwise_dists_fast,
            train_nn_targets_fast,
        ) = _make_fast_regress_tensors(
            self.kernel.metric, nn_indices, train, targets
        )
        K = self.kernel(pairwise_dists_fast)

        return self._build_fast_regress_coeffs(
            K, self.eps(), train_nn_targets_fast
        )

    @staticmethod
    def _build_fast_regress_coeffs(
        K: np.ndarray,
        eps: float,
        train_nn_targets_fast: np.ndarray,
    ) -> np.ndarray:

        return _muygps_fast_regress_precompute(K, eps, train_nn_targets_fast)

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
                A matrix of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance matrix corresponding
                to each of the batch elements.
            batch_nn_targets:
                A tensor of shape `(batch_count, nn_count, response_count)`
                whose last dimension lists the vector-valued responses for the
                nearest neighbors of each batch element.
            variance_mode:
                Specifies the type of variance to return. Currently supports
                `"diagonal"` and None. If None, report no variance term.
            apply_sigma_sq:
                Indicates whether to scale the posterior variance by `sigma_sq`.
                Unused if `variance_mode is None` or
                `sigma_sq.trained() is False`.

        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        diagonal_variance:
            A vector of shape `(batch_count,)` consisting of the diagonal
            elements of the posterior variance, or a matrix of shape
            `(batch_count, response_count)` for a multidimensional response.
            Only returned where `variance_mode == "diagonal"`.
        """
        return self._regress(
            K,
            Kcross,
            batch_nn_targets,
            self.eps(),
            self.sigma_sq(),
            variance_mode=variance_mode,
            apply_sigma_sq=(apply_sigma_sq and self.sigma_sq.trained()),
        )

    @staticmethod
    def _regress(
        K: np.array,
        Kcross: np.array,
        batch_nn_targets: np.array,
        eps: float,
        sigma_sq: np.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        responses = MuyGPS._compute_solve(K, Kcross, batch_nn_targets, eps)
        if variance_mode is None:
            return responses
        elif variance_mode == "diagonal":
            diagonal_variance = MuyGPS._compute_diagonal_variance(
                K, Kcross, eps
            )
            if apply_sigma_sq is True:
                if len(sigma_sq) == 1:
                    diagonal_variance *= sigma_sq
                else:
                    diagonal_variance = np.array(
                        [ss * diagonal_variance for ss in sigma_sq]
                    ).T
            return responses, diagonal_variance
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )

    def fast_regress_from_indices(
        self,
        indices: np.ndarray,
        nn_indices: np.ndarray,
        test_features: np.ndarray,
        train_features: np.ndarray,
        closest_index: np.ndarray,
        coeffs_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Performs fast regression using provided
        cross-covariance, the index of the training point closest to the
        queried test point, and precomputed coefficient matrix.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations, as computed in
        Equation (9) of [dunton2022fast]_. For each test point
        :math:`\\mathbf{z}`, we compute

        .. math::
            \\widehat{Y} (\\mathbf{z} \\mid X) =
                K_\\theta (\\mathbf{z}, X_{N^*}) \mathbf{C}_{N^*}.

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the queried
        test point :math:`\\mathbf{z}` and the nearest neighbors of that
        training point, :math:`K_\\theta` is the kernel functor specified
        by `self.kernel`, and :math:`\mathbf{C}_{N^*}` is the matrix of
        precomputed coefficients given in Equation (8) of [dunton2022fast]_.

        Args:
            indices:
                A vector of shape `('batch_count,)` providing the indices of the
                test features to be queried in the formation of the crosswise
                distance tensor.
            nn_indices:
                A matrix of shape `('batch_count, nn_count)` providing the index
                of the closest training point to each queried test point, as
                well as the `nn_count - 1` closest neighbors of that point.
            test_features:
                A matrix of shape `(batch_count, feature_count)` containing
                the test data points.
            train_features:
                A matrix of shape `(train_count, feature_count)` containing the
                training data.
            closest_index:
                A vector of shape `('batch_count,)` for which each
                entry is the index of the training point closest to
                each queried point.
            coeffs_mat:
                A matrix of shape `('batch_count, nn_count)` providing
                precomputed coefficients for fast regression.

        Returns:
            A matrix of shape `(batch_count,)` whose rows are
            the predicted response for each of the given indices.
        """
        crosswise_dists = crosswise_distances(
            test_features,
            train_features,
            indices,
            nn_indices,
        )

        Kcross = self.kernel(crosswise_dists)
        return self.fast_regress(
            Kcross,
            coeffs_mat[closest_index, :],
        )

    def fast_regress(
        self,
        Kcross: np.ndarray,
        coeffs_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Performs fast regression using provided
        cross-covariance and precomputed coefficient matrix.

        Assumes that cross-covariance matrix `Kcross` is already computed and
        given as an argument. To implicitly construct these values from indices
        instead use :func:`~MuyGPyS.gp.muygps.MuyGPS.fast_regress_from_indices`.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations, as computed in
        Equation (9) of [dunton2022fast]_. For each test point
        :math:`\\mathbf{z}`, we compute

        .. math::
            \\widehat{Y} (\\mathbf{z} \\mid X) =
                K_\\theta (\\mathbf{z}, X_{N^*}) \mathbf{C}_{N^*}.

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the queried
        test point :math:`\\mathbf{z}` and the nearest neighbors of that
        training point, :math:`K_\\theta` is the kernel functor
        specified by `self.kernel`, and :math:`\mathbf{C}_{N^*}` is
        the matrix of precomputed coefficients given in Equation (8)
        of [dunton2022fast]_.

        Args:
            Kcross:
                A matrix of shape `(batch_count, nn_count)` containing the
                `1 x nn_count` -shaped cross-covariance vector corresponding
                to each of the batch elements.
            coeffs_mat:
                A matrix of shape `(batch_count, nn_count)` whose rows
                are given by precomputed coefficients for fast regression.


        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        return self._fast_regress(
            Kcross,
            coeffs_mat,
        )

    @staticmethod
    def _fast_regress(
        Kcross: np.ndarray,
        coeffs_mat: np.ndarray,
    ) -> np.ndarray:
        responses = _muygps_fast_regress_solve(Kcross, coeffs_mat)
        return responses

    def get_opt_mean_fn(self, opt_method) -> Callable:
        """
        Return a posterior mean function for use in optimization.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()`. The
        `opt_method` parameter determines the format of the returned function.

        Returns:
            A function implementing a posterior mean, where `eps` is either
            fixed or takes updating values during optimization. The format of
            the function depends upon `opt_method`.
        """
        return _switch_on_opt_method(
            opt_method, self.get_kwargs_opt_mean_fn, self.get_array_opt_mean_fn
        )

    def get_array_opt_mean_fn(self) -> Callable:
        """
        Return a posterior mean function for use in optimization.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
        `opt_method="scipy"`, and assumes that the optimization parameters will
        be passed in an `(optim_count,)` vector where `eps` is either the last
        element or is not included.

        Returns:
            A function implementing posterior mean, where `eps` is either fixed
            or takes updating values during optimization. The function expects a
            list of current hyperparameter values for unfixed parameters, which
            are expected to occur in a certain order matching how they are set
            in `~MuyGPyS.gp.muygps.MuyGPS.get_optim_params()`.
        """
        return self._get_array_opt_mean_fn(_muygps_compute_solve, self.eps)

    @staticmethod
    def _get_array_opt_mean_fn(
        solve_fn: Callable, eps: Hyperparameter
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, batch_nn_targets, x0):
                return solve_fn(K, Kcross, batch_nn_targets, x0[-1])

        else:

            def caller_fn(K, Kcross, batch_nn_targets, x0):
                return solve_fn(K, Kcross, batch_nn_targets, eps())

        return caller_fn

    def get_kwargs_opt_mean_fn(self) -> Callable:
        """
        Return a posterior mean function for use in optimization.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()` with
        `opt_method="bayesian"`, and assumes that either `eps` will be passed
        via a keyword argument or not at all.

        Returns:
            A function implementing the posterior mean, where `eps` is either
            fixed or takes updating values during optimization. The function
            expects keyword arguments corresponding to current hyperparameter
            values for unfixed parameters.
        """
        return self._get_kwargs_opt_mean_fn(_muygps_compute_solve, self.eps)

    @staticmethod
    def _get_kwargs_opt_mean_fn(
        solve_fn: Callable, eps: Hyperparameter
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, batch_nn_targets, **kwargs):
                return solve_fn(K, Kcross, batch_nn_targets, kwargs["eps"])

        else:

            def caller_fn(K, Kcross, batch_nn_targets, **kwargs):
                return solve_fn(K, Kcross, batch_nn_targets, eps())

        return caller_fn

    def get_opt_var_fn(self, opt_method) -> Callable:
        """
        Return a posterior variance function for use in optimization.

        This function is designed for use with
        :func:`MuyGPyS.optimize.chassis.optimize_from_tensors()`. The
        `opt_method` parameter determines the format of the returned function.

        Returns:
            A function implementing posterior variance, where `eps` is either
            fixed or takes updating values during optimization. The format of
            the function depends upon `opt_method`.
        """
        return _switch_on_opt_method(
            opt_method, self.get_kwargs_opt_var_fn, self.get_array_opt_var_fn
        )

    def get_array_opt_var_fn(self) -> Callable:
        return self._get_array_opt_var_fn(
            _muygps_compute_diagonal_variance, self.eps
        )

    @staticmethod
    def _get_array_opt_var_fn(
        var_fn: Callable, eps: Hyperparameter
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, x0):
                return var_fn(K, Kcross, x0[-1])

        else:

            def caller_fn(K, Kcross, x0):
                return var_fn(K, Kcross, eps())

        return caller_fn

    def get_kwargs_opt_var_fn(self) -> Callable:
        return self._get_kwargs_opt_var_fn(
            _muygps_compute_diagonal_variance, self.eps
        )

    @staticmethod
    def _get_kwargs_opt_var_fn(
        var_fn: Callable, eps: Hyperparameter
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, **kwargs):
                return var_fn(K, Kcross, kwargs["eps"])

        else:

            def caller_fn(K, Kcross, **kwargs):
                return var_fn(K, Kcross, eps())

        return caller_fn


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
        indices_by_rank: bool = False,
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
                Unused if `variance_mode is None` or
                `sigma_sq.trained() is False`.
            return_distances:
                If `True`, returns a `(test_count, nn_count)` matrix containing
                the crosswise distances between the test elements and their
                nearest neighbor sets and a `(test_count, nn_count, nn_count)`
                tensor containing the pairwise distances between the test data's
                nearest neighbor sets.
            indices_by_rank:
                If `True`, construct the tensors using local indices with no
                communication. Only for use in MPI mode.
        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count)` whose rows are
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
            A tensor of shape `(test_count, nn_count, nn_count)` whose latter
            two dimensions contain square matrices containing the pairwise
            distances between the nearest neighbors of the test elements. Only
            returned if `return_distances is True`.
        """
        tensor_fn = (
            _make_regress_tensors_n
            if _is_mpi_mode() is True and indices_by_rank is True
            else _make_regress_tensors
        )

        (crosswise_dists, pairwise_dists, batch_nn_targets,) = tensor_fn(
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
                Unused if `variance_mode is None` or
                `sigma_sq.leanred() is False`.


        Returns
        -------
        responses:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        diagonal_variance:
            A vector of shape `(batch_count, response_count)` consisting of the
            diagonal elements of the posterior variance for each model. Only
            returned where `variance_mode == "diagonal"`.
        """
        return self._regress(
            self.models,
            pairwise_dists,
            crosswise_dists,
            batch_nn_targets,
            self.sigma_sq,
            variance_mode=variance_mode,
            apply_sigma_sq=(apply_sigma_sq and self.sigma_sq.trained()),
        )

    @staticmethod
    def _regress(
        models: List[MuyGPS],
        pairwise_dists: np.ndarray,
        crosswise_dists: np.ndarray,
        batch_nn_targets: np.ndarray,
        sigma_sq: SigmaSq,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        for i, model in enumerate(models):
            K = model.kernel(pairwise_dists)
            Kcross = model.kernel(crosswise_dists)
            responses[:, i] = model._compute_solve(
                K,
                Kcross,
                batch_nn_targets[:, :, i].reshape(batch_count, nn_count, 1),
                model.eps(),
            ).reshape(batch_count)
            if variance_mode == "diagonal":
                diagonal_variance[:, i] = model._compute_diagonal_variance(
                    K, Kcross, model.eps()
                ).reshape(batch_count)
                if apply_sigma_sq:
                    diagonal_variance[:, i] *= sigma_sq()[i]
        if variance_mode == "diagonal":
            return responses, diagonal_variance
        return responses

    def build_fast_regress_coeffs(
        self,
        train: np.ndarray,
        nn_indices: np.ndarray,
        targets: np.ndarray,
        indices_by_rank: bool = False,
    ) -> np.ndarray:
        """
        Produces coefficient tensor for fast regression given in Equation
        (8) of [dunton2022fast]_. To form the tensor, we compute

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
            regression.

        """
        (
            pairwise_dists_fast,
            train_nn_targets_fast,
        ) = _make_fast_regress_tensors(self.metric, nn_indices, train, targets)

        return self._build_fast_regress_coeffs(
            self.models, pairwise_dists_fast, train_nn_targets_fast
        )

    @staticmethod
    def _build_fast_regress_coeffs(
        models: List[MuyGPS],
        pairwise_dists_fast: np.ndarray,
        train_nn_targets_fast: np.ndarray,
    ) -> np.ndarray:
        train_count, nn_count, response_count = train_nn_targets_fast.shape
        coeffs_mat = np.zeros((train_count, nn_count, response_count))
        for i, model in enumerate(models):
            K = model.kernel(pairwise_dists_fast)
            coeffs_mat[:, :, i] = _muygps_fast_regress_precompute(
                K, model.eps(), train_nn_targets_fast[:, :, i]
            )

        return coeffs_mat

    def fast_regress_from_indices(
        self,
        indices: np.ndarray,
        nn_indices: np.ndarray,
        test_features: np.ndarray,
        train_features: np.ndarray,
        closest_index: np.ndarray,
        coeffs_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Performs fast multivariate regression using provided
        vectors and matrices used in constructed the crosswise distances matrix,
        the index of the training point closest to the queried test point,
        and precomputed coefficient matrix.

        Returns the predicted response in the form of a posterior
        mean for each element of the batch of observations, as computed in
        Equation (9) of [dunton2022fast]_. For each test point
        :math:`\\mathbf{z}`, we compute

        .. math::
            \\widehat{Y} (\\mathbf{z} \\mid X) =
                K_\\theta (\\mathbf{z}, X_{N^*}) \mathbf{C}_{N^*}.

        Here :math:`X_{N^*}` is the union of the nearest neighbor of the queried
        test point :math:`\\mathbf{z}` and the nearest neighbors of that
        training point, :math:`K_\\theta` is the kernel functor specified
        by `self.kernel`, and :math:`\mathbf{C}_{N^*}` is the matrix of
        precomputed coefficients given in Equation (8) of [dunton2022fast]_.

        Args:
            indices:
                A vector of shape `('batch_count,)` providing the indices of the
                test features to be queried in the formation of the crosswise
                distance tensor.
            nn_indices:
                A matrix of shape `('batch_count, nn_count)` providing the index
                of the closest training point to each queried test point, as
                well as the `nn_count - 1` closest neighbors of that point.
            test_features:
                A matrix of shape `(batch_count, feature_count)` containing
                the test data points.
            train_features:
                A matrix of shape `(train_count, feature_count)` containing the
                training data.
            closest_index:
                A vector of shape `(batch_count,)` for which each entry is
                the index of the training point closest to each queried
                test point.
            coeffs_mat:
                A tensor of shape `(batch_count, nn_count, response_count)`
                providing the precomputed coefficients for fast regression.

        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """

        crosswise_dists = crosswise_distances(
            test_features,
            train_features,
            indices,
            nn_indices,
        )

        return self.fast_regress(
            crosswise_dists,
            coeffs_mat[closest_index, :, :],
        )

    def fast_regress(
        self,
        crosswise_dists: np.ndarray,
        coeffs_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Performs fast regression using provided
        crosswise distances and precomputed coefficient matrix.

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
            coeffs_mat:
                A tensor of shape `(batch_count, nn_count, response_count)`
                providing the precomputed coefficients for fast regression.


        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        models = self.models
        responses = self._fast_regress(models, crosswise_dists, coeffs_mat)
        return responses

    @staticmethod
    def _fast_regress(
        models: List[MuyGPS],
        crosswise_dists: np.ndarray,
        coeffs_mat: np.ndarray,
    ) -> np.ndarray:
        Kcross = np.zeros(coeffs_mat.shape)
        for i, model in enumerate(models):
            Kcross[:, :, i] = model.kernel(crosswise_dists)
        return _muygps_fast_regress_solve(Kcross, coeffs_mat)
