# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
MuyGPs implementation
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import MuyGPyS._src.math as mm
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
from MuyGPyS._src.gp.noise import _homoscedastic_perturb
from MuyGPyS._src.mpi_utils import _is_mpi_mode
from MuyGPyS.gp.distance import crosswise_distances
from MuyGPyS.gp.kernels import (
    _get_kernel,
    _init_hyperparameter,
    SigmaSq,
)
from MuyGPyS.gp.noise import HomoscedasticNoise
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
        response_count: int = 1,
        **kwargs,
    ):
        self.kern = kern.lower()
        self.kernel = _get_kernel(self.kern, **kwargs)
        self.eps = _init_hyperparameter(
            1e-14, "fixed", HomoscedasticNoise, **eps
        )
        self.sigma_sq = SigmaSq(response_count)

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
        names, params, bounds = self.kernel.get_optim_params()
        if not self.eps.fixed():
            names.append("eps")
            params.append(self.eps())
            bounds.append(self.eps.get_bounds())
        return names, mm.array(params), mm.array(bounds)

    @staticmethod
    def _compute_solve(
        K: mm.ndarray,
        Kcross: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        eps: float,
    ) -> mm.ndarray:
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
        return _muygps_compute_solve(
            _homoscedastic_perturb(K, eps), Kcross, batch_nn_targets
        )

    @staticmethod
    def _compute_diagonal_variance(
        K: mm.ndarray,
        Kcross: mm.ndarray,
        eps: float,
    ) -> mm.ndarray:
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
        return _muygps_compute_diagonal_variance(
            _homoscedastic_perturb(K, eps), Kcross
        )

    def regress_from_indices(
        self,
        indices: mm.ndarray,
        nn_indices: mm.ndarray,
        test: mm.ndarray,
        train: mm.ndarray,
        targets: mm.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
        return_distances: bool = False,
        indices_by_rank: bool = False,
    ) -> Union[
        mm.ndarray,
        Tuple[mm.ndarray, mm.ndarray],
        Tuple[mm.ndarray, mm.ndarray, mm.ndarray],
        Tuple[mm.ndarray, mm.ndarray, mm.ndarray, mm.ndarray],
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
                `sigma_sq.trained is False`.
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
        train: mm.ndarray,
        nn_indices: mm.ndarray,
        targets: mm.ndarray,
        indices_by_rank: bool = False,
    ) -> mm.ndarray:
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
        K: mm.ndarray,
        eps: float,
        train_nn_targets_fast: mm.ndarray,
    ) -> mm.ndarray:

        return _muygps_fast_regress_precompute(
            _homoscedastic_perturb(K, eps), train_nn_targets_fast
        )

    def regress(
        self,
        K: mm.ndarray,
        Kcross: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
    ) -> Union[mm.ndarray, Tuple[mm.ndarray, mm.ndarray]]:
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
                `sigma_sq.trained is False`.

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
            apply_sigma_sq=(apply_sigma_sq and self.sigma_sq.trained),
        )

    @staticmethod
    def _regress(
        K: mm.ndarray,
        Kcross: mm.ndarray,
        batch_nn_targets: mm.ndarray,
        eps: float,
        sigma_sq: mm.ndarray,
        variance_mode: Optional[str] = None,
        apply_sigma_sq: bool = True,
    ) -> Union[mm.ndarray, Tuple[mm.ndarray, mm.ndarray]]:
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
                    diagonal_variance = mm.array(
                        [ss * diagonal_variance for ss in sigma_sq]
                    ).T
            return responses, diagonal_variance
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )

    def fast_regress_from_indices(
        self,
        indices: mm.ndarray,
        nn_indices: mm.ndarray,
        test_features: mm.ndarray,
        train_features: mm.ndarray,
        closest_index: mm.ndarray,
        coeffs_tensor: mm.ndarray,
    ) -> mm.ndarray:
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
            coeffs_tensor:
                A matrix of shape `('batch_count, nn_count, response_count)` providing
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
            coeffs_tensor[closest_index, :],
        )

    def fast_regress(
        self,
        Kcross: mm.ndarray,
        coeffs_tensor: mm.ndarray,
    ) -> mm.ndarray:
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
            coeffs_tensor:
                A matrix of shape `(batch_count, nn_count, response_count)` whose rows
                are given by precomputed coefficients for fast regression.


        Returns:
            A matrix of shape `(batch_count, response_count)` whose rows are
            the predicted response for each of the given indices.
        """
        return self._fast_regress(
            Kcross,
            coeffs_tensor,
        )

    @staticmethod
    def _fast_regress(
        Kcross: mm.ndarray,
        coeffs_tensor: mm.ndarray,
    ) -> mm.ndarray:
        responses = _muygps_fast_regress_solve(Kcross, coeffs_tensor)
        return responses

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
        if isinstance(self.eps, HomoscedasticNoise):
            return self._get_opt_mean_fn(
                _muygps_compute_solve, _homoscedastic_perturb, self.eps
            )
        else:
            raise TypeError(
                f"Noise parameter type {type(self.eps)} is not supported for "
                f"optimization!"
            )

    @staticmethod
    def _get_opt_mean_fn(
        solve_fn: Callable, perturb_fn: Callable, eps: HomoscedasticNoise
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, batch_nn_targets, **kwargs):
                return solve_fn(
                    perturb_fn(K, kwargs["eps"]), Kcross, batch_nn_targets
                )

        else:

            def caller_fn(K, Kcross, batch_nn_targets, **kwargs):
                return solve_fn(perturb_fn(K, eps()), Kcross, batch_nn_targets)

        return caller_fn

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
        if isinstance(self.eps, HomoscedasticNoise):
            return self._get_opt_var_fn(
                _muygps_compute_diagonal_variance,
                _homoscedastic_perturb,
                self.eps,
            )
        else:
            raise TypeError(
                f"Noise parameter type {type(self.eps)} is not supported for "
                f"optimization!"
            )

    @staticmethod
    def _get_opt_var_fn(
        var_fn: Callable, perturb_fn: Callable, eps: HomoscedasticNoise
    ) -> Callable:
        if not eps.fixed():

            def caller_fn(K, Kcross, **kwargs):
                return var_fn(perturb_fn(K, kwargs["eps"]), Kcross)

        else:

            def caller_fn(K, Kcross, **kwargs):
                return var_fn(perturb_fn(K, eps()), Kcross)

        return caller_fn

    def __eq__(self, rhs) -> bool:
        if isinstance(rhs, self.__class__):
            return all(
                (
                    self.kern == rhs.kern,
                    all(
                        self.kernel.hyperparameters[h]()
                        == rhs.kernel.hyperparameters[h]()
                        for h in self.kernel.hyperparameters
                    ),
                    self.eps() == rhs.eps(),
                    self.sigma_sq() == rhs.sigma_sq(),
                )
            )
        else:
            return False
