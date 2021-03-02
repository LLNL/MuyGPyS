#!/usr/bin/env python
# encoding: utf-8
"""
@file lkgp.py

Created by priest2 on 2020-10-19

Implements the local kriging GP approximation logic.
"""

import numpy as np

from sklearn.gaussian_process.kernels import Matern, RBF

from muyscans.gp.kernels import NNGP


class LKGP:
    """
    Local Kriging Gaussian Process.

    Performs approximate GP inference by locally approximating an observation's
    response using its nearest neighbors.
    """

    def __init__(self, kern="matern", **kwargs):
        """
        Initialize.

        Parameters
        ----------
        eps : float
            Homoscedastic noise prior variance.
        kern : str
            The kernel to be used. Currently supports ``matern'', ``rbf'' and
            ``nngp''.
        """
        self.kern = kern.lower()
        self.set_params(**kwargs)

    def set_params(self, **params):
        # NOTE[bwp] this logic should get moved into kernel functors once
        # implemented
        self.params = {
            p: params[p] for p in params if p != "eps" and p != "sigma_sq"
        }
        self.eps = params.get("eps", 0.015)
        self.sigma_sq = params.get("sigma_sq", np.array(1.0))
        if self.kern == "matern":
            self.kernel = Matern(
                length_scale=self.params.get("length_scale", 10.0),
                nu=self.params.get("nu", 0.5),
            )
            unset_params = {"eps", "sigma_sq", "length_scale", "nu"}.difference(
                params.keys()
            )
        elif self.kern == "rbf":
            self.kernel = RBF(length_scale=self.params.get("length_scale", 0.5))
            unset_params = {"eps", "sigma_sq", "length_scale"}.difference(
                params.keys()
            )
        elif self.kern == "nngp":
            self.kernel = NNGP(
                sigma_b_sq=self.params.get("sigma_b_sq", 0.5),
                sigma_w_sq=self.params.get("sigma_w_sq", 0.5),
            )
            unset_params = {
                "eps",
                "sigma_sq",
                "sigma_b_sq",
                "sigma_w_sq",
            }.difference(params.keys())
        else:
            raise NotImplementedError(f"{self.kern} is not implemented yet!")
        return sorted(list(unset_params))

    def set_param_array(self, names, values):
        # NOTE[bwp] this logic should get moved into kernel functors once
        # implemented
        names = list(names)
        # this is going to break if we add a hyperparameter that occurs earlier
        # in alphabetical order.
        if names[0] == "eps":
            self.eps = values[0]
            names = names[1:]
            values = values[1:]
        for i, name in enumerate(names):
            self.params[name] = values[i]
        if self.kern == "matern":
            self.kernel = Matern(**self.params)
        elif self.kern == "rbf":
            self.kernel = RBF(**self.params)
        elif self.kern == "nngp":
            self.kernel = NNGP(**self.params)

    def optim_bounds(self, names, eps=1e-6):
        """
        Return hyperparameter bounds.

        NOTE[bwp]: Currently hard-coded. Do we want this to be configurable?
        NOTE[bwp] this logic should get moved into kernel functors once
        implemented
        """
        ret = list()
        if "eps" in names:
            ret.append((eps, 0.2))
        if self.kern == "matern":
            if "length_scale" in names:
                ret.append((eps, 40.0))
            if "nu" in names:
                ret.append((eps, 2.0))
        elif self.kern == "rbf":
            if "length_scale" in names:
                ret.append((eps, 40.0))
        elif self.kern == "nngp":
            if "sigma_b_sq" in names:
                ret.append((eps, 2.0))
            if "sigma_w_sq" in names:
                ret.append((eps, 2.0))
        return ret

    def regress(
        self,
        indices,
        nn_indices,
        test,
        train,
        targets,
        variance_mode=None,
    ):
        """
        Performs simultaneous regression on a list of observations.

        Parameters
        ----------
        index : np.ndarray, type = int, shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray, type=int, shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray, type = float, shape = ``(train_count, dim)''
            The full training data matrix.
        test : numpy.ndarray, type = float, shape = ``(test_count, dim)''
            The full testing data matrix.
        targets : numpy.ndarray, type = float,
                  shape = ``(train_count, ouput_dim)''
            Vector-valued responses for each training element.
        variance_mode : str or None
            Specifies the type of variance to return. Currently supports
            ``diagonal'' and None. If None, report no variance term.

        Returns
        -------
        responses : numpy.ndarray, type = float,
                    shape = ``(batch_count, output_dim,)''
            The predicted response for each of the given indices.
        diagonal_variance : numpy.ndarray, type = float,
                   shape = ``(batch_count, )
            The diagonal elements of the posterior variance. Only returned where
            ``variance_mode == "diagonal"''.
        """
        batch_size = len(indices)
        output_dim = targets.shape[1]
        nn_count = nn_indices.shape[1]
        dim = test.shape[1]
        # NOTE[bwp] This is clugy and terrible. Need to reenginer nngp.
        # NOTE[bwp] In fact, should reengineer all kernels so as to use on-node
        # parallelism. This is one of the main bottlenecks right now.
        if type(self.kernel) == NNGP:
            Kfull = np.array(
                [
                    self.kernel(np.vstack((mat, vec.reshape(1, dim))))
                    for vec, mat in zip(test[indices], train[nn_indices])
                ]
            )
            K = Kfull[:, :-1, :-1]
            Kcross = Kfull[:, -1, :-1].reshape((batch_size, 1, nn_count))
        else:
            K = np.array([self.kernel(mat) for mat in train[nn_indices]])
            Kcross = np.array(
                [
                    self.kernel(vec.reshape(1, dim), mat)
                    for vec, mat in zip(test[indices], train[nn_indices])
                ]
            )
        solve = Kcross @ np.linalg.solve(
            K + self.eps * np.eye(nn_count), targets[nn_indices, :]
        )
        responses = solve.reshape(batch_size, output_dim)
        if variance_mode is None:
            return responses
        elif variance_mode == "diagonal":
            Kcross = Kcross.reshape(batch_size, nn_count)
            diagonal_variance = np.array(
                [
                    1.0
                    - Kcross[i, :]
                    @ np.linalg.solve(
                        K[i, :, :] + self.eps * np.eye(nn_count), Kcross[i, :]
                    )
                    for i in range(batch_size)
                ]
            )
            return responses, diagonal_variance
        else:
            raise NotImplementedError(
                f"Variance mode {variance_mode} is not implemented."
            )

    def sigma_sq_optim(
        self,
        indices,
        nn_indices,
        train,
        targets,
    ):
        """
        Optimize the value of the sigma^2 scale parameter.

        sigma^2 = 1/n * Y^T @ K^{-1} @ Y

        Parameters
        ----------
        index : np.ndarray, type = int, shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray, type=int, shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray, type = float, shape = ``(train_count, dim)''
            The full training data matrix.
        targets : numpy.ndarray, type = float,
                  shape = ``(train_count, ouput_dim)''
            Vector-valued responses for each training element.

        Returns
        -------
        sigmas : numpy.ndarray, type = float, shape = ``(output_dim,)''
            The value of sigma^2 for each dimension.
        """
        batch_size, nn_count = nn_indices.shape
        out_dim = targets.shape[1]

        K = np.array([self.kernel(mat) for mat in train[nn_indices]])
        # sigmas = np.zeros((batch_size, out_dim))
        sigmas = np.zeros((out_dim,))
        for i in range(out_dim):
            sigmas[i] = sum(self._get_sigma_sq(K, targets[:, i], nn_indices))
        self.sigma_sq = sigmas / (nn_count * batch_size)

    def get_sigma_sq(
        self,
        indices,
        nn_indices,
        train,
        target_col,
    ):
        """
        Return the series of sigma^2 scale parameters for each neighborhood
        solve.
        NOTE[bwp]: This function is only for testing purposes.

        Parameters
        ----------
        index : np.ndarray, type = int, shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray, type=int, shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray, type = float, shape = ``(train_count, dim)''
            The full training data matrix.
        target_col : numpy.ndarray, type = float, shape = ``(train_count,)''
            The target vector consisting of the target for each nearest
            neighbor.

        Returns
        -------
        sigmas : numpy.ndarray, type = float, shape = ``(output_dim,)''
            The value of sigma^2 for each dimension.
        """
        batch_size, nn_count = nn_indices.shape

        K = np.array([self.kernel(mat) for mat in train[nn_indices]])
        # sigmas = np.zeros((batch_size, out_dim))
        sigmas = np.zeros((batch_size,))
        for i, el in enumerate(self._get_sigma_sq(K, target_col, nn_indices)):
            sigmas[i] = el
        return sigmas / nn_count

    def _get_sigma_sq(self, K, target_col, nn_indices):
        """
        Generate series of sigma^2 scale parameters for each individual solve
        along a single dimension:

        sigma^2 = 1/nn * Y_{nn}^T @ K_{nn}^{-1} @ Y_{nn}

        Parameters
        ----------
        K : np.ndarray, type = float,
                shape = ``(batch_count, nn_count, nn_count)''
            Kernel tensor containing nearest neighbor kernels for each local
            neighborhood.
        target_col : numpy.ndarray, type = float, shape = ``(batch_count,)''
            The target vector consisting of the target for each nearest
            neighbor.
        nn_indices : numpy.ndarray, type=int, shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.

        Yields
        -------
        sigmas : numpy.ndarray, type = float, shape = ``(batch_count,)''
            The optimal value of sigma^2 for each neighborhood for the given
            output dimension.
        """
        batch_size, nn_count = nn_indices.shape
        for j in range(batch_size):
            Y_0 = target_col[nn_indices[j, :]]
            yield Y_0 @ np.linalg.solve(
                K[j, :, :] + self.eps * np.eye(nn_count), Y_0
            )

    def classify(self, indices, nn_indices, test, train, labels):
        """
        Performs simultaneous classification on a list of observations.

        Parameters
        ----------
        index : np.ndarray, type = int, shape = ``(batch_count,)''
            The integer indices of the observations to be approximated.
        nn_indices : numpy.ndarray, type=int, shape = ``(batch_size, nn_count)''
            A matrix listing the nearest neighbor indices for all observations
            in the testing batch.
        train : numpy.ndarray, type = float, shape = ``(train_count, dim)''
            The full training data matrix.
        test : numpy.ndarray, type = float, shape = ``(test_count, dim)''
            The full testing data matrix.
        labels : numpy.ndarray, type = int, shape = ``(train_count, n_classes)''
            One-hot encoding of class labels for all training data.

        Returns
        -------
        labels, numpy.ndarray, type = int, shape = ``(batch_count,)''
            The predicted class labels for each of the given indices.
        """
        return np.argmax(
            self.regress(indices, nn_indices, test, train, labels),
            axis=1,
        )


#     # def do_GP(
#     #     kernel,
#     #     test_indices,
#     #     nn_indices,
#     #     embedded_test,
#     #     embedded_train,
#     #     train_diags,
#     #     test_diags,
#     # ):
#     # dist = 1 - (
#     #     (embedded_train[coords2, :] @ embedded_train[coords2, :].T)
#     #     / np.outer(
#     #         np.sqrt(train_diags[coords2]), np.sqrt(train_diags[coords2])
#     #     )
#     # )
#     # dist[dist < 0] = 0.0
#     # cross_dist = 1 - (
#     #     (embedded_test[test_index, :] @ embedded_train[coords2, :].T)
#     #     / np.sqrt(train_diags[coords2])
#     #     * np.sqrt(test_diags[test_index])
#     # )
#     # cross_dist[cross_dist < 0] = 0
#     # K = kernel(dist)
#     # Kcross = kernel(cross_dist)

#     # label = np.argmax(
#     #     Kcross
#     #     @ np.linalg.solve(K + eps * np.eye(K.shape[0]), labels[coords2, :])
#     # )
#     # return label


# def do_GP_tensor(
#     nu,
#     test_indices,
#     nn_indices,
#     embedded_test,
#     embedded_train,
#     train_labels,
#     eps=0.015,
#     kern="matern",
# ):
#     """
#     Using https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
#     """
#     #     diff_tensor = 1 - np.einsum('bij, bjk -> bik',
#     #                                 embedded_train[batch_nn_indices],
#     #                                 embedded_train[batch_nn_indices].transpose(0,2,1))
#     #     diag_tensor = np.einsum("bi, bj -> bij",
#     #                             np.sqrt(train_diags[batch_nn_indices]),
#     #                             np.sqrt(train_diags[batch_nn_indices]))
#     #     dist_tensor = diff_tensor / diag_tensor
#     #     dist_tensor[dist_tensor < 0] = 0.

#     #     cross_diff_tensor = 1 - np.einsum('bj, bij -> bi',
#     #                                       embedded_test[test_indices],
#     #                                       embedded_train[batch_nn_indices])
#     #     cross_diag_tensor = np.einsum("b, bi -> bi",
#     #                                   np.sqrt(test_diags[test_indices]),
#     #                                   np.sqrt(train_diags[batch_nn_indices]))
#     #     cross_dist_tensor = cross_diff_tensor / cross_diag_tensor
#     #     cross_dist_tensor[cross_dist_tensor < 0] = 0.
