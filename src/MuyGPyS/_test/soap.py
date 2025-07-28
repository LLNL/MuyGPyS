# Copyright 2021-2024 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT
import MuyGPyS._src.math.numpy as np

from absl.testing import parameterized

from MuyGPyS.gp import MuyGPS
from MuyGPyS.gp.deformation import DifferenceIsotropy, dot
from MuyGPyS.gp.hyperparameter import Parameter
from MuyGPyS.gp.kernels.experimental import SOAPKernel
from MuyGPyS.gp.noise import HomoscedasticNoise


def pad_atom_count(train_features, test_features):
    a_train = train_features.shape[3]
    a_test = test_features.shape[3]
    a_max = max(a_train, a_test)

    train_pad = a_max - a_train
    test_pad = a_max - a_test

    train_pad_width = ((0, 0), (0, 0), (0, 0), (0, train_pad), (0, 0))

    test_pad_width = ((0, 0), (0, 0), (0, 0), (0, test_pad), (0, 0))

    train_features_padded = np.pad(
        array=train_features,
        pad_width=train_pad_width,
        mode="mean",
        # constant_values=0,
    )

    test_features_padded = np.pad(
        array=test_features,
        pad_width=test_pad_width,
        mode="mean",
        # constant_values=0,
    )

    return train_features_padded, test_features_padded


def explicit_crosswise(data, nn_data, indices, nn_indices):
    locations = data[indices]
    points = nn_data[nn_indices].swapaxes(2, 1)

    crosswise_similarity = np.zeros(shape=(2, 3, 2, 3, 2, 2, 2, 10))

    for i in range(2):
        for c1 in range(3):
            for k in range(2):
                for c2 in range(3):
                    for q1 in range(2):
                        for q2 in range(2):
                            for a1 in range(2):
                                for a2 in range(10):
                                    crosswise_similarity[
                                        i, c1, k, c2, q1, q2, a1, a2
                                    ] = (
                                        locations[i, c1, q1, a1]
                                        * points[i, c2, k, q2, a2]
                                    ).sum()
    return crosswise_similarity.reshape(2, 3, 2, 3, 4, 2, 10)


def explicit_pairwise(data, nn_indices):
    points = data[nn_indices].swapaxes(2, 1)

    pairwise_similarity = np.zeros(shape=(2, 3, 2, 3, 2, 2, 2, 10, 10))

    for i in range(2):
        for c1 in range(3):
            for k1 in range(2):
                for c2 in range(3):
                    for k2 in range(2):
                        for q1 in range(2):
                            for q2 in range(2):
                                for a1 in range(10):
                                    for a2 in range(10):
                                        pairwise_similarity[
                                            i, c1, k1, c2, k2, q1, q2, a1, a2
                                        ] = (
                                            points[i, c1, k1, q1, a1]
                                            * points[i, c2, k2, q2, a2]
                                        ).sum()

    return pairwise_similarity.reshape(2, 3, 2, 3, 2, 4, 10, 10)


def unwrap_feature_vectors(features, desc_dim):
    """
    unwraps feature vectors and constructs the descriptor and data objects
    as outlined in kernel slides written by J. Stimac

    ARGs
    ----
      - features: np matrix of feature vectors for a set

      - desc_dim: int descritptor dimensionality

    Returns
    -------
        *** A more detailed description for these returned values is given ***
            in the slides on the implementation; the data objects are given
            the same names as on the slides.

      - X_dot:      np tensor with all descriptors for a given set
      - Delta:      np tensor with all desriptor derivatives for a given set


    """
    n = int(
        features.shape[1] / 2 / desc_dim
    )  # equal to max number of atoms per frame for the set
    tot_num_desc_per_feature_row = int(n * desc_dim)

    X_dot = np.zeros((features.shape[0], n, desc_dim))
    Delta = np.zeros((features.shape[0], n, desc_dim))

    for i in np.arange(features.shape[0]):
        X_dot[i, :, :] = np.reshape(
            features[i, :tot_num_desc_per_feature_row], (n, desc_dim), "C"
        )
        Delta[i, :, :] = np.reshape(
            features[i, tot_num_desc_per_feature_row:], (n, desc_dim), "C"
        )

    return X_dot, Delta


def cov_dot_prod(
    X_dot1, Delta1, X_dot2, Delta2, hyperparams, loop_over_n=False
):
    """
    NOTE:

    """
    var = hyperparams[0] * hyperparams[0]  # variance over the prior
    sensativity = hyperparams[1]

    # get feature vector lens
    # X1_len = np.linalg.norm(X_dot1, 2, 2)[:, :, None]  # (i, n, 0)
    # X2_len = np.linalg.norm(X_dot2, 2, 2)[:, :, None]  # (j, m, 0)

    K = np.zeros((X_dot1.shape[0], X_dot2.shape[0]))
    # n = X_dot1.shape[1]

    if loop_over_n:
        raise Exception(
            " DID NOT IMPLEMENT LOOP VERSION, SEE RBF COV FUNCTION FOR HOW "
            "THAT WOULD BE DONE"
        )

    else:
        # vectorized version
        X_hat1 = X_dot1  # /X1_len # (i, n, k)
        X_hat2 = X_dot2  # /X2_len # (j, m, k)

        # /X1_len - X_dot1 * np.sum(
        #     Delta1 * X_dot1, 2, keepdims=True
        # )/(X1_len**3) # (i, n, k)
        Delta1_hat = Delta1
        # /X2_len - X_dot2 * np.sum(
        #     Delta2 * X_dot2, 2, keepdims=True
        # )/(X2_len**3) # (j, m, k)
        Delta2_hat = Delta2

        omega = np.sum(
            X_hat1[:, None, :, None, :] * X_hat2[None, :, None, :, :], 4
        )  # (i, j, n, m)
        T1 = np.sum(
            Delta2_hat[None, :, None, :, :] * Delta1_hat[:, None, :, None, :], 4
        )  # (i, j, n, m)
        T2 = np.sum(
            Delta2_hat[None, :, None, :, :] * X_hat1[:, None, :, None], 4
        )  # (i, j, n, m)
        T3 = np.sum(
            Delta1_hat[:, None, :, None, :] * X_hat2[None, :, None, :, :], 4
        )  # (i, j, n, m)
        K = np.sum(
            (sensativity - 1) * (omega ** (sensativity - 2)) * (T2 * T3)
            + omega ** (sensativity - 1) * T1,
            (2, 3),
        ).squeeze()  # (i, j)

    K *= var * sensativity

    return K


def cov_mat_muygps(
    features1, features2, hyperparams, desc_dim, N_rows_per_iter
):
    features1 = np.asarray(features1)
    features2 = np.asarray(features2)

    (X_dot1, Delta1) = unwrap_feature_vectors(features1, desc_dim)
    (X_dot2, Delta2) = unwrap_feature_vectors(features2, desc_dim)

    # print(np.min(X_dot1), np.max(X_dot1), np.min(Delta1), np.max(Delta1))
    # print(np.min(X_dot2), np.max(X_dot2), np.min(Delta2), np.max(Delta2))

    K = np.zeros((X_dot1.shape[0], X_dot2.shape[0]))

    # loop over different sections of rows of the cov matrix to avoid OOM
    N_sections = np.ceil(X_dot1.shape[0] / N_rows_per_iter)
    for section in np.arange(N_sections):
        # percent_done = 100 * section/N_sections
        # print(f"    COMPLETE WITH {percent_done:.2f}% OF COVARIANCE MATRIX")

        ind_start = int(section * N_rows_per_iter)
        if section == (N_sections - 1):
            ind_stop = X_dot1.shape[0]
        else:
            ind_stop = int((section + 1) * N_rows_per_iter)

        K[ind_start:ind_stop, :] = cov_dot_prod(
            X_dot1[ind_start:ind_stop, :, :],
            Delta1[ind_start:ind_stop, :, :],
            X_dot2,
            Delta2,
            hyperparams,
        )
    # return np.asnumpy(K)
    return np.asarray(K)


def base_implmementation_mean(
    nn_envs, test_features, train_features, train_forces, noise_prior
):
    test_count = test_features.shape[0] // 3
    train_count = train_features.shape[0] // 3
    nn_count = nn_envs.shape[1]
    train_atom_count = train_features.shape[-1] // (2 * 116)
    # test_atom_count = test_features.shape[-1] // (2 * 116)

    neighbor_envs_reshaped = np.repeat(nn_envs, repeats=3, axis=0)
    neighbor_envs_modified = neighbor_envs_reshaped * 3
    env_adjust = (np.arange(neighbor_envs_modified.shape[0]) % 3)[:, None]
    neighbor_envs = (neighbor_envs_modified + env_adjust).reshape(
        test_count, 3, nn_count
    )

    hyperparams = np.array([1.0, 4.0])
    forces_pred_test = np.array([])  # where to store predicted test forces
    # loop over all env in the test set
    nn_list = np.array(nn_envs)
    for ind_test_env in np.arange(neighbor_envs.shape[0]):
        if np.mod(ind_test_env, 10) == 0:
            print(
                " Percent done with test data "
                f"{100 * ind_test_env / nn_list.shape[0]} "
            )

        # down select test features for current env
        ind_test_features = np.arange(3 * ind_test_env, (3 * ind_test_env) + 3)
        print(ind_test_features)
        features_test_select = test_features[ind_test_features, :]

        # down select which forces in the training env to use
        # - translate the index of environments to keep to which forces/force
        #     features to keep
        # n_env_train = nn_list.shape[0]
        # print(n_env_train)
        # ind_forces_2_envs = np.repeat(np.arange(n_env_train), 3)
        # # index of which env each of the force/features rows corresponds to
        # print(ind_forces_2_envs)
        # mask = np.isin(ind_forces_2_envs, nn_list[ind_test_env])
        # ind_forces_keep = np.where(mask)[0]
        # print(ind_forces_keep)

        features_train_NN = train_features[neighbor_envs[ind_test_env]].reshape(
            train_count * 3, 2 * train_atom_count * 116
        )
        forces_train_NN = train_forces[neighbor_envs[ind_test_env]].reshape(
            3 * train_count,
        )

        # evaluate covariance matrix between test and training set
        desc_dim = 116
        Ktn = cov_mat_muygps(
            features_test_select, features_train_NN, hyperparams, desc_dim, 1
        )

        # evaluate covariance matrix for training set with itself
        Knn = cov_mat_muygps(
            features_train_NN, features_train_NN, hyperparams, desc_dim, 1
        )

        # diag_ind = np.arange(Knn.shape[0])
        Knn_ = Knn + np.diag(
            noise_prior**2 * np.ones((Knn.shape[0], Knn.shape[0]))
        )
        Knn_inv = np.linalg.pinv(Knn_)
        forces_pred_test = np.append(
            forces_pred_test, Ktn @ Knn_inv @ forces_train_NN
        )

    return forces_pred_test


class BenchmarkTestCase(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        super(BenchmarkTestCase, cls).setUpClass()
        cls.nn_count = 2
        cls.zeta = 2.0
        cls.noise_prior = 1e-5
        cls.nn_envs = [[1, 3], [1, 3]]

        # features shape (env_count, 3, 2, atom_count, desc_count)
        cls.raw_train_features = np.random.rand(10, 3, 2, 10, 116)
        cls.raw_test_features = np.random.rand(2, 3, 2, 2, 116)
        cls.train_features = pad_atom_count(
            cls.raw_train_features, cls.raw_test_features
        )[0]
        cls.test_features = pad_atom_count(
            cls.raw_train_features, cls.raw_test_features
        )[1]
        cls.train_forces = np.random.rand(10, 3)

        cls.test_count = cls.test_features.shape[0]

        cls.sim_fn = DifferenceIsotropy(metric=dot, length_scale=Parameter(1.0))
        cls.model = MuyGPS(
            kernel=SOAPKernel(deformation=cls.sim_fn),
            noise=HomoscedasticNoise(cls.noise_prior),
        )
