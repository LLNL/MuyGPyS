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


def get_nearest_neighbors(
    desc_test,
    desc_train,
    desc_filtered_train,
    L_train,
    N_neigh_env,
    N_neigh_frame,
    dist_metric="cos",
):

    if dist_metric == "cos":

        desc_train_len = np.linalg.norm(desc_train, 2, 1)[:, None]
        desc_filtered_train_len = np.linalg.norm(desc_filtered_train, 2, 1)[
            :, None
        ]
        desc_test_len = np.linalg.norm(desc_test, 2, 1)[:, None]

        dist_matrix = (
            1 - (desc_test / desc_test_len) @ (desc_train / desc_train_len).T
        )
        dist_filtered_matrix = (
            1
            - (desc_test / desc_test_len)
            @ (desc_filtered_train / desc_filtered_train_len).T
        )

    if dist_metric == "euclidean":

        dist_matrix = np.linalg.norm(
            desc_test[:, :, None] - desc_train.T[None, :, :], 2, 1
        )
        dist_filtered_matrix = np.linalg.norm(
            desc_test[:, :, None] - desc_filtered_train.T[None, :, :], 2, 1
        )

    average_dist_to_frame = (dist_matrix @ L_train.T) / np.sum(L_train, 1)

    # check if have more training set frames than NN being asked for
    if N_neigh_frame >= average_dist_to_frame.shape[1]:
        neigh_ind_frames = np.tile(
            np.arange(average_dist_to_frame.shape[1]),
            (average_dist_to_frame.shape[0], 1),
        )
    else:
        neigh_ind_frames = np.argpartition(
            average_dist_to_frame, N_neigh_frame, axis=1
        )

    # check if have more training env than NN being asked for
    if N_neigh_env >= dist_filtered_matrix.shape[1]:
        neigh_ind_frames = np.tile(
            np.arange(dist_filtered_matrix.shape[1]),
            (dist_filtered_matrix.shape[0], 1),
        )
    else:
        neigh_ind_envs = np.argpartition(
            dist_filtered_matrix, N_neigh_env, axis=1
        )

    return (
        neigh_ind_envs[:, :N_neigh_env],
        neigh_ind_frames[:, :N_neigh_frame],
    )


def explicit_crosswise(data, nn_data, indices, nn_indices):
    """
    Crosswise unit test.
    Takes in train and test data sets and related index information.
    """

    nn_indices = nn_indices

    locations = data[indices]
    points = nn_data[nn_indices].swapaxes(1, 2)

    nn_count = nn_indices.shape[1]
    test_count = locations.shape[0]
    test_atom_count = locations.shape[-2]
    train_atom_count = points.shape[-2]

    crosswise_similarity = np.zeros(
        shape=(test_count, 3, nn_count, 3, 4, test_atom_count, train_atom_count)
    )

    # crosswise
    for (
        i_env_test,
        i_xyz_2,
        i_nn,
        i_xyz_1,
        i_combo,
        i_atom_1,
        i_atom_2,
    ), _ in np.ndenumerate(crosswise_similarity):
        if i_combo == 0:  # should be q1 dot q2
            q_1 = locations[i_env_test, i_xyz_1, 0, i_atom_1]
            # i_env_train = nn_indices[i_env_test, i_nn]
            q_2 = points[i_env_test, i_xyz_2, i_nn, 0, i_atom_2]
            q1_dot_q2 = np.sum(q_1 * q_2)
            crosswise_similarity[
                i_env_test, i_xyz_2, i_nn, i_xyz_1, i_combo, i_atom_1, i_atom_2
            ] = q1_dot_q2
        elif i_combo == 1:  # should be q1 dot dq2
            q_1 = locations[i_env_test, i_xyz_1, 0, i_atom_1]
            # i_env_test = nn_indices[i_env_test, i_nn]
            dq_2 = points[i_env_test, i_xyz_2, i_nn, 1, i_atom_2]
            q1_dot_dq2 = np.sum(q_1 * dq_2)
            crosswise_similarity[
                i_env_test, i_xyz_2, i_nn, i_xyz_1, i_combo, i_atom_1, i_atom_2
            ] = q1_dot_dq2
        elif i_combo == 2:  # should be dq1 dot q2
            dq_1 = locations[i_env_test, i_xyz_1, 1, i_atom_1]
            # i_env_test = nn_indices[i_env_test, i_nn]
            q_2 = points[i_env_test, i_xyz_2, i_nn, 0, i_atom_2]
            dq1_dot_q2 = np.sum(dq_1 * q_2)
            crosswise_similarity[
                i_env_test, i_xyz_2, i_nn, i_xyz_1, i_combo, i_atom_1, i_atom_2
            ] = dq1_dot_q2
        elif i_combo == 3:  # should be dq1 dot dq2
            dq_1 = locations[i_env_test, i_xyz_1, 1, i_atom_1]
            # i_env_test = nn_indices[i_env_test, i_nn]
            dq_2 = points[i_env_test, i_xyz_2, i_nn, 1, i_atom_2]
            dq1_dot_dq2 = np.sum(dq_1 * dq_2)
            crosswise_similarity[
                i_env_test, i_xyz_2, i_nn, i_xyz_1, i_combo, i_atom_1, i_atom_2
            ] = dq1_dot_dq2

    return crosswise_similarity


def explicit_pairwise(data, nn_indices):
    """
    pairwise unit test.
    Takes in train and test data sets and related index information.
    """

    nn_indices = nn_indices

    points = data[nn_indices].swapaxes(1, 2)

    nn_count = nn_indices.shape[1]
    train_atom_count = points.shape[-2]
    test_count = points.shape[0]

    pairwise_similarity = np.zeros(
        shape=(
            test_count,
            3,
            nn_count,
            3,
            nn_count,
            4,
            train_atom_count,
            train_atom_count,
        )
    )

    # pairwise
    for (
        i_env_test,
        i_xyz_1,
        i_nn_1,
        i_xyz_2,
        i_nn_2,
        i_combo,
        i_atom_1,
        i_atom_2,
    ), _ in np.ndenumerate(pairwise_similarity):
        if i_combo == 0:  # should be q1 dot q2
            q_1 = points[i_env_test, i_xyz_1, i_nn_1, 0, i_atom_1]
            q_2 = points[i_env_test, i_xyz_2, i_nn_2, 0, i_atom_2]
            q1_dot_q2 = np.sum(q_1 * q_2)
            pairwise_similarity[
                i_env_test,
                i_xyz_1,
                i_nn_1,
                i_xyz_2,
                i_nn_2,
                i_combo,
                i_atom_1,
                i_atom_2,
            ] = q1_dot_q2
        elif i_combo == 1:  # should be q1 dot dq2
            q_1 = points[i_env_test, i_xyz_1, i_nn_1, 0, i_atom_1]
            q_2 = points[i_env_test, i_xyz_2, i_nn_2, 1, i_atom_2]
            q1_dot_q2 = np.sum(q_1 * q_2)
            pairwise_similarity[
                i_env_test,
                i_xyz_1,
                i_nn_1,
                i_xyz_2,
                i_nn_2,
                i_combo,
                i_atom_1,
                i_atom_2,
            ] = q1_dot_q2
        elif i_combo == 2:  # should be q1 dot dq2
            q_1 = points[i_env_test, i_xyz_1, i_nn_1, 1, i_atom_1]
            q_2 = points[i_env_test, i_xyz_2, i_nn_2, 0, i_atom_2]
            q1_dot_q2 = np.sum(q_1 * q_2)
            pairwise_similarity[
                i_env_test,
                i_xyz_1,
                i_nn_1,
                i_xyz_2,
                i_nn_2,
                i_combo,
                i_atom_1,
                i_atom_2,
            ] = q1_dot_q2
        elif i_combo == 3:  # should be q1 dot dq2
            q_1 = points[i_env_test, i_xyz_1, i_nn_1, 1, i_atom_1]
            q_2 = points[i_env_test, i_xyz_2, i_nn_2, 1, i_atom_2]
            q1_dot_q2 = np.sum(q_1 * q_2)
            pairwise_similarity[
                i_env_test,
                i_xyz_1,
                i_nn_1,
                i_xyz_2,
                i_nn_2,
                i_combo,
                i_atom_1,
                i_atom_2,
            ] = q1_dot_q2

    return pairwise_similarity


def create_tensors_for_muygps(desc, derivatives, forces, frames):
    L = get_L(frames)
    desc_new, deriv_new = reshape_desc_for_deriv_memfix(
        L, desc, derivatives
    )  # (i, n, d)
    # derivatives (i, n, 3, d)

    env_count = desc_new.shape[0]
    atom_count = deriv_new.shape[1]
    desc_count = desc_new.shape[-1]

    # get features
    features = np.zeros((env_count, 3, 2, atom_count, desc_count))

    for a in range(env_count):
        for c in range(3):
            features[a, c, 0, :, :] = desc_new[a, :, :]
            features[a, c, 1, :, :] = deriv_new[a, :, c, :]

    return (features, forces)


def reshape_features_for_muygps(desc, derivatives, forces, frames):

    L = get_L(frames)
    # desc_4_deriv = reshape_desc_for_deriv(L, desc, max_env) # (i, n, d)
    desc_4_deriv, derivatives = reshape_desc_for_deriv_memfix(
        L, desc, derivatives
    )
    # derivatives (i, n, 3, d)

    # get features
    num_feature_rows = int(3 * desc.shape[0])
    num_feature_col = int((desc_4_deriv.shape[1] * 2) * desc.shape[1])
    half_num_feature_col = int(0.5 * num_feature_col)
    features = np.zeros((num_feature_rows, num_feature_col))
    row_ind = 0
    for a in np.arange(desc.shape[0]):
        for c in np.arange(3):
            features[row_ind, :] = np.block(
                [
                    np.reshape(
                        desc_4_deriv[a, :, :], (1, half_num_feature_col), "C"
                    ),
                    np.reshape(
                        derivatives[a, :, c, :], (1, half_num_feature_col), "C"
                    ),
                ]
            )
            row_ind += 1

    forces = np.reshape(forces, (int(forces.shape[0] * 3)), "C")

    return (features, forces)


def get_L(frames):

    frames = frames.squeeze()
    _, ind = np.unique(frames, return_index=True)
    frame_list = frames[np.sort(ind)]
    tot_num_frames = frame_list.shape[0]
    n = frames.shape[0]
    L = np.zeros((tot_num_frames, n))
    for a in range(0, tot_num_frames):
        L[a, frames == frame_list[a]] = 1
    return L


def reshape_desc_for_deriv_memfix(L, desc, deriv):

    d = desc.shape[1]
    n_tot = desc.shape[0]

    # get max_env by finding atom with most neighbors
    atom_is_neigh = np.logical_not(np.all(np.isclose(deriv, 0), axis=(2, 3)))

    # get neighbor list
    row_indices, col_indices = np.nonzero(atom_is_neigh)
    neigh_list = [[] for _ in range(n_tot)]
    for r, c in zip(row_indices, col_indices):
        neigh_list[r].append(int(c))

    num_atoms_in_frames = np.sum(L, 1).astype("int")
    frame_starts = np.cumsum(np.insert(num_atoms_in_frames, 0, 0))[:-1]
    atom_to_frame = np.repeat(
        np.arange(len(num_atoms_in_frames)), num_atoms_in_frames
    )
    offsets = frame_starts[atom_to_frame]

    flat_neigh = np.concatenate(neigh_list)
    lens = np.array([len(n) for n in neigh_list]).astype("int")
    i_for_flat = np.repeat(np.arange(n_tot), lens)
    offsets_flat = offsets[i_for_flat]
    flat_indices = np.array(flat_neigh + offsets_flat).astype("int")

    # desc_for_deriv
    max_neighbors = max(lens)
    # desc_for_deriv = np.zeros((n_tot, max_neighbors, d), dtype=desc.dtype)
    desc_for_deriv = np.ones((n_tot, max_neighbors, d), dtype=desc.dtype)
    desc_for_deriv_flat = desc[flat_indices, :]

    # deriv_reshaped
    X, Y = deriv.shape[2], deriv.shape[3]
    deriv_reshaped = np.zeros((n_tot, max_neighbors, X, Y), dtype=deriv.dtype)
    deriv_reshaped_flat = deriv[i_for_flat, flat_neigh, :, :]

    # Scatter back
    starts = np.cumsum(np.insert(lens, 0, 0))[:-1]
    for i, (start, length) in enumerate(zip(starts, lens)):
        desc_for_deriv[i, :length, :] = desc_for_deriv_flat[
            start : start + length, :
        ]
        deriv_reshaped[i, :length, :, :] = deriv_reshaped_flat[
            start : start + length, :, :
        ]

    return desc_for_deriv, deriv_reshaped


def unwrap_feature_vectors(features, desc_dim):

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
    return np.asarray(K)


def base_implmementation_mean(
    nn_envs, test_features, train_features, train_forces, noise_prior
):
    test_count = test_features.shape[0] // 3
    train_count = train_features.shape[0] // 3
    nn_count = nn_envs.shape[1]
    train_atom_count = train_features.shape[-1] // (2 * 116)

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
        # print(ind_test_features)
        features_test_select = test_features[ind_test_features, :]

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
        cls.nn_count = 10
        cls.zeta = 4.0
        cls.noise_prior = 1e-15
        cls.train_count = 10
        cls.test_count = 7
        cls.desc_count = 116
        cls.nn_envs = np.array(
            [
                np.random.choice(
                    cls.train_count, size=(cls.nn_count), replace=False
                )
                for i in range(cls.test_count)
            ]
        )

        cls.train_forces_raw = np.random.normal(
            loc=0, scale=1, size=(cls.train_count, 3)
        )
        cls.train_desc = np.random.normal(
            loc=0, scale=1, size=(cls.train_count, cls.desc_count)
        )
        cls.train_derivs = np.random.normal(
            loc=0,
            scale=1,
            size=(cls.train_count, cls.train_count, 3, cls.desc_count),
        )

        cls.test_forces_raw = np.random.normal(
            loc=0, scale=1, size=(cls.test_count, 3)
        )
        cls.test_desc = np.random.normal(
            loc=0, scale=1, size=(cls.test_count, cls.desc_count)
        )
        cls.test_derivs = np.random.normal(
            loc=0,
            scale=1,
            size=(cls.test_count, cls.test_count, 3, cls.desc_count),
        )

        cls.train_features = create_tensors_for_muygps(
            cls.train_desc,
            cls.train_derivs,
            cls.train_forces_raw,
            np.zeros(cls.train_count),
        )[0]
        cls.train_forces = create_tensors_for_muygps(
            cls.train_desc,
            cls.train_derivs,
            cls.train_forces_raw,
            np.zeros(cls.train_count),
        )[1]
        cls.test_features = create_tensors_for_muygps(
            cls.test_desc,
            cls.test_derivs,
            cls.test_forces_raw,
            np.zeros(cls.test_count),
        )[0]
        cls.test_forces = create_tensors_for_muygps(
            cls.test_desc,
            cls.test_derivs,
            cls.test_forces_raw,
            np.zeros(cls.test_count),
        )[1]

        cls.sim_fn = DifferenceIsotropy(metric=dot, length_scale=Parameter(1.0))
        cls.model = MuyGPS(
            kernel=SOAPKernel(
                deformation=cls.sim_fn, sensitivity=Parameter(cls.zeta)
            ),
            noise=HomoscedasticNoise(cls.noise_prior),
        )
