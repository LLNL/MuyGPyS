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


def explicit_crosswise(
    data, nn_data, indices, nn_indices, test_index, neighbor_index
):
    """
    Crosswise unit test.
    Takes in train and test data sets and related index information.
    """
    desc_count = data.shape[-1]
    test_atom_count = data.shape[-2]
    train_atom_count = nn_data.shape[-2]

    locations = data[indices]
    points = nn_data[nn_indices]

    test_point = locations[test_index]
    neighbor_point = points[test_index, neighbor_index]

    crosswise_product = np.zeros(
        shape=(3, 3, 2, 2, test_atom_count, train_atom_count)
    )

    for c1 in range(3):
        for c2 in range(3):
            for d1 in range(2):
                for d2 in range(2):
                    for a1 in range(test_atom_count):
                        for a2 in range(train_atom_count):
                            for q1 in range(desc_count):
                                for q2 in range(desc_count):
                                    crosswise_product[
                                        c1, c2, d1, d2, a1, a2
                                    ] = np.sum(
                                        test_point[c1, d1, a1, q1]
                                        * neighbor_point[c2, d2, a2, q2],
                                        axis=(-1, -2),
                                    )

    crosswise_similarity = crosswise_product.reshape(
        *crosswise_product.shape[:2], -1, *crosswise_product.shape[-2:]
    )

    return crosswise_similarity


def explicit_pairwise(nn_data, nn_indices, test_index, neighbor_index):
    """
    Crosswise unit test.
    Takes in train and test data sets and related index information.
    """
    train_atom_count = nn_data.shape[-2]

    points = nn_data[nn_indices]

    neighbor_point = points[test_index, neighbor_index]

    pairwise_product = np.zeros(
        shape=(3, 3, 2, 2, train_atom_count, train_atom_count)
    )

    for c1 in range(3):
        for c2 in range(3):
            for d1 in range(2):
                for d2 in range(2):
                    for a1 in range(train_atom_count):
                        for a2 in range(train_atom_count):
                            pairwise_product[c1, c2, d1, d2, a1, a2] = np.sum(
                                neighbor_point[c1, d1, a1]
                                * neighbor_point[c2, d2, a2],
                                axis=-1,
                            )

    pairwise_similarity = pairwise_product.reshape(
        *pairwise_product.shape[:2], -1, *pairwise_product.shape[-2:]
    )

    return pairwise_similarity


def test_random(
    data=None, nn_data=None, indices=None, nn_indices=None, N_tests=5
):
    train_count = nn_data.shape[0]
    if data is not None:
        test_count = data.shape[0]
    else:
        test_count = nn_indices.shape[0]

    random_test_index = np.random.randint(low=0, high=test_count, size=N_tests)
    random_neighbor_index = np.random.randint(
        low=0, high=nn_count, size=N_tests
    )

    if data is not None:
        sim = _crosswise_similarity(data, nn_data, indices, nn_indices)
    else:
        sim = _pairwise_similarity(data=nn_data, nn_indices=nn_indices)

    check_bool = []

    for idx in range(N_tests):
        print(
            f"Testing: test_index={random_test_index[idx]}, nn_index={random_neighbor_index[idx]}"
        )
        if data is not None:
            check = crosswise_check(
                data,
                nn_data,
                indices,
                nn_indices,
                random_test_index[idx],
                random_neighbor_index[idx],
            )
            check_bool.append(
                np.allclose(
                    check,
                    sim[random_test_index[idx], :, random_neighbor_index[idx]],
                )
            )
        else:
            check = pairwise_check(
                nn_data,
                nn_indices,
                random_test_index[idx],
                random_neighbor_index[idx],
            )
            check_bool.append(
                np.allclose(
                    check,
                    sim[
                        random_test_index[idx],
                        :,
                        random_neighbor_index[idx],
                        :,
                        random_neighbor_index[idx],
                    ],
                )
            )

    if all(check_bool):
        return True
    else:
        raise ValueError(
            f"Slices are the same shape but do not match value-wise."
        )


def create_tensors_for_muygps(desc, derivatives, forces, frames):
    L = get_L(frames)
    max_env = derivatives.shape[1]
    desc_4_deriv = reshape_desc_for_deriv(L, desc, max_env)  # (i, n, d)
    # derivatives (i, n, 3, d)

    frame_count = desc.shape[0]
    atom_count = derivatives.shape[1]
    desc_count = desc.shape[-1]

    # get features
    features = np.zeros((frame_count, 3, 2, atom_count, desc_count))
    for a in range(frame_count):
        for c in range(3):
            features[a, c, 0, :, :] = desc_4_deriv[a, :, :]
            # print(f'a={a}, {features[a, c, 0, 0 ,0]}')
            features[a, c, 1, :, :] = derivatives[a, :, c, :]

    return (features, forces)


def reshape_features_for_muygps(desc, derivatives, forces, frames):
    """

    function will setup the features for muygps. This is assuming that we are only
    fitting to forces. Each feature vector will include the descriptors followed
    by the associated derivatives.

    ARGS:
    -----
        desc - np array, matrix of descriptors (i x d) where i is the number of atomic environments and d is the dimension of descriptors

        deriv - np array, tensor of descriptor derivatives (i, n, c, d), where
                n is the max number of atoms that could be found in a frame for a dataset
                and c is the number of cartesian components, which will always be 3

        forces - np array of atomic forces of size (i, c)

        frames  -  np.array the size of the total number of atomic environments (or atoms)
                    in a set of frames; each element corresponds to which frame an
                    atom belongs to

                    eg.) [0, 0, 0, 1, 1, 1, 2, 2, 2] for a set that has three atoms in
                     frame 0, three atoms in frame, 1, and two atoms in frame 2

    RETURNS
    -------
        features - np array of feature vector rows (3 * i x 2 * n * d).
                    In a given row, the descriptors for all atoms in a frame will be
                    first listed, followed by the descriptor derivatives. The derivatives
                    will correspond to the forces in the same row in the returned forces
                    variable

        forces - np. array vector of the forces but just reshaped to be size (3 * i)
                 when reshaped, they will be in order [fx1, fy1, fz1, fx2, ....]
                 where the xyz are the cartesian coordiantes and the numbers are the environment index

    """
    L = get_L(frames)
    max_env = derivatives.shape[1]
    desc_4_deriv = reshape_desc_for_deriv(L, desc, max_env)  # (i, n, d)
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
    """
    function contructs object called L, an np.array such  that elements (i,j)
     is 1 if atomic environment j is in frame (also referred to as frame) i, and
     othewise that element is 0.

     ARGS
     ----
        frames: np.array the size of the total number of atomic environments (or atoms)
                in a set of frames; each element corresponds to which frame an
                atom belongs to

                eg.) [0, 0, 0, 1, 1, 1, 2, 2, 2] for a set that has three atoms in
                 frames 0, three atoms in frame, 1, and two atoms in frame 2
     RETURNS
     -------
        L:      np.array described above; in the example used in description for
                frames, L would be:

                        [[1, 1, 1, 0, 0, 0, 0, 0]
                        [[0, 0, 0, 1, 1, 1, 0, 0]
                        [[0, 0, 0, 0, 0, 0, 1, 1]]

    """
    frames = frames.squeeze()
    _, ind = np.unique(frames, return_index=True)
    frame_list = frames[np.sort(ind)]
    tot_num_frames = frame_list.shape[0]
    n = frames.shape[0]
    L = np.zeros((tot_num_frames, n))
    for a in range(0, tot_num_frames):
        L[a, frames == frame_list[a]] = 1
    return L


def reshape_desc_for_deriv(L, desc, max_env):
    """
    constructs desc_for_derivs, an tensor constructed out of the descriptor matrix that
    replicates rows in a fashion that is sensible for the cov matrix force entries

    ARGS:
    ----
         L: np.array produced by the get_L() function in this module, which
             contains information about number of atoms in each frame
         desc: np.array of dataset descriptors of size [n, d]

         max_env: int, maxium number of environments an atom can be for the entire data set

    returns
    -------

         desc_for_deriv [n, max_env, d] where n is the total number of atoms in the set
             which is also the 0th dimension of desc, max_env is the max number of env and atom can
             be in; this will be taken as the max number of atoms in the largest frame
             of the set; d is the number of descriptor dimensions.

             -> for frames that have less than max_env, the rest of the max_env
                 dimension will be filled with ones. Although it might seem
                 more reasonable to fill with zeros, the some kerenls require
                 dividing by this quantity, which will lead to NANs. If we use
                 ones, this should not hurt because they should always be mulilied
                 with descriptor derivatives that are zeros.


    """
    d = desc.shape[1]
    n_tot = desc.shape[0]
    # desc_for_deriv = np.zeros((n_tot, max_env, d))
    desc_for_deriv = np.ones((n_tot, max_env, d))
    n_frames = L.shape[0]
    n_env_so_far = 0
    for a in np.arange(n_frames):
        n_env = int(np.sum(L[a, :]))
        desc_for_deriv[
            n_env_so_far : n_env_so_far + n_env, :n_env, :
        ] = np.tile(desc[n_env_so_far : n_env_so_far + n_env, :], (n_env, 1, 1))
        n_env_so_far += n_env

    return desc_for_deriv


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
