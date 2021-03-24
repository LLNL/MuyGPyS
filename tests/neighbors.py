import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from muyscans.neighbors import NN_Wrapper
from muyscans.testing.test_utils import (
    _make_gaussian_matrix,
    _make_gaussian_dict,
    _make_gaussian_data,
)


class NeighborsTest(parameterized.TestCase):
    @parameterized.parameters(
        (
            (1000, f, nn, 100, e)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for e in [True, False]
        )
    )
    def test_neighbors_batch_shape(
        self, data_count, feature_count, nn_count, batch_count, is_exact
    ):
        data = _make_gaussian_matrix(data_count, feature_count)
        nbrs_lookup = NN_Wrapper(data, nn_count, is_exact)
        indices = np.random.choice(data_count, batch_count, replace=False)
        nn_indices = nbrs_lookup.get_batch_nns(indices)
        self.assertEqual(nn_indices.shape, (batch_count, nn_count))

    @parameterized.parameters(
        (
            (1000, f, nn, 100, e)
            for f in [100, 10, 2, 1]
            for nn in [5, 10, 100]
            for e in [True, False]
        )
    )
    def test_neighbors_query_shape(
        self,
        train_count,
        feature_count,
        nn_count,
        test_count,
        is_exact,
    ):
        train = _make_gaussian_matrix(train_count, feature_count)
        test = _make_gaussian_matrix(test_count, feature_count)
        nbrs_lookup = NN_Wrapper(train, nn_count, is_exact)
        nn_indices = nbrs_lookup.get_nns(test)
        self.assertEqual(nn_indices.shape, (test_count, nn_count))

    ## NOTE[bwp] Should we validate actual KNN behavior, or just trust that we
    # are using the APIs correctly and that the libraries work internally? I
    # don't want to try to develop tests for third-party software...


if __name__ == "__main__":
    absltest.main()
