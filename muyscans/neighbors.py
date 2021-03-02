#!/usr/bin/env python
# encoding: utf-8
"""
@file neighbors.py

Created by priest2 on 2020-11-23

Nearest Neighbor logic.
"""

import numpy as np

from time import perf_counter


class NN_Wrapper:
    """
    Nearest Neighbors lookup datastructure wrapper.

    Wraps the logic driving nearest neighbor data structure training and
    querying. Currently supports `sklearn.neighbors.NearestNeighbors` for exact
    computation and `hnswlib.Index` for approximate nearest neighbors.
    """

    def __init__(self, train, nn_count, exact):
        """
        Initialize.

        Parameters
        ----------
        train : numpy.ndarray, type = float, shape = ``(train_count, dim)''
            The full training data that will construct the nearest neighbor
            query datastructure.
            NOTE[bwp] Will need to be replaced with a data stream in the future.
        nn_count : int
            The number of nearest neighbors to return in queries.
        exact : Boolean
            Flag indicating whether to use `sklearn.neighbors.NearestNeighbors`
            (if ``True'') or `hnswlib.Index` otherwise.
        """
        self.train = train
        self.train_count, self.dim = self.train.shape
        self.nn_count = nn_count
        self.exact = exact
        if self.exact is True:
            from sklearn.neighbors import NearestNeighbors

            self.nbrs = NearestNeighbors(
                n_neighbors=(nn_count + 1),
                algorithm="ball_tree",
                n_jobs=-1,
            ).fit(self.train)
        else:
            import hnswlib

            self.nbrs = hnswlib.Index(space="cosine", dim=self.dim)
            self.nbrs.init_index(
                max_elements=self.train_count,
                ef_construction=100,
                M=16,
            )
            self.nbrs.add_items(self.train)

    def get_nns(self, test):
        """
        Get the nearest neighbors for each row of `test` dataset.

        Parameters
        ----------
        test : numpy.ndarray, type = float, shape = ``(test_count, dim)''
            Testing data matrix.

        Returns
        -------
        nn_indices, numpy.ndarray, type = int, shape= ``(test_count, nn_count)''
            The nearest neighbors for each row of the test data.
        """
        return self._get_nns(test, self.nn_count)

    def get_batch_nns(self, batch_indices):
        """
        Get the nearest neighbors for each index of the training data, ignoring
        self in the neighbor sets.

        Parameters
        ----------
        batch_indices : numpy.ndarray, type = float,
                shape = ``(batch_size,)''
            Indices into the training data.

        Returns
        -------
        nn_indices, numpy.ndarray, type = int, shape= ``(test_count, nn_count)''
            The nearest neighbors for each row of the test data.
        """
        batch_nn_indices = self._get_nns(
            self.train[batch_indices, :],
            self.nn_count + 1,
        )
        return batch_nn_indices[:, 1:]

    def _get_nns(self, samples, nn_count):
        """
        Get the nearest neighbors for each row of `samples` dataset.

        Parameters
        ----------
        samples : numpy.ndarray, type = float, shape = ``(sample_count, dim)''
            Data matrix whose rows include samples to be queried.
        nn_count : int
            THe number of nearest neighbors to query.

        Returns
        -------
        nn_indices, numpy.ndarray, type = int, shape= ``(test_count, nn_count)''
            The nearest neighbors for each row of the samples matrix.
        """
        if self.exact is True:
            _, nn_indices = self.nbrs.kneighbors(
                samples,
                n_neighbors=nn_count,
            )
        else:
            nn_indices, _ = self.nbrs.knn_query(samples, k=nn_count)
        return nn_indices
