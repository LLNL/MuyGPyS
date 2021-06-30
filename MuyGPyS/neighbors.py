# Copyright 2021 Lawrence Livermore National Security, LLC and other MuyGPyS
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import numpy as np

from time import perf_counter

from MuyGPyS.utils import safe_apply


class NN_Wrapper:
    """
    Nearest Neighbors lookup datastructure wrapper.

    Wraps the logic driving nearest neighbor data structure training and
    querying. Currently supports `sklearn.neighbors.NearestNeighbors` for exact
    computation and `hnswlib.Index` for approximate nearest neighbors.
    """

    def __init__(self, train, nn_count, nn_method="exact", **kwargs):
        """
        Initialize.

        Parameters
        ----------
        train : numpy.ndarray(float), shape = ``(train_count, feature_count)''
            The full training data that will construct the nearest neighbor
            query datastructure.
            NOTE[bwp] Will need to be replaced with a data stream in the future.
        nn_count : int
            The number of nearest neighbors to return in queries.
        nn_method : str
            Inicates which nearest neighbor algorithm should be used.
            NOTE[bwp] currently "exact" indicates
            `sklearn.neighbors.NearestNeighbors`, while "hnsw" indicates
            `hnswlib.Index`.
        kwargs : dict
            Additional kwargs used for lookup data structure construction.
            `nn_method="exact"` supports "radius", "algorithm", "leaf_size",
            "metric", "p", and "metric_params" kwargs. `nn_method="hnsw"
            supports "space", "ef_construction", "M", and "random_seed" kwargs.
        """
        self.train = train
        self.train_count, self.feature_count = self.train.shape
        self.nn_count = nn_count
        self.nn_method = nn_method.lower()
        if self.nn_method == "exact":
            from sklearn.neighbors import NearestNeighbors

            exact_kwargs = {
                k: kwargs[k]
                for k in kwargs
                if k
                in {
                    "radius",
                    "algorithm",
                    "leaf_size",
                    "metric",
                    "p",
                    "metric_params",
                    "n_jobs",
                }
            }

            exact_kwargs["n_neighbors"] = nn_count + 1
            exact_kwargs["n_jobs"] = exact_kwargs.get("n_jobs", -1)
            print(exact_kwargs["n_jobs"])
            self.nbrs = NearestNeighbors(**exact_kwargs).fit(self.train)
        elif self.nn_method == "hnsw":
            import hnswlib

            self.nbrs = hnswlib.Index(
                space=kwargs.get("space", "l2"), dim=self.feature_count
            )
            hnsw_kwargs = {
                k: kwargs[k]
                for k in kwargs
                if k in {"ef_construction", "M", "random_seed"}
            }
            hnsw_kwargs["max_elements"] = self.train_count
            self.nbrs.init_index(**hnsw_kwargs)
            self.nbrs.add_items(self.train)
        else:
            raise NotImplementedError(
                f"Nearest Neighbor algorithm {self.nn_method} is not implemented."
            )

    def get_nns(self, test):
        """
        Get the nearest neighbors for each row of `test` dataset.

        Parameters
        ----------
        test : numpy.ndarray(float), shape = ``(test_count, feature_count)''
            Testing data matrix.

        Returns
        -------
        nn_indices, numpy.ndarray(int), shape= ``(test_count, nn_count)''
            The nearest neighbors for each row of the test data.
        """
        return self._get_nns(test, self.nn_count)

    def get_batch_nns(self, batch_indices):
        """
        Get the nearest neighbors for each index of the training data, ignoring
        self in the neighbor sets.

        Parameters
        ----------
        batch_indices : numpy.ndarray(float), shape = ``(batch_size,)''
            Indices into the training data.

        Returns
        -------
        nn_indices, numpy.ndarray(int), shape= ``(test_count, nn_count)''
            The nearest neighbors for each row of the test data.
        """
        batch_nn_indices, batch_nn_dists = self._get_nns(
            self.train[batch_indices, :],
            self.nn_count + 1,
        )
        return batch_nn_indices[:, 1:], batch_nn_dists[:, 1:]

    def _get_nns(self, samples, nn_count):
        """
        Get the nearest neighbors for each row of `samples` dataset.

        Parameters
        ----------
        samples : numpy.ndarray(float),
                  shape = ``(sample_count, feature_count)''
            Data matrix whose rows include samples to be queried.
        nn_count : int
            THe number of nearest neighbors to query.

        Returns
        -------
        nn_indices, numpy.ndarray(int), shape = ``(test_count, nn_count)''
            The nearest neighbors for each row of the samples matrix.
        """
        if self.nn_method == "exact":
            nn_dists, nn_indices = self.nbrs.kneighbors(
                samples,
                n_neighbors=nn_count,
            )
            if self.nbrs.metric == "minkowski" and self.nbrs.p == 2:
                # We do this so that both implementations return the squared l2
                # for downstream consistency. Taking the square root is much
                # more expensive, so this should not produce much overhead.
                nn_dists = nn_dists ** 2
        elif self.nn_method == "hnsw":
            # Although hnsw uses 'l2' as the name of its metric, it returns
            # F2 values as distances in order to avoid the square root
            # computations.
            nn_indices, nn_dists = self.nbrs.knn_query(samples, k=nn_count)
        else:
            raise NotImplementedError(
                f"Nearest Neighbor algorithm {self.nn_method} is not implemented."
            )

        return nn_indices, nn_dists
