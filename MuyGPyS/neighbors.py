# Copyright 2021-2022 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""
KNN lookup management

`MuyGPyS.neighbors.NN_Wrapper` is an api for tasking several KNN libraries with
the construction of lookup indexes that empower fast training and inference.
The wrapper constructor expects the training features, the number of nearest
neighbors, and a method string specifying which algorithm to use, as well as any
additional kwargs used by the methods.
Currently supported implementations include exact KNN using
`sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_
("exact") and approximate KNN using `hnsw <https://github.com/nmslib/hnswlib>`_
("hnsw").
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors
from typing import Tuple

from MuyGPyS import config

if config.muygpys_hnswlib_enabled is True:  # type: ignore
    import hnswlib


class NN_Wrapper:
    """
    Nearest Neighbors lookup datastructure wrapper.

    Wraps the logic driving nearest neighbor data structure training and
    querying. Currently supports `sklearn.neighbors.NearestNeighbors` for exact
    computation and `hnswlib.Index` for approximate nearest neighbors.

    An example constructing exact and approximate KNN data lookups with k = 10.

    Example:
        >>> from MuyGPyS.neighors import NN_Wrapper
        >>> train_features = load_train_features()
        >>> nn_count = 10
        >>> exact_nbrs_lookup = NN_Wrapper(
        ...         train_features, nn_count, nn_method="exact", algorithm="ball_tree"
        ... )
        >>> approx_nbrs_lookup = NN_Wrapper(
        ...         train_features, nn_count, nn_method="hnsw", space="l2", M=16
        ... )


    Args:
        train:
            The full training data of shape `(train_count, feature_count)` that
            will construct the nearest neighbor query datastructure.
        nn_count:
            The number of nearest neighbors to return in queries.
        nn_method:
            Indicates which nearest neighbor algorithm should be used.
            Currently "exact" indicates `sklearn.neighbors.NearestNeighbors`,
            while "hnsw" indicates `hnswlib.Index` (requires installing MuyGPyS
            with the "hnswlib" extras flag).
        kwargs:
            Additional kwargs used for lookup data structure construction.
            `nn_method="exact"` supports "radius", "algorithm", "leaf_size",
            "metric", "p", "metric_params", and "n_jobs" kwargs.
            `nn_method="hnsw"` supports "space", "ef_construction", "M", and
            "random_seed" kwargs.
    """

    def __init__(
        self,
        train: np.ndarray,
        nn_count: int,
        nn_method: str = "exact",
        **kwargs,
    ):
        """
        NOTE[bwp] Will need to replace `train` with a data stream in the future.
        """
        self.train = train
        self.train_count, self.feature_count = self.train.shape
        self.nn_count = nn_count
        self.nn_method = nn_method.lower()
        if self.nn_method == "exact":
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
            self.nbrs = NearestNeighbors(**exact_kwargs).fit(self.train)
        elif self.nn_method == "hnsw":
            if config.muygpys_hnswlib_enabled is True:  # type: ignore
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
                raise ModuleNotFoundError("Module hnswlib is not installed!")
        else:
            raise NotImplementedError(
                f"Nearest Neighbor algorithm {self.nn_method} is not "
                f"implemented."
            )

    def get_nns(
        self,
        test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the nearest neighbors for each row of `test` dataset.

        Find the nearest neighbors and associated distances for each element of
        the given test dataset. Here we assume that the test dataset is distinct
        from the train dataset used in the construction of the nearest neighbor
        lookup data structure.

        Example:
            >>> from MuyGPyS.neighbors import NN_Wrapper
            >>> train_features = load_train_features()
            >>> test_features = load_test_features()
            >>> nn_count = 10
            >>> nbrs_lookup = NN_Wrapper(
            ...         train_features, nn_count, nn_method="exact", algorithm="ball_tree"
            ... )
            >>> nn_indices, nn_dists = nbrs_lookup.get_nns(test_features)


        Args:
            test:
                Testing data matrix of shape `(test_count, feature_count)`.

        Returns
        -------
        nn_indices:
            Matrix of nearest neighbor indices of shape
            `(test_count, nn_count)`. Each row lists the nearest neighbor
            indices of the corresponding test element.
        nn_dists:
            Matrix of distances of shape `(test_count, nn_count)`. Each row
            lists the distance to the test element of the corresponding element
            in `nn_indices`.
        """
        return self._get_nns(test, self.nn_count)

    def get_batch_nns(
        self,
        batch_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the non-self nearest neighbors for indices into the training data.

        Find the nearest neighbors and associated distances for each specified
        index into the training data.

        Example:
            >>> from MuyGPyS.neighbors import NN_Wrapper
            >>> from numpy.random import choice
            >>> train_features = load_train_features()
            >>> nn_count = 10
            >>> nbrs_lookup = NN_Wrapper(
            ...         train_features, nn_count, nn_method="exact", algorithm="ball_tree"
            ... )
            >>> train_count, _ = train_features.shape
            >>> batch_count = 50
            >>> batch_indices = choice(train_count, batch_count, replace=False)
            >>> nn_indices, nn_dists = nbrs_lookup.get_nns(batch_indices)

        Args:
            batch_indices:
                Indices into the training data of shape `(batch_count,)`.

        Returns
        -------
        batch_nn_indices:
            Matrix of nearest neighbor indices of shape
            `(batch_count, nn_count)`. Each row lists the nearest neighbor
            indices (self excluded) of the corresponding batch element.
        batch_nn_dists : numpy.ndarray(int), shape=(batch_count, nn_count)
            Matrix of distances of shape `(batch_count, nn_count)`. Each row
            lists the distance to the batch element of the corresponding element
            in `batch_nn_indices`.
        """
        batch_nn_indices, batch_nn_dists = self._get_nns(
            self.train[batch_indices, :],
            self.nn_count + 1,
        )
        return batch_nn_indices[:, 1:], batch_nn_dists[:, 1:]

    def _get_nns(
        self,
        samples: np.ndarray,
        nn_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the nearest neighbors for each row of `samples` dataset.

        Args:
            samples:
                Data matrix of shape `(sample_count, feature_count)` whose rows
                include samples to be queried.
            nn_count:
                The number of nearest neighbors to query.

        Returns
        -------
        nn_indices:
            Matrix of nearest neighbor indices of shape
            `(sample_count, nn_count)`. Each row lists the nearest neighbor
            indices of the corresponding samples element.
        nn_dists:
            Matrix of distances of shape `(sample_count, nn_count)`. Each row
            lists the distance to the sample element of the corresponding
            element in `nn_indices`.
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
                nn_dists = nn_dists**2
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
