###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from dcmppln.utils.utils import get_instance_non_private_attributes

from dcmppln.utils.utils import timeit
import numpy as np
from dcmppln.newman import (newman_partition, capacity_newman_partition,
                           validate_community_capacity)

clustering_methods = ("newman",)
default_clustering_method = "newman"


class Clustering:

    def __init__(
        self,
        max_community_size=0,
        active=True,
        clustering_method="newman",
        take_absolute_value=True,
        cluster_on_correlation=True,
        seed=42,
    ):
        """
        Community detection uses only igraph Newman's leading-eigenvector method.
        take_absolute_value: flag to take absolute value of the matrix before clustering
        cluster_on_correlation: if True, cluster on (denoised) correlation else cluster on covariance

        :param max_community_size: 0 for natural Newman; positive integer for
            induced-subgraph recursion, stopping when each block fits. Raise
            RuntimeError if an oversized block cannot be split by igraph.
        :param active: bool to activate or deactivate the component, if deactivated other parameters are meaningless
        :param clustering_method: newman; other method names are rejected
        :param take_absolute_value: bool to apply abs to matrix before sending it to the clustering method
        :param cluster_on_correlation: bool to cluster on correlation otherwise cluster on covariance
        """
        if clustering_method not in clustering_methods:
            raise ValueError(f"Unknown clustering method: {clustering_method}; only 'newman' is supported")

        self.method = clustering_method
        self.active = active
        self.take_absolute_value = take_absolute_value
        self.max_community_size = validate_community_capacity(max_community_size)
        self.cluster_on_correlation = cluster_on_correlation
        self.seed = int(seed)

        self.params = get_instance_non_private_attributes(self)

    @timeit
    def cluster(self, matrix):
        if not self.active:
            return np.zeros(len(matrix), dtype=int)
        if self.take_absolute_value:
            matrix = np.abs(matrix)
        if self.max_community_size:
            partitions, self.modularity, self.split_trace = capacity_newman_partition(
                matrix, self.max_community_size, seed=self.seed)
        else:
            partitions, self.modularity = newman_partition(matrix, seed=self.seed)
            self.split_trace = []
        print(
            "Max community size:", max(np.unique(partitions, return_counts=True)[1], default=0)
        )
        return partitions

    # use this structure to ease profile readability
    def __call__(self, matrix):
        return self.cluster(matrix)
