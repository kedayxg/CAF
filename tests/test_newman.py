"""Checks for the revision default, distinct from archived upstream defects."""
import numpy as np
import pytest

from dcmppln.clustering import Clustering
from dcmppln.newman import newman_partition, capacity_newman_partition
import dcmppln.newman as newman_module
from experiments.classical.run_capacity_benchmark import capacity_partition
from experiments.classical.run_library_clustering_benchmark import library_partition


def test_default_newman_matches_independent_benchmark_adapter():
    a = np.full((16, 16), .005)
    for start in range(0, 16, 4):
        a[start:start + 4, start:start + 4] = 1.
    labels, _, _ = Clustering()(a)
    reference, _ = library_partition(a, "igraph_spectral")
    np.testing.assert_array_equal(labels, reference)
    groups = {frozenset(np.flatnonzero(labels == label)) for label in np.unique(labels)}
    assert groups == {frozenset(range(i, i+4)) for i in range(0, 16, 4)}
    permutation = np.random.default_rng(7).permutation(16)
    permuted, _ = newman_partition(a[np.ix_(permutation, permutation)])
    assert {frozenset(permutation[permuted == label]) for label in np.unique(permuted)} == groups


@pytest.mark.parametrize("size", [0, 4])
def test_newman_edgeless_graphs(size):
    labels, q = newman_partition(np.zeros((size, size)))
    np.testing.assert_array_equal(labels, np.zeros(size, dtype=int))
    assert q == 0


@pytest.mark.parametrize("matrix", [np.ones((2, 3)), np.array([[0., 1.], [0., 0.]]),
                                   -np.eye(2), np.array([[np.nan]])])
def test_newman_rejects_invalid_graphs(matrix):
    with pytest.raises(ValueError):
        newman_partition(matrix)


@pytest.mark.parametrize("method", ["legacy_spectral", "louvain", "kmeans", "typo"])
def test_removed_and_unknown_methods_are_rejected(method):
    with pytest.raises(ValueError, match="Unknown"):
        Clustering(clustering_method=method)


@pytest.mark.parametrize("capacity", [-1, 1.5, True, np.inf, np.nan])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="max_community_size"):
        Clustering(max_community_size=capacity)


@pytest.mark.parametrize("matrix", [np.zeros((9, 9)), np.ones((9, 9))])
def test_unsplittable_oversized_block_fails_explicitly(matrix):
    with pytest.raises(RuntimeError, match="cannot split"):
        Clustering(max_community_size=3)(matrix)


@pytest.mark.parametrize("size", [0, 4])
def test_capacity_accepts_edgeless_graph_that_fits(size):
    labels, q, trace = capacity_newman_partition(np.zeros((size, size)), 4)
    np.testing.assert_array_equal(labels, np.zeros(size, dtype=int))
    assert q == 0 and trace == []


def test_capacity_recursion_maps_local_positions_to_global_assets(monkeypatch):
    matrix = np.full((8, 8), .1)
    np.fill_diagonal(matrix, np.arange(1, 9))
    called = []

    def split(block, seed):
        called.append((np.diag(block).astype(int).tolist(), seed))
        return np.arange(len(block)) % 2, 0.

    monkeypatch.setattr(newman_module, "newman_partition", split)
    labels, q, trace = capacity_newman_partition(matrix, 2, seed=7)
    assert {frozenset(np.flatnonzero(labels == label)) for label in np.unique(labels)} == {
        frozenset([0, 4]), frozenset([1, 5]), frozenset([2, 6]), frozenset([3, 7])}
    assert called == [(list(range(1, 9)), 7), ([1, 3, 5, 7], 7), ([2, 4, 6, 8], 7)]
    assert [r['parent_size'] for r in trace] == [8, 4, 4]
    # Report modularity against the full graph, not a sum of child scores.
    degree = matrix.sum(axis=1)
    b = matrix-np.outer(degree, degree)/matrix.sum()
    expected = sum(b[np.ix_(np.flatnonzero(labels == k), np.flatnonzero(labels == k))].sum()
                   for k in np.unique(labels))/matrix.sum()
    assert q == pytest.approx(expected)


def test_public_capacity_adapter_and_experiment_share_the_partition():
    matrix = np.full((16, 16), .005)
    for start in range(0, 16, 4):
        matrix[start:start+4, start:start+4] = 1.
    adapter = Clustering(max_community_size=5)
    labels, _, _ = adapter(matrix)
    reference, trace = capacity_partition(matrix, 5)
    np.testing.assert_array_equal(labels, reference)
    assert sorted(np.unique(labels, return_counts=True)[1]) == [4, 4, 4, 4]
    assert adapter.split_trace == trace
