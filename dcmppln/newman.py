"""Newman leading-eigenvector partitioning through python-igraph.

The matrix convention is k_i=sum_j A_ij, T=sum_ij A_ij. An undirected
igraph loop contributes twice its weight to degree, hence loop weight A_ii/2.
This preserves Q=sum_ij (A_ij-k_i*k_j/T) [c_i=c_j]/T, including the diagonal.
"""
import random
import operator

import numpy as np


def _validated_adjacency(adjacency):
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("adjacency must be finite and nonnegative")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("adjacency must be symmetric")
    return matrix


def matrix_graph_edges(adjacency):
    """Validate a nonnegative symmetric matrix and return graph edge triples."""
    matrix = _validated_adjacency(adjacency)
    rows, cols = np.triu_indices(len(matrix))
    weights = matrix[rows, cols].copy()
    weights[rows == cols] *= .5
    edges = [(int(i), int(j), float(w)) for i, j, w in zip(rows, cols, weights) if w > 0]
    return matrix, edges


def newman_partition(adjacency, seed=42):
    """Return canonical labels and igraph modularity; infer the community count.

    No forced splits, maximum block size, or Kernighan--Lin fine tuning is added.
    Empty graphs have no labels; edgeless graphs use a single community and Q=0.
    """
    import igraph as ig

    matrix, edges = matrix_graph_edges(adjacency)
    if not edges:
        return np.zeros(len(matrix), dtype=int), 0.
    graph = ig.Graph(n=len(matrix), edges=[(i, j) for i, j, _ in edges], directed=False)
    graph.es["weight"] = [w for _, _, w in edges]
    ig.set_random_number_generator(random.Random(int(seed)))
    options = ig.ARPACKOptions()
    options.maxiter = 10000
    options.tol = 1e-10
    partition = graph.community_leading_eigenvector(weights="weight", arpack_options=options)
    groups = sorted((sorted(group) for group in partition), key=lambda group: group[0])
    labels = np.empty(len(matrix), dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    return labels, float(partition.modularity)


def validate_community_capacity(value):
    """Return a nonnegative integer capacity; zero means unrestricted."""
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        capacity = operator.index(value)
    except TypeError:
        raise ValueError("max_community_size must be a nonnegative integer") from None
    if capacity < 0:
        raise ValueError("max_community_size must be a nonnegative integer")
    return capacity


def capacity_newman_partition(adjacency, max_community_size, seed=42):
    """Restart igraph Newman on oversized induced subgraphs.

    Stop whenever a block fits the capacity, including the initial graph.
    Each call uses the same seed and the original matrix's induced submatrix;
    no denoising or forced bisection is added. An oversized block that igraph
    cannot split raises RuntimeError instead of violating the capacity.
    Return canonical labels, full-graph modularity and the split trace.
    """
    capacity = validate_community_capacity(max_community_size)
    if capacity == 0:
        raise ValueError("capacity_newman_partition requires max_community_size > 0")
    matrix = _validated_adjacency(adjacency)
    if len(matrix) == 0:
        return np.zeros(0, dtype=int), 0., []
    trace = []

    def recurse(nodes):
        if len(nodes) <= capacity:
            return [nodes]
        labels, q = newman_partition(matrix[np.ix_(nodes, nodes)], seed=seed)
        children = [nodes[labels == label] for label in np.unique(labels)]
        trace.append(dict(parent_size=len(nodes), child_sizes=list(map(len, children)),
                          induced_modularity=q))
        if len(children) == 1:
            raise RuntimeError(
                f"Newman cannot split community of size {len(nodes)} below {capacity}")
        return [leaf for child in children for leaf in recurse(child)]

    groups = sorted(recurse(np.arange(len(matrix))), key=lambda group: int(group[0]))
    labels = np.empty(len(matrix), dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    degree, total = matrix.sum(axis=1), float(matrix.sum())
    if total:
        b = matrix - np.outer(degree, degree)/total
        modularity = float(sum(b[np.ix_(g, g)].sum() for g in groups)/total)
    else:
        modularity = 0.
    return labels, modularity, trace
