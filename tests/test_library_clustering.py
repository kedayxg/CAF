"""Validate graph conventions and independent-library partition adapters."""
import networkx as nx
import numpy as np
import pytest

from experiments.classical.run_library_clustering_benchmark import matrix_to_graph,library_partition


def test_loop_conversion_preserves_matrix_degrees_and_modularity():
    matrix=np.array([[2.,.8,.1],[.8,4.,.2],[.1,.2,6.]])
    graph=matrix_to_graph(matrix)
    np.testing.assert_allclose([graph.degree(i,weight='weight') for i in range(3)],matrix.sum(axis=1))
    assert graph[1][1]['weight']==2.
    groups=[{0,2},{1}]
    degree=matrix.sum(axis=1)
    b=matrix-np.outer(degree,degree)/matrix.sum()
    expected=sum(b[np.ix_(sorted(g),sorted(g))].sum() for g in groups)/matrix.sum()
    assert nx.community.modularity(graph,groups,weight='weight')==pytest.approx(expected)


@pytest.mark.parametrize('method,count',[('igraph_spectral',4),('cnm9',9),('louvain',4)])
def test_public_library_partitions_cover_known_block_graph(method,count):
    if method=='igraph_spectral':
        pytest.importorskip('igraph')
    matrix=np.full((16,16),.005)
    matrix[:8,:8]=.08
    matrix[8:,8:]=.12
    for start in range(0,16,4):
        matrix[start:start+4,start:start+4]=1.
    np.fill_diagonal(matrix,0)
    labels,info=library_partition(matrix,method,42)
    assert len(np.unique(labels))==count
    assert sorted(i for g in info['communities'] for i in g)==list(range(16))
    if count==4:
        assert {frozenset(g) for g in info['communities']}=={frozenset(range(i,i+4)) for i in range(0,16,4)}
    assert info['modularity']==pytest.approx(info['networkx_modularity'])


@pytest.mark.parametrize('matrix',[np.ones((2,3)),np.array([[0.,1.],[0.,0.]]),-np.eye(2)])
def test_invalid_graph_conversion_is_rejected(matrix):
    with pytest.raises(ValueError):
        matrix_to_graph(matrix)
