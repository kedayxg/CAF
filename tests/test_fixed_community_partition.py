"""Check fixed-count partition structure and modularity accounting."""
import numpy as np
import pytest

from experiments.classical.run_fixed_community_benchmark import corrected_fixed_partition


def hierarchy():
    matrix = np.full((16,16), .005)
    matrix[:8,:8] = .08
    matrix[8:,8:] = .12
    for start in range(0,16,4):
        matrix[start:start+4,start:start+4] = 1.
    np.fill_diagonal(matrix,0)
    return matrix


def test_corrected_natural_partition_and_forced_count():
    matrix = hierarchy()
    labels, info = corrected_fixed_partition(matrix,9)
    expected = {frozenset(range(start,start+4)) for start in range(0,16,4)}
    assert {frozenset(g) for g in info['natural_communities']} == expected
    assert len(np.unique(labels)) == 9
    assert sorted(i for g in info['communities'] for i in g) == list(range(16))
    assert len([r for r in info['split_trace'] if r['forced']]) == 5
    assert all(abs(r['child_sizes'][0]-r['child_sizes'][1])<=1
               for r in info['split_trace'] if r['forced'])
    assert info['modularity'] < info['natural_modularity']
    assert info['modularity'] == pytest.approx(sum(r['modularity_gain'] for r in info['split_trace']))


def test_edgeless_fixed_count_terminates_with_full_coverage():
    labels, info = corrected_fixed_partition(np.zeros((6,6)),4)
    assert len(np.unique(labels)) == 4
    assert all(info['community_sizes'])
    assert info['modularity'] == 0.


@pytest.mark.parametrize('count',[0,17,True,2.5])
def test_invalid_counts_are_rejected(count):
    with pytest.raises(ValueError,match='target_count'):
        corrected_fixed_partition(hierarchy(),count)


def test_target_below_natural_count_does_not_silently_merge():
    with pytest.raises(ValueError,match='smaller than the natural'):
        corrected_fixed_partition(hierarchy(),3)


def test_asymmetric_graph_is_rejected():
    with pytest.raises(ValueError,match='symmetric'):
        corrected_fixed_partition(np.array([[0.,1.],[0.,0.]]),2)
