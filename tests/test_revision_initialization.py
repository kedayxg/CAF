"""Risk sweeps must not silently reuse another partition's static portfolios."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.run_revision import shared_initial_selected


def example():
    return dict(config=dict(max_community_size=100),
        partition=dict(communities_solver_order=[[0, 1], [2, 3]]),
        per_run_records=[dict(seed=42, jpm=dict(metrics=dict(weights=[1, 0, 1, 0])))])


def test_shared_initialization_uses_matching_partition_and_seed():
    assert shared_initial_selected(example(), 0, [np.array([0, 1]), np.array([2, 3])]) == [0, 2]


@pytest.mark.parametrize('change', ['capacity', 'members', 'seed'])
def test_shared_initialization_rejects_mismatched_protocol(change):
    data = deepcopy(example())
    if change == 'capacity':
        data['config']['max_community_size'] = 0
    elif change == 'members':
        data['partition']['communities_solver_order'] = [[0, 2], [1, 3]]
    else:
        data['per_run_records'][0]['seed'] = 43
    with pytest.raises(ValueError):
        shared_initial_selected(data, 0, [np.array([0, 1]), np.array([2, 3])])
