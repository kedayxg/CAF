"""Regressions for the vendored decomposition and optimization helpers."""
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcmppln.modularity_maximization.modularity_spectral_optimization import (
    modularity_spectral_optimization,
)
from dcmppln.pipeline import Pipeline
from dcmppln.utils.objective_after_clustering import get_model
from dcmppln.utils.risk_rebalance import rebalancing_risk_factor


def _block_adjacency(size, block_size, background=0.001):
    adjacency = np.full((size, size), background)
    for start in range(0, size, block_size):
        adjacency[start : start + block_size, start : start + block_size] = 1.0
    np.fill_diagonal(adjacency, 0)
    return adjacency


def test_recursive_clustering_keeps_separate_strong_blocks():
    communities, labels = modularity_spectral_optimization(
        _block_adjacency(6, 2), return_communities=True
    )
    assert {frozenset(group) for group in communities} == {
        frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))
    }
    assert len(labels) == 6


def test_recursive_clustering_many_splits_do_not_overflow_labels():
    adjacency = _block_adjacency(128, 2, background=0.00001)
    communities, labels = modularity_spectral_optimization(
        adjacency, return_communities=True
    )
    assert sorted(asset for group in communities for asset in group) == list(range(128))
    assert len(np.unique(labels)) == len(communities)
    # The recursion reaches enough splits to overflow the old exponential labels.
    assert len(communities) >= 64
    assert all(len({asset // 2 for asset in group}) == 1 for group in communities)


def test_get_model_preserves_zero_budget_and_risk_factor():
    model, weights = get_model(
        np.arange(1, 5, dtype=float), np.eye(4), budget=0, risk_factor=0
    )
    assert model.get_constraint_by_index(0).rhs.constant == 0
    assert not model.objective_expr.is_quad_expr()
    assert [model.objective_expr.get_coef(weight) for weight in weights] == [-1, -2, -3, -4]


def test_risk_rebalancing_uses_communities_and_ignores_label_names():
    covariance = np.array(
        [[2.0, 0.1, 0.2, 0.0], [0.1, 1.0, 0.0, 0.1],
         [0.2, 0.0, 3.0, 0.3], [0.0, 0.1, 0.3, 4.0]]
    )
    returns = np.array([0.2, 0.5, 0.8, 1.0])
    groups = {"first": [0, 2], "second": [1, 3]}
    expected = 0.7 * np.linalg.norm(covariance) / sum(
        np.linalg.norm(covariance[np.ix_(group, group)]) for group in groups.values()
    ) * sum(np.linalg.norm(returns[group]) for group in groups.values()) / np.linalg.norm(returns)
    for communities in (
        np.array([0, 1, 0, 1]),
        np.array([23, -7, 23, -7]),
        np.array(["alpha", "beta", "alpha", "beta"]),
        groups,
    ):
        assert rebalancing_risk_factor(0.7, covariance, returns, communities) == pytest.approx(expected)


@pytest.mark.parametrize("covariance,returns", [(np.zeros((2, 2)), np.ones(2)), (np.eye(2), np.zeros(2))])
def test_degenerate_risk_rebalancing_stays_finite(covariance, returns):
    assert rebalancing_risk_factor(0.7, covariance, returns, [0, 1]) == 0.7


@pytest.mark.parametrize("communities", [{"a": [0, 0], "b": [1]}, {"a": [0]}, [0]])
def test_risk_rebalancing_rejects_invalid_partition(communities):
    with pytest.raises(ValueError):
        rebalancing_risk_factor(0.7, np.eye(2), np.ones(2), communities)


def test_pipeline_run_forwards_risk_rebalancing_option():
    pipeline = Pipeline(np.eye(4), np.eye(4), np.ones(4))
    pipeline.optimize_and_score = Mock(return_value={"score": 0.0})
    pipeline.run(run_optimizer=True, risk_rebalancing=True)
    assert pipeline.optimize_and_score.call_args.kwargs["risk_rebalancing"] is True


class _BudgetOptimizer:
    params = {}

    def __call__(self, model, weights, **kwargs):
        budget = int(model.get_constraint_by_index(0).rhs.constant)
        return np.array([1] * budget + [0] * (len(weights) - budget)), 0.0, 0.0


def _score(pipeline, **kwargs):
    return pipeline.optimize_and_score(
        np.eye(20), np.eye(20), np.repeat([0, 1], 10),
        0.7, "test", False, True, False, 0.0, 0.0, 0.0, 0.0,
        **kwargs,
    )


@pytest.mark.parametrize("cardinality", [0, 8, 20])
def test_pipeline_accepts_exact_cardinality(cardinality):
    pipeline = Pipeline(np.eye(20), np.eye(20), np.ones(20), optimize_func=_BudgetOptimizer())
    result = _score(pipeline, cardinality=cardinality)
    assert np.sum(result["recombined_solution"]) == cardinality
    assert pipeline.__state__["key_parameters"]["problem_parameters"]["cardinality"] == cardinality


def test_pipeline_keeps_legacy_cardinality_divider_default():
    pipeline = Pipeline(np.eye(20), np.eye(20), np.ones(20), optimize_func=_BudgetOptimizer())
    result = _score(pipeline, cardinality_divider=3)
    assert np.sum(result["recombined_solution"]) == 6
