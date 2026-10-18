"""Regression checks for feasible CAF starts and cardinality preservation."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from caf_core import evaluate_portfolio, run_caf_pipeline


def _exact_local_components(seed_calls=None):
    def get_model(adjusted_returns, sub_covariance, budget, risk_factor):
        return SimpleNamespace(), list(range(len(adjusted_returns)))

    def optimizer_factory(num_reads=None, seed=None):
        if seed_calls is not None:
            seed_calls.append(seed)

        def optimize(model, weights):
            covariance = model._caf_sub_covariance
            returns = model._caf_adjusted_returns
            candidates = []
            for chosen in combinations(range(len(weights)), model._caf_budget):
                state = np.zeros(len(weights), dtype=int)
                state[list(chosen)] = 1
                objective = model._caf_risk_factor * (state @ covariance @ state) - returns @ state
                candidates.append((float(objective), state))
            objective, state = min(candidates, key=lambda item: item[0])
            return state, objective, 0.0

        return optimize

    return get_model, optimizer_factory


def _run(covariance, returns, communities, cards, **kwargs):
    get_model, optimizer_factory = _exact_local_components()
    return run_caf_pipeline(
        covariance,
        returns,
        communities,
        cards,
        get_model_func=get_model,
        optimizer_factory=optimizer_factory,
        max_workers=1,
        **kwargs,
    )


@pytest.mark.parametrize("use_incremental", [True, False])
def test_default_start_cannot_get_stuck_in_infeasible_empty_portfolio(use_incremental):
    # Every feasible objective is positive. Previously the all-zero state had
    # objective zero, so the monotone acceptance rule rejected every proposal.
    result = _run(
        np.eye(2),
        np.array([0.1, 0.2]),
        [[0, 1]],
        [1],
        use_incremental=use_incremental,
        iterations=2,
    )

    assert result["selected_indices"] == [1]
    assert result["metrics"]["objective"] == pytest.approx(0.3)
    assert result["history"]["objective"][0] == pytest.approx(0.3)


@pytest.mark.parametrize("use_incremental", [True, False])
def test_zero_iterations_still_returns_a_feasible_default_start(use_incremental):
    result = _run(
        np.eye(5),
        np.array([0.1, 0.2, 5.0, 6.0, 0.4]),
        [[0, 1], [2, 3], [4]],
        [1, 0, 1],
        use_incremental=use_incremental,
        iterations=0,
    )

    assert result["selected_indices"] == [1, 4]
    assert len(result["history"]["objective"]) == 1
    assert result["timing"]["community_seconds"] == []


@pytest.mark.parametrize("use_incremental", [True, False])
def test_valid_warm_start_is_preserved_with_zero_iterations(use_incremental):
    result = _run(
        np.eye(4),
        np.array([0.1, 2.0, 0.1, 2.0]),
        [[0, 1], [2, 3]],
        [1, 1],
        initial_selected_indices=[0, 2],
        use_incremental=use_incremental,
        iterations=0,
    )

    assert result["selected_indices"] == [0, 2]


def test_feasible_warm_start_modes_match_and_preserve_zero_cardinality_and_seeds():
    rng = np.random.default_rng(29)
    matrix = rng.normal(size=(6, 6)) * 0.1
    covariance = matrix.T @ matrix + np.eye(6)
    returns = np.array([1.0, 3.0, 0.1, 0.1, 4.0, 1.0])
    communities = [[0, 1], [2, 3], [4, 5]]
    cards = [1, 0, 1]
    results = []
    mode_seeds = []
    for use_incremental in (True, False):
        seeds = []
        get_model, optimizer_factory = _exact_local_components(seeds)
        result = run_caf_pipeline(
            covariance,
            returns,
            communities,
            cards,
            get_model_func=get_model,
            optimizer_factory=optimizer_factory,
            initial_selected_indices=[0, 5],
            risk_factor=0.5,
            effective_risk_factor=1.1,
            iterations=3,
            patience=2,
            use_incremental=use_incremental,
            return_diagnostics=True,
            max_workers=1,
            seed=17,
        )
        assert result["selected_indices"] == [1, 4]
        assert result["history"]["objective"][-1] < result["history"]["objective"][0]
        assert np.all(np.diff(result["history"]["objective"]) <= 1e-9)
        assert all(item["candidate_feasible"] for item in result["proposal_diagnostics"])
        assert result["metrics"]["objective"] == pytest.approx(
            evaluate_portfolio([1, 4], covariance, returns)["objective"]
        )
        results.append(result)
        mode_seeds.append(seeds)

    assert results[0]["history"]["objective"] == pytest.approx(results[1]["history"]["objective"])
    assert mode_seeds[0] == mode_seeds[1] == [
        17 + iteration * 100003 + pos for iteration in range(3) for pos in range(3)
    ]


@pytest.mark.parametrize(
    "communities,cards,message",
    [
        ([[0, 1], [2, 3]], [1], "equal lengths"),
        ([[0, 1], [2, 3]], [1, 1, 0], "equal lengths"),
        ([[0, 1]], [1], "partition all asset indices"),
        ([[0, 1], [1, 2, 3]], [1, 1], "partition all asset indices"),
        ([[0, 0], [1, 2, 3]], [1, 1], "duplicate asset indices"),
        ([[-1, 0], [1, 2, 3]], [1, 1], "outside"),
        ([[0, 1], [2, 4]], [1, 1], "outside"),
        ([[0.0, 1.0], [2, 3]], [1, 1], "integer indices"),
        ([[0, 1], [2, 3]], [-1, 1], "between 0"),
        ([[0, 1], [2, 3]], [3, 1], "between 0"),
        ([[0, 1], [2, 3]], [1.5, 1], "must be an integer"),
        ([[0, 1], [2, 3]], [True, 1], "must be an integer"),
    ],
)
def test_invalid_partition_or_cardinality_is_rejected_before_optimization(communities, cards, message):
    with pytest.raises(ValueError, match=message):
        _run(np.eye(4), np.ones(4), communities, cards, iterations=0)


@pytest.mark.parametrize(
    "selected,message",
    [
        ([], "satisfy every community cardinality"),
        ([0, 1], "satisfy every community cardinality"),
        ([0, 2], "satisfy every community cardinality"),
        ([0, 0], "duplicate asset indices"),
        ([-1], "outside"),
        ([4], "outside"),
        ([0.0], "integer indices"),
    ],
)
@pytest.mark.parametrize("use_incremental", [True, False])
def test_invalid_warm_start_is_not_silently_accepted(selected, message, use_incremental):
    with pytest.raises(ValueError, match=message):
        _run(
            np.eye(4),
            np.ones(4),
            [[0, 1], [2, 3]],
            [1, 0],
            initial_selected_indices=selected,
            use_incremental=use_incremental,
            iterations=0,
        )
