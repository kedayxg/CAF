"""Check feasibility and cached acceptance against exact enumeration."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from caf_core import run_caf_pipeline, normalize_local_solution


def build_model(adjusted_returns, sub_covariance, budget, risk_factor):
    return SimpleNamespace(), []


def exact_factory(num_reads=None, seed=None):
    def solve(model, weights):
        n = len(model._caf_adjusted_returns)
        candidates = []
        for selected in itertools.combinations(range(n), model._caf_budget):
            x = np.zeros(n, dtype=int)
            x[list(selected)] = 1
            value = model._caf_risk_factor * (x @ model._caf_sub_covariance @ x) - model._caf_adjusted_returns @ x
            candidates.append((value, x))
        return min(candidates, key=lambda item: item[0])[1], 0., 0.
    return solve


def kwargs():
    return dict(covariance_matrix=np.eye(4), expected_returns=np.zeros(4),
                communities=[np.array([0, 1]), np.array([2, 3])],
                subproblem_cardinalities=[1, 1], get_model_func=build_model,
                optimizer_factory=exact_factory, max_workers=2, iterations=2)


def test_missing_initial_state_is_made_feasible_before_descent():
    result = run_caf_pipeline(**kwargs())
    assert len(result["selected_indices"]) == 2
    assert result["history"]["objective"][0] == pytest.approx(1.0)
    assert all(value == pytest.approx(1.0) for value in result["history"]["objective"])


@pytest.mark.parametrize("indices", [[0], [0, 1], [0, 2, 2], [-1, 2], [0.0, 2.0]])
def test_infeasible_or_invalid_initial_state_is_rejected(indices):
    with pytest.raises(ValueError, match="initial_selected_indices"):
        run_caf_pipeline(**kwargs(), initial_selected_indices=indices)


@pytest.mark.parametrize("groups,cards", [
    ([[0, 1], [1, 2, 3]], [1, 1]), ([[0, 1]], [1]),
    ([[0, 1], [2, 3]], [1]), ([[0, 1], [2, 3]], [1, 3]),
])
def test_bad_partition_or_budgets_are_rejected(groups, cards):
    options = kwargs()
    options.update(communities=groups, subproblem_cardinalities=cards)
    with pytest.raises(ValueError):
        run_caf_pipeline(**options)


def test_normalization_distinguishes_raw_feasibility_from_repair():
    vector, info = normalize_local_solution([0, 0, 0], 3, 1, return_info=True)
    assert vector.sum() == 1
    assert info == {"raw_candidate_feasible": False, "candidate_repaired": True}
    vector, info = normalize_local_solution([2], 3, 1, return_info=True)
    assert vector.tolist() == [0, 0, 1]
    assert info["raw_candidate_feasible"] and not info["candidate_repaired"]
    with pytest.raises(ValueError, match="finite"):
        normalize_local_solution([float("nan")], 3, 1)


def test_repair_is_reported_in_pipeline_diagnostics():
    options = kwargs()
    options["optimizer_factory"] = lambda num_reads, seed=None: lambda model, weights: ([0, 0], 0., 0.)
    result = run_caf_pipeline(**options, return_diagnostics=True)
    assert all(row["candidate_repaired"] for row in result["initialization_diagnostics"])
    assert all(row["candidate_repaired"] and not row["raw_candidate_feasible"]
               for row in result["proposal_diagnostics"])


@pytest.mark.parametrize("incremental", [True, False])
def test_zero_and_full_budget_do_not_call_backend(incremental):
    options = kwargs()
    options["subproblem_cardinalities"] = [0, 2]
    def no_backend(**kwargs):
        raise AssertionError("trivial budgets do not require a local solver")
    options["optimizer_factory"] = no_backend
    result = run_caf_pipeline(**options, use_incremental=incremental, return_diagnostics=True)
    assert result["selected_indices"] == [2, 3]
    assert all(not row["backend_called"] for row in result["proposal_diagnostics"])


def test_cached_and_full_acceptance_agree_with_exact_local_solves():
    rng = np.random.default_rng(34)
    for _ in range(8):
        a = rng.normal(size=(8, 8))
        covariance = a.T @ a
        returns = rng.normal(size=8)
        options = dict(covariance_matrix=covariance, expected_returns=returns,
                       communities=[np.arange(0, 4), np.arange(4, 8)],
                       subproblem_cardinalities=[2, 2], get_model_func=build_model,
                       optimizer_factory=exact_factory, initial_selected_indices=[0, 1, 4, 5],
                       max_workers=2, iterations=6)
        cached = run_caf_pipeline(**options, use_incremental=True)
        full = run_caf_pipeline(**options, use_incremental=False)
        assert cached["selected_indices"] == full["selected_indices"]
        assert len(cached["selected_indices"]) == 4
        np.testing.assert_allclose(cached["history"]["objective"], full["history"]["objective"])
        assert np.all(np.diff(cached["history"]["objective"]) <= 1e-10)
