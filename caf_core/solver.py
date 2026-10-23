"""CAF solver utilities.

This module contains helper functions for:
- normalizing local solver outputs into feasible binary selections,
- evaluating portfolio states,
- running the CAF (Context-Aware Folding) iterative optimization pipeline.

The implementation supports two update modes:
- incremental mode: uses cached covariance-vector updates for faster acceptance,
- original mode: recomputes the full objective when evaluating a candidate move.
"""

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np


def _to_numpy_values(array_like):
    """Return the underlying NumPy array for pandas/NumPy inputs."""
    return array_like.values if hasattr(array_like, "values") else array_like


def _enforce_local_cardinality(binary_vec, score_vec, community_cardinality):
    """Project a local binary vector onto an exact-cardinality solution.

    If the vector already satisfies the requested cardinality, it is returned as-is.
    Otherwise, the top-`k` positions ranked by `score_vec` are selected.
    """
    k = int(community_cardinality)
    n = int(binary_vec.size)

    if k <= 0:
        return np.zeros(n, dtype=int)
    if k >= n:
        return np.ones(n, dtype=int)
    if int(binary_vec.sum()) == k:
        return binary_vec.astype(int)

    # Repair to an exact-k solution using descending score ranking.
    order = np.argsort(-score_vec, kind="mergesort")
    repaired = np.zeros(n, dtype=int)
    repaired[order[:k]] = 1
    return repaired


def _normalize_local_solution(raw_solution, local_state_len, community_cardinality):
    """Normalize backend output into a feasible local binary vector.

    Some optimizers may return:
    - a full binary vector of length `local_state_len`,
    - a list of selected local indices,
    - or an irregular array in edge cases.

    This helper converts all of them into a length-matched binary vector and
    enforces the requested local cardinality.
    """
    arr = np.asarray(raw_solution).reshape(-1)

    if arr.size == local_state_len:
        vec = (arr > 0.5).astype(int)
        score = arr.astype(float)
        return _enforce_local_cardinality(vec, score, community_cardinality)

    # Heuristic: treat output as a list of chosen local indices.
    if (
        arr.size > 0
        and arr.size <= local_state_len
        and np.issubdtype(arr.dtype, np.integer)
        and np.all(arr >= 0)
        and np.all(arr < local_state_len)
        and np.unique(arr).size == arr.size
        and (arr.size == community_cardinality or np.any(arr > 1))
    ):
        vec = np.zeros(local_state_len, dtype=int)
        vec[arr.astype(int)] = 1
        return _enforce_local_cardinality(vec, vec.astype(float), community_cardinality)

    # Fallback: binarize and pad/truncate to protect downstream linear algebra.
    vec = (arr > 0.5).astype(int)
    score = arr.astype(float)
    if vec.size < local_state_len:
        vec = np.pad(vec, (0, local_state_len - vec.size), mode="constant")
        score = np.pad(score, (0, local_state_len - score.size), mode="constant")
    elif vec.size > local_state_len:
        vec = vec[:local_state_len]
        score = score[:local_state_len]
    return _enforce_local_cardinality(vec, score, community_cardinality)


def _attach_local_problem_metadata(model, adjusted_returns, sub_covariance, budget, risk_factor):
    """Expose the local surrogate data to optimizers that need warm-start metadata."""
    model._caf_adjusted_returns = np.asarray(adjusted_returns, dtype=float)
    model._caf_sub_covariance = np.asarray(sub_covariance, dtype=float)
    model._caf_budget = int(budget)
    model._caf_risk_factor = float(risk_factor)


def evaluate_portfolio(selected_indices, covariance_matrix, expected_returns, risk_factor=0.5):
    """Evaluate a portfolio described by the selected asset indices."""
    weights = np.zeros(covariance_matrix.shape[0])
    weights[selected_indices] = 1

    covariance_values = _to_numpy_values(covariance_matrix)
    returns_values = _to_numpy_values(expected_returns)

    risk = float(weights.T @ covariance_values @ weights)
    expected_return = float(returns_values @ weights)
    objective = float(risk_factor * risk - expected_return)

    return {
        "risk": risk,
        "expected_return": expected_return,
        "objective": objective,
        "weights": weights,
    }


def evaluate_state_vector(state_vector, covariance_matrix, expected_returns, risk_factor=0.5):
    """Evaluate a binary state vector by converting it to selected indices."""
    selected_indices = np.where(
        np.asarray(state_vector).astype(np.int64) == 1
    )[0].tolist()
    return evaluate_portfolio(
        selected_indices,
        covariance_matrix,
        expected_returns,
        risk_factor=risk_factor,
    )


def run_caf_pipeline(
    covariance_matrix,
    expected_returns,
    communities,
    subproblem_cardinalities,
    get_model_func,
    optimizer_factory,
    risk_factor=0.5,
    effective_risk_factor=None,
    initial_selected_indices=None,
    iterations=6,
    use_incremental=True,
    patience=2,
    return_diagnostics=False,
    max_workers=None,
    seed=None,
):
    """Run the CAF iterative optimization pipeline.

    Args:
        covariance_matrix: Full covariance matrix of the asset universe.
        expected_returns: Expected return vector.
        communities: Partition of asset indices into local subproblems.
        subproblem_cardinalities: Required number of selections per community.
        get_model_func: Builds the local optimization model.
        optimizer_factory: Produces a callable optimizer for the local model.
        risk_factor: Global portfolio risk-aversion coefficient.
        effective_risk_factor: Optional risk factor used inside local proposals.
        initial_selected_indices: Optional warm-start selection.
        iterations: Maximum CAF outer iterations.
        use_incremental: Whether to use cached incremental objective updates.
        patience: Early-stopping patience in outer iterations.
        return_diagnostics: Whether to collect per-proposal diagnostics.
        max_workers: Number of worker threads for community solves.
        seed: Optional base seed for deterministic optimizer construction.
    """
    print(
        f"\n--- Running CAF (Context-Aware Folding) Pipeline "
        f"[Incremental={use_incremental}] ---"
    )

    if effective_risk_factor is None:
        effective_risk_factor = risk_factor

    total_start = perf_counter()

    # Preserve backward compatibility with factories that do not expose `seed`.
    try:
        factory_accepts_seed = "seed" in inspect.signature(optimizer_factory).parameters
    except (TypeError, ValueError):
        factory_accepts_seed = False

    dimension = covariance_matrix.shape[0]
    covariance_values = _to_numpy_values(covariance_matrix)
    returns_values = _to_numpy_values(expected_returns)

    current_state = np.zeros(dimension, dtype=int)
    if initial_selected_indices is not None:
        current_state[initial_selected_indices] = 1

    current_metrics = evaluate_state_vector(
        current_state,
        covariance_matrix,
        expected_returns,
        risk_factor=risk_factor,
    )
    all_indices = np.arange(dimension)
    current_cov_vector = covariance_values @ current_state if use_incremental else None

    iteration_seconds = []
    community_seconds = []
    time_build = 0.0
    time_solve = 0.0
    time_accept = 0.0

    # History is recorded once per outer iteration for plotting or diagnostics.
    history = {
        "objective": [current_metrics["objective"]],
        "risk": [current_metrics["risk"]],
        "expected_return": [current_metrics["expected_return"]],
    }
    proposal_diagnostics = []

    best_objective = current_metrics["objective"]
    no_improve_count = 0

    for iteration_idx in range(iterations):
        iteration_start = perf_counter()

        def local_effective_objective(local_x, sub_covariance, adjusted_returns):
            """Evaluate the local surrogate objective used for proposal comparison."""
            return float(
                effective_risk_factor * (local_x @ sub_covariance @ local_x)
                - (adjusted_returns @ local_x)
            )

        def solve_community(community, community_cardinality, community_pos):
            """Build and solve one community subproblem."""
            community_start = perf_counter()
            community_cardinality = int(community_cardinality)
            local_build_time = 0.0
            local_solve_time = 0.0

            c_size = len(community)
            if c_size > 80:
                dynamic_reads = 200
            elif c_size > 30:
                dynamic_reads = 100
            else:
                dynamic_reads = 20

            if factory_accepts_seed and seed is not None:
                call_seed = int(seed) + iteration_idx * 100003 + int(community_pos)
                local_optimizer = optimizer_factory(
                    num_reads=dynamic_reads,
                    seed=call_seed,
                )
            else:
                local_optimizer = optimizer_factory(num_reads=dynamic_reads)

            thread_id = threading.get_ident()

            if use_incremental:
                t0 = perf_counter()
                local_state = current_state[community]
                sub_covariance = covariance_values[np.ix_(community, community)]

                if community_cardinality <= 0:
                    local_solution = np.zeros_like(local_state)
                    adjusted_returns = returns_values[community].copy()
                    local_build_time += perf_counter() - t0
                else:
                    # Remove the local block contribution from the cached global
                    # covariance-vector product, leaving only the external context.
                    external_shift = 2 * effective_risk_factor * (
                        current_cov_vector[community] - sub_covariance @ local_state
                    )
                    adjusted_returns = returns_values[community] - external_shift
                    model, weights = get_model_func(
                        adjusted_returns,
                        sub_covariance,
                        budget=community_cardinality,
                        risk_factor=effective_risk_factor,
                    )
                    _attach_local_problem_metadata(
                        model,
                        adjusted_returns,
                        sub_covariance,
                        community_cardinality,
                        effective_risk_factor,
                    )
                    model.name = f"model_{thread_id}"  # Prevent temp file collision.
                    local_build_time += perf_counter() - t0

                    t1 = perf_counter()
                    local_solution, _, _ = local_optimizer(model, weights)
                    local_solution = _normalize_local_solution(
                        local_solution,
                        local_state_len=len(local_state),
                        community_cardinality=community_cardinality,
                    )
                    local_solve_time += perf_counter() - t1

                return (
                    community,
                    community_cardinality,
                    local_solution,
                    local_state,
                    sub_covariance,
                    adjusted_returns,
                    dynamic_reads,
                    perf_counter() - community_start,
                    local_build_time,
                    local_solve_time,
                )
            else:
                t0 = perf_counter()
                if community_cardinality <= 0:
                    local_build_time += perf_counter() - t0
                    return (
                        community,
                        community_cardinality,
                        np.zeros(len(community), dtype=int),
                        current_state[community],
                        covariance_values[np.ix_(community, community)],
                        returns_values[community].copy(),
                        dynamic_reads,
                        perf_counter() - community_start,
                        local_build_time,
                        local_solve_time,
                    )

                fixed_indices = np.setdiff1d(all_indices, community, assume_unique=False)
                external_shift = (
                    2
                    * effective_risk_factor
                    * covariance_values[np.ix_(community, fixed_indices)]
                    @ current_state[fixed_indices]
                )
                adjusted_returns = returns_values[community] - external_shift
                sub_covariance = covariance_values[np.ix_(community, community)]
                model, weights = get_model_func(
                    adjusted_returns,
                    sub_covariance,
                    budget=community_cardinality,
                    risk_factor=effective_risk_factor,
                )
                _attach_local_problem_metadata(
                    model,
                    adjusted_returns,
                    sub_covariance,
                    community_cardinality,
                    effective_risk_factor,
                )
                model.name = f"model_{thread_id}"  # Prevent temp file collision.
                local_build_time += perf_counter() - t0

                t1 = perf_counter()
                local_solution, _, _ = local_optimizer(model, weights)
                local_solution = _normalize_local_solution(
                    local_solution,
                    local_state_len=len(community),
                    community_cardinality=community_cardinality,
                )
                local_solve_time += perf_counter() - t1
                return (
                    community,
                    community_cardinality,
                    local_solution,
                    current_state[community],
                    sub_covariance,
                    adjusted_returns,
                    dynamic_reads,
                    perf_counter() - community_start,
                    local_build_time,
                    local_solve_time,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(solve_community, community, card, pos)
                for pos, (community, card) in enumerate(zip(communities, subproblem_cardinalities))
            ]
            results = [f.result() for f in futures]

        for (
            community,
            community_cardinality,
            local_solution,
            local_state,
            sub_covariance,
            adjusted_returns,
            dynamic_reads,
            comm_time,
            local_build_time,
            local_solve_time,
        ) in results:
            time_build += local_build_time
            time_solve += local_solve_time
            community_seconds.append(comm_time)
            t2 = perf_counter()
            local_obj_before = local_effective_objective(local_state, sub_covariance, adjusted_returns)
            local_obj_candidate = local_effective_objective(local_solution, sub_covariance, adjusted_returns)
            local_improved = bool(local_obj_candidate < local_obj_before - 1e-9)
            candidate_feasible = bool(np.sum(local_solution) == int(community_cardinality))
            accepted_globally = False
            global_delta_if_committed = 0.0
            if use_incremental:
                delta_x = local_solution - local_state
                if np.any(delta_x):
                    delta_obj = risk_factor * (
                        delta_x @ sub_covariance @ delta_x
                        + 2 * (current_cov_vector[community] @ delta_x)
                    ) - (returns_values[community] @ delta_x)
                    global_delta_if_committed = float(delta_obj)
                    if delta_obj <= -1e-5:
                        current_state[community] = local_solution
                        current_cov_vector += covariance_values[:, community] @ delta_x
                        current_metrics["objective"] += float(delta_obj)
                        accepted_globally = True
            else:
                if community_cardinality > 0:
                    candidate_state = current_state.copy()
                    candidate_state[community] = local_solution
                    candidate_metrics = evaluate_state_vector(
                        candidate_state,
                        covariance_matrix,
                        expected_returns,
                        risk_factor=risk_factor,
                    )
                    global_delta_if_committed = float(
                        candidate_metrics["objective"] - current_metrics["objective"]
                    )
                    if candidate_metrics["objective"] < current_metrics["objective"] - 1e-5:
                        current_state = candidate_state
                        current_metrics = candidate_metrics
                        accepted_globally = True
            time_accept += (perf_counter() - t2)
            if return_diagnostics:
                proposal_diagnostics.append(
                    {
                        "iteration_idx": int(iteration_idx),
                        "community_size": int(len(community)),
                        "local_cardinality": int(community_cardinality),
                        "dynamic_reads": int(dynamic_reads),
                        "candidate_feasible": bool(candidate_feasible),
                        "local_obj_before": float(local_obj_before),
                        "local_obj_candidate": float(local_obj_candidate),
                        "local_improved": bool(local_improved),
                        "global_delta_if_committed": float(global_delta_if_committed),
                        "accepted_globally": bool(accepted_globally),
                    }
                )

        iteration_seconds.append(perf_counter() - iteration_start)

        # Early stopping is based on the true global objective after the iteration.
        if current_metrics["objective"] < best_objective - 1e-5:
            best_objective = current_metrics["objective"]
            no_improve_count = 0
        else:
            no_improve_count += 1

        current_full_metrics = evaluate_state_vector(
            current_state,
            covariance_matrix,
            expected_returns,
            risk_factor=risk_factor,
        )
        history["objective"].append(current_full_metrics["objective"])
        history["risk"].append(current_full_metrics["risk"])
        history["expected_return"].append(current_full_metrics["expected_return"])

        if no_improve_count >= patience:
            print(
                f"    -> Early stopping at iteration {iteration_idx + 1} "
                f"(No improvement for {patience} rounds)"
            )
            break

    selected_indices = np.where(current_state == 1)[0].tolist()
    final_metrics = evaluate_state_vector(
        current_state,
        covariance_matrix,
        expected_returns,
        risk_factor=risk_factor,
    )

    result = {
        "selected_indices": selected_indices,
        "metrics": final_metrics,
        "history": history,
        "timing": {
            "mode": "incremental" if use_incremental else "original",
            "total_seconds": perf_counter() - total_start,
            "iteration_seconds": iteration_seconds,
            "community_seconds": community_seconds,
            "time_build": time_build,
            "time_solve": time_solve,
            "time_accept": time_accept,
        },
    }
    if return_diagnostics:
        result["proposal_diagnostics"] = proposal_diagnostics

    return result
