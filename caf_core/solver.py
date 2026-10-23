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


def _normalize_local_solution(raw_solution, local_state_len, community_cardinality, *, return_info=False):
    """Normalize binary vectors or local indices, recording any feasibility repair.

    With return_info=True, return (vector, info). The default public API still
    returns only the vector. A repaired candidate must not be called a raw sample.
    """
    n, k = int(local_state_len), int(community_cardinality)
    if n != local_state_len or n < 0 or k != community_cardinality or not 0 <= k <= n:
        raise ValueError("local cardinality must be an integer between zero and local size")
    arr = np.asarray(raw_solution).reshape(-1)
    if not np.issubdtype(arr.dtype, np.number) and arr.dtype != np.bool_:
        raise ValueError("local solver output must contain finite numbers")
    if not np.all(np.isfinite(arr)):
        raise ValueError("local solver output must contain finite numbers")
    raw_feasible = False
    if arr.size == n:
        vec = (arr > 0.5).astype(int)
        score = arr.astype(float)
        raw_feasible = bool(np.all((arr == 0) | (arr == 1)) and vec.sum() == k)
    elif (
        arr.size > 0 and arr.size <= n
        and np.issubdtype(arr.dtype, np.integer)
        and np.all(arr >= 0) and np.all(arr < n)
        and np.unique(arr).size == arr.size
        and (arr.size == k or np.any(arr > 1))
    ):
        vec = np.zeros(n, dtype=int)
        vec[arr.astype(int)] = 1
        score = vec.astype(float)
        raw_feasible = bool(arr.size == k)
    else:
        vec = (arr > 0.5).astype(int)[:n]
        score = arr.astype(float)[:n]
        if vec.size < n:
            vec = np.pad(vec, (0, n - vec.size))
            score = np.pad(score, (0, n - score.size))
    result = _enforce_local_cardinality(vec, score, k)
    info = {"raw_candidate_feasible": raw_feasible, "candidate_repaired": not raw_feasible}
    return (result, info) if return_info else result


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


def _validated_problem(covariance, returns, communities, cardinalities):
    covariance = np.asarray(_to_numpy_values(covariance), dtype=float)
    returns = np.asarray(_to_numpy_values(returns), dtype=float)
    if returns.ndim != 1 or covariance.shape != (returns.size, returns.size):
        raise ValueError("covariance must be square and match the returns vector")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(returns)):
        raise ValueError("covariance and returns must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    groups, cards = list(communities), list(cardinalities)
    if len(groups) != len(cards):
        raise ValueError("each community must have one cardinality")
    validated = []
    for community, card in zip(groups, cards):
        indices = np.asarray(community)
        if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("communities must be nonempty integer index arrays")
        if not np.isfinite(card) or int(card) != card or not 0 <= card <= len(indices):
            raise ValueError("community cardinality must be an integer within its size")
        validated.append(indices.astype(int))
    covered = np.concatenate(validated) if validated else np.array([], dtype=int)
    if not np.array_equal(np.sort(covered), np.arange(returns.size)):
        raise ValueError("communities must partition all assets exactly once")
    return covariance, returns, validated, [int(card) for card in cards]


def _reads_for_size(size):
    return 200 if size > 80 else 100 if size > 30 else 20


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
        initial_selected_indices: Feasible warm-start indices. If omitted, solve\n            independent local problems to build a feasible state before refinement.
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

    covariance_values, returns_values, communities, subproblem_cardinalities = _validated_problem(
        covariance_matrix, expected_returns, communities, subproblem_cardinalities
    )
    dimension = len(returns_values)
    for name, value in (("risk_factor", risk_factor), ("effective_risk_factor", effective_risk_factor)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if int(iterations) != iterations or iterations < 0 or int(patience) != patience or patience < 1:
        raise ValueError("iterations must be non-negative and patience must be positive integers")
    iterations, patience = int(iterations), int(patience)

    def make_optimizer(reads, call_seed):
        kwargs = {"num_reads": reads}
        if factory_accepts_seed and call_seed is not None:
            kwargs["seed"] = int(call_seed)
        return optimizer_factory(**kwargs)

    current_state = np.zeros(dimension, dtype=int)
    initialization_diagnostics = []
    if initial_selected_indices is not None:
        indices = np.asarray(initial_selected_indices)
        if indices.size == 0:
            indices = np.array([], dtype=int)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("initial_selected_indices must be a one-dimensional integer sequence")
        if np.any(indices < 0) or np.any(indices >= dimension) or len(np.unique(indices)) != len(indices):
            raise ValueError("initial_selected_indices must be unique and in range")
        current_state[indices] = 1
        if any(current_state[c].sum() != k for c, k in zip(communities, subproblem_cardinalities)):
            raise ValueError("initial_selected_indices must satisfy every community budget")
    else:
        # Feasibility is established before monotonic acceptance; zero is not a
        # feasible starting portfolio when any community has a nonzero budget.
        for pos, (community, card) in enumerate(zip(communities, subproblem_cardinalities)):
            if card in (0, len(community)):
                current_state[community] = int(card > 0)
                continue
            sub_covariance = covariance_values[np.ix_(community, community)]
            model, weights = get_model_func(
                returns_values[community], sub_covariance, budget=card, risk_factor=effective_risk_factor
            )
            _attach_local_problem_metadata(
                model, returns_values[community], sub_covariance, card, effective_risk_factor
            )
            optimizer = make_optimizer(_reads_for_size(len(community)), None if seed is None else int(seed) + pos)
            raw, _, _ = optimizer(model, weights)
            local, info = _normalize_local_solution(raw, len(community), card, return_info=True)
            current_state[community] = local
            initialization_diagnostics.append({"community_pos": pos, **info})

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
            """Generate a proposal using the state frozen until all futures finish."""
            community_start = perf_counter()
            t0 = perf_counter()
            local_state = current_state[community]
            sub_covariance = covariance_values[np.ix_(community, community)]
            dynamic_reads = _reads_for_size(len(community))
            if use_incremental:
                shift = current_cov_vector[community] - sub_covariance @ local_state
            else:
                external = np.setdiff1d(all_indices, community)
                shift = covariance_values[np.ix_(community, external)] @ current_state[external]
            adjusted_returns = returns_values[community] - 2 * effective_risk_factor * shift
            info = {"raw_candidate_feasible": None, "candidate_repaired": False}
            backend_called = community_cardinality not in (0, len(community))
            if backend_called:
                model, weights = get_model_func(
                    adjusted_returns, sub_covariance, budget=community_cardinality,
                    risk_factor=effective_risk_factor,
                )
                _attach_local_problem_metadata(
                    model, adjusted_returns, sub_covariance, community_cardinality, effective_risk_factor
                )
                model.name = f"model_{threading.get_ident()}"
                call_seed = None if seed is None else int(seed) + iteration_idx * 100003 + community_pos
                optimizer = make_optimizer(dynamic_reads, call_seed)
                build_time = perf_counter() - t0
                t1 = perf_counter()
                raw, _, _ = optimizer(model, weights)
                local_solution, info = _normalize_local_solution(
                    raw, len(community), community_cardinality, return_info=True
                )
                solve_time = perf_counter() - t1
            else:
                local_solution = np.full(len(community), int(community_cardinality > 0), dtype=int)
                build_time, solve_time = perf_counter() - t0, 0.0
            return (
                community, community_cardinality, local_solution, local_state,
                sub_covariance, adjusted_returns, dynamic_reads,
                perf_counter() - community_start, build_time, solve_time, info, backend_called,
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
            normalization_info,
            backend_called,
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
                    if global_delta_if_committed <= -1e-5:
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
                        "backend_called": bool(backend_called),
                        **normalization_info,
                        "local_obj_before": float(local_obj_before),
                        "local_obj_candidate": float(local_obj_candidate),
                        "local_improved": bool(local_improved),
                        "global_delta_if_committed": float(global_delta_if_committed),
                        "accepted_globally": bool(accepted_globally),
                    }
                )

        iteration_seconds.append(perf_counter() - iteration_start)

        # Early stopping is based on the true global objective after the iteration.
        if current_metrics["objective"] < best_objective:
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
        "initialization_diagnostics": initialization_diagnostics,
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
