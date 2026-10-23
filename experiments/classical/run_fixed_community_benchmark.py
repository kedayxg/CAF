"""Offline SA/CAF study with corrected spectral partitioning and a fixed count.

Natural positive-gain Newman splits are followed by largest-block-first balanced
spectral bisections. Forced splits may decrease modularity. The vendored default
partitioner is not changed by this experiment.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import helmert

from caf_core import evaluate_portfolio, run_caf_pipeline
from experiments._common import (
    BUNDLED_DATA_DIR, extract_communities_from_partitions, load_local_jpm_data,
    load_upstream_components, split_cardinality_constraint, to_jsonable,
)
from experiments.classical.run_benchmark import paired_objective_statistics

ROOT = Path(__file__).resolve().parents[2]


def corrected_fixed_partition(adjacency, target_count):
    """Return canonical labels and an audit of natural and forced splits."""
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.all(np.isfinite(adjacency)) or np.any(adjacency < 0):
        raise ValueError("adjacency must be finite and nonnegative")
    if not np.allclose(adjacency, adjacency.T, rtol=1e-10, atol=1e-12):
        raise ValueError("adjacency must be symmetric")
    n = len(adjacency)
    if isinstance(target_count, bool) or int(target_count) != target_count or not 1 <= target_count <= n:
        raise ValueError("target_count must be an integer between one and the node count")
    target_count = int(target_count)
    degree, total = adjacency.sum(axis=1), float(adjacency.sum())
    full_b = adjacency - np.outer(degree, degree) / total if total else np.zeros_like(adjacency)
    trace = []

    def block_matrix(nodes):
        block = full_b[np.ix_(nodes, nodes)].copy()
        return block - np.diag(block.sum(axis=1))

    def record(nodes, left, right, block, forced):
        signs = np.where(np.isin(nodes, left), 1., -1.)
        gain = float(signs @ block @ signs / (2 * total)) if total else 0.
        trace.append({"forced": forced, "parent_nodes": nodes.tolist(),
                      "child_nodes": [left.tolist(), right.tolist()],
                      "parent_size": len(nodes), "child_sizes": [len(left), len(right)],
                      "modularity_gain": gain})
        return gain

    pending, groups = [np.arange(n)], []
    while pending:
        nodes = pending.pop()
        if len(nodes) < 2 or total == 0:
            groups.append(nodes)
            continue
        block = block_matrix(nodes)
        _, vectors = np.linalg.eigh(block)
        mask = vectors[:, -1] >= 0
        left, right = nodes[mask], nodes[~mask]
        signs = np.where(mask, 1., -1.)
        gain = float(signs @ block @ signs / (2 * total))
        if not len(left) or not len(right) or gain <= 1e-12:
            groups.append(nodes)
        else:
            record(nodes, left, right, block, False)
            pending.extend((left, right))

    natural_groups = sorted(groups, key=lambda g: int(g[0]))
    if len(groups) > target_count:
        raise ValueError("target_count is smaller than the natural partition; merging is not implemented")
    while len(groups) < target_count:
        position = min(range(len(groups)), key=lambda i: (-len(groups[i]), int(groups[i][0])))
        nodes = groups.pop(position)
        block = block_matrix(nodes)
        # Remove the constant eigenvector. Otherwise a nonpositive-gain block's
        # leading vector can be constant, providing no usable spectral ordering.
        basis = helmert(len(nodes), full=False)
        _, vectors = np.linalg.eigh(basis @ block @ basis.T)
        vector = basis.T @ vectors[:, -1]
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        order = np.lexsort((nodes, vector))
        middle = len(nodes) // 2
        left, right = np.sort(nodes[order[:middle]]), np.sort(nodes[order[middle:]])
        record(nodes, left, right, block, True)
        groups.extend((left, right))

    groups = sorted(groups, key=lambda g: int(g[0]))
    labels = np.empty(n, dtype=int)
    for label, nodes in enumerate(groups):
        labels[nodes] = label
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(n))
    natural_q = float(sum(full_b[np.ix_(g,g)].sum() for g in natural_groups) / total) if total else 0.
    final_q = float(sum(full_b[np.ix_(g,g)].sum() for g in groups) / total) if total else 0.
    same = labels[:, None] == labels[None, :]
    info = {"natural_count": len(natural_groups), "target_count": target_count,
            "natural_community_sizes": sorted(map(len, natural_groups), reverse=True),
            "community_sizes": sorted(map(len, groups), reverse=True),
            "natural_modularity": natural_q, "modularity": final_q,
            "cross_block_absolute_denoised_correlation_fraction": float(adjacency[~same].sum()/total) if total else 0.,
            "natural_communities": [g.tolist() for g in natural_groups],
            "communities": [g.tolist() for g in groups], "split_trace": trace,
            "forced_policy": "largest_block_first_nonconstant_spectral_vector_median_bisection"}
    return labels, info


class FixedPartition:
    def __init__(self, labels, count):
        self.labels = labels
        self.params = {"method": "corrected_spectral_fixed_count", "count": count}

    def __call__(self, matrix):
        if len(matrix) != len(self.labels):
            raise ValueError("fixed partition dimension mismatch")
        return self.labels.copy(), 0., 0.


def source_digest():
    result = hashlib.sha256()
    for parent in ("caf_core", "dcmppln", "experiments", "tests"):
        for path in sorted((ROOT/parent).rglob("*.py")):
            if path.name != "config.py":
                result.update(path.relative_to(ROOT).as_posix().encode()+b"\0"+path.read_bytes()+b"\0")
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--communities", type=int, default=9)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--check-no-cache-first", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    _, Denoiser, SA, Pipeline, get_model, _ = load_upstream_components(None)
    covariance, correlation, returns = load_local_jpm_data(BUNDLED_DATA_DIR)
    filtered, _, _ = Denoiser(active=True, q=.5, q_fit=True)(correlation.values)
    labels, partition = corrected_fixed_partition(np.abs(filtered), args.communities)
    communities = extract_communities_from_partitions(labels)
    cards = split_cardinality_constraint([len(c) for c in communities], len(returns)//2)
    partition["communities_solver_order"] = [c.tolist() for c in communities]
    partition["subproblem_cardinalities"] = cards.tolist()
    partition["caf_reads_per_outer_iteration"] = sum(200 if len(c)>80 else 100 if len(c)>30 else 20 for c in communities)
    same = labels[:,None] == labels[None,:]
    partition["cross_block_covariance_squared_frobenius_fraction"] = float(np.sum(covariance.values[~same]**2)/np.sum(covariance.values**2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir/"benchmark_summary.json"
    summary = {"experiment": "corrected spectral clustering with a forced fixed community count",
               "source_sha256": source_digest(), "runs": args.runs, "runs_completed": 0,
               "config": {"assets": len(returns), "cardinality": len(returns)//2,
                          "risk_factor": .5, "risk_rebalancing": False, "num_reads": 200,
                          "iterations": 6, "patience": 2, "base_seed": args.base_seed,
                          "communities": args.communities, "check_no_cache_first": args.check_no_cache_first},
               "partition": partition,
               "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(BUNDLED_DATA_DIR.glob("1_2016-01-01*.npy"))},
               "per_run_records": []}

    def factory(num_reads, seed=None):
        return SA(num_reads=num_reads, seed=seed)

    print(json.dumps({"partition_sizes": partition["community_sizes"],
        "modularity": partition["modularity"], "reads_per_iteration": partition["caf_reads_per_outer_iteration"]}), flush=True)
    for index in range(args.runs):
        start = perf_counter()
        seed = args.base_seed+index
        np.random.seed(seed)
        random.seed(seed)
        pipeline = Pipeline(correlation.values, covariance.values, returns.values,
            denoiser=Denoiser(active=True, q=.5, q_fit=True),
            cluster=FixedPartition(labels, args.communities), optimize_func=SA(num_reads=200))
        initial = pipeline.run(run_optimizer=True, risk_rebalancing=False,
            cluster_on_correlation=True, optimize_on_correlation=False, input_risk_factor=.5)
        selected = np.flatnonzero(np.asarray(initial["recombined_solution"]).astype(int)==1).tolist()
        baseline = evaluate_portfolio(selected, covariance, returns, .5)
        options = dict(covariance_matrix=covariance, expected_returns=returns,
            communities=communities, subproblem_cardinalities=cards, get_model_func=get_model,
            optimizer_factory=factory, risk_factor=.5, effective_risk_factor=.5,
            initial_selected_indices=selected, iterations=6, patience=2, seed=seed)
        result = run_caf_pipeline(**options, use_incremental=True, return_diagnostics=True)
        for metrics in (baseline, result["metrics"]):
            weights = np.asarray(metrics["weights"])
            assert weights.sum() == len(returns)//2
            assert all(weights[c].sum()==k for c,k in zip(communities,cards))
        assert np.all(np.diff(result["history"]["objective"]) <= 1e-10)
        record = {"seed": seed, "jpm": {"metrics": baseline}, "caf": result,
                  "elapsed_seconds": perf_counter()-start}
        if index == 0 and args.check_no_cache_first:
            uncached = run_caf_pipeline(**options, use_incremental=False)
            assert np.array_equal(uncached["metrics"]["weights"], result["metrics"]["weights"])
            assert np.allclose(uncached["history"]["objective"], result["history"]["objective"], rtol=0, atol=1e-12)
            record["no_cache_check"] = {"final_weights_match": True,
                "objective_delta": uncached["metrics"]["objective"]-result["metrics"]["objective"],
                "histories_match_at_1e_minus_12": True}
        summary["per_run_records"].append(record)
        summary["runs_completed"] = index+1
        summary.update(paired_objective_statistics(
            [r["jpm"]["metrics"]["objective"] for r in summary["per_run_records"]],
            [r["caf"]["metrics"]["objective"] for r in summary["per_run_records"]]))
        for side in ("jpm", "caf"):
            values = [r[side]["metrics"]["objective"] for r in summary["per_run_records"]]
            summary[side] = {"objective_mean": float(np.mean(values)), "objective_std": float(np.std(values))}
        output.write_text(json.dumps(to_jsonable(summary), indent=2)+"\n")
        print(json.dumps({"run": index+1, "seed": seed, "baseline": baseline["objective"],
            "caf": result["metrics"]["objective"], "improvement_pct":
            (baseline["objective"]-result["metrics"]["objective"])/abs(baseline["objective"])*100}), flush=True)
    print(f"Finished: {output}", flush=True)


if __name__ == "__main__":
    main()
