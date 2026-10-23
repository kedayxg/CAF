"""Compare public Newman spectral, fixed-nine CNM, and Louvain partitions.

All methods receive the same weighted graph. Diagonal matrix weight A[i,i]
is represented by a graph self-loop of weight A[i,i]/2, so weighted degrees
and modularity match the original matrix convention exactly.
"""
import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np

from caf_core import evaluate_portfolio, run_caf_pipeline
from dcmppln.newman import matrix_graph_edges, newman_partition
from experiments._common import (
    BUNDLED_DATA_DIR, extract_communities_from_partitions, load_local_jpm_data,
    load_upstream_components, split_cardinality_constraint, to_jsonable,
)
from experiments.classical.run_benchmark import paired_objective_statistics
from experiments.classical.run_fixed_community_benchmark import FixedPartition, source_digest

METHODS = ("igraph_spectral", "cnm9", "louvain")


def matrix_to_graph(adjacency):
    matrix, edges = matrix_graph_edges(adjacency)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(matrix)))
    graph.add_weighted_edges_from(edges)
    np.testing.assert_allclose([graph.degree(i, weight="weight") for i in range(len(matrix))],
                               matrix.sum(axis=1), rtol=1e-12, atol=1e-12)
    return graph


def library_partition(adjacency, method, seed=42):
    """Return canonical labels and directly checked graph/matrix modularities."""
    matrix = np.asarray(adjacency, dtype=float)
    graph = matrix_to_graph(matrix)
    start = perf_counter()
    reported_q = None
    if method == "igraph_spectral":
        labels, reported_q = newman_partition(matrix, seed=seed)
        groups = [np.flatnonzero(labels == label).tolist() for label in np.unique(labels)]
        algorithm = "Newman leading-eigenvector recursive modularity splitting"
    elif method == "cnm9":
        groups = [sorted(g) for g in nx.community.greedy_modularity_communities(
            graph, weight="weight", resolution=1., cutoff=9, best_n=9)]
        algorithm = "Clauset-Newman-Moore greedy modularity, cutoff=best_n=9"
    elif method == "louvain":
        groups = [sorted(g) for g in nx.community.louvain_communities(
            graph, weight="weight", resolution=1., threshold=1e-7, seed=seed)]
        algorithm = "Louvain modularity, resolution=1"
    else:
        raise ValueError(f"unknown method: {method}")
    seconds = perf_counter()-start
    groups = sorted(groups, key=lambda g:g[0])
    assert sorted(i for g in groups for i in g) == list(range(len(matrix)))
    labels = np.empty(len(matrix), dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    degree, total = matrix.sum(axis=1), float(matrix.sum())
    full_b = matrix - np.outer(degree, degree)/total
    matrix_q = float(sum(full_b[np.ix_(g,g)].sum() for g in groups)/total)
    graph_q = float(nx.community.modularity(graph, groups, weight="weight", resolution=1.))
    np.testing.assert_allclose(matrix_q, graph_q, rtol=0, atol=1e-10)
    if reported_q is not None:
        np.testing.assert_allclose(matrix_q, reported_q, rtol=0, atol=1e-10)
    same = labels[:,None] == labels[None,:]
    return labels, {"method":method, "algorithm":algorithm, "partition_seed":seed,
        "community_count":len(groups), "community_sizes":sorted(map(len, groups), reverse=True),
        "communities":groups, "modularity":matrix_q, "networkx_modularity":graph_q,
        "igraph_reported_modularity":reported_q, "partition_seconds":seconds,
        "arpack_options":{"maxiter":10000,"tol":1e-10} if method=="igraph_spectral" else None,
        "cross_block_absolute_denoised_correlation_fraction":float(matrix[~same].sum()/total),
        "graph_convention":"undirected weighted graph; self-loop weight A_ii/2; weighted degree equals matrix row sum"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--partition-only", action="store_true")
    parser.add_argument("--check-no-cache-first", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    _, Denoiser, SA, Pipeline, get_model, _ = load_upstream_components(None)
    covariance, correlation, returns = load_local_jpm_data(BUNDLED_DATA_DIR)
    filtered, _, _ = Denoiser(active=True, q=.5, q_fit=True)(correlation.values)
    labels, partition = library_partition(np.abs(filtered), args.method, args.partition_seed)
    communities = extract_communities_from_partitions(labels)
    cards = split_cardinality_constraint([len(c) for c in communities], len(returns)//2)
    partition["communities_solver_order"] = [c.tolist() for c in communities]
    partition["subproblem_cardinalities"] = cards.tolist()
    partition["caf_reads_per_full_scan"] = sum(200 if len(c)>80 else 100 if len(c)>30 else 20
        for c,k in zip(communities,cards) if 0<k<len(c))
    partition["static_reads_per_run"] = 200*len(communities)
    same = labels[:,None] == labels[None,:]
    partition["cross_block_covariance_squared_frobenius_fraction"] = float(np.sum(covariance.values[~same]**2)/np.sum(covariance.values**2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    summary = {"experiment":"public community-library comparison", "source_sha256":source_digest(),
        "library_versions":{name:importlib.metadata.version(name) for name in ("igraph","networkx","numpy","scipy","texttable")},
        "config":{"method":args.method,"assets":len(returns),"cardinality":len(returns)//2,
                  "risk_factor":.5,"risk_rebalancing":False,"num_reads":200,
                  "iterations":6,"patience":2,"base_seed":args.base_seed,
                  "partition_seed":args.partition_seed,"check_no_cache_first":args.check_no_cache_first},
        "data_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(BUNDLED_DATA_DIR.glob("1_2016-01-01*.npy"))},
        "partition":partition,"runs":args.runs,"runs_completed":0,"per_run_records":[]}
    output = args.out_dir/"benchmark_summary.json"
    output.write_text(json.dumps(to_jsonable(summary),indent=2)+"\n")
    print(json.dumps({k:v for k,v in partition.items() if k not in ("communities","communities_solver_order")}),flush=True)
    if args.partition_only:
        return

    def factory(num_reads, seed=None):
        return SA(num_reads=num_reads,seed=seed)

    for index in range(args.runs):
        seed=args.base_seed+index
        np.random.seed(seed)
        random.seed(seed)
        cluster=FixedPartition(labels,len(communities))
        cluster.params={"method":args.method,"partition_seed":args.partition_seed,"fixed":True}
        start=perf_counter()
        pipeline=Pipeline(correlation.values,covariance.values,returns.values,
            denoiser=Denoiser(active=True,q=.5,q_fit=True),cluster=cluster,optimize_func=SA(num_reads=200))
        initial=pipeline.run(run_optimizer=True,risk_rebalancing=False,
            cluster_on_correlation=True,optimize_on_correlation=False,input_risk_factor=.5)
        selected=np.flatnonzero(np.asarray(initial["recombined_solution"]).astype(int)==1).tolist()
        baseline=evaluate_portfolio(selected,covariance,returns,.5)
        baseline_seconds=perf_counter()-start
        options=dict(covariance_matrix=covariance,expected_returns=returns,communities=communities,
            subproblem_cardinalities=cards,get_model_func=get_model,optimizer_factory=factory,
            risk_factor=.5,effective_risk_factor=.5,initial_selected_indices=selected,
            iterations=6,patience=2,seed=seed)
        caf=run_caf_pipeline(**options,use_incremental=True,return_diagnostics=True)
        for metrics in (baseline,caf["metrics"]):
            w=np.array(metrics["weights"])
            assert np.all((w==0)|(w==1)) and w.sum()==len(returns)//2
            assert all(w[c].sum()==k for c,k in zip(communities,cards))
        assert np.all(np.diff(caf["history"]["objective"])<=1e-10)
        record={"seed":seed,"jpm":{"metrics":baseline,"seconds":baseline_seconds},"caf":caf,
                "total_seconds_excluding_partition":perf_counter()-start}
        if index==0 and args.check_no_cache_first:
            uncached=run_caf_pipeline(**options,use_incremental=False)
            assert np.array_equal(uncached["metrics"]["weights"],caf["metrics"]["weights"])
            assert np.allclose(uncached["history"]["objective"],caf["history"]["objective"],rtol=0,atol=1e-12)
            record["no_cache_check"]={"final_weights_match":True,"histories_match_at_1e_minus_12":True,
                "objective_delta":uncached["metrics"]["objective"]-caf["metrics"]["objective"]}
        summary["per_run_records"].append(record)
        summary["runs_completed"]=index+1
        summary.update(paired_objective_statistics(
            [r["jpm"]["metrics"]["objective"] for r in summary["per_run_records"]],
            [r["caf"]["metrics"]["objective"] for r in summary["per_run_records"]]))
        for side in ("jpm","caf"):
            values=[r[side]["metrics"]["objective"] for r in summary["per_run_records"]]
            summary[side]={"objective_mean":float(np.mean(values)),"objective_std":float(np.std(values))}
        output.write_text(json.dumps(to_jsonable(summary),indent=2)+"\n")
        print(json.dumps({"method":args.method,"run":index+1,"seed":seed,
            "static":baseline["objective"],"caf":caf["metrics"]["objective"],
            "improvement_pct":(baseline["objective"]-caf["metrics"]["objective"])/abs(baseline["objective"])*100}),flush=True)
    print(f"Finished: {output}",flush=True)


if __name__=="__main__":
    main()
