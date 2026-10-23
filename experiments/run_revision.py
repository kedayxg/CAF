"""Checkpointed offline reruns for the igraph Newman manuscript revision.

Run from the repository root: python -m experiments.run_revision --workers 4
Each independent seed is an isolated subprocess with an immutable JSON result.
No hardware jobs or publication actions are performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np

from caf_core import evaluate_portfolio, run_caf_pipeline
from experiments._common import (BUNDLED_DATA_DIR, extract_communities_from_partitions,
    load_local_jpm_data, load_upstream_components, split_cardinality_constraint,
    subset_problem, to_jsonable)
from experiments.classical.run_fixed_community_benchmark import source_digest
from experiments.revision_protocol import CLASSICAL_CAPACITY, QUANTUM_CAPACITY, PROTOCOL_NAME

ROOT = Path(__file__).resolve().parents[1]


def tasks():
    """Fixed protocol, declared before the revision reruns."""
    result = []
    for index in range(20):
        result.append(dict(group="panel2016", index=index))
        for depth in (2, 3):
            result.append(dict(group=f"qaoa{depth}", index=index, depth=depth))
        result.append(dict(group="sqa", index=index))
        result.append(dict(group="panel2018", index=index))
        for n in (100, 200, 300, 484):
            result.append(dict(group="scaling_sa", n=n, k=n//2, index=index))
        result.append(dict(group="sparsity_sa", n=200, k=50, index=index))
        for qi, q in enumerate((.1, .3, .5, .7, 1.)):
            result.append(dict(group="q_sweep", q=q, qi=qi, index=index))
        if index < 10:
            for n in (100, 200, 300, 484):
                result.append(dict(group="suppression", n=n, index=index))
    for n, k in ((100, 50), (200, 100), (300, 150), (484, 242), (200, 50)):
        result.append(dict(group="gurobi", n=n, k=k, index=0))
    return result


def task_id(task):
    return "_".join(f"{k}-{v}" for k, v in sorted(task.items()))


def shared_initial_selected(main_data, index, communities):
    """Reject shared initializations from a different decomposition protocol."""
    if main_data['config'].get('max_community_size') != CLASSICAL_CAPACITY:
        raise ValueError('Risk sweep requires the capacity-protocol main initialization')
    actual = [list(map(int, group)) for group in communities]
    if actual != main_data['partition']['communities_solver_order']:
        raise ValueError('Risk sweep initialization and partition do not match')
    record = main_data['per_run_records'][index]
    if record['seed'] != 42+index:
        raise ValueError('Risk sweep initialization seed does not match')
    return np.flatnonzero(np.asarray(record['jpm']['metrics']['weights']) == 1).tolist()


def classical(task, main_result):
    from dcmppln.optimizer import GurobiOptimizer
    Clustering, Denoiser, SA, Pipeline, get_model, _ = load_upstream_components(None)
    group, index = task["group"], task["index"]
    prefix = "1_2018-01-01" if group == "panel2018" else "1_2016-01-01"
    cov, corr, ret = load_local_jpm_data(BUNDLED_DATA_DIR, prefix)
    if "n" in task:
        cov, corr, ret = subset_problem(cov, corr, ret, task["n"], 123+task["n"])
    k = task.get("k", len(ret)//2)
    q = task.get("q", .5)
    seed = (42+10000+task["qi"]*100+index if group == "q_sweep" else
            42+index if group in ("panel2016", "panel2018") else 42+len(ret)*1000+k*10+index)
    np.random.seed(seed)
    random.seed(seed)
    factory = (lambda num_reads=None, seed=None: GurobiOptimizer()) if group == "gurobi" else (
        lambda num_reads=200, seed=None: SA(num_reads=num_reads, seed=seed))
    start = perf_counter()
    cluster = Clustering(active=True, clustering_method="newman", seed=42,
                         max_community_size=CLASSICAL_CAPACITY)
    pipe = Pipeline(corr.values, cov.values, ret.values,
        denoiser=Denoiser(active=True, q=.5, q_fit=True), cluster=cluster,
        optimize_func=factory(seed=seed))
    if group == "q_sweep":
        _, labels = pipe.run(run_optimizer=False, risk_rebalancing=False)
        # Shared initialization from the independently completed twenty-run study.
        main_data = json.loads(main_result.read_text())
        selected = shared_initial_selected(main_data, index, extract_communities_from_partitions(labels))
    else:
        if k == len(ret)//2:
            initial = pipe.run(run_optimizer=True, risk_rebalancing=False,
                cluster_on_correlation=True, optimize_on_correlation=False, input_risk_factor=q)
        else:
            filtered, labels = pipe.run(run_optimizer=False, risk_rebalancing=False)
            initial = pipe.optimize_and_score(matrix_to_cluster=filtered,
                matrix_to_optimize=cov.values, partitions=labels, risk_factor=q,
                unique_identifier_base=f"revision_{seed}", log_best_feasible=False,
                cluster_on_correlation=True, optimize_on_correlation=False,
                denoise_process_time=0., clustering_process_time=0.,
                denoise_time=0., clustering_time=0., risk_rebalancing=False, cardinality=k)
        labels = pipe.__state__["result"]["partitions"]
        selected = np.flatnonzero(np.asarray(initial["recombined_solution"]) == 1).tolist()
    seconds = perf_counter()-start
    communities = extract_communities_from_partitions(labels)
    cards = split_cardinality_constraint([len(c) for c in communities], k)
    static = evaluate_portfolio(selected, cov, ret, q)
    caf = run_caf_pipeline(cov, ret, communities, cards, get_model_func=get_model,
        optimizer_factory=factory, risk_factor=q, effective_risk_factor=q,
        initial_selected_indices=selected, iterations=6, patience=2,
        max_workers=1, seed=seed, return_diagnostics=True)
    for metrics in (static, caf["metrics"]):
        w = np.asarray(metrics["weights"])
        assert np.isin(w, (0, 1)).all() and w.sum() == k
        assert all(w[c].sum() == card for c, card in zip(communities, cards))
    assert np.all(np.diff(caf["history"]["objective"]) <= 1e-10)
    return dict(seed=seed, risk_factor=q, communities=communities,
        partition_max_community_size=CLASSICAL_CAPACITY,
        community_sizes=[len(c) for c in communities], subproblem_cardinalities=cards,
        static=static, caf=caf["metrics"], history=caf["history"],
        static_time_sec=seconds, caf_time_sec=caf["timing"]["total_seconds"],
        caf_timing=caf["timing"], proposal_diagnostics=caf["proposal_diagnostics"],
        initialization_scope="shared q=0.5 baseline" if group == "q_sweep" else "fresh static solve")


def quantum(task):
    from experiments.quantum.run_qaoa import (QAOAConfig, QiskitQAOAOptimizer,
        SQAOptimizer, make_denoiser_factory, run_one_setting, run_reference_backend)
    Clustering, Denoiser, _, Pipeline, get_model, rebal = load_upstream_components(None)
    cov, corr, ret = subset_problem(*load_local_jpm_data(BUNDLED_DATA_DIR), 40, 1234)
    common = dict(cov=cov, corr=corr, ret=ret, cardinality=20, risk_factor=.5,
        runs=1, base_seed=42+task["index"], iterations=4, patience=2, max_workers=1,
        Clustering=Clustering, denoiser_factory=make_denoiser_factory(Denoiser),
        Pipeline=Pipeline, get_model=get_model, rebalancing_risk_factor=rebal,
        max_community_size=QUANTUM_CAPACITY)
    if task["group"] == "sqa":
        _, records = run_reference_backend(optimizer_cls=SQAOptimizer,
            seed_offset=30000, default_num_reads=20, num_sweeps=1000, **common)
    else:
        def factory(seed, num_reads=None):
            return QiskitQAOAOptimizer(QAOAConfig(p=task["depth"], restarts=6,
                maxiter=80, shots=4000, decode="shots", warmstart=False, seed=seed))
        records = run_one_setting(optimizer_factory_static=factory,
            optimizer_factory_caf=factory, **common)
    record = records[0]
    if task["group"] == "sqa":
        record["backend_protocol"] = dict(sampler="PathIntegralAnnealingSampler",
            Hd_field="linspace(1,0,1000)", Gamma=1., num_sweeps=1000,
            longitudinal_schedule="default geometric; automatic beta range from each BQM")
    for side in ("static", "caf"):
        w = np.asarray(record[side]["weights"])
        assert np.isin(w, (0, 1)).all() and w.sum() == 20
    assert record["caf"]["objective"] <= record["static"]["objective"]+1e-10
    return record


def execute(task, main_result):
    if task["group"] in ("qaoa2", "qaoa3", "sqa"):
        return quantum(task)
    if task["group"] == "suppression":
        from types import SimpleNamespace
        from experiments.classical.run_suppression import run_one_setting
        args = SimpleNamespace(subset_seed=123, sa_num_reads=100, risk_factor=.5,
                               max_community_size=CLASSICAL_CAPACITY)
        return run_one_setting(*load_local_jpm_data(BUNDLED_DATA_DIR),
            task["n"], task["index"], args)
    return classical(task, main_result)


def write_main_result(directory, target):
    """Export fresh main-panel records in the manuscript's established schema."""
    items = [json.loads((directory/(task_id(dict(group='panel2016', index=i))+'.json')).read_text())
             for i in range(20)]
    records = [item['record'] for item in items]
    first = records[0]
    for i, r in enumerate(records):
        assert items[i]['source_sha256'] == source_digest()
        assert items[i]['task'] == dict(group='panel2016', index=i)
        assert r['seed'] == 42+i
        assert r['communities'] == first['communities']
        assert r['subproblem_cardinalities'] == first['subproblem_cardinalities']
    result = dict(experiment='Fresh capacity-protocol main 2016 experiment',
        source_sha256=source_digest(), runs=20, runs_completed=20,
        config=dict(max_community_size=CLASSICAL_CAPACITY, partition_seed=42,
                    risk_factor=.5, risk_rebalancing=False),
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(BUNDLED_DATA_DIR.glob('1_2016-01-01*.npy'))},
        partition=dict(communities_solver_order=first['communities'],
            subproblem_cardinalities=first['subproblem_cardinalities'],
            community_sizes=first['community_sizes'], community_count=len(first['communities'])),
        per_run_records=[dict(seed=r['seed'], jpm=dict(metrics=r['static'],seconds=r['static_time_sec']),
            caf=dict(metrics=r['caf'],history=r['history'],timing=r['caf_timing'],
                     proposal_diagnostics=r['proposal_diagnostics'])) for r in records])
    target.write_text(json.dumps(to_jsonable(result),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"revision_results_v2")
    parser.add_argument("--main-result", type=Path,
        help="Main initialization export; defaults to OUT_DIR/main_2016.json")
    parser.add_argument("--groups", default="all", help="Comma-separated groups, or all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.main_result is None:
        args.main_result = args.out_dir/'main_2016.json'
    if args.worker_task:
        task = json.loads(args.worker_task)
        start = perf_counter()
        result = dict(task=task, source_sha256=source_digest(),
            protocol=dict(name=PROTOCOL_NAME,classical_capacity=CLASSICAL_CAPACITY,
                          quantum_capacity=QUANTUM_CAPACITY),
            library_versions={name:importlib.metadata.version(name)
                for name in ("igraph", "numpy", "scipy", "qiskit", "dwave-samplers", "gurobipy")},
            record=execute(task, args.main_result), total_seconds=perf_counter()-start)
        target = args.out_dir/(task_id(task)+".json")
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(to_jsonable(result), indent=2)+"\n")
        temp.replace(target)
        return
    if args.workers < 1:
        parser.error("workers must be positive")
    wanted = set(args.groups.split(","))
    selected = [t for t in tasks() if "all" in wanted or t["group"] in wanted]
    if not selected:
        parser.error("no task groups selected")
    data_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(BUNDLED_DATA_DIR.glob("*.npy"))}
    expected_source = source_digest()
    def run(task):
        name = task_id(task)
        target = args.out_dir/(name+".json")
        if target.exists():
            existing = json.loads(target.read_text())
            if existing["task"] != task or existing["source_sha256"] != expected_source:
                raise RuntimeError(f"Checkpoint protocol/source mismatch: {target}")
            return name, "cached"
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
            MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
        with (args.out_dir/(name+".log")).open("w") as log:
            subprocess.run([sys.executable, "-m", "experiments.run_revision",
                "--out-dir", str(args.out_dir.resolve()), "--main-result", str(args.main_result.resolve()),
                "--worker-task", json.dumps(task)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        return name, "completed"
    def run_stage(stage):
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run, t):t for t in stage}
            for f in as_completed(futures):
                try:
                    print(f.result(), flush=True)
                except Exception as error:
                    failed.append(dict(task=futures[f], error=str(error)))
                    print("FAILED", failed[-1], flush=True)
        if failed:
            raise RuntimeError(f"{len(failed)} tasks failed; completed checkpoints retained")

    preparation = dict(tasks=selected,source_sha256=expected_source,data_sha256=data_hashes,
        workers=args.workers,protocol=PROTOCOL_NAME,
        classical_capacity=CLASSICAL_CAPACITY,quantum_capacity=QUANTUM_CAPACITY)
    manifest_path = args.out_dir/f"manifest-{args.groups.replace(',', '-')}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != preparation:
        raise RuntimeError("Existing manifest does not match source or protocol")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(preparation,indent=2)+'\n')
    main_tasks = [t for t in selected if t['group']=='panel2016']
    if main_tasks:
        run_stage(main_tasks)
        write_main_result(args.out_dir,args.main_result)
    remaining = [t for t in selected if t['group']!='panel2016']
    if any(t['group']=='q_sweep' for t in remaining) and not args.main_result.exists():
        raise RuntimeError('Run the panel2016 group before the shared-initialization q sweep')
    if args.main_result.exists():
        (args.out_dir/'main-initialization-sha256.json').write_text(json.dumps(dict(
            main_result_sha256=hashlib.sha256(args.main_result.read_bytes()).hexdigest()),indent=2)+'\n')
    run_stage(remaining)


if __name__ == "__main__":
    main()
