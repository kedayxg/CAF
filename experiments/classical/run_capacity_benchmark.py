"""Checkpointed SA/CAF experiment with induced-subgraph Newman size limits.

This is a separate capacity-oriented study; the manuscript default is unchanged.
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
from dcmppln.newman import capacity_newman_partition
from experiments._common import (BUNDLED_DATA_DIR, extract_communities_from_partitions,
    load_local_jpm_data, load_upstream_components, split_cardinality_constraint, to_jsonable)
from experiments.classical.run_benchmark import paired_objective_statistics
from experiments.classical.run_fixed_community_benchmark import FixedPartition, source_digest

ROOT = Path(__file__).resolve().parents[2]


def capacity_partition(matrix, capacity):
    """Use the same igraph capacity implementation as the public adapter."""
    labels, _, trace = capacity_newman_partition(matrix, capacity, seed=42)
    return labels, trace


def worker(out, seed):
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['source_sha256'] == source_digest()
    cfg = manifest['config']
    _, Denoiser, SA, Pipeline, get_model, _ = load_upstream_components(None)
    cov, corr, ret = load_local_jpm_data(BUNDLED_DATA_DIR, cfg['data_prefix'])
    for path in BUNDLED_DATA_DIR.glob(cfg['data_prefix']+'*.npy'):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['data_sha256'][path.name]
    labels = np.asarray(manifest['partition']['labels'])
    communities = extract_communities_from_partitions(labels)
    cards = split_cardinality_constraint(list(map(len, communities)), cfg['cardinality'])
    np.random.seed(seed)
    random.seed(seed)
    cluster = FixedPartition(labels, len(communities))
    cluster.params = dict(method='induced_subgraph_newman_capacity', capacity=cfg['capacity'], seed=42)
    factory = lambda num_reads=200, seed=None: SA(num_reads=num_reads, seed=seed)
    start = perf_counter()
    pipe = Pipeline(corr.values, cov.values, ret.values,
        denoiser=Denoiser(active=True, q=.5, q_fit=True), cluster=cluster,
        optimize_func=factory(seed=seed))
    initial = pipe.run(run_optimizer=True, risk_rebalancing=False,
        cluster_on_correlation=True, optimize_on_correlation=False, input_risk_factor=.5)
    selected = np.flatnonzero(np.asarray(initial['recombined_solution']) == 1).tolist()
    static = evaluate_portfolio(selected, cov, ret, .5)
    static_seconds = perf_counter()-start
    caf = run_caf_pipeline(cov, ret, communities, cards, get_model_func=get_model,
        optimizer_factory=factory, risk_factor=.5, effective_risk_factor=.5,
        initial_selected_indices=selected, iterations=6, patience=2,
        max_workers=1, seed=seed, return_diagnostics=True)
    record = dict(seed=seed, static=static, caf=caf['metrics'], history=caf['history'],
        static_time_sec=static_seconds, caf_timing=caf['timing'],
        proposal_diagnostics=caf['proposal_diagnostics'], source_sha256=source_digest())
    target = out/f'seed-{seed}.json'
    target.with_suffix('.tmp').write_text(json.dumps(to_jsonable(record), indent=2)+'\n')
    target.with_suffix('.tmp').replace(target)


def collect(out):
    manifest = json.loads((out/'manifest.json').read_text())
    cov, _, ret = load_local_jpm_data(BUNDLED_DATA_DIR, manifest['config']['data_prefix'])
    groups = manifest['partition']['communities_solver_order']
    cards = manifest['partition']['subproblem_cardinalities']
    records = [json.loads((out/f'seed-{seed}.json').read_text()) for seed in manifest['seeds']]
    for record in records:
        assert record['source_sha256'] == manifest['source_sha256']
        for side in ('static', 'caf'):
            m = record[side]
            w = np.asarray(m['weights'])
            assert np.isin(w, (0, 1)).all() and w.sum() == manifest['config']['cardinality']
            assert all(w[g].sum() == k for g, k in zip(groups, cards))
            risk, returns = float(w @ cov.values @ w), float(ret.values @ w)
            np.testing.assert_allclose([risk, returns, .5*risk-returns],
                [m['risk'], m['expected_return'], m['objective']], rtol=0, atol=1e-10)
        h = record['history']['objective']
        assert np.all(np.diff(h) <= 1e-10)
        np.testing.assert_allclose([h[0], h[-1]],
            [record['static']['objective'], record['caf']['objective']], rtol=0, atol=1e-10)
    result = dict(manifest, runs_completed=len(records), per_run_records=records,
        validation=dict(independent_full_matrix_portfolios=2*len(records),
                        feasible_fixed_budgets=True, monotone_histories=len(records)))
    result.update(paired_objective_statistics(
        [r['static']['objective'] for r in records], [r['caf']['objective'] for r in records]))
    for side in ('static', 'caf'):
        result[side] = {f'{metric}_{suffix}': float(fn([r[side][metric] for r in records]))
            for metric in ('objective', 'risk', 'expected_return')
            for suffix, fn in (('mean', np.mean), ('std', np.std))}
    (out/'benchmark_summary.json').write_text(json.dumps(to_jsonable(result), indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('objective_') or
        k in ('static','caf','runs_completed','validation')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=(2016, 2018), default=2018)
    parser.add_argument('--capacity', type=int, default=100)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--worker-seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker_seed is not None:
        worker(out, args.worker_seed)
        return
    if min(args.capacity, args.runs, args.workers) < 1:
        parser.error('capacity, runs, and workers must be positive')
    _, Denoiser, _, _, _, _ = load_upstream_components(None)
    data_prefix = f'1_{args.year}-01-01'
    cov, corr, ret = load_local_jpm_data(BUNDLED_DATA_DIR, data_prefix)
    matrix = np.abs(Denoiser(active=True, q=.5, q_fit=True)(corr.values)[0])
    labels, trace = capacity_partition(matrix, args.capacity)
    groups = extract_communities_from_partitions(labels)
    degree, total = matrix.sum(axis=1), matrix.sum()
    b = matrix - np.outer(degree, degree)/total
    manifest = dict(source_sha256=source_digest(), seeds=list(range(42, 42+args.runs)),
        config=dict(data_prefix=data_prefix, assets=len(ret), cardinality=len(ret)//2,
            capacity=args.capacity, risk_factor=.5, risk_rebalancing=False,
            static_reads_per_community=200, iterations=6, patience=2,
            caf_reads='20 if n<=30; 100 if 30<n<=80; 200 if n>80',
            partition_seed=42, workers=args.workers),
        library_versions={n:importlib.metadata.version(n) for n in
            ('igraph','numpy','scipy','dwave-neal','dimod')},
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(BUNDLED_DATA_DIR.glob(data_prefix+'*.npy'))},
        protocol='Restart Newman on oversized original-denoised induced subgraphs; no forced fallback; '
            'allocate final fixed community budgets by existing proportional rule; '
            'fresh static baseline per seed; exploratory study, manuscript default unchanged.',
        partition=dict(labels=labels.tolist(), communities_solver_order=groups,
            community_sizes=list(map(len, groups)), community_count=len(groups),
            subproblem_cardinalities=split_cardinality_constraint(list(map(len, groups)), len(ret)//2),
            global_modularity=float(sum(b[np.ix_(g,g)].sum() for g in groups)/total), trace=trace))
    manifest = to_jsonable(manifest)
    target = out/'manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise RuntimeError('Existing experiment manifest does not match protocol/source')
    else:
        target.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(community_sizes=manifest['partition']['community_sizes'],
        cards=manifest['partition']['subproblem_cardinalities'])), flush=True)

    def run(seed):
        if not (out/f'seed-{seed}.json').exists():
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MPLBACKEND='Agg', PYTHONPATH=str(ROOT))
            with (out/f'seed-{seed}.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'experiments.classical.run_capacity_benchmark',
                    '--out-dir', str(out), '--worker-seed', str(seed)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        r = json.loads((out/f'seed-{seed}.json').read_text())
        return dict(seed=seed, static=r['static']['objective'], caf=r['caf']['objective'])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, seed) for seed in manifest['seeds']]):
            print(json.dumps(future.result()), flush=True)
    collect(out)


if __name__ == '__main__':
    main()
