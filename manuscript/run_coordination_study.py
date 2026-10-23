"""Predeclared CAF/BCD, community-capacity and input-subset follow-up.

The frozen experiment core is unchanged. Three methods share a fresh Gurobi
static portfolio and six complete sweeps; method order rotates by case index.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_core import run_caf_pipeline
from caf_core.solver import _attach_local_problem_metadata
from dcmppln.denoiser import Denoiser
from dcmppln.newman import capacity_newman_partition
from dcmppln.utils.objective_after_clustering import get_model
from experiments._common import (BUNDLED_DATA_DIR, load_local_jpm_data,
    extract_communities_from_partitions, split_cardinality_constraint, to_jsonable)
from experiments.classical.run_fixed_community_benchmark import source_digest

PARAMETERS = dict(Threads=1, Seed=0, MIPGap=1e-4, TimeLimit=300)
METHODS = ['caf_serial', 'sequential_bcd', 'caf_parallel4']
ROUNDS = 6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    result = []
    for year in (2016, 2018):
        for subset_seed in (None, 1729, 2718, 3141):
            for capacity in (75, 100, 150):
                index = len(result)
                result.append(dict(index=index, year=year, subset_seed=subset_seed,
                    assets=None if subset_seed is None else 300, capacity=capacity,
                    method_order=METHODS[index % 3:]+METHODS[:index % 3]))
    return result


def case_id(case):
    sample = 'full' if case['subset_seed'] is None else f"n300-seed{case['subset_seed']}"
    return f"panel{case['year']}-{sample}-cap{case['capacity']}"


def selected_problem(case):
    cov, corr, ret = load_local_jpm_data(BUNDLED_DATA_DIR, f"1_{case['year']}-01-01")
    indices = (np.arange(len(ret)) if case['subset_seed'] is None else
        np.sort(np.random.default_rng(case['subset_seed']).choice(len(ret), 300, replace=False)))
    return cov.iloc[indices, indices], corr.iloc[indices, indices], ret.iloc[indices], indices


def sequential_bcd(cov, ret, groups, cards, initial, factory):
    """Latest-context cyclic BCD with the same cache and acceptance threshold."""
    covariance, returns = cov.values, ret.values
    x = np.asarray(initial, dtype=int).copy()
    v = covariance @ x
    history = dict(objective=[], risk=[], expected_return=[])
    diagnostics, round_seconds = [], []
    def metrics():
        risk, expected_return = float(x @ covariance @ x), float(returns @ x)
        return dict(risk=risk, expected_return=expected_return,
                    objective=.5*risk-expected_return, weights=x.copy())
    def append():
        m = metrics()
        for k in history: history[k].append(m[k])
    append()
    start = perf_counter()
    for iteration in range(ROUNDS):
        round_start = perf_counter()
        for ci, (g, card) in enumerate(zip(groups, cards)):
            old = x[g].copy()
            block = covariance[np.ix_(g, g)]
            adjusted = returns[g] - (v[g] - block @ old)
            called = card not in (0, len(g))
            if called:
                model, weights = get_model(adjusted, block, budget=card, risk_factor=.5)
                _attach_local_problem_metadata(model, adjusted, block, card, .5)
                candidate, _, _ = factory()(model, weights)
            else:
                candidate = np.full(len(g), int(card > 0))
            delta_x = candidate-old
            delta = float(.5*(delta_x@block@delta_x+2*v[g]@delta_x)-returns[g]@delta_x)
            accepted = delta <= -1e-5
            if accepted:
                x[g] = candidate
                v += covariance[:, g] @ delta_x
            diagnostics.append(dict(iteration_idx=iteration, community_pos=ci,
                backend_called=called, global_delta_if_committed=delta,
                accepted_globally=accepted))
        append()
        round_seconds.append(perf_counter()-round_start)
    return dict(metrics=metrics(), history=history, proposal_diagnostics=diagnostics,
                timing=dict(total_seconds=perf_counter()-start, iteration_seconds=round_seconds))


def worker(out, case):
    import gurobipy as gp
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['runner_sha256'] == sha(Path(__file__))
    assert manifest['core_source_sha256'] == source_digest()
    np.random.seed(42)
    cov, corr, ret, indices = selected_problem(case)
    name = case_id(case)
    folder = out/name
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(case=case, core_source_sha256=source_digest(),
        runner_sha256=sha(Path(__file__)), input_indices=indices.tolist(),
        dataset_sha256={p.name:sha(p) for p in BUNDLED_DATA_DIR.glob(f"1_{case['year']}-01-01*.npy")})
    before = perf_counter()
    filtered = np.abs(Denoiser(active=True, q=.5, q_fit=True)(corr.values)[0])
    try:
        labels, modularity, trace = capacity_newman_partition(filtered, case['capacity'], seed=42)
    except RuntimeError as e:
        if not str(e).startswith('Newman cannot split community'): raise
        result = dict(metadata, status='partition_failure', reason=str(e),
                      preprocessing_seconds=perf_counter()-before)
        (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        return
    preprocessing_seconds = perf_counter()-before
    groups = extract_communities_from_partitions(labels)
    cards = split_cardinality_constraint(list(map(len, groups)), len(ret)//2)
    lookup = {(len(g), cov.values[np.ix_(g,g)].tobytes()):ci for ci,g in enumerate(groups)}
    assert len(lookup) == len(groups)
    calls, lock = [], threading.Lock()
    counters, serial = {}, [0]
    phase = ['static']

    class AuditedGurobi:
        def __call__(self, model, weights, **kwargs):
            service_start = perf_counter()
            ci = lookup[(len(weights), np.asarray(model._caf_sub_covariance).tobytes())]
            with lock:
                seq = serial[0]; serial[0] += 1
                key = (phase[0], ci)
                iteration = counters.get(key, 0); counters[key] = iteration+1
                current_phase = phase[0]
            logpath = folder/f'{seq:05d}-{current_phase}-r{iteration}-c{ci}.log'
            with tempfile.NamedTemporaryFile(mode='w',suffix='.lp',delete=False) as f:
                f.write(model.lp_string.replace('*',' * ')); lp = Path(f.name)
            try:
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag',0); env.start()
                    with gp.read(str(lp),env=env) as m:
                        for k,value in PARAMETERS.items():m.setParam(k,value)
                        m.setParam('LogToConsole',0);m.setParam('OutputFlag',1)
                        m.setParam('LogFile',str(logpath.resolve()))
                        wall, cpu = perf_counter(), process_time()
                        m.optimize()
                        elapsed, processing = perf_counter()-wall, process_time()-cpu
                        if m.SolCount == 0:raise RuntimeError(f'No incumbent: status {m.Status}')
                        raw = np.array([m.getVarByName(w.name).X for w in weights])
                        x = np.rint(raw).astype(int)
                        assert np.allclose(raw,x,atol=1e-7,rtol=0) and x.sum()==cards[ci]
                        adjusted = np.asarray(model._caf_adjusted_returns)
                        actual = float(.5*x@cov.values[np.ix_(groups[ci],groups[ci])]@x-adjusted@x)
                        np.testing.assert_allclose(actual,m.ObjVal,atol=1e-8,rtol=0)
                        row = dict(phase=current_phase, community_pos=ci, round=iteration,
                            status=int(m.Status), objective=float(m.ObjVal), bound=float(m.ObjBound),
                            gap=float(m.MIPGap), solution_count=int(m.SolCount),
                            runtime_sec=float(m.Runtime), solve_wall_sec=elapsed,
                            service_wall_sec=perf_counter()-service_start,
                            node_count=float(m.NodeCount), x=x.tolist(), model_returns=adjusted.tolist(),
                            log=logpath.relative_to(out).as_posix())
                        with lock:calls.append(row)
                        return x, elapsed, processing
            finally:lp.unlink(missing_ok=True)

    factory = lambda num_reads=None, seed=None:AuditedGurobi()
    before = perf_counter()
    static = run_caf_pipeline(cov,ret,groups,cards,get_model,factory,
        risk_factor=.5,effective_risk_factor=.5,iterations=0,max_workers=1,seed=42,
        return_diagnostics=True)
    static_seconds = perf_counter()-before
    initial = np.asarray(static['metrics']['weights'],dtype=int)
    selected = np.flatnonzero(initial).tolist()
    methods = {}
    for method in case['method_order']:
        phase[0] = method
        method_start = perf_counter()
        if method == 'sequential_bcd':
            methods[method] = sequential_bcd(cov,ret,groups,cards,initial,factory)
        else:
            methods[method] = run_caf_pipeline(cov,ret,groups,cards,get_model,factory,
                risk_factor=.5,effective_risk_factor=.5,initial_selected_indices=selected,
                iterations=ROUNDS,patience=ROUNDS+1,max_workers=4 if method=='caf_parallel4' else 1,
                seed=42,return_diagnostics=True)
        # Include method initialization/cache/history setup for every backend.
        methods[method]['timing']['internal_seconds'] = methods[method]['timing']['total_seconds']
        methods[method]['timing']['total_seconds'] = perf_counter()-method_start
    calls.sort(key=lambda c:(['static']+METHODS).index(c['phase'])*100000+c['round']*len(groups)+c['community_pos'])
    for c in calls:c['log_sha256']=sha(out/c['log'])
    result = dict(metadata,status='complete',parameters=PARAMETERS,rounds=ROUNDS,
        communities=groups,cards=cards,labels=labels,partition_trace=trace,modularity=modularity,
        preprocessing_seconds=preprocessing_seconds,static=static['metrics'],static_seconds=static_seconds,
        methods=methods,local_solves=calls,
        versions={n:importlib.metadata.version(n) for n in ['gurobipy','igraph','numpy','docplex']},
        cpu_affinity_count=len(os.sched_getaffinity(0)),
        timing_note='One measurement per method; case processes run serially. CAF parallel4 uses four independent one-thread solver environments, BCD/CAF serial use one. Timings exclude shared preprocessing/static initialization; no equal peak CPU or hardware-queue assumption.')
    target=out/(name+'.json')
    target.with_suffix('.tmp').write_text(json.dumps(to_jsonable(result),indent=2)+'\n')
    target.with_suffix('.tmp').replace(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--worker-case',type=int)
    p.add_argument('--case-indices',help='Optional comma-separated preflight/resume subset')
    args=p.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=dict(protocol='caf-bcd-capacity-subsets-v1',cases=cases(),core_source_sha256=source_digest(),
        runner_sha256=sha(Path(__file__)),capacities=[75,100,150],subset_assets=300,
        subset_seeds=[1729,2718,3141],methods=METHODS,rounds=ROUNDS,patience_disabled=True,
        partition_seed=42,risk_factor=.5,parameters=PARAMETERS,case_workers=1,
        fixed_budgets=True,no_repartition_within_refinement=True,
        report_all_partition_failures=True,
        timing_interpretation='Single timings, not repeated statistical trials. Four-worker CAF uses more simultaneous solver resources; quality/call-count comparison uses equal six-sweep budgets.')
    if args.worker_case is not None:
        worker(out,cases()[args.worker_case]);return
    path=out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    selected=set(map(int,args.case_indices.split(','))) if args.case_indices else set(range(len(cases())))
    for case in cases():
        if case['index'] not in selected:continue
        target=out/(case_id(case)+'.json')
        if target.exists():
            d=json.loads(target.read_text())
            assert d['runner_sha256']==manifest['runner_sha256'] and d['core_source_sha256']==source_digest()
        else:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
            with (out/(case_id(case)+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--out-dir',str(out),
                    '--worker-case',str(case['index'])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        d=json.loads(target.read_text())
        print(case_id(case),d['status'],
            {m:round(v['metrics']['objective'],9) for m,v in d.get('methods',{}).items()},flush=True)


if __name__=='__main__':main()
