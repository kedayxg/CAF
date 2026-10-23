"""Audited Gurobi main panels; preserve the frozen SA batch and core source.

One fixed-configuration pipeline per panel, not repeated statistical trials.
The companion's hash supplements the unchanged experimental core hash.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caf_core import run_caf_pipeline, evaluate_portfolio
from dcmppln.clustering import Clustering
from dcmppln.denoiser import Denoiser
from dcmppln.utils.objective_after_clustering import get_model
from experiments._common import BUNDLED_DATA_DIR, load_local_jpm_data, to_jsonable
from experiments.classical.run_fixed_community_benchmark import source_digest

PARAMETERS = dict(Threads=1, Seed=0, MIPGap=1e-4, TimeLimit=300)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(results, out, year):
    import gurobipy as gp
    source = results / f'group-panel{year}_index-0.json'
    primary = json.loads(source.read_text())
    assert primary['source_sha256'] == source_digest()
    cov, corr, ret = load_local_jpm_data(BUNDLED_DATA_DIR, f'1_{year}-01-01')
    groups = primary['record']['communities']
    cards = primary['record']['subproblem_cardinalities']
    # Reconstruct rather than trusting the archived memberships alone.
    filtered = Denoiser(active=True, q=.5, q_fit=True)(corr.values)[0]
    labels = Clustering(clustering_method='newman', seed=42,
                        max_community_size=100)(filtered)[0]
    assert {tuple(sorted(g)) for g in groups} == {
        tuple(np.flatnonzero(labels == label)) for label in np.unique(labels)}
    lookup = {(len(g), cov.values[np.ix_(g, g)].tobytes()): i
              for i, g in enumerate(groups)}
    assert len(lookup) == len(groups)
    calls = []
    phase = ['static']
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / f'logs-{year}'
    logdir.mkdir(exist_ok=True)
    started = perf_counter()

    class AuditedGurobi:
        def __call__(self, model, weights, **kwargs):
            index = len(calls)
            logpath = logdir / f'{index:04d}-{phase[0]}.log'
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as f:
                f.write(model.lp_string.replace('*', ' * '))
                lp = Path(f.name)
            try:
                with gp.Env(empty=True) as env:
                    env.setParam('OutputFlag', 0)
                    env.start()
                    with gp.read(str(lp), env=env) as m:
                        for name, value in PARAMETERS.items():
                            m.setParam(name, value)
                        m.setParam('OutputFlag', 1)
                        m.setParam('LogToConsole', 0)
                        m.setParam('LogFile', str(logpath.resolve()))
                        t, cpu = perf_counter(), process_time()
                        m.optimize()
                        elapsed, processing = perf_counter()-t, process_time()-cpu
                        if m.SolCount == 0:
                            raise RuntimeError(f'Gurobi returned no incumbent: {m.Status}')
                        x = np.array([m.getVarByName(w.name).X for w in weights])
                        rounded = np.rint(x).astype(int)
                        if not np.allclose(x, rounded, atol=1e-7, rtol=0):
                            raise RuntimeError('Nonbinary Gurobi incumbent')
                        ci = lookup[(len(weights), np.asarray(model._caf_sub_covariance).tobytes())]
                        # The unshifted control deliberately overrides metadata attached by CAF.
                        returns = (ret.values[groups[ci]] if phase[0] in ('static', 'control')
                                   else np.asarray(model._caf_adjusted_returns))
                        risk = float(.5 * rounded @ cov.values[np.ix_(groups[ci], groups[ci])] @ rounded)
                        np.testing.assert_allclose(risk - returns @ rounded, m.ObjVal, atol=1e-8, rtol=0)
                        calls.append(dict(phase=phase[0], community_pos=ci, status=int(m.Status),
                            solution_count=int(m.SolCount), objective=float(m.ObjVal),
                            bound=float(m.ObjBound), gap=float(m.MIPGap), runtime_sec=float(m.Runtime),
                            wall_sec=elapsed, node_count=float(m.NodeCount), x=rounded.tolist(),
                            model_returns=returns.tolist(), log=logpath.relative_to(out).as_posix()))
                        return rounded, elapsed, processing
            finally:
                lp.unlink(missing_ok=True)

    factory = lambda num_reads=None, seed=None: AuditedGurobi()
    # The same CAF feasibility-initialization code solves independent local models.
    static_started = perf_counter()
    initial = run_caf_pipeline(cov, ret, groups, cards, get_model, factory,
        risk_factor=.5, effective_risk_factor=.5, iterations=0, max_workers=1,
        seed=42, return_diagnostics=True)
    static_time = perf_counter()-static_started
    selected = np.flatnonzero(np.asarray(initial['metrics']['weights']) == 1).tolist()
    phase[0] = 'caf'
    caf = run_caf_pipeline(cov, ret, groups, cards, get_model, factory,
        risk_factor=.5, effective_risk_factor=.5, initial_selected_indices=selected,
        iterations=6, patience=2, max_workers=1, seed=42, return_diagnostics=True)
    rounds = len(caf['history']['objective'])-1

    def unshifted(_returns, block, budget, risk_factor):
        ci = lookup[(len(block), np.asarray(block).tobytes())]
        return get_model(ret.values[groups[ci]], block, budget=budget, risk_factor=risk_factor)

    phase[0] = 'control'
    control = run_caf_pipeline(cov, ret, groups, cards, unshifted, factory,
        risk_factor=.5, effective_risk_factor=.5, initial_selected_indices=selected,
        iterations=rounds, patience=rounds+1, max_workers=1, seed=42,
        return_diagnostics=True)
    for call in calls:
        call['log_sha256'] = digest(out/call['log'])
    for m in [initial['metrics'], caf['metrics'], control['metrics']]:
        x = np.asarray(m['weights'])
        assert np.isin(x, (0, 1)).all() and x.sum() == len(ret)//2
        assert all(x[g].sum() == k for g, k in zip(groups, cards))
        recomputed = evaluate_portfolio(np.flatnonzero(x).tolist(), cov, ret, .5)
        np.testing.assert_allclose(m['objective'], recomputed['objective'], atol=1e-10, rtol=0)
    result = dict(year=year, assets=len(ret), cardinality=len(ret)//2, capacity=100,
        core_source_sha256=source_digest(), runner_sha256=digest(Path(__file__)),
        partition_reference_sha256=digest(source), parameters=PARAMETERS,
        versions={name:importlib.metadata.version(name) for name in ['gurobipy','igraph','numpy','docplex']},
        dataset_sha256={p.name:digest(p) for p in BUNDLED_DATA_DIR.glob(f'1_{year}-01-01*.npy')},
        communities=groups, cards=cards, static=initial['metrics'], static_time_sec=static_time,
        caf=caf, control=control, local_solves=calls, total_seconds=perf_counter()-started,
        interpretation='One fixed-configuration flow per panel. Local MIP bounds do not certify full-portfolio global optimality. No equal aggregate effort or runtime claim.')
    target = out/f'panel-{year}.json'
    target.with_suffix('.tmp').write_text(json.dumps(to_jsonable(result), indent=2)+'\n')
    target.with_suffix('.tmp').replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--worker', type=int, choices=[2016,2018])
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.results,args.out_dir,args.worker)
        return
    manifest = dict(protocol='gurobi-primary-two-panel-v2', core_source_sha256=source_digest(),
        runner_sha256=digest(Path(__file__)), years=[2016,2018], capacity=100,
        q=.5, max_rounds=6, patience=2, parameters=PARAMETERS,
        note='Fixed memberships/budgets from the already published-in-workspace SA batch; both initialization and refinement use Gurobi. SA evidence retained unchanged.')
    manifest_path = args.out_dir/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    def run(year):
        path = args.out_dir/f'panel-{year}.json'
        if path.exists():
            d=json.loads(path.read_text())
            assert d['runner_sha256']==manifest['runner_sha256'] and d['core_source_sha256']==source_digest()
        else:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')
            with (args.out_dir/f'panel-{year}.log').open('w') as f:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--results',str(args.results.resolve()),
                    '--out-dir',str(args.out_dir.resolve()),'--worker',str(year)],cwd=ROOT,env=env,
                    stdout=f,stderr=subprocess.STDOUT,check=True)
        return year
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,year) for year in [2016,2018]]):
            print('Completed',f.result(),flush=True)


if __name__ == '__main__':
    main()
