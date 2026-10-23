"""Sampling-matched no-context SA control; separate from the frozen v2 batch.

The core experiment source remains unchanged. This companion records its own
file hash and the core hash; it is an additional mechanism check, not more
independent trials of the primary static-versus-CAF comparison.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from caf_core import run_caf_pipeline
from dcmppln.optimizer import SimulatedAnnealingDwave
from dcmppln.utils.objective_after_clustering import get_model
from experiments._common import load_local_jpm_data, BUNDLED_DATA_DIR, to_jsonable
from experiments.classical.run_fixed_community_benchmark import source_digest
from experiments.classical.run_benchmark import bootstrap_ci_mean, sign_test_two_sided


def worker(results, output, year, index):
    path=results/f'group-panel{year}_index-{index}.json'
    data=json.loads(path.read_text());r=data['record']
    cov,_,ret=load_local_jpm_data(BUNDLED_DATA_DIR,f'1_{year}-01-01')
    groups=[np.asarray(g) for g in r['communities']]
    cards=r['subproblem_cardinalities']
    lookup={}
    for g in groups:
        block=cov.values[np.ix_(g,g)]
        key=(block.shape,block.tobytes())
        if key in lookup:raise ValueError('Ambiguous covariance block in no-context model mapping')
        lookup[key]=g
    def no_context_model(_injected_returns, block, budget, risk_factor):
        g=lookup[(block.shape,np.asarray(block).tobytes())]
        return get_model(ret.values[g],block,budget=budget,risk_factor=risk_factor)
    def factory(num_reads,seed=None):
        return SimulatedAnnealingDwave(num_reads=num_reads,seed=seed)
    rounds=len(r['history']['objective'])-1
    initial=np.flatnonzero(np.asarray(r['static']['weights'])==1).tolist()
    start=perf_counter()
    control=run_caf_pipeline(cov,ret,groups,cards,get_model_func=no_context_model,
        optimizer_factory=factory,risk_factor=.5,effective_risk_factor=.5,
        initial_selected_indices=initial,iterations=rounds,patience=rounds+1,
        max_workers=1,seed=r['seed'],return_diagnostics=True)
    # The SA adapter consumes model.lp_string, not CAF's metadata attributes.
    # Thus these candidates really optimize the unshifted financial local model.
    caf_reads=sum(d['dynamic_reads'] for d in r['proposal_diagnostics'] if d['backend_called'])
    control_reads=sum(d['dynamic_reads'] for d in control['proposal_diagnostics'] if d['backend_called'])
    assert caf_reads==control_reads
    assert len(r['proposal_diagnostics'])==len(control['proposal_diagnostics'])
    for m in (r['static'],r['caf'],control['metrics']):
        w=np.asarray(m['weights'])
        assert np.isin(w,(0,1)).all() and w.sum()==len(ret)//2
        assert all(w[g].sum()==k for g,k in zip(groups,cards))
        np.testing.assert_allclose(.5*w@cov.values@w-ret.values@w,m['objective'],rtol=0,atol=1e-10)
    assert np.all(np.diff(control['history']['objective'])<=1e-10)
    record=dict(year=year,index=index,seed=r['seed'],core_source_sha256=data['source_sha256'],
        control_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        current_core_source_sha256=source_digest(),rounds=rounds,reads_per_method=caf_reads,
        initial=r['static'],caf=r['caf'],control=control['metrics'],history=control['history'],
        communities=r['communities'],cards=cards,elapsed_sec=perf_counter()-start,
        control_proposal_count=len(control['proposal_diagnostics']),
        control_accepted=sum(d['accepted_globally'] for d in control['proposal_diagnostics']),
        note='Same initial weights, community order, call seeds, SA sweeps/reads, rounds and global '
             'acceptance. Control models use original returns. CAF local surrogate-improvement '
             'diagnostics are not interpreted for this overridden model.')
    assert record['current_core_source_sha256']==record['core_source_sha256']
    target=output/f'panel-{year}_index-{index}.json'
    target.with_suffix('.tmp').write_text(json.dumps(to_jsonable(record),indent=2)+'\n')
    target.with_suffix('.tmp').replace(target)


def collect(output):
    records=[json.loads((output/f'panel-{year}_index-{i}.json').read_text())
             for year in (2016,2018) for i in range(20)]
    summary={}
    for year in (2016,2018):
        rows=[r for r in records if r['year']==year]
        static=np.array([r['initial']['objective'] for r in rows])
        caf=np.array([r['caf']['objective'] for r in rows])
        control=np.array([r['control']['objective'] for r in rows])
        advantage=(control-caf)/np.abs(static)*100
        wins=int(np.sum(caf<control-1e-9));losses=int(np.sum(caf>control+1e-9))
        summary[str(year)]=dict(runs=20,static_mean=float(static.mean()),
            caf_mean=float(caf.mean()),no_context_mean=float(control.mean()),
            caf_gain_mean_pct=float(np.mean((static-caf)/abs(static)*100)),
            control_gain_mean_pct=float(np.mean((static-control)/abs(static)*100)),
            caf_advantage_mean_percentage_points=float(advantage.mean()),
            caf_advantage_ci95_percentage_points=bootstrap_ci_mean(advantage),
            caf_wins=wins,caf_losses=losses,ties=20-wins-losses,
            paired_sign_pvalue=sign_test_two_sided(wins,wins+losses),
            reads_per_method_mean=float(np.mean([r['reads_per_method'] for r in rows])))
    result=dict(status='complete',protocol='sampling-matched iterative no-context SA control',
        control_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_source_sha256=source_digest(),summary=summary,records=records,
        interpretation='Matches annealing calls, sweeps, reads and random seeds, not elapsed '
            'time or every arithmetic operation. It tests the complete injected model path, '
            'including automatic penalty scaling; it does not establish generic quantum advantage.')
    (ROOT/'docs/sampling-control-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--worker',nargs=2,type=int,help=argparse.SUPPRESS)
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(args.results,args.out_dir,*args.worker);return
    expected=source_digest()
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def run(year,index):
        target=args.out_dir/f'panel-{year}_index-{index}.json'
        if target.exists():
            r=json.loads(target.read_text())
            assert r['core_source_sha256']==expected and r['control_script_sha256']==script_hash
            return year,index,'cached'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
        with (args.out_dir/f'panel-{year}_index-{index}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--results',str(args.results.resolve()),
                '--out-dir',str(args.out_dir.resolve()),'--worker',str(year),str(index)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return year,index,'completed'
    pending={(year,i) for year in (2016,2018) for i in range(20)}
    # Schedule available immutable primary records; unfinished primary tasks are
    # picked up on a later invocation, without sleeping or changing their order.
    ready=sorted((y,i) for y,i in pending if (args.results/f'group-panel{y}_index-{i}.json').exists())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run,y,i) for y,i in ready]):print(f.result(),flush=True)
    complete=sum((args.out_dir/f'panel-{y}_index-{i}.json').exists() for y,i in pending)
    print(f'{complete}/40 control records',flush=True)
    if complete==40:collect(args.out_dir)


if __name__=='__main__':main()
