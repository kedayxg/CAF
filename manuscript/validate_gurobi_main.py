"""Independently replay every Gurobi-main proposal against full matrices."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import BUNDLED_DATA_DIR, load_local_jpm_data, split_cardinality_constraint
from experiments.classical.run_fixed_community_benchmark import source_digest


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--sa-results',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.results/'manifest.json').read_text())
    summary={}
    totals=Counter()
    source_files={}
    for year in [2016,2018]:
        path=args.results/f'panel-{year}.json'
        d=json.loads(path.read_text())
        assert d['core_source_sha256']==manifest['core_source_sha256']==source_digest()
        assert d['runner_sha256']==manifest['runner_sha256']==sha(ROOT/'manuscript/run_gurobi_main.py')
        assert d['parameters']==manifest['parameters']
        primary=args.sa_results/f'group-panel{year}_index-0.json'
        assert sha(primary)==d['partition_reference_sha256']
        old=json.loads(primary.read_text())['record']
        assert d['communities']==old['communities'] and d['cards']==old['subproblem_cardinalities']
        cov,_,ret=load_local_jpm_data(BUNDLED_DATA_DIR,f'1_{year}-01-01')
        cov,ret=cov.values,ret.values
        for n,h in d['dataset_sha256'].items():assert sha(BUNDLED_DATA_DIR/n)==h
        groups,cards=d['communities'],d['cards']
        assert sorted(i for g in groups for i in g)==list(range(len(ret)))
        assert max(map(len,groups))<=100 and sum(cards)==len(ret)//2
        assert np.array_equal(cards,split_cardinality_constraint(list(map(len,groups)),len(ret)//2))

        def obj(x):return float(.5*x@cov@x-ret@x)
        def check_metrics(m,x):
            assert np.isin(x,(0,1)).all() and x.sum()==len(ret)//2
            assert all(x[g].sum()==k for g,k in zip(groups,cards))
            np.testing.assert_allclose(x,m['weights'],atol=0,rtol=0)
            np.testing.assert_allclose([float(x@cov@x),float(ret@x),obj(x)],
                [m['risk'],m['expected_return'],m['objective']],atol=1e-10,rtol=0)
            totals['verified_portfolios']+=1
        def check_call(c,ci,returns):
            assert c['community_pos']==ci
            assert sha(args.results/c['log'])==c['log_sha256']
            log=(args.results/c['log']).read_text()
            assert 'Optimal solution found' in log and 'Time limit reached' not in log
            assert c['status']==2 and c['solution_count']>=1
            assert 0<=c['gap']<=1.00001e-4
            assert c['bound']<=c['objective']+1e-8
            np.testing.assert_allclose(c['model_returns'],returns,atol=1e-11,rtol=0)
            x=np.array(c['x']);assert np.isin(x,(0,1)).all() and x.sum()==cards[ci]
            actual=.5*x@cov[np.ix_(groups[ci],groups[ci])]@x-returns@x
            np.testing.assert_allclose(actual,c['objective'],atol=1e-8,rtol=0)
            totals['verified_local_solves']+=1
            return x
        calls={phase:[c for c in d['local_solves'] if c['phase']==phase]
               for phase in ['static','caf','control']}
        assert len(calls['static'])==len(groups)
        initial=np.zeros(len(ret),dtype=int)
        for ci,g in enumerate(groups):initial[g]=check_call(calls['static'][ci],ci,ret[g])
        check_metrics(d['static'],initial)
        method_stats={}
        for phase in ['caf','control']:
            data=d[phase]
            state=initial.copy()
            count=len(groups); rounds=len(data['history']['objective'])-1
            assert len(calls[phase])==count*rounds
            accepted=0;deltas=[]
            history=[obj(state)]
            for iteration in range(rounds):
                frozen=state.copy()
                for ci,g in enumerate(groups):
                    c=calls[phase][iteration*count+ci]
                    external=np.array(sorted(set(range(len(ret)))-set(g)))
                    returns=ret[g] if phase=='control' else ret[g]-cov[np.ix_(g,external)]@frozen[external]
                    local=check_call(c,ci,returns)
                    candidate=state.copy();candidate[g]=local
                    delta=obj(candidate)-obj(state)
                    diag=data['proposal_diagnostics'][iteration*count+ci]
                    np.testing.assert_allclose(delta,diag['global_delta_if_committed'],atol=1e-10,rtol=0)
                    accept=delta<=-1e-5
                    assert diag['accepted_globally']==accept
                    if accept:state=candidate;accepted+=1;deltas.append(delta)
                    assert diag['raw_candidate_feasible'] and not diag['candidate_repaired']
                    totals['verified_proposals']+=1
                history.append(obj(state))
                np.testing.assert_allclose([state@cov@state,ret@state],
                    [data['history']['risk'][iteration+1],data['history']['expected_return'][iteration+1]],atol=1e-10,rtol=0)
            np.testing.assert_allclose(history,data['history']['objective'],atol=1e-10,rtol=0)
            assert np.all(np.diff(history)<=1e-10)
            check_metrics(data['metrics'],state)
            method_stats[phase]=dict(rounds=rounds,accepted_updates=accepted,
                objective=obj(state),time_sec=data['timing']['total_seconds'],
                same_weights_as_static=bool(np.array_equal(initial,state)))
            totals['verified_histories']+=1
            totals['accepted_updates']+=accepted
        static=d['static']['objective'];final=d['caf']['metrics']['objective']
        summary[str(year)]=dict(assets=len(ret),cardinality=len(ret)//2,communities=len(groups),
            community_sizes=list(map(len,groups)),static_objective=static,caf_objective=final,
            control_objective=d['control']['metrics']['objective'],gain_pct=(static-final)/abs(static)*100,
            static_risk=d['static']['risk'],caf_risk=d['caf']['metrics']['risk'],
            static_return=d['static']['expected_return'],caf_return=d['caf']['metrics']['expected_return'],
            static_time_sec=d['static_time_sec'],caf_time_sec=d['caf']['timing']['total_seconds'],
            static_local_solves=len(calls['static']),caf_local_solves=len(calls['caf']),
            control_local_solves=len(calls['control']),local_solves=len(d['local_solves']),
            max_local_gap=max(c['gap'] for c in d['local_solves']),methods=method_stats)
        source_files[path.name]=sha(path)
    validation=dict(status='passed',core_source_sha256=source_digest(),
        runner_sha256=sha(ROOT/'manuscript/run_gurobi_main.py'),validator_sha256=sha(Path(__file__)),
        validated_source_files=source_files,checks=dict(totals),
        scope='Independent full-matrix replay of all proposals, frozen-context model coefficients, global acceptance decisions, histories, feasibility, local statuses/bounds/gaps and per-call log hashes.')
    (ROOT/'docs/gurobi-main-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    result=dict(status='complete',protocol=manifest,summary=summary,
        source_files=source_files,validation_sha256=sha(ROOT/'docs/gurobi-main-validation.json'))
    (ROOT/'docs/gurobi-main-results.json').write_text(json.dumps(result,indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'docs/gurobi-main-raw-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(args.results.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(args.results))
        for n in ['run_gurobi_main.py','validate_gurobi_main.py']:
            z.write(ROOT/'manuscript'/n,'scripts/'+n)
    print(json.dumps(dict(summary=summary,checks=dict(totals)),indent=2))


if __name__=='__main__':main()
