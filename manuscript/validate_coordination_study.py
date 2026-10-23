"""Full-matrix replay of CAF/BCD candidates and every declared study case."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'manuscript'))
from run_coordination_study import (cases,case_id,selected_problem,sha,PARAMETERS,METHODS,ROUNDS)
from experiments.classical.run_fixed_community_benchmark import source_digest
from dcmppln.denoiser import Denoiser
from dcmppln.newman import capacity_newman_partition
from experiments._common import extract_communities_from_partitions,split_cardinality_constraint,BUNDLED_DATA_DIR


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();out=args.results
    manifest=json.loads((out/'manifest.json').read_text())
    declaration=json.loads((ROOT/'docs/coordination-study-protocol.json').read_text())
    assert manifest['cases']==cases()==declaration['cases']
    assert manifest['parameters']==PARAMETERS==declaration['parameters']
    assert manifest['runner_sha256']==declaration['runner_sha256']==sha(ROOT/'manuscript/run_coordination_study.py')
    assert manifest['core_source_sha256']==declaration['core_source_sha256']==source_digest()
    totals=Counter();summaries=[];failures=[];sources={};missing=[];filtered_cache={}
    for case in cases():
        name=case_id(case);path=out/(name+'.json')
        if not path.exists():missing.append(name);continue
        d=json.loads(path.read_text());sources[path.name]=sha(path)
        assert d['case']==case and d['runner_sha256']==manifest['runner_sha256']
        assert d['core_source_sha256']==source_digest()
        cov,corr,ret,indices=selected_problem(case)
        assert d['input_indices']==indices.tolist()
        for n,h in d['dataset_sha256'].items():assert sha(BUNDLED_DATA_DIR/n)==h
        key=(case['year'],case['subset_seed'])
        if key not in filtered_cache:
            np.random.seed(42)
            filtered_cache[key]=np.abs(Denoiser(active=True,q=.5,q_fit=True)(corr.values)[0])
        try:
            labels,modularity,trace=capacity_newman_partition(filtered_cache[key],case['capacity'],seed=42)
        except RuntimeError as e:
            assert d['status']=='partition_failure' and d['reason']==str(e)
            failures.append(dict(case=case,reason=str(e)));totals['partition_failures']+=1
            continue
        assert d['status']=='complete'
        groups=[g.tolist() for g in extract_communities_from_partitions(labels)]
        cards=split_cardinality_constraint(list(map(len,groups)),len(ret)//2).tolist()
        assert d['communities']==groups and d['cards']==cards and d['labels']==labels.tolist()
        np.testing.assert_allclose(d['modularity'],modularity,atol=1e-12,rtol=0)
        assert max(map(len,groups))<=case['capacity']
        assert sorted(i for g in groups for i in g)==list(range(len(ret)))
        assert d['parameters']==PARAMETERS and d['rounds']==ROUNDS
        sigma,mu=cov.values,ret.values
        def objective(x):return float(.5*x@sigma@x-mu@x)
        def check_metrics(m,x):
            assert np.isin(x,(0,1)).all() and x.sum()==len(ret)//2
            assert all(x[g].sum()==k for g,k in zip(groups,cards))
            np.testing.assert_array_equal(m['weights'],x)
            np.testing.assert_allclose([x@sigma@x,mu@x,objective(x)],
                [m['risk'],m['expected_return'],m['objective']],atol=1e-10,rtol=0)
            totals['verified_portfolios']+=1
        calls={}
        for call in d['local_solves']:
            k=(call['phase'],call['round'],call['community_pos'])
            assert k not in calls;calls[k]=call
        def check_call(phase,round_idx,ci,returns):
            c=calls[(phase,round_idx,ci)]
            assert sha(out/c['log'])==c['log_sha256']
            assert c['solution_count']>=1 and c['bound']<=c['objective']+1e-8
            if c['status']==2:
                assert 0<=c['gap']<=1.00001e-4
                assert 'Optimal solution found' in (out/c['log']).read_text()
                totals['optimal_local_solves']+=1
            elif c['status']==9:
                assert 'Time limit reached' in (out/c['log']).read_text()
                totals['time_limited_local_solves']+=1
            else:raise AssertionError(f"Unreviewed termination status {c['status']}")
            np.testing.assert_allclose(c['model_returns'],returns,atol=1e-10,rtol=0)
            x=np.array(c['x']);assert np.isin(x,(0,1)).all() and x.sum()==cards[ci]
            val=.5*x@sigma[np.ix_(groups[ci],groups[ci])]@x-returns@x
            np.testing.assert_allclose(c['objective'],val,atol=1e-8,rtol=0)
            totals['verified_local_solves']+=1
            return x
        initial=np.zeros(len(ret),dtype=int)
        for ci,(g,k) in enumerate(zip(groups,cards)):
            initial[g]=(np.full(len(g),int(k>0)) if k in (0,len(g))
                        else check_call('static',0,ci,mu[g]))
        check_metrics(d['static'],initial)
        active=sum(k not in (0,len(g)) for g,k in zip(groups,cards))
        assert len(calls)==active*(1+ROUNDS*len(METHODS))
        summary=dict(case=case,assets=len(ret),cardinality=len(ret)//2,
            community_count=len(groups),community_sizes=list(map(len,groups)),
            static_objective=objective(initial),static_seconds=d['static_seconds'],
            preprocessing_seconds=d['preprocessing_seconds'],methods={})
        for method in METHODS:
            m=d['methods'][method];state=initial.copy();history=[objective(state)]
            assert len(m['history']['objective'])==ROUNDS+1
            assert len(m['proposal_diagnostics'])==ROUNDS*len(groups)
            accepted=0;accepted_rounds=[]
            for round_idx in range(ROUNDS):
                frozen=state.copy()
                round_accepts=0
                for ci,(g,k) in enumerate(zip(groups,cards)):
                    reference=state if method=='sequential_bcd' else frozen
                    external=np.array(sorted(set(range(len(ret)))-set(g)))
                    returns=mu[g]-sigma[np.ix_(g,external)]@reference[external]
                    called=k not in (0,len(g))
                    local=check_call(method,round_idx,ci,returns) if called else np.full(len(g),int(k>0))
                    candidate=state.copy();candidate[g]=local
                    delta=objective(candidate)-objective(state)
                    diag=m['proposal_diagnostics'][round_idx*len(groups)+ci]
                    assert diag['iteration_idx']==round_idx and diag['backend_called']==called
                    np.testing.assert_allclose(delta,diag['global_delta_if_committed'],atol=1e-10,rtol=0)
                    accept=delta<=-1e-5;assert diag['accepted_globally']==accept
                    if accept:state=candidate;accepted+=1;round_accepts+=1
                    totals['verified_proposals']+=1
                history.append(objective(state));accepted_rounds.append(round_accepts)
                np.testing.assert_allclose([state@sigma@state,mu@state],
                    [m['history']['risk'][round_idx+1],m['history']['expected_return'][round_idx+1]],atol=1e-10,rtol=0)
            np.testing.assert_allclose(history,m['history']['objective'],atol=1e-10,rtol=0)
            assert np.all(np.diff(history)<=1e-10)
            check_metrics(m['metrics'],state);totals['verified_histories']+=1
            local=[c for c in d['local_solves'] if c['phase']==method]
            first_final=next(i for i,value in enumerate(history) if abs(value-history[-1])<=1e-9)
            summary['methods'][method]=dict(objective=objective(state),
                gain_pct=(history[0]-history[-1])/abs(history[0])*100,
                risk=float(state@sigma@state),expected_return=float(mu@state),
                calls=len(local),wall_seconds=m['timing']['total_seconds'],
                solver_wall_sum=sum(c['solve_wall_sec'] for c in local),
                service_wall_sum=sum(c['service_wall_sec'] for c in local),
                ideal_unlimited_service_makespan=sum(max([c['service_wall_sec'] for c in local if c['round']==i],default=0) for i in range(ROUNDS)),
                history=history,first_final_round=first_final,accepted_updates=accepted,
                accepted_by_round=accepted_rounds,
                time_limits=sum(c['status']==9 for c in local),
                max_local_gap=max([c['gap'] for c in local],default=0))
        # Concurrency is intended to affect scheduling only. Report, do not assume,
        # equality: time-limited solver trajectories could legitimately differ.
        summary['parallel_serial_objective_delta']=(summary['methods']['caf_parallel4']['objective']-
                                                   summary['methods']['caf_serial']['objective'])
        summary['bcd_minus_caf_objective']=(summary['methods']['sequential_bcd']['objective']-
                                           summary['methods']['caf_serial']['objective'])
        summaries.append(summary);totals['complete_cases']+=1
    if missing and not args.allow_partial:raise SystemExit(f'Missing {len(missing)} declared cases')
    validation=dict(status='passed' if not missing else 'partial',core_source_sha256=source_digest(),
        runner_sha256=manifest['runner_sha256'],validator_sha256=sha(Path(__file__)),
        manifest_sha256=sha(out/'manifest.json'),validated_source_files=sources,
        checks=dict(totals),missing_cases=missing,
        summary_sha256=hashlib.sha256(json.dumps(summaries,sort_keys=True).encode()).hexdigest(),
        scope='Independent full-matrix candidate and history replay; latest versus frozen proposal coefficients; native budgets; seeded input indices, full partitions/failures, per-solve log hashes, bounds/status and equal six-sweep call budgets.')
    suffix='-partial' if missing else ''
    vp=ROOT/f'docs/coordination-study-validation{suffix}.json'
    vp.write_text(json.dumps(validation,indent=2)+'\n')
    result=dict(status='complete' if not missing else 'partial',protocol=manifest,
        cases=summaries,partition_failures=failures,missing_cases=missing,
        source_files=sources,validation_sha256=sha(vp),checks=dict(totals))
    (ROOT/f'docs/coordination-study-results{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    if not missing:
        with zipfile.ZipFile(ROOT/'docs/coordination-study-raw-results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(out))
            for name in ['run_coordination_study.py','validate_coordination_study.py']:
                z.write(ROOT/'manuscript'/name,'scripts/'+name)
    print(json.dumps(dict(checks=dict(totals),missing=len(missing),
        results=[dict(case=case_id(c['case']),caf=c['methods']['caf_serial']['objective'],
                      bcd=c['methods']['sequential_bcd']['objective']) for c in summaries],
        failures=failures),indent=2))


if __name__=='__main__':main()
