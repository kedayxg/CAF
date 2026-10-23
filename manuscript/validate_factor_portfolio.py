"""Audit all synthetic outcomes and validate both selected examples in CAF/Gurobi."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'manuscript'))
from search_factor_portfolio import generate,digest
from caf_core import run_caf_pipeline
from dcmppln.utils.objective_after_clustering import get_model
from experiments.classical.run_fixed_community_benchmark import source_digest
from experiments._common import to_jsonable
from run_coordination_study import sequential_bcd

LOCAL=np.zeros((6,4),dtype=int)
for j,pair in enumerate(itertools.combinations(range(4),2)):LOCAL[j,list(pair)]=1
GROUPS=[list(range(i,i+4)) for i in (0,4,8)]
ALL=np.array([np.concatenate([LOCAL[a],LOCAL[b],LOCAL[c]])
              for a,b,c in itertools.product(range(6),repeat=3)])


def matrix_dynamics(sigma,mu,initial,frozen):
    """Independent 12-variable replay; choose candidates by full global costs."""
    objective=lambda x:float(.5*x@sigma@x-mu@x)
    x=initial.copy();history=[objective(x)]
    for _ in range(217):
        snapshot=x.copy();proposals=[]
        if frozen:
            for g in GROUPS:
                choices=np.tile(snapshot,(6,1));choices[:,g]=LOCAL
                costs=.5*np.einsum('bi,ij,bj->b',choices,sigma,choices)-choices@mu
                proposals.append(LOCAL[int(np.argmin(costs))])
        changes=0
        for ci,g in enumerate(GROUPS):
            if frozen:local=proposals[ci]
            else:
                choices=np.tile(x,(6,1));choices[:,g]=LOCAL
                costs=.5*np.einsum('bi,ij,bj->b',choices,sigma,choices)-choices@mu
                local=LOCAL[int(np.argmin(costs))]
            candidate=x.copy();candidate[g]=local
            if objective(candidate)-objective(x)<=-1e-5:x=candidate;changes+=1
        history.append(objective(x))
        if not changes:break
    return x,history


def audit_example(example,folder):
    import gurobipy as gp
    sigma=np.array(example['covariance']);mu=np.array(example['returns'])
    assert np.linalg.eigvalsh(sigma).min()>0
    cov=pd.DataFrame(sigma);ret=pd.Series(mu)
    results={};calls=[];phase=[''];counters={}
    objective=lambda x:float(.5*x@sigma@x-mu@x)
    for backend in ('enumeration','gurobi'):
        class LocalSolver:
            def __call__(self,model,weights,**kwargs):
                ci=next(i for i,g in enumerate(GROUPS)
                    if np.array_equal(sigma[np.ix_(g,g)],model._caf_sub_covariance))
                block=np.asarray(model._caf_sub_covariance)
                returns=np.asarray(model._caf_adjusted_returns)
                scores=.5*np.einsum('bi,ij,bj->b',LOCAL,block,LOCAL)-LOCAL@returns
                t=perf_counter();key=(backend,phase[0],ci)
                iteration=counters.get(key,0);counters[key]=iteration+1
                row=dict(backend=backend,phase=phase[0],round=iteration,community_pos=ci,
                         model_returns=returns.tolist())
                if backend=='enumeration':
                    x=LOCAL[int(np.argmin(scores))].copy()
                    row.update(status='exact enumeration',objective=float(scores.min()))
                else:
                    log=folder/f'{len(calls):03d}-{phase[0]}-c{ci}.log'
                    with tempfile.NamedTemporaryFile(mode='w',suffix='.lp',delete=False) as f:
                        f.write(model.lp_string.replace('*',' * '));lp=Path(f.name)
                    try:
                        with gp.Env(empty=True) as env:
                            env.setParam('OutputFlag',0);env.start()
                            with gp.read(str(lp),env=env) as m:
                                for k,v in dict(Threads=1,Seed=0,MIPGap=0.,MIPGapAbs=1e-12,TimeLimit=300).items():m.setParam(k,v)
                                m.setParam('LogToConsole',0);m.setParam('OutputFlag',1)
                                m.setParam('LogFile',str(log.resolve()));m.optimize()
                                assert m.Status==2 and m.SolCount>0
                                raw=np.array([m.getVarByName(w.name).X for w in weights])
                                x=np.rint(raw).astype(int);assert np.allclose(raw,x,atol=1e-7,rtol=0)
                                row.update(status=int(m.Status),objective=float(m.ObjVal),
                                    bound=float(m.ObjBound),gap=float(m.MIPGap),log=log.name)
                                np.testing.assert_allclose(m.ObjVal,scores.min(),atol=1e-10,rtol=0)
                    finally:lp.unlink(missing_ok=True)
                assert x.sum()==2 and np.isin(x,(0,1)).all()
                np.testing.assert_allclose(objective_local:=float(.5*x@block@x-returns@x),scores.min(),atol=1e-10,rtol=0)
                row['x']=x.tolist();row['model_objective_recomputed']=objective_local
                calls.append(row)
                return x,perf_counter()-t,0.
        factory=lambda num_reads=None,seed=None:LocalSolver()
        phase[0]='static'
        static=run_caf_pipeline(cov,ret,GROUPS,[2]*3,get_model,factory,risk_factor=.5,
            effective_risk_factor=.5,iterations=0,max_workers=1,return_diagnostics=True)
        initial=np.array(static['metrics']['weights'],dtype=int)
        expected_initial=np.concatenate([LOCAL[j] for j in example['initial']])
        np.testing.assert_array_equal(initial,expected_initial)
        phase[0]='caf'
        caf=run_caf_pipeline(cov,ret,GROUPS,[2]*3,get_model,factory,risk_factor=.5,
            effective_risk_factor=.5,initial_selected_indices=np.flatnonzero(initial).tolist(),
            iterations=6,patience=7,max_workers=1,return_diagnostics=True)
        phase[0]='bcd'
        bcd=sequential_bcd(cov,ret,GROUPS,[2]*3,initial,factory)
        for method,data in [('caf',caf),('bcd',bcd)]:
            expected=example[method]['history']
            padded=expected+[expected[-1]]*(7-len(expected))
            np.testing.assert_allclose(data['history']['objective'],padded,atol=1e-10,rtol=0)
            endpoint=np.array(data['metrics']['weights'],dtype=int)
            assert endpoint.sum()==6 and all(endpoint[g].sum()==2 for g in GROUPS)
            np.testing.assert_allclose(objective(endpoint),example[method+'_objective'],atol=1e-10,rtol=0)
            # Enumerate every one-block move at the actual endpoint.
            deltas=[]
            for g in GROUPS:
                candidates=np.tile(endpoint,(6,1));candidates[:,g]=LOCAL
                costs=.5*np.einsum('bi,ij,bj->b',candidates,sigma,candidates)-candidates@mu
                deltas.append(float(costs.min()-objective(endpoint)))
            assert min(deltas)>-1e-5-1e-12
            data['block_minimum_deltas']=deltas
            data['global_optimality_gap']=objective(endpoint)-example['optimum']
        results[backend]=dict(static=static['metrics'],caf=caf,bcd=bcd)
    for row in calls:
        if 'log' in row:row['log_sha256']=digest(folder/row['log'])
    # Reconstruct all Gurobi/enumeration proposals from the logged local states.
    trace={}
    for backend in ('enumeration','gurobi'):
        initial=np.array(results[backend]['static']['weights'],dtype=int)
        for method in ('caf','bcd'):
            state=initial.copy();events=[]
            local=[c for c in calls if c['backend']==backend and c['phase']==method]
            assert len(local)==18
            for round_idx in range(6):
                frozen=state.copy()
                for ci,g in enumerate(GROUPS):
                    c=next(c for c in local if c['round']==round_idx and c['community_pos']==ci)
                    reference=frozen if method=='caf' else state
                    external=np.array(sorted(set(range(12))-set(g)))
                    adjusted=mu[g]-sigma[np.ix_(g,external)]@reference[external]
                    np.testing.assert_allclose(c['model_returns'],adjusted,atol=1e-12,rtol=0)
                    candidate=state.copy();candidate[g]=c['x']
                    delta=objective(candidate)-objective(state);accepted=delta<=-1e-5
                    diag=results[backend][method]['proposal_diagnostics'][round_idx*3+ci]
                    assert accepted==diag['accepted_globally']
                    np.testing.assert_allclose(delta,diag['global_delta_if_committed'],atol=1e-12,rtol=0)
                    before=state.copy()
                    if accepted:state=candidate
                    events.append(dict(round=round_idx,community_pos=ci,before=before.tolist(),
                        candidate=candidate.tolist(),accepted=accepted,delta=delta,objective=objective(state)))
                np.testing.assert_allclose(objective(state),results[backend][method]['history']['objective'][round_idx+1],atol=1e-12,rtol=0)
            trace[backend+'_'+method]=events
    return dict(seed=example['seed'],example=example,results=results,calls=calls,trace=trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    d=json.loads(args.results.read_text());protocol=d['protocol']
    assert protocol['runner_sha256']==digest(ROOT/'manuscript/search_factor_portfolio.py')
    assert protocol['core_source_sha256']==source_digest()
    checks=Counter();counts=Counter();first={}
    for record in d['records']:
        sigma,mu,_=generate(record['seed'])
        assert np.linalg.eigvalsh(sigma)[0]>0
        costs=.5*np.einsum('bi,ij,bj->b',ALL,sigma,ALL)-ALL@mu
        np.testing.assert_allclose(costs.min(),record['optimum'],atol=1e-12,rtol=0)
        initial=np.zeros(12,dtype=int)
        for g in GROUPS:
            local=.5*np.einsum('bi,ij,bj->b',LOCAL,sigma[np.ix_(g,g)],LOCAL)-LOCAL@mu[g]
            initial[g]=LOCAL[int(np.argmin(local))]
        outcomes={}
        for method,frozen in [('caf',True),('bcd',False)]:
            endpoint,h=matrix_dynamics(sigma,mu,initial,frozen)
            np.testing.assert_allclose(h[-1],record[method+'_objective'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(h[min(6,len(h)-1)],record[method+'_six_sweep_objective'],atol=1e-12,rtol=0)
            counts[method+'_global']+=int(abs(h[-1]-costs.min())<=1e-9)
            outcomes[method]=(h[-1],len(h)-1)
            checks['full_matrix_dynamics_replays']+=1
        a,b=outcomes['caf'][0],outcomes['bcd'][0]
        winner='caf_better' if a<b-1e-9 else 'bcd_better' if b<a-1e-9 else 'tied'
        assert record['winner']==winner;counts[winner]+=1
        for label,good,bad in [('caf_example',a,b),('bcd_example',b,a)]:
            eligible=(abs(good-costs.min())<=1e-9 and bad-costs.min()>=.001
                and bad-costs.min()>=.2*(record['initial_objective']-costs.min())
                and max(t[1] for t in outcomes.values())<=6)
            if eligible and label not in first:first[label]=record['seed']
        checks['enumerated_global_states']+=len(ALL);checks['instances']+=1
    assert dict(counts)==d['counts'] and len(d['records'])==5000
    assert {k:v['seed'] for k,v in d['selected'].items()}==first
    examples={}
    for label,example in d['selected'].items():
        folder=out/label;folder.mkdir(exist_ok=True)
        audited=audit_example(example,folder);examples[label]=audited
        (folder/'audit.json').write_text(json.dumps(to_jsonable(audited),indent=2)+'\n')
        checks['native_gurobi_local_solves']+=sum(c['backend']=='gurobi' for c in audited['calls'])
        checks['native_exact_enum_local_solves']+=sum(c['backend']=='enumeration' for c in audited['calls'])
    validation=dict(status='passed',core_source_sha256=source_digest(),
        search_sha256=digest(ROOT/'manuscript/search_factor_portfolio.py'),
        validator_sha256=digest(Path(__file__)),results_sha256=digest(args.results),
        counts=dict(counts),selected_first_seed=first,checks=dict(checks),
        interpretation='All 5000 outcomes independently replayed. Selected favorable and reverse examples validated with actual CAF/BCD code, exhaustive local solve and native Gurobi. Illustrative postselection, not general superiority.')
    (ROOT/'docs/factor-portfolio-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (ROOT/'docs/factor-portfolio-search-results.json').write_text(args.results.read_text())
    (ROOT/'docs/factor-portfolio-examples.json').write_text(json.dumps(to_jsonable(examples),indent=2)+'\n')
    with zipfile.ZipFile(ROOT/'docs/factor-portfolio-raw-results.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(args.results,'results.json')
        z.write(args.results.parent/'protocol.json','protocol.json')
        for path in out.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(out))
        for name in ['search_factor_portfolio.py','validate_factor_portfolio.py']:
            z.write(ROOT/'manuscript'/name,'scripts/'+name)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
