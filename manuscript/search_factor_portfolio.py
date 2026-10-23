"""Exploratory, fully reported search for different CAF/BCD attraction basins.

This is a synthetic fixed-sector portfolio task, not a replacement for the
historical-panel benchmark. All 5,000 seeds and both directions are retained.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.classical.run_fixed_community_benchmark import source_digest

SEEDS=5000
STATES=np.array(list(itertools.combinations(range(4),2)),dtype=int)
LOCAL=np.array([[int(i in pair) for i in range(4)] for pair in STATES])
CONFIGS=np.array(list(itertools.product(range(6),repeat=3)),dtype=int)
PORTFOLIOS=np.concatenate([LOCAL[CONFIGS[:,i]] for i in range(3)],axis=1)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(seed):
    rng=np.random.default_rng(seed)
    loadings=rng.normal(size=(12,3))
    covariance=.04*(loadings@loadings.T/3+.2*np.eye(12))
    returns=rng.uniform(.01,.05,size=12)
    return covariance,returns,loadings


def exact_dynamics(energy,initial,method):
    """Exact conditional minimizers, with frozen/updated references explicit."""
    state=np.array(initial).copy()
    trajectory=[state.tolist()]
    history=[float(energy[tuple(state)])]
    accepted=0
    # At most 215 accepted changes on 216 states, then one no-change sweep.
    for iteration in range(217):
        frozen=state.copy();proposals=[]
        if method=='caf':
            for block in range(3):
                candidates=np.tile(frozen,(6,1));candidates[:,block]=np.arange(6)
                proposals.append(int(np.argmin(energy[tuple(candidates.T)])))
        changes=0
        for block in range(3):
            if method=='bcd':
                candidates=np.tile(state,(6,1));candidates[:,block]=np.arange(6)
                proposal=int(np.argmin(energy[tuple(candidates.T)]))
            else:proposal=proposals[block]
            candidate=state.copy();candidate[block]=proposal
            if energy[tuple(candidate)]-energy[tuple(state)]<=-1e-5:
                state=candidate;changes+=1;accepted+=1
        trajectory.append(state.tolist());history.append(float(energy[tuple(state)]))
        if changes==0:break
    else:raise AssertionError('Finite-state bound exceeded')
    # Independent exact check of a complete no-improvement sweep at the endpoint.
    for block in range(3):
        candidates=np.tile(state,(6,1));candidates[:,block]=np.arange(6)
        assert energy[tuple(state)]-min(energy[tuple(candidates.T)])<1e-5+1e-12
    return dict(state=state.tolist(),objective=float(energy[tuple(state)]),history=history,
        trajectory=trajectory,accepted=accepted,sweeps=len(history)-1,
        six_sweep_objective=history[min(6,len(history)-1)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(study='Exploratory synthetic factor portfolio basin search',
        core_source_sha256=source_digest(),runner_sha256=digest(Path(__file__)),
        seeds=list(range(SEEDS)),assets=12,groups=[list(range(i,i+4)) for i in (0,4,8)],
        cards=[2,2,2],risk_factor=.5,
        covariance='.04*(L@L.T/3 + .2*I), L standard-normal 12x3; SPD by construction',
        returns='Uniform(.01,.05) after drawing L with default_rng(seed)',
        initial='Exact independent unshifted local optimum; common to both methods',
        order=[0,1,2],local_solver='Enumerate all six feasible local states; deterministic lexicographic ties',
        stopping='One full no-improvement sweep, finite-state cap 217; also compare six-sweep endpoints',
        selection='First seed where CAF reaches enumerated global optimum, BCD gap>=.001 and >=20% of initial-to-optimum gap, both settled by six sweeps; symmetric reverse example also retained.',
        interpretation='Purposefully selected mechanism examples. Full seed outcomes retained, not a general superiority claim. Sector blocks externally specified, not Newman partitions.')
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    records=[];selected={};counts=dict(caf_better=0,bcd_better=0,tied=0,caf_global=0,bcd_global=0)
    for seed in range(SEEDS):
        sigma,mu,loadings=generate(seed)
        assert np.linalg.eigvalsh(sigma)[0]>0
        energies=(.5*np.einsum('bi,ij,bj->b',PORTFOLIOS,sigma,PORTFOLIOS)-PORTFOLIOS@mu).reshape((6,6,6))
        independent=[.5*np.einsum('bi,ij,bj->b',LOCAL,sigma[i:i+4,i:i+4],LOCAL)-LOCAL@mu[i:i+4] for i in (0,4,8)]
        initial=[int(np.argmin(a)) for a in independent]
        caf=exact_dynamics(energies,initial,'caf');bcd=exact_dynamics(energies,initial,'bcd')
        optimum=float(energies.min());start=float(energies[tuple(initial)])
        delta=caf['objective']-bcd['objective']
        winner='caf_better' if delta < -1e-9 else 'bcd_better' if delta>1e-9 else 'tied'
        counts[winner]+=1
        counts['caf_global']+=int(abs(caf['objective']-optimum)<=1e-9)
        counts['bcd_global']+=int(abs(bcd['objective']-optimum)<=1e-9)
        record=dict(seed=seed,initial_objective=start,optimum=optimum,
            caf_objective=caf['objective'],bcd_objective=bcd['objective'],
            caf_six_sweep_objective=caf['six_sweep_objective'],bcd_six_sweep_objective=bcd['six_sweep_objective'],
            caf_sweeps=caf['sweeps'],bcd_sweeps=bcd['sweeps'],winner=winner)
        records.append(record)
        for label,good,bad in [('caf_example',caf,bcd),('bcd_example',bcd,caf)]:
            eligible=(abs(good['objective']-optimum)<=1e-9 and bad['objective']-optimum>=.001
                and bad['objective']-optimum>=.2*(start-optimum)
                and caf['sweeps']<=6 and bcd['sweeps']<=6)
            if eligible and label not in selected:
                selected[label]=dict(record,covariance=sigma.tolist(),returns=mu.tolist(),
                    factor_loadings=loadings.tolist(),initial=initial,caf=caf,bcd=bcd,
                    global_argmin=list(map(int,np.unravel_index(np.argmin(energies),energies.shape))),
                    all_feasible_objectives=energies.reshape(-1).tolist())
        if (seed+1)%1000==0:print(seed+1,counts,flush=True)
    result=dict(status='complete',protocol=protocol,counts=counts,records=records,selected=selected)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SELECTED',[(k,v['seed']) for k,v in selected.items()],flush=True)


if __name__=='__main__':main()
