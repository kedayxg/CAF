"""Audit frozen hardware records offline, without optimization or cloud access.

Reconstruct the ideal distribution from the saved theta and warm-start angles;
do not retrain QAOA. Recompute financial objectives from the complete matrix,
decode all raw-count states, and compute exact uniform best-of-shots controls.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import dimod
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=1e-11):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=atol)


def normalized_counts(raw, width):
    result = Counter()
    for key, count in raw.items():
        assert isinstance(key, str) and isinstance(count, int) and not isinstance(count, bool)
        assert count >= 0
        if key.lower().startswith('0x'):
            index = int(key, 16)
        else:
            assert key and set(key) <= {'0', '1'}
            index = int(key, 2)
        assert 0 <= index < 2**width
        if count:
            result[format(index, f'0{width}b')] += count
    return dict(result)


def ideal_distribution(record):
    """Independent circuit construction using only recorded numerical inputs."""
    labels = record['var_labels']
    pos = {label: i for i, label in enumerate(labels)}
    n, p = record['n'], record['p']
    theta = np.asarray(record['theta_star'])
    assert theta.shape == (2*p,) and len(labels) == n == len(set(labels))
    circuit = QuantumCircuit(n)
    angles = record['warmstart_angles']
    for i in range(n):
        circuit.ry(float(angles[i]), i) if angles is not None else circuit.h(i)
    for layer in range(p):
        gamma, beta = theta[layer], theta[p+layer]
        for label, h in record['ising_linear'].items():
            circuit.rz(float(-2*gamma*h), pos[label])
        for a, b, j in record['ising_quadratic']:
            circuit.rzz(float(2*gamma*j), pos[a], pos[b])
        for i in range(n):
            if angles is None:
                circuit.rx(float(2*beta), i)
            else:
                circuit.ry(float(-angles[i]), i)
                circuit.rz(float(2*beta), i)
                circuit.ry(float(angles[i]), i)
    probabilities = Statevector.from_instruction(circuit).probabilities()
    close(probabilities.sum(), 1)
    return probabilities


def lower_tail_log10(k, n, p):
    if p >= 1:
        return None if k < n else 0.
    if p <= 0:
        return 0.
    a = np.arange(k+1, dtype=float)
    terms = (gammaln(n+1)-gammaln(a+1)-gammaln(n-a+1)
             + a*np.log(p)+(n-a)*np.log1p(-p))
    return float(logsumexp(terms)/np.log(10))


def hit_probability(single_draw_probability, shots):
    if single_draw_probability == 0:
        return 0.
    if single_draw_probability == 1:
        return 1.
    return float(-np.expm1(shots*np.log1p(-single_draw_probability)))


def audit_record(record, expected, sdk_local=False):
    from dcmppln.utils.objective_after_clustering import get_model
    n, k, shots, ci = [record[key] for key in ['n', 'K', 'shots', 'community']]
    cov, ret, groups, budgets = expected
    c = np.asarray(groups[ci]);base = np.asarray(record['static_state'], dtype=int)
    assert record['communities'] == groups and record['subproblem_cardinalities'] == budgets
    assert record['community_indices'] == c.tolist() and n == len(c) and k == budgets[ci]
    assert set(base) <= {0, 1} and len(base) == len(ret)
    assert all(base[group].sum() == budget for group, budget in zip(groups, budgets))
    close(record['covariance'], cov);close(record['expected_returns'], ret)
    q = record['risk_factor'];assert q == record['q_eff'] == .5
    assert record['partition_max_community_size'] == 16 and not record['local_risk_rebalancing']
    assert record['shots'] == 4000 and record['tau'] == .7
    assert record['base_seed'] == 42 and record['subset_seed'] == 1234 and record['num_reads'] == 200
    assert record['p'] == 2 and record['qaoa_restarts'] == 6 and record['qaoa_maxiter'] == 80
    assert record['warmstart'] and record['status'] == 'completed'
    sub = cov[np.ix_(c,c)]
    adjusted = ret[c]-2*q*((cov@base)[c]-sub@base[c])
    close(record['local_covariance'], sub);close(record['adjusted_returns'], adjusted)
    objective = lambda x: float(q*(x@cov@x)-ret@x)
    base_obj = objective(base);close(base_obj, record['static_objective'])
    # Rebuild constrained model and penalty conversion, independently of the
    # production hardware helper, to check coefficients and variable ordering.
    model, weights = get_model(adjusted, sub, budget=k, risk_factor=q)
    cqm = dimod.lp.loads(model.export_as_lp_string())
    bqm, _ = dimod.cqm_to_bqm(cqm)
    h, interactions, offset = bqm.to_ising()
    labels = record['var_labels'];weight_labels = [w.name for w in weights]
    assert labels == list(bqm.variables) and set(labels) == set(weight_labels)
    assert set(record['ising_linear']) == set(h)
    for label in h:close(record['ising_linear'][label], h[label])
    saved_j = {frozenset((a,b)):j for a,b,j in record['ising_quadratic']}
    assert set(saved_j) == {frozenset(pair) for pair in interactions}
    for pair,j in interactions.items():close(saved_j[frozenset(pair)], j)
    close(record['offset'], offset)
    states = np.arange(2**n, dtype=np.uint64)
    bits = ((states[:,None] >> np.arange(n, dtype=np.uint64)) & 1).astype(int)
    weight_positions = [labels.index(label) for label in weight_labels]
    bits_c_order = bits[:,weight_positions]
    full = np.tile(base, (len(bits),1));full[:,c] = bits_c_order
    deltas = q*np.einsum('bi,ij,bj->b',full,cov,full)-full@ret-base_obj
    financial = q*np.einsum('bi,ij,bj->b',bits_c_order,sub,bits_c_order)-bits_c_order@adjusted
    feasible = bits_c_order.sum(axis=1) == k
    spin = 2*bits-1
    energies = np.full(len(bits), offset)
    for label,value in h.items():energies += value*spin[:,labels.index(label)]
    for (a,b),value in interactions.items():energies += value*spin[:,labels.index(a)]*spin[:,labels.index(b)]
    close(energies[feasible], financial[feasible])
    def candidate_index(candidate):
        candidate = np.asarray(candidate,dtype=int)
        assert candidate.shape == (n,) and set(candidate) <= {0,1} and candidate.sum() == k
        values = dict(zip(weight_labels, candidate))
        return sum(int(values[label]) << i for i,label in enumerate(labels))
    sim_idx = candidate_index(record['sim_candidate']);hw_idx = candidate_index(record['hw_candidate'])
    close(deltas[sim_idx], record['dO_sim']);close(deltas[hw_idx], record['dO_hw'])
    probabilities = ideal_distribution(record)
    close(probabilities@energies, record['E_sim'])
    counts = normalized_counts(record['counts_full'], n)
    assert counts == record['counts_full']
    assert sum(counts.values()) == shots == record['counts_total'] == record['distribution_total']
    assert len(counts) == record['counts_distinct']
    empirical_counts = np.zeros(2**n, dtype=int)
    for key,value in counts.items():empirical_counts[int(key,2)] = value
    assert empirical_counts[hw_idx] > 0
    sampled_feasible = feasible & (empirical_counts > 0)
    close(energies[hw_idx], energies[sampled_feasible].min())
    close(deltas[hw_idx], deltas[sampled_feasible].min())
    empirical = empirical_counts/shots
    debug = record['qpu_debug']
    if record['mode'] == 'submit':
        assert record['machine'] == 'WK_C180' and debug['backend_mode'] == 'qpu'
        assert debug['final_status'] == 'FINISHED' and debug['distribution_kind'] == 'raw_counts'
        assert not debug['readout_amend'] and not record['fallback_used']
        receipt = record['job_receipt']
        assert receipt['job_id'] == debug['job_id'] and receipt['final_status'] == 'FINISHED'
        assert normalized_counts(receipt['raw_counts_received'], n) == counts
        origin = json.loads(debug['origin_data'])
        assert origin['success'] is True and origin['code'] == 10000
        obj = origin['obj'];assert obj['taskId'] == debug['job_id'] and obj['chipId'] == 'WK_C180'
        assert str(obj['taskState']) == '3'
        if 'errCode' in obj:
            assert int(obj['errCode']) == 0
        entries = obj['probCount'];assert len(entries) == 1
        channel = json.loads(entries[0]) if isinstance(entries[0],str) else entries[0]
        assert len(channel['key']) == len(channel['value'])
        assert normalized_counts(dict(zip(channel['key'],channel['value'])), n) == counts
        close(empirical@energies, record['E_hw'])
        compiled_circuit_available = any(item is not None for item in obj.get('QProg',[]))
    else:
        assert record['mode'] == 'local' and debug['backend_mode'] == 'local'
        close(record['E_hw'], record['E_sim'])
        compiled_circuit_available = False
    accepted_sim = deltas[sim_idx] <= -1e-5;accepted_hw = deltas[hw_idx] <= -1e-5
    ratio = float(deltas[hw_idx]/deltas[sim_idx]) if abs(deltas[sim_idx]) > 1e-14 else None
    passed = bool(accepted_sim and accepted_hw and ratio is not None and ratio >= record['tau'])
    assert bool(record['accepted_sim']) == bool(accepted_sim)
    assert bool(record['accepted_hw']) == bool(accepted_hw) and record['criterion_pass'] == passed
    close(record['hamming_hw_sim'], np.count_nonzero(bits[sim_idx] != bits[hw_idx]))
    if ratio is None:assert record['ratio_dO'] is None
    else:close(record['ratio_dO'], ratio)
    # Uniform controls are exact probabilities, not 2000/2000 interpreted as certainty.
    target_p = float(probabilities[sim_idx])
    uniform_hit = hit_probability(1/2**n, shots)
    eligible = feasible & (deltas <= -1e-5) & (deltas <= record['tau']*deltas[sim_idx])
    if not accepted_sim:eligible[:] = False
    uniform_pass = hit_probability(float(eligible.sum()/2**n), shots)
    b = np.arange(shots+1)
    uniform_tv_mean = float(2**n/(2*shots)*np.sum(np.abs(b-shots/2**n)*binom.pmf(b,shots,1/2**n)))
    sdk_gap = None
    if sdk_local:
        from experiments.quantum import run_hardware_qaoa as hardware
        hardware.setup_pyqpanda(local=True)
        prog = hardware.build_qaoa_prog(h,interactions,np.asarray(record['theta_star']),record['p'],
                                      np.asarray(record['warmstart_angles']),n,labels)
        sdk_energy = hardware.hw_energy(prog,h,interactions,offset,labels)
        sdk_gap = float(sdk_energy-probabilities@energies);close(sdk_gap,0)
    return dict(community=ci,n=n,K=k,mode=record['mode'],
        job_id=debug.get('job_id'),dO_sim=float(deltas[sim_idx]),dO_hw=float(deltas[hw_idx]),
        accepted_sim=bool(accepted_sim),accepted_hw=bool(accepted_hw),criterion_pass=passed,
        post_acceptance_delta=float(deltas[hw_idx]) if accepted_hw else 0.,
        shots=shots,distinct_bitstrings=len(counts),feasible_shots=int(empirical_counts[feasible].sum()),
        candidate_hamming=int(record['hamming_hw_sim']),ideal_candidate_probability=target_p,
        expected_ideal_candidate_hits=target_p*shots,observed_ideal_candidate_hits=int(empirical_counts[sim_idx]),
        ideal_candidate_lower_tail_log10_p=lower_tail_log10(int(empirical_counts[sim_idx]),shots,target_p),
        ideal_probability_of_zero_target_hits_log10=float(shots*np.log1p(-min(target_p,1-1e-15))/np.log(10)),
        tv_hardware_ideal=float(.5*np.abs(empirical-probabilities).sum()),
        tv_hardware_uniform=float(.5*np.abs(empirical-1/2**n).sum()),
        uniform_expected_empirical_tv=uniform_tv_mean,uniform_target_hit_probability=uniform_hit,
        uniform_criterion_pass_probability=uniform_pass,uniform_single_draw_qualifying_states=int(eligible.sum()),
        feasible_fraction_hw=float(empirical[feasible].sum()),feasible_fraction_uniform=float(feasible.mean()),
        exact_conditional_optimum_delta=float(deltas[feasible].min()),
        sim_is_conditional_optimum=bool(abs(deltas[sim_idx]-deltas[feasible].min())<1e-12),
        max_empirical_state_probability=float(empirical.max()),max_ideal_state_probability=float(probabilities.max()),
        sdk_saved_theta_energy_gap=sdk_gap,submitted_compiled_circuit_available=compiled_circuit_available)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=ROOT/'results/hardware')
    ap.add_argument('--out',type=Path,default=ROOT/'docs/hardware-validation.json')
    ap.add_argument('--sdk-local',action='store_true')
    args = ap.parse_args();directory=args.results
    manifest = directory/('PUBLIC_ARTIFACT_SHA256.txt' if (directory/'PUBLIC_ARTIFACT_SHA256.txt').exists()
                          else 'ARTIFACT_SHA256_20261023.txt')
    manifest_files = {}
    for line in manifest.read_text().splitlines():
        digest,name = line.split(None,1)
        path=directory/name;assert path.resolve().is_relative_to(directory.resolve())
        assert sha(path)==digest,name
        manifest_files[name]=digest
    from experiments._common import (load_local_jpm_data,subset_problem,extract_communities_from_partitions,
                                     split_cardinality_constraint,BUNDLED_DATA_DIR)
    from dcmppln.denoiser import Denoiser
    from dcmppln.clustering import Clustering
    from experiments.classical.run_fixed_community_benchmark import source_digest
    cov,corr,ret = subset_problem(*load_local_jpm_data(BUNDLED_DATA_DIR),40,1234)
    with contextlib.redirect_stdout(io.StringIO()):
        filtered,_,_ = Denoiser(active=True,q=.5,q_fit=True)(corr.values)
        partition,_,_ = Clustering(max_community_size=16,seed=42)(filtered)
    groups = [c.tolist() for c in extract_communities_from_partitions(partition)]
    budgets = split_cardinality_constraint(list(map(len,groups)),20).tolist()
    expected = cov.values,ret.values,groups,budgets
    rows=[];local_rows=[];sources={}
    for mode,output in [('qpu',rows),('local',local_rows)]:
        paths=sorted((directory/mode).glob('HW_*.json'));assert len(paths)==4
        for path in paths:
            record=json.loads(path.read_text())
            assert record['source_sha256']==source_digest()
            assert record['script_sha256']==sha(ROOT/'experiments/quantum/run_hardware_qaoa.py')
            for name,digest in record['data_sha256'].items():assert sha(BUNDLED_DATA_DIR/name)==digest
            result=audit_record(record,expected,sdk_local=args.sdk_local and mode=='qpu')
            result['file']=str(path.relative_to(directory));result['file_sha256']=sha(path)
            output.append(result);sources[str(path.relative_to(directory))]=sha(path)
        output.sort(key=lambda r:r['community']);assert [r['community'] for r in output]==list(range(4))
    for row,local in zip(rows,local_rows):
        raw=json.loads((directory/row['file']).read_text());loc=json.loads((directory/local['file']).read_text())
        for key in ['theta_star','warmstart_angles','static_state','communities','adjusted_returns','sim_candidate']:
            if key != 'communities':
                close(raw[key],loc[key])
        assert raw['communities']==loc['communities']
    proof=dict(status='passed',protocol='Four independent frozen iteration-0 community updates; no hardware outer loop',
        source_sha256=source_digest(),validator_sha256=sha(Path(__file__)),
        manifest_sha256=sha(manifest),manifest_verified_files=len(manifest_files),
        validated_source_files=sources,qpu_records=rows,local_records=local_rows,
        all_qpu_jobs_finished=True,qpu_shots=16000,counts_crosschecked_with_platform_origin=True,
        complete_matrix_objectives_replayed=True,ising_coefficients_and_all_feasible_state_energies_verified=True,
        conditional_feasible_states=sum(math.comb(row['n'],row['K']) for row in rows),
        ideal_distribution_method='Direct saved-theta and saved-angle circuit reconstruction; no reoptimization',
        sdk_saved_theta_replayed=args.sdk_local,uniform_control_method='Exact independent uniform best-of-shots event probabilities',
        criterion_passes=[r['community'] for r in rows if r['criterion_pass']],
        rejected_nonimproving=[r['community'] for r in rows if not r['accepted_hw']],
        scope='Best-feasible-sample proposal/acceptance executability; no demonstrated sampler advantage or ideal-distribution preservation',
        limitations=['One batch per community, no independent hardware repetitions.',
                    'Task receipts are archived files; no live platform account verification was performed.',
                    'Compiled cloud instructions/mapping are absent (origin QProg is null); noise versus cloud compilation/mapping is not identified.',
                    'Diagnostic uniform control is post hoc, not a preregistered competitive benchmark.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    for row in rows:
        print(f"community {row['community']}: delta={row['dO_hw']:+.12f}, "
              f"ideal p={row['ideal_candidate_probability']:.6f}, target hits={row['observed_ideal_candidate_hits']}, "
              f"uniform pass={row['uniform_criterion_pass_probability']:.9f}")
    print('Audit passed:',args.out)


if __name__=='__main__':
    main()
