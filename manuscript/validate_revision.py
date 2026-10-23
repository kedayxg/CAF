"""Independent full-matrix, partition and frozen-move audit for arXiv v2."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_revision import tasks, task_id
from experiments._common import (BUNDLED_DATA_DIR, load_local_jpm_data, subset_problem,
    extract_communities_from_partitions, split_cardinality_constraint)
from experiments.revision_protocol import CLASSICAL_CAPACITY, QUANTUM_CAPACITY, PROTOCOL_NAME
from dcmppln.clustering import Clustering
from dcmppln.denoiser import Denoiser
from dcmppln.optimizer import SimulatedAnnealingDwave
from dcmppln.utils.objective_after_clustering import get_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'revision_results_v2')
    parser.add_argument('--main-result', type=Path)
    parser.add_argument('--sampling-control', type=Path, default=ROOT/'docs/sampling-control-results.json')
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true',
                        help='Audit completed records without producing a final validation certificate')
    args = parser.parse_args()
    main_path = args.main_result or args.results/'main_2016.json'
    main_data = json.loads(main_path.read_text())
    expected_source = main_data['source_sha256']
    manifest = json.loads((args.results/'manifest-all.json').read_text())
    assert manifest['source_sha256'] == expected_source
    for name, expected in manifest['data_sha256'].items():
        assert digest(BUNDLED_DATA_DIR/name) == expected
    protocol = dict(name=PROTOCOL_NAME, classical_capacity=CLASSICAL_CAPACITY,
                    quantum_capacity=QUANTUM_CAPACITY)
    full2016 = load_local_jpm_data(BUNDLED_DATA_DIR)
    full2018 = load_local_jpm_data(BUNDLED_DATA_DIR, '1_2018-01-01')
    folded = subset_problem(*full2016, 40, 1234)
    fixed = {n:subset_problem(*full2016, n, 123+n) for n in (100, 200, 300, 484)}
    counts, source_counts = Counter(), Counter()
    validated_files, partition_cache, suppression_audits, missing = {}, {}, [], []

    def partition(problem, capacity, key):
        cache_key = (key, capacity)
        if cache_key not in partition_cache:
            filtered = Denoiser(active=True, q=.5, q_fit=True)(problem[1].values)[0]
            labels = Clustering(max_community_size=capacity, seed=42)(filtered)[0]
            groups = extract_communities_from_partitions(labels)
            assert sorted(i for g in groups for i in g) == list(range(len(problem[2])))
            assert max(map(len, groups), default=0) <= capacity
            partition_cache[cache_key] = groups
        return partition_cache[cache_key]

    def check(metrics, cov, ret, q, k, groups, cards):
        w = np.asarray(metrics['weights'])
        assert np.isin(w, (0, 1)).all() and w.sum() == k
        assert all(w[g].sum() == card for g, card in zip(groups, cards))
        risk, returns = float(w @ cov @ w), float(ret @ w)
        np.testing.assert_allclose([risk, returns, q*risk-returns],
            [metrics['risk'], metrics['expected_return'], metrics['objective']], rtol=0, atol=1e-10)
        counts['verified_portfolios'] += 1

    def audit_suppression(task, rows):
        n, index = task['n'], task['index']
        problem = subset_problem(*full2016, n, 123+100*n+index)
        cov, _, ret = problem
        groups = partition(problem, CLASSICAL_CAPACITY, ('suppression', n, index))
        cards = split_cardinality_constraint(list(map(len, groups)), n//2)
        # Independently replay the seeded static solves, then use a direct full
        # external-portfolio multiplication, rather than cached-vector subtraction.
        state = np.zeros(n, dtype=int)
        solver = SimulatedAnnealingDwave(num_reads=100, seed=123+100*n+index)
        for group, card in zip(groups, cards):
            model, weights = get_model(ret.values[group], cov.values[np.ix_(group, group)],
                                       budget=int(card), risk_factor=.5)
            state[group] = np.asarray(solver(model, weights)[0], dtype=int)
        assert state.sum() == n//2 and np.isin(state, (0, 1)).all()
        expected_indices = [i for i, k in enumerate(cards) if k > 0]
        assert [row['community_idx'] for row in rows] == expected_indices
        for row in rows:
            group = groups[row['community_idx']]
            external = state.copy()
            external[group] = 0
            adjusted = ret.values[group] - cov.values[group] @ external
            static_fraction = 100*float(np.mean(ret.values[group] <= 0))
            caf_fraction = 100*float(np.mean(adjusted <= 0))
            np.testing.assert_allclose([static_fraction, caf_fraction, caf_fraction-static_fraction],
                [row['suppressed_static_pct'], row['suppressed_caf_pct'], row['suppressed_delta_pct']],
                rtol=0, atol=1e-10)
            assert row['community_size'] == len(group)
            if row['active_static'] is None or row['active_caf'] is None:
                assert row['reduction_vs_static_pct'] is None
                counts['unavailable_presolve_comparisons'] += 1
            else:
                assert 0 <= row['active_static'] <= len(group)
                assert 0 <= row['active_caf'] <= len(group)
            counts['suppression_community_observations'] += 1
        suppression_audits.append(dict(task=task, static_weights=state.tolist(),
            communities=[g.tolist() for g in groups], cards=cards.tolist(),
            directly_verified_observations=len(rows)))
        counts['suppression_seeded_replays'] += 1

    for task in tasks():
        path = args.results/(task_id(task)+'.json')
        if not path.exists():
            if args.allow_partial:
                missing.append(task)
                continue
            raise SystemExit(f'Missing result: {path}')
        data = json.loads(path.read_text())
        assert data['task'] == task and data['protocol'] == protocol
        assert data['source_sha256'] == expected_source
        source_counts[data['source_sha256']] += 1
        validated_files[path.name] = digest(path)
        r, group = data['record'], task['group']
        if group == 'suppression':
            audit_suppression(task, r)
            continue
        if group in ('qaoa2', 'qaoa3', 'sqa'):
            problem, capacity, key = folded, QUANTUM_CAPACITY, 'folded'
            q, k = .5, 20
            if group == 'sqa':
                assert r['backend_protocol']['Hd_field'] == 'linspace(1,0,1000)'
        else:
            problem = full2018 if group == 'panel2018' else fixed[task['n']] if 'n' in task else full2016
            capacity = CLASSICAL_CAPACITY
            key = '2018' if group == 'panel2018' else ('fixed', task['n']) if 'n' in task else '2016'
            q, k = r['risk_factor'], task.get('k', len(problem[2])//2)
        groups = partition(problem, capacity, key)
        assert r['partition_max_community_size'] == capacity
        assert r['communities'] == [g.tolist() for g in groups]
        cards = split_cardinality_constraint(list(map(len, groups)), k)
        np.testing.assert_array_equal(cards, r['subproblem_cardinalities'])
        cov, _, ret = problem
        for side in ('static', 'caf'):
            check(r[side], cov.values, ret.values, q, k, groups, cards)
        h = np.asarray(r['history']['objective'])
        assert np.all(np.diff(h) <= 1e-10)
        np.testing.assert_allclose([h[0], h[-1]],
            [r['static']['objective'], r['caf']['objective']], rtol=0, atol=1e-10)
        assert r['caf']['objective'] <= r['static']['objective']+1e-10
        for i in range(len(h)-1):
            accepted_delta = sum(d['global_delta_if_committed'] for d in r['proposal_diagnostics']
                if d['iteration_idx'] == i and d['accepted_globally'])
            np.testing.assert_allclose(h[i+1]-h[i], accepted_delta, rtol=0, atol=1e-10)
            counts['verified_iteration_deltas'] += 1
        if group == 'q_sweep':
            expected = main_data['per_run_records'][task['index']]['jpm']['metrics']['weights']
            np.testing.assert_array_equal(r['static']['weights'], expected)
        counts['verified_histories'] += 1
        counts['paired_revision_runs'] += 1
        if group == 'panel2016':
            mirror = main_data['per_run_records'][task['index']]
            assert mirror['seed'] == r['seed']
            for side, export in [('static', mirror['jpm']['metrics']), ('caf', mirror['caf']['metrics'])]:
                assert export == r[side]
            assert mirror['caf']['history'] == r['history']
            counts['main_2016_runs'] += 1
    counts['new_revision_portfolios'] = counts['verified_portfolios']
    gurobi_audit_path = ROOT/'docs/gurobi-v2-validation.json'
    gurobi_audit_sha = None
    if len([name for name in validated_files if name.startswith('group-gurobi')]) == 5:
        gurobi_audit = json.loads(gurobi_audit_path.read_text())
        for row in gurobi_audit['logs']:
            assert digest(args.results/row['file']) == row['sha256']
        gurobi_audit_sha = digest(gurobi_audit_path)

    # Independently audit the appendix's original natural-partition references.
    data = json.loads((ROOT/'docs/library-igraph_spectral-results.json').read_text())
    natural_path = ROOT/'docs/natural-newman-revision-results.json'
    natural_summary = json.loads(natural_path.read_text())
    def check_reference_summary(rows, statistics):
        for side in ('static', 'caf'):
            for raw_key, stat_key in [('objective', 'objective'), ('risk', 'risk'), ('expected_return', 'return')]:
                values = np.array([r[side][raw_key] for r in rows])
                np.testing.assert_allclose([values.mean(), values.std()],
                    [statistics[side][stat_key+'_mean'], statistics[side][stat_key+'_std']],
                    rtol=0, atol=1e-12)
        gain = np.mean([(r['static']['objective']-r['caf']['objective'])/
            abs(r['static']['objective'])*100 for r in rows])
        np.testing.assert_allclose(gain, statistics['objective_relative_improvement_mean_pct'], rtol=0, atol=1e-12)
    check_reference_summary([dict(static=r['jpm']['metrics'], caf=r['caf']['metrics'])
        for r in data['per_run_records']], natural_summary['main'])
    groups = [np.asarray(g) for g in data['partition']['communities_solver_order']]
    cards = data['partition']['subproblem_cardinalities']
    for r in data['per_run_records']:
        for side in ('jpm', 'caf'):
            check(r[side]['metrics'], full2016[0].values, full2016[2].values, .5, 242, groups, cards)
    with zipfile.ZipFile(ROOT/'docs/natural-newman-raw-results.zip') as z:
        reference_rows = []
        for i in range(20):
            r = json.loads(z.read(f'group-panel2018_index-{i}.json'))['record']
            reference_rows.append(r)
            groups = [np.asarray(g) for g in r['communities']]
            for side in ('static', 'caf'):
                check(r[side], full2018[0].values, full2018[2].values, .5, 698,
                      groups, r['subproblem_cardinalities'])
    check_reference_summary(reference_rows, next(iter(natural_summary['groups']['panel2018'].values())))
    counts['natural_reference_portfolios'] = counts['verified_portfolios']-counts['new_revision_portfolios']

    controls = json.loads(args.sampling_control.read_text()) if args.sampling_control.exists() else None
    if controls and controls.get('status') != 'complete':
        controls = None
    if not controls and not args.allow_partial:
        raise SystemExit('Missing completed sampling-matched control')
    if controls:
        assert controls['core_source_sha256'] == expected_source
        script_sha = digest(ROOT/'manuscript/run_sampling_control.py')
        assert controls['control_script_sha256'] == script_sha
        assert len(controls['records']) == 40
        assert {(r['year'], r['index']) for r in controls['records']} == {
            (year, i) for year in (2016, 2018) for i in range(20)}
        for row in controls['records']:
            path = args.results/f"group-panel{row['year']}_index-{row['index']}.json"
            primary = json.loads(path.read_text())['record']
            assert row['primary_record_sha256'] == digest(path)
            assert row['control_script_sha256'] == script_sha
            assert row['core_source_sha256'] == expected_source
            for side, key in [('initial', 'static'), ('caf', 'caf')]:
                np.testing.assert_array_equal(row[side]['weights'], primary[key]['weights'])
            problem = full2016 if row['year'] == 2016 else full2018
            groups = [np.asarray(g) for g in primary['communities']]
            check(row['control'], problem[0].values, problem[2].values, .5,
                  len(problem[2])//2, groups, primary['subproblem_cardinalities'])
            h = np.asarray(row['history']['objective'])
            assert np.all(np.diff(h) <= 1e-10)
            assert len(h) == len(primary['history']['objective']) == row['rounds']+1
            np.testing.assert_allclose([h[0], h[-1]],
                [primary['static']['objective'], row['control']['objective']], rtol=0, atol=1e-10)
            assert row['reads_per_method'] == sum(d['dynamic_reads'] for d in
                primary['proposal_diagnostics'] if d['backend_called'])
            assert row['control_proposal_count'] == len(primary['proposal_diagnostics'])
            counts['sampling_matched_control_portfolios'] += 1

    frozen = json.loads(args.frozen.read_text())
    assert frozen['q_eff'] == .5 and frozen['partition_max_community_size'] == QUANTUM_CAPACITY
    assert frozen['source_sha256'] == expected_source
    groups = partition(folded, QUANTUM_CAPACITY, 'folded')
    cov, ret = folded[0].values, folded[2].values
    x = np.asarray(frozen['current_state'])
    base = float(.5*x @ cov @ x-ret @ x)
    assert abs(base-frozen['base_objective']) < 1e-10
    exact_checks = []
    for row in frozen['scan']:
        group, k = groups[row['idx']], row['K']
        choices = list(combinations(range(len(group)), k))
        states = np.zeros((len(choices), len(group)))
        for i, selected in enumerate(choices):
            states[i, list(selected)] = 1
        sub = cov[np.ix_(group, group)]
        external = x.copy()
        external[group] = 0
        adjusted = ret[group]-cov[group] @ external
        quadratic = .5*np.einsum('bi,ij,bj->b', states, sub, states)
        candidates = dict(injected=states[np.argmin(quadratic-states @ adjusted)],
                          uninjected=states[np.argmin(quadratic-states @ ret[group])])
        record = dict(community=row['idx'], assets=group.tolist())
        for name, candidate in candidates.items():
            new = x.copy()
            new[group] = candidate
            delta = float(.5*new @ cov @ new-ret @ new-base)
            expected = row['dO_inject'] if name == 'injected' else row['dO_noinject']
            assert abs(delta-expected) < 1e-10
            record[name] = dict(candidate=candidate.tolist(), global_delta=delta)
        exact_checks.append(record)
    suffix = '-partial' if missing or not controls else ''
    suppression_path = ROOT/f'docs/suppression-v2-validation{suffix}.json'
    suppression_path.write_text(json.dumps(dict(source_sha256=expected_source,
        protocol=protocol, audits=suppression_audits), indent=2)+'\n')
    result = dict(status='incomplete' if suffix else 'passed', missing_tasks=missing,
        sampling_control_complete=bool(controls),
        protocol=protocol, source_sha256=expected_source,
        checks=dict(counts), task_source_sha256_counts=dict(source_counts),
        dataset_sha256=manifest['data_sha256'],
        main_experiment_sha256=digest(main_path),
        gurobi_audit_sha256=gurobi_audit_sha,
        validator_script_sha256=digest(Path(__file__)),
        natural_reference=dict(file=natural_path.name, sha256=digest(natural_path),
            main_reference_sha256=digest(ROOT/'docs/library-igraph_spectral-results.json'),
            raw_archive='natural-newman-raw-results.zip',
            raw_archive_sha256=digest(ROOT/'docs/natural-newman-raw-results.zip')),
        validated_source_files=validated_files, frozen_sha256=digest(args.frozen),
        frozen_independent_enumeration=exact_checks,
        suppression_audit_sha256=digest(suppression_path),
        scope='Full-matrix objectives, exact partitions and budgets, histories and accepted deltas; '
              'seeded suppression replays with direct external multiplication; independently enumerated frozen moves')
    if controls:
        result['sampling_control_sha256'] = digest(args.sampling_control)
    (ROOT/f'docs/revision-validation{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['checks']))


if __name__ == '__main__':
    main()
