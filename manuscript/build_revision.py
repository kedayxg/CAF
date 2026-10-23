"""Build evidence-linked revision tables and figures; refuse incomplete final builds."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_revision import tasks, task_id
from experiments.classical.run_benchmark import paired_objective_statistics
from experiments.quantum.run_qaoa import aggregate_results
from experiments.revision_protocol import CLASSICAL_CAPACITY, QUANTUM_CAPACITY, PROTOCOL_NAME
from coordination_study_section import render_section, write_report
from hardware_results_section import render_hardware


def aggregate(records):
    result = aggregate_results(records)
    result.update(paired_objective_statistics(
        [r["static"]["objective"] for r in records],
        [r["caf"]["objective"] for r in records]))
    return result


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def pm(value, std, digits=4):
    return rf"${fmt(value,digits)} \pm {fmt(std,digits)}$"


def sci(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def plot_main(records, figures):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.4))
    obj = np.array([np.pad(r["caf"]["history"]["objective"],
        (0, 7-len(r["caf"]["history"]["objective"])), mode="edge") for r in records])
    axes[0].plot(range(7), obj.mean(0), marker="o", color="#146b8c")
    axes[0].fill_between(range(7), obj.mean(0)-obj.std(0), obj.mean(0)+obj.std(0),
        color="#146b8c", alpha=.18)
    axes[0].set(xlabel="CAF outer iteration", ylabel="Global objective",
        title="2016, N=484, Newman capacity 100")
    risk = np.array([np.pad(r["caf"]["history"]["risk"],
        (0, 7-len(r["caf"]["history"]["risk"])), mode="edge") for r in records])
    ret = np.array([np.pad(r["caf"]["history"]["expected_return"],
        (0, 7-len(r["caf"]["history"]["expected_return"])), mode="edge") for r in records])
    axes[1].plot(risk.mean(0), ret.mean(0), marker="o", color="#146b8c")
    axes[1].annotate("0",(risk.mean(0)[0],ret.mean(0)[0]),
        xytext=(-14,-14),textcoords="offset points")
    axes[1].annotate("final",(risk.mean(0)[-1],ret.mean(0)[-1]),
        xytext=(4,7),textcoords="offset points")
    axes[1].set(xlabel="Risk", ylabel="Expected return", title="Iteration 0 = static initialization")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figures/"revision_main.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gurobi_main(panels, figures):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.0))
    for ax, year in zip(axes, ('2016', '2018')):
        h = panels[year]['caf']['history']['objective']
        ax.plot(range(len(h)), h, marker='o', color='#146b8c', label='CAF')
        ax.axhline(h[0], linestyle='--', color='#9b6238', label='Static / CAF without injection')
        ax.set(xlabel='CAF outer iteration', ylabel='Global objective',
               title=f'{year}: Gurobi local solves, capacity 100')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures/'revision_main.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_flow(figures):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.3, 4.1))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    boxes = [(1.2,4.3,"Market data"), (4.6,4.3,"RMT + capacity Newman\nIndependent static solves"),
        (8.2,4.3,"Feasible portfolio\nFixed community budgets"),
        (2.1,2.7,"Freeze portfolio\nand covariance cache"),
        (6.6,2.7,"Inject external context\nGenerate local proposals"),
        (6.6,1.1,"Sequential global test\nAccept only ΔO ≤ −ε"),
        (2.1,1.1,"Update portfolio/cache\nCheck stopping rule")]
    for x, y, label in boxes:
        ax.text(x,y,label,ha="center",va="center",fontsize=10,
            bbox=dict(boxstyle="round,pad=.5",fc="#eef5f8",ec="#28657d"))
    for start, end in [((2.15,4.3),(3.05,4.3)),((6.15,4.3),(7.0,4.3)),
        ((3.45,2.7),(4.8,2.7)),
        ((6.6,2.25),(6.6,1.55)),((4.7,1.1),(3.6,1.1)),((2.1,1.55),(2.1,2.25))]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#28657d",lw=1.5))
    ax.plot([8.2,8.2,2.1], [3.88,3.5,3.5], color="#28657d",lw=1.5)
    ax.annotate("",xy=(2.1,3.13),xytext=(2.1,3.5),arrowprops=dict(arrowstyle="->",color="#28657d"))
    ax.text(2.25,1.9,"continue",fontsize=8)
    ax.plot([2.1,2.1,8.4], [.68,.25,.25], color="#28657d",lw=1.5)
    ax.text(5.2,.4,"stopping rule met",fontsize=9,ha="center")
    ax.text(9.1,.25,"Final portfolio",ha="center",va="center",fontsize=10)
    ax.annotate("",xy=(8.5,.25),xytext=(8.2,.25),arrowprops=dict(arrowstyle="->",color="#28657d"))
    fig.tight_layout()
    fig.savefig(figures/"caf_overall_flow.pdf",bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"revision_results_v2")
    parser.add_argument("--main-result", type=Path)
    parser.add_argument("--sampling-control", type=Path, default=ROOT/'docs/sampling-control-results.json')
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--sqa-results", type=Path,
        help="Optional separate complete SQA rerun after correcting the zero-field default")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    main_path = args.main_result or args.results/"main_2016.json"
    main_data = json.loads(main_path.read_text())
    assert main_data["config"]["max_community_size"] == CLASSICAL_CAPACITY
    rows = main_data["per_run_records"]
    assert len(rows) == 20 and main_data["runs_completed"] == 20
    groups = defaultdict(list)
    paths, missing, digests = [], [], Counter()
    for task in tasks():
        directory = args.sqa_results if task["group"] == "sqa" and args.sqa_results else args.results
        path = directory/(task_id(task)+".json")
        if not path.exists():
            missing.append(task)
            continue
        data = json.loads(path.read_text())
        assert data["task"] == task
        if data.get('protocol') != dict(name=PROTOCOL_NAME, classical_capacity=CLASSICAL_CAPACITY, quantum_capacity=QUANTUM_CAPACITY):
            raise SystemExit(f"Wrong or undocumented protocol: {path}")
        if data['source_sha256'] != main_data['source_sha256']:
            raise SystemExit(f"Mixed execution source: {path}")
        if task["group"] == "sqa" and data["record"].get("backend_protocol",{}).get("Hd_field") != "linspace(1,0,1000)":
            missing.append(task)
            continue
        digests[data["source_sha256"]] += 1
        groups[task["group"]].append(data)
        paths.append(path)
    if missing and not args.allow_partial:
        raise SystemExit(f"Incomplete: {len(paths)}/{len(tasks())} tasks. No final manuscript generated.")
    frozen = json.loads(args.frozen.read_text())
    assert frozen['partition_max_community_size'] == QUANTUM_CAPACITY
    assert frozen['source_sha256'] == main_data['source_sha256']
    main_sizes = main_data['partition']['community_sizes']
    controls = json.loads(args.sampling_control.read_text()) if args.sampling_control.exists() else None
    if controls and (controls.get('status') != 'complete' or len(controls.get('records', [])) != 40):
        controls = None
    if controls:
        assert controls['core_source_sha256'] == main_data['source_sha256']
    if not controls and not args.allow_partial:
        raise SystemExit('Incomplete sampling-matched control; finish all forty additional pairs')
    gurobi_path = ROOT/'docs/gurobi-main-results.json'
    gurobi = json.loads(gurobi_path.read_text())
    gurobi_validation_path = ROOT/'docs/gurobi-main-validation.json'
    gurobi_validation = json.loads(gurobi_validation_path.read_text())
    assert gurobi['status'] == 'complete' and gurobi_validation['status'] == 'passed'
    assert gurobi['source_files'] == gurobi_validation['validated_source_files']
    assert gurobi['validation_sha256'] == hashlib.sha256(gurobi_validation_path.read_bytes()).hexdigest()
    assert gurobi['protocol']['core_source_sha256'] == main_data['source_sha256']
    with zipfile.ZipFile(ROOT/'docs/gurobi-main-raw-results.zip') as z:
        gurobi_panels = {year:json.loads(z.read(f'panel-{year}.json')) for year in ('2016','2018')}
        for year in gurobi_panels:
            assert hashlib.sha256(z.read(f'panel-{year}.json')).hexdigest() == gurobi['source_files'][f'panel-{year}.json']
    incomplete = bool(missing or not controls)
    output = dict(status="complete" if not incomplete else "DRAFT: reruns or controls incomplete",
        planned_tasks=len(tasks()), completed_tasks=len(paths), missing_tasks=missing,
        execution_source_sha256_counts=dict(digests),
        main_experiment_reference=main_path.name,
        protocol=dict(name=PROTOCOL_NAME,classical_capacity=CLASSICAL_CAPACITY,quantum_capacity=QUANTUM_CAPACITY),
        frozen_sha256=hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
        main_experiment_sha256=hashlib.sha256(main_path.read_bytes()).hexdigest(),
        source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        groups={}, frozen=frozen)
    output['sampling_control_complete'] = bool(controls)
    output['sampling_control_role'] = 'CAF injection ablation: both variants retain full-global objective verification and greedy sequential acceptance; only local proposal injection is disabled. This is not an independent static-decomposition baseline.'
    output['primary_backend'] = 'Gurobi'
    output['gurobi_main'] = gurobi['summary']
    output['gurobi_main_sha256'] = hashlib.sha256(gurobi_path.read_bytes()).hexdigest()
    output['gurobi_main_validation_sha256'] = hashlib.sha256(gurobi_validation_path.read_bytes()).hexdigest()
    if controls:
        output['sampling_control'] = controls['summary']
        output['sampling_control_sha256'] = hashlib.sha256(args.sampling_control.read_bytes()).hexdigest()
    classical_rows = []
    for r in rows:
        classical_rows.append(dict(seed=r["seed"],static=r["jpm"]["metrics"],caf=r["caf"]["metrics"],
            static_time_sec=r["jpm"]["seconds"],caf_time_sec=r["caf"]["timing"]["total_seconds"]))
    main_stats = aggregate(classical_rows)
    output["main"] = main_stats
    text = []
    if incomplete:
        text.append(r"\noindent\textbf{Working draft: revision reruns are incomplete. Do not submit this build.}")
    text.extend([r"\section{Experimental Results}",
        r"\subsection{Corrected Community Detection and Protocol}",
        r"The v1 implementation inherited a modified spectral routine whose recursion mixed local and global asset indices and used nonstandard split-gain handling. We now use only python-igraph 0.11.9 (C igraph 0.10.16) Newman's leading-eigenvector method~\cite{newman2006finding,igraph2006}. The original source is retained in paper commit \texttt{356c1852}, and its runtime entry points have been removed. This replacement changes the experimental decomposition, while CAF's context and acceptance equations remain unchanged. The previous label `Louvain' did not describe the executed algorithm.",
        r"RMT preprocessing uses the bundled denoiser with initial aspect-ratio parameter 0.5 and fitting enabled. We form $A=|C_{\rm RMT}|$ from its filtered correlation output, retaining the diagonal; financial objectives retain the original sample covariance and signed cross-covariances. Undirected graph self-loops have weight $A_{ii}/2$, preserving $k_i=\sum_j A_{ij}$ and $T=\sum_{ij}A_{ij}$. On each current graph the score is $Q=T^{-1}\sum_{ij}(A_{ij}-k_i k_j/T)\mathbf{1}[c_i=c_j]$. Each igraph call uses leading-eigenvector splitting to seek high Newman modularity, with seed 42, ARPACK tolerance $10^{-10}$ and at most 10,000 iterations. To model bounded local solvers, we restart igraph on an oversized community's original induced submatrix and repeat until each block fits. We stop when a block fits, including a full instance already below the capacity; we neither re-denoise submatrices nor force balanced splits. An oversized block that igraph cannot split is an explicit failure. Restarting changes the local degree null model: this is a resource-oriented hierarchical heuristic, not a claim to optimize constrained full-graph modularity. These declared capacities are benchmark resource scenarios, not measured QPU limits or a reported gain-maximizing capacity sweep.",
        f"The classical capacity is {CLASSICAL_CAPACITY} assets and the statevector-study capacity is {QUANTUM_CAPACITY}. The main 2016 partition has {len(main_sizes)} communities, with sizes $"+", ".join(map(str,main_sizes))+r"$. Static and CAF share this partition and its fixed proportional integer budgets; CAF does not move budget across communities. Unrestricted Newman reference results are reported in the appendix, rather than substituted for these capacity-constrained experiments.",
        r"The bundled matrices are derived from asset-return panels in the upstream tests/data release~\cite{jpmorgan_dcmppln}: expected returns are column-wise panel means, covariance uses NumPy sample covariance, and correlation is derived from covariance. No extra annualization factor is introduced here; the manifests hash all six matrices. The 2016 panel has $N=484,K=242$, and the additional 2018 panel has $N=1397,K=698$. They are separate in-sample historical cross-sections, not a held-out investment evaluation. Unless stated otherwise, local and global objectives use $q=0.5$ without risk rescaling. This records the actual unscaled initialization behavior; v1 incorrectly described that switch as enabled.",
        r"The supplementary static SA baseline uses 200 reads per block. CAF uses 20 reads for $n\le30$, 100 for $30<n\le80$, and 200 for $n>80$, at most six outer iterations, patience two and acceptance tolerance $10^{-5}$. The 2016 SA study uses seeds 42--61. Bootstrap intervals use 5,000 paired resamples; sign tests exclude ties at $10^{-9}$. These repeats measure local-solver randomness on fixed inputs and partitions, not uncertainty across market samples. CAF times refer to refinement after initialization; end-to-end time is static plus refinement. Per-solve caps are matched, but total computational effort is higher for repeated CAF solves. Thus the aggregate comparison does not by itself isolate context from additional search.",
        r"\subsection{Simulated-Annealing Sensitivity on 2016}",
        r"\begin{table}[t]\caption{SA sensitivity on 2016, 20 paired runs ($q=0.5$).}\label{tab:benchmark}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{lccc}\toprule Method & Objective & Risk & Expected return\\\midrule"])
    for label, side in (("Static", "static"), ("CAF", "caf")):
        m = main_stats[side]
        text.append(f"{label} & {pm(m['objective_mean'],m['objective_std'])} & {pm(m['risk_mean'],m['risk_std'])} & {pm(m['return_mean'],m['return_std'])}"+r"\\")
    text.append(r"\bottomrule\end{tabular}}\end{table}")
    main_ci = main_stats['objective_relative_improvement_ci95_pct']
    text.append(f"CAF attains a mean paired objective improvement of ${main_stats['objective_relative_improvement_mean_pct']:.4f}\\%$: {main_stats['objective_improvement_runs']} improving runs, {main_stats['objective_tied_runs']} ties and no losses. The 95\\% bootstrap interval is $[{main_ci[0]:.4f}\\%,{main_ci[1]:.4f}\\%]$. Every portfolio satisfies all budgets and every history is nonincreasing. These gains describe recovery from this capacity-constrained static initialization, not improvement over unrestricted full-portfolio optimization. The risk and expected-return columns must be read together; an improved scalar objective need not raise expected return.")
    text.append(r"The sign calculation uses a symmetric-sign reference model. Because monotone acceptance precludes losses against initialization by design, that p-value is descriptive rather than an independent mechanism test. The bootstrap interval describes solver variability on these fixed data. A separate sampling-matched iterative control below examines the injected proposal model beyond additional search.")
    text.append(r"An independent audit recomputes all stored portfolio objectives from the original full covariance matrices and checks binary decisions, fixed budgets, history endpoints and accepted-update increments. This verification does not rely solely on the incremental objective cache.")
    gtext = [r"\subsection{Primary Gurobi Coordination Experiments}",
        r"The primary experiment uses Gurobi for both independent static initialization and CAF refinement, to study coordination when the local problems are solved to a declared MIP tolerance. This is a classical diagnostic of the coordination layer; it does not make capacity 100 a Gurobi hardware limit. Each panel has one fixed-configuration pipeline run, not twenty independent trials. This primary-backend emphasis was selected after examining the SA controls and is a diagnostic follow-up, not a preregistered backend ranking. We retain all SA runs and sampling-matched controls below to expose heuristic-backend sensitivity rather than replacing that evidence.",
        r"Gurobi 13.0.2 uses Threads=1, Seed=0, MIPGap=$10^{-4}$ and a 300-second limit per local call. CAF uses at most six frozen-proposal/sequential-commit rounds and patience two. The unshifted control is CAF without injection: it repeats the original local models from the same Gurobi static portfolio for the same number of rounds, retaining CAF's full-global verification and greedy sequential acceptance. There is no equality of total work between one-shot static optimization and CAF. We record each local incumbent, bound, relative gap and termination status, and independently replay all candidate commits against the full covariance matrix.",
        r"\begin{table}[t]\caption{Primary Gurobi experiments, one fixed-configuration pipeline per panel ($q=0.5$, capacity 100).}\label{tab:gurobi-main}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{llccc}\toprule Panel & Method & Objective & Risk & Expected return\\\midrule"]
    for year, stat in gurobi['summary'].items():
        gtext.append(f"{year} & Static & {stat['static_objective']:.6f} & {stat['static_risk']:.6f} & {stat['static_return']:.6f}"+r"\\")
        gtext.append(f"{year} & CAF & {stat['caf_objective']:.6f} & {stat['caf_risk']:.6f} & {stat['caf_return']:.6f}"+r"\\")
    gtext.append(r"\bottomrule\end{tabular}}\end{table}")
    for year, stat in gurobi['summary'].items():
        gtext.append(f"On {year}, the objective decreases from {stat['static_objective']:.9f} to {stat['caf_objective']:.9f}, a {stat['gain_pct']:.4f}\\% improvement, with {stat['methods']['caf']['accepted_updates']} accepted updates. The repeated unshifted control finishes at {stat['control_objective']:.9f}; its final weights are identical to the static initialization.")
    gtext.append(r"With unique exact local optima, repeating an unchanged unshifted model cannot improve its initialization. The observed equality on both panels is checked directly at the stated numerical tolerance, without assuming uniqueness.")
    gtext.append(r"\begin{table}[t]\caption{Primary Gurobi cost and local certificates. Times exclude shared RMT/partition preprocessing; CAF time excludes initialization. Control solves are additional.}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrr}\toprule Panel & Static (s) & CAF (s) & Static solves & CAF solves & Max gap\\\midrule")
    for year, stat in gurobi['summary'].items():
        gtext.append(f"{year} & {stat['static_time_sec']:.3f} & {stat['caf_time_sec']:.3f} & {stat['static_local_solves']} & {stat['caf_local_solves']} & ${sci(stat['max_local_gap'])}$"+r"\\")
    gtext.append(r"\bottomrule\end{tabular}}\end{table}")
    total_g = sum(s['local_solves'] for s in gurobi['summary'].values())
    gtext.append(f"All {total_g} local solves across initialization, CAF and the unshifted controls terminate as optimal within the requested tolerance, with no time limits. A local MIP bound certifies its own frozen local model, not the coupled full portfolio. Nor do patience or the six-round cap establish a block-local fixed point. These results isolate coordination beyond re-solving the same well-solved unshifted models; they do not establish universal superiority over other global or block-search methods.")
    gtext.append(r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/revision_main.pdf}\caption{Primary Gurobi convergence on both fixed panels. Dashed lines mark static initialization and the unchanged unshifted-control endpoints. Each trajectory is one pipeline run.}\label{fig:obj_conv}\end{figure}")
    insertion = text.index(r"\subsection{Simulated-Annealing Sensitivity on 2016}")
    text[insertion:insertion] = gtext
    study_text, study_binding = render_section()
    insertion = text.index(r"\subsection{Simulated-Annealing Sensitivity on 2016}")
    text[insertion:insertion] = [
        r"\subsection{Supplementary Update-Timing and Sensitivity Checks}",
        f"A latest-context sequential-update ablation, following cyclic discrete BCD and sharing CAF's partition, budgets, initialization, local solver and acceptance rule, matches CAF's final objectives in {study_binding['tied']}/{study_binding['complete_cases']} configurations after six matched sweeps. The study covers capacities 75, 100 and 150 on both full panels and three 300-asset subsets per panel. Changing capacity also changes the fixed-budget feasible space. Appendix~\\ref{{app:update-timing}} reports the complete ablation, capacity/subset results and resource measurements; no consistent local CPU speedup is established."]
    output['coordination_study'] = study_binding
    hardware_main,hardware_appendix,hardware_binding = render_hardware()
    output['hardware_study'] = hardware_binding
    output['manuscript_scope'] = dict(
        purpose='Implementation correction and revalidation of the existing portfolio coordination study',
        primary_comparison='Independent static initialization versus CAF refinement',
        coordination_study_role='Supplementary update-timing ablation and capacity/input sensitivity; full results in appendix',
        bcd_role='Technical foundation and latest-context update-rule variant, not a separate primary competing pipeline',
        synthetic_search_in_main_paper=False,
        additional_experiments=False)
    for group in ("panel2018", "scaling_sa", "sparsity_sa", "gurobi", "q_sweep"):
        indexed = defaultdict(list)
        for item in groups[group]:
            task = item["task"]
            key = str(task.get("q")) if group == "q_sweep" else f"N={task.get('n',1397)},K={task.get('k',698)}"
            indexed[key].append(item["record"])
        output["groups"][group] = {key:aggregate(records) for key, records in indexed.items()}
    text.append(r"\subsection{Additional SA Panel and Scaling Checks}")
    if len(groups["panel2018"]) == 20:
        stats = next(iter(output["groups"]["panel2018"].values()))
        text.append(f"The twenty-run 2018 check gives static objective {pm(stats['static']['objective_mean'],stats['static']['objective_std'])} and CAF objective {pm(stats['caf']['objective_mean'],stats['caf']['objective_std'])}. Mean paired improvement is ${stats['objective_relative_improvement_mean_pct']:.4f}\\%$, with {stats['objective_improvement_runs']} improvements, {stats['objective_tied_runs']} ties and {stats['objective_worsened_runs']} losses. This panel is a separate historical cross-section, not an out-of-sample investment test.")
        ci = stats["objective_relative_improvement_ci95_pct"]
        p = stats["objective_sign_test_pvalue"]
        p_text = sci(p) if p < .001 else fmt(p)
        text.append(f"The paired-gain 95\\% bootstrap interval is $[{ci[0]:.4f}\\%,{ci[1]:.4f}\\%]$; the tie-excluding sign-test p-value is ${p_text}$.")
        if stats["objective_improvement_runs"] == 0:
            text.append(r"Under this fixed protocol CAF accepts no improving move in any trial. These revised data do not support the v1 claim of a positive gain on the 2018 panel. This is a limit of the observed refinement behavior, not a local-optimality certificate.")
    text.append(r"For the size and sparsity studies, subsets are fixed using seed $123+N$, with SA run seeds $42+1000N+10K+i$, $i=0,\ldots,19$. These differ from the dedicated main-benchmark seeds. Gurobi 13.0.2 uses one thread and relative MIP-gap tolerance $10^{-4}$; we perform one paired pipeline execution per setting, comprising multiple local MIP solves, and do not represent it as 20 independent trials. Each local call has a 300-second limit. A nonzero MIP tolerance is not an exact global-optimality certificate.")
    text.append(r"\begin{table}[t]\caption{Revision size and sparsity checks. SA entries have 20 paired runs; Gurobi entries one deterministic run.}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{llccc}\toprule Backend & Setting & Static objective & CAF objective & Gain (\%)\\\midrule")
    for group, label in (("scaling_sa","SA"),("sparsity_sa","SA"),("gurobi","Gurobi")):
        for key, stats in output["groups"][group].items():
            count = stats["total_runs"]
            required = 1 if group == "gurobi" else 20
            if count == required:
                text.append(f"{label} & {key} & {stats['static']['objective_mean']:.6f} & {stats['caf']['objective_mean']:.6f} & {stats['objective_relative_improvement_mean_pct']:.3f}"+r"\\")
    text.append(r"\bottomrule\end{tabular}}\end{table}")
    text.append(r"At $N=100$ the complete instance already fits the classical capacity and is kept as one block. Its external context is zero, so any SA refinement reflects additional search rather than cross-community coordination. This endpoint is retained to show where decomposition is unnecessary; no monotone increase of benefit with instance size or community count is asserted.")
    if len(groups['gurobi']) == 5:
        audit_path = ROOT/'docs/gurobi-v2-validation.json'
        audit = json.loads(audit_path.read_text())
        for row in audit['logs']:
            assert hashlib.sha256((args.results/row['file']).read_bytes()).hexdigest() == row['sha256']
        output['gurobi_audit_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        text.append(f"The five Gurobi pipeline runs comprise {audit['local_solves']} local solves: {audit['optimal_with_requested_tolerance']} terminate as optimal within the requested tolerance, {audit['time_limits']} hit the time limit, and {audit['other_terminations']} have other terminations. These local statuses do not certify a globally optimal full portfolio.")
    text.append(r"\subsection{Sampling-Matched Injection Ablation within CAF}")
    text.append(r"For each SA panel seed, we compare full CAF to CAF without injection, an ablation that uses original, unshifted local returns for proposals. Both variants retain CAF's full-global objective verification and greedy sequential acceptance, as well as the same initial portfolio, partition and budgets. This comparison isolates the contribution of proposal injection within that architecture; it is not CAF versus an independent static-decomposition method. The ablation matches full CAF's actual number of rounds, local call seeds, sweeps and reads; its patience is disabled until that round budget is exhausted. Automatic constrained-to-binary penalty conversion remains part of each model path, so the tested component includes the complete injected-model implementation. Matching sampling does not prove identical elapsed time or arithmetic work. The common verification and acceptance components are not independently ablated here.")
    if controls:
        text.append(r"\begin{table}[t]\caption{CAF injection ablation, 20 paired SA seeds per panel. Both variants retain global verification and greedy sequential acceptance. Extra gain is relative to the common static objective, in percentage points. Counts describe full CAF versus its ablation.}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{lcccc}\toprule Panel & Without injection & Full CAF & Extra gain & Better/tied/worse\\\midrule")
        for year in ('2016', '2018'):
            stat = controls['summary'][year]
            text.append(f"{year} & {stat['no_context_mean']:.6f} & {stat['caf_mean']:.6f} & {stat['caf_advantage_mean_percentage_points']:.3f} & {stat['caf_wins']}/{stat['ties']}/{stat['caf_losses']}"+r"\\")
        text.append(r"\bottomrule\end{tabular}}\end{table}")
        for year in ('2016', '2018'):
            stat = controls['summary'][year]
            lo, hi = stat['caf_advantage_ci95_percentage_points']
            text.append(f"For {year}, the full-minus-ablated gain difference has mean {stat['caf_advantage_mean_percentage_points']:.3f} percentage points and paired bootstrap 95\\% interval $[{lo:.3f},{hi:.3f}]$; full CAF is better in {stat['caf_wins']} pairs, tied in {stat['ties']} and worse in {stat['caf_losses']}. These solver-seed comparisons qualify the contribution of injection, without measuring the independent effect of the shared global acceptance mechanism.")
    else:
        text.append(r"The sampling-matched control is still running; this working draft does not claim a completed mechanism comparison.")
    text.append(r"\subsection{SA Risk-Aversion Sensitivity}")
    text.append(r"The sweep $q\in\{0.1,0.3,0.5,0.7,1.0\}$ keeps the twenty $q=0.5$ static initial portfolios fixed and changes both the CAF proposal objective and global evaluation. It tests refinement from a shared initialization, not reoptimized static baselines at each $q$. The seed is $42+10000+100j+i$ for risk-setting index $j$ and run index $i$.")
    text.append(r"\begin{table}[t]\caption{Shared-initialization risk sweep, 20 runs per entry.}\centering\begin{tabular}{lcc}\toprule $q$ & Mean objective gain (\%) & Improving / tied\\\midrule")
    for key, stats in output["groups"]["q_sweep"].items():
        if stats["total_runs"] == 20:
            text.append(f"{key} & {stats['objective_relative_improvement_mean_pct']:.3f} & {stats['objective_improvement_runs']} / {stats['objective_tied_runs']}"+r"\\")
    text.append(r"\bottomrule\end{tabular}\end{table}")
    text.append(r"\subsection{Landscape Diagnostic and Frozen Context Ablation}")
    suppression = [row for item in groups["suppression"] for row in item["record"]]
    if suppression:
        from experiments.classical.run_suppression import summarize
        stats = summarize(suppression, [100,200,300,484])
        output["groups"]["suppression"] = stats
        if len(groups["suppression"]) == 40:
            changes = np.array([r["suppressed_delta_pct"] for r in suppression])
            text.append(f"Across ten sampled subsets at each $N\\in\\{{100,200,300,484\\}}$ ({len(suppression)} community observations), injection changes the fraction of nonpositive local linear returns by an unweighted community mean of {changes.mean():.2f} percentage points (standard deviation {changes.std():.2f}; 10th--90th percentiles {np.percentile(changes,10):.2f}--{np.percentile(changes,90):.2f}). A nonpositive linear return is a surrogate diagnostic, not proof that an asset is fixed or removable from the constrained quadratic model. Community observations within the same portfolio are dependent.")
            available = stats["overall"]["presolve_available_samples"]
            reduction = stats["overall"]["reduction_vs_static_mean_pct"]
            text.append(f"Auxiliary Gurobi presolve is available for {available} of {len(suppression)} community comparisons; the remaining presolve entries, if any, are unavailable because of the local license's quadratic-model size limit. All linear-return diagnostic observations are retained. Among available pairs, the mean active-variable reduction relative to static presolve is {reduction:.2f}\\%. Missing presolve values are not assigned zero reductions.")
    selected = next(r for r in frozen["scan"] if r["idx"] == frozen["selected"])
    if frozen.get("local_risk_rebalancing", True):
        local_convention = f"The local surrogate uses group-based norm rescaling $q_{{\\rm eff}}={frozen['q_eff']:.9f}$, while acceptance uses the original global $q=0.5$."
    else:
        assert frozen["q_eff"] == .5
        local_convention = "Local solves and global acceptance both use $q=0.5$, without local risk rescaling."
    text.append(f"On the frozen $N=40$ snapshot (subset seed 1234, static seed 42), the declared small-community selector returns $n={frozen['n_local']},K={frozen['k_local']}$. {local_convention} Without injection the exact update has $\\Delta\\mathcal O={selected['dO_noinject']:.8f}$; with injection it has $\\Delta\\mathcal O={selected['dO_inject']:.8f}$. Warm-start QAOA ($p=2$, six restarts, 80 COBYLA evaluations, 4,000 shots) obtains $\\Delta\\mathcal O={selected['dO_sim']:.8f}$, at Hamming distance {selected['hamming']} from the exact candidate. This is a single controlled local example, not an aggregate equal-effort comparison. No improving $n\\le6$ target satisfied this selection rule; a subsequent hardware check of all four communities, including this eight-asset block, is reported below.")
    text.append(r"Here the smallest detected block already has eight assets, so the absence of an $n\le6$ hardware target follows from the partition, rather than from a hardware execution test. The full four-community frozen scan is archived, including communities where injection does not change or improve the selected update.")
    text.append(r"\subsection{Quantum-Simulator Compatibility}")
    text.append(r"We fix the $N=40,K=20$ subset using seed 1234. With capacity 16 the community sizes are 14, 10, 8 and 8. The imposed simulator capacity is part of the protocol; the observed maximum of 14 is instance-specific. Matched plain QAOA uses $p=2$ or $p=3$, six random restarts, up to 80 COBYLA evaluations per restart, 4,000-shot feasible-sample decoding, and no warm start. Static and CAF use the same backend settings within each depth, with at most four CAF iterations and patience two. Seeds are 1042--1061. SQA uses D-Wave's path-integral sampler~\cite{dwave_samplers_docs,martonak2002pimc}, 1,000 sweeps, 20 static reads and CAF's size-adaptive reads, with seeds 31042--31061. We explicitly set the inverse-temperature-scaled transverse sequence $H_d=\mathrm{linspace}(1,0,1000)$ and $\Gamma=1$, retaining the library's geometric longitudinal schedule with automatically computed beta range. This declared schedule is not optimized over alternatives. These are simulation compatibility checks, not evidence of quantum advantage.")
    text.append(r"This also addresses a separate caller-configuration omission. In the pinned reproduction environment, an unspecified transverse sequence defaults to zero; the provided earlier wrapper omitted it. That checked execution path is a zero-transverse-field classical limit, and the original complete runtime environment was not supplied. The revised SQA column therefore uses a fresh complete twenty-seed run with the explicit nonzero sequence, while archived zero-field reproduction checkpoints are identified separately.")
    text.append(r"QAOA parameter optimization uses exact statevector energy expectations; the 4,000 samples are drawn from the final optimized distribution for decoding. These runs therefore do not model shot noise during training, gate noise, queueing or hardware latency. The frozen warm-start state and mixer use the local continuous relaxation, so that example also includes classical preprocessing. CAF normalizes backend outputs to fixed budgets and records raw proposal feasibility and repairs; a repaired vector is not evidence of a directly sampled quantum state.")
    text.append(r"The cost circuit uses the direct Pauli convention: dimod's spin $s=2x-1$ equals $-Z$, hence the target diagonal Hamiltonian is $H=\mathrm{offset}-\sum_i h_i Z_i+\sum_{i<j}J_{ij}Z_iZ_j$. The gates are $R_Z(-2\gamma h_i)$ and $R_{ZZ}(2\gamma J_{ij})$. The old linear-sign convention is related by a parameter redefinition for standard plain QAOA, but arbitrary warm starts and finite optimizer trajectories need not coincide.")
    text.append(r"\begin{table}[t]\caption{Corrected folded quantum-simulator comparisons, 20 runs per column.}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{lccc}\toprule Backend & Improving / tied & Mean $\Delta\mathcal O$ & Std. $\Delta\mathcal O$\\\midrule")
    for group, label in (("qaoa2","QAOA $p=2$"),("qaoa3","QAOA $p=3$"),("sqa","SQA")):
        records = [item["record"] for item in groups[group]]
        if records:
            stats = aggregate(records)
            output["groups"][group] = stats
            if len(records) == 20:
                text.append(f"{label} & {stats['objective_improvement_runs']} / {stats['objective_tied_runs']} & ${sci(stats['absolute_delta_mean'])}$ & ${sci(stats['absolute_delta_std'])}$"+r"\\")
    text.append(r"\bottomrule\end{tabular}}\end{table}")
    if all(len(groups[g]) == 20 for g in ('qaoa2', 'qaoa3', 'sqa')):
        diag = [d for group in ('qaoa2', 'qaoa3', 'sqa') for item in groups[group]
                for d in item['record']['proposal_diagnostics']]
        raw = sum(bool(d.get('raw_candidate_feasible')) for d in diag)
        repaired = sum(bool(d.get('candidate_repaired')) for d in diag)
        text.append(f"Among the {len(diag)} recorded CAF proposals across these three simulator groups, {raw} are raw feasible candidates and {repaired} require cardinality repair. These are simulator proposal diagnostics; the subsequent hardware study has separate raw-count records.")
    text.extend(hardware_main)
    text.append(r"\section{Conclusion}")
    g16, g18 = gurobi['summary']['2016'], gurobi['summary']['2018']
    text.append(f"CAF coordinates fixed community subproblems through an external-risk linear bias and sequential global acceptance. In the primary capacity-100 Gurobi runs, objective gains are {g16['gain_pct']:.2f}\\% on 2016 and {g18['gain_pct']:.2f}\\% on 2018. Both initialization and refinement use the same local solver settings. Repeated unshifted local solves leave the respective initial portfolios unchanged; CAF improves them by accounting for external covariance. These are two fixed-instance pipeline runs, not statistical market samples or full-portfolio global-optimality certificates.")
    if len(groups['panel2018']) == 20:
        panel = next(iter(output['groups']['panel2018'].values()))
        text.append(f"The supplementary SA experiments give mean gains of {main_stats['objective_relative_improvement_mean_pct']:.2f}\\% on 2016 and {panel['objective_relative_improvement_mean_pct']:.2f}\\% on 2018 against their respective static baselines. These quantify finite-budget heuristic behavior.")
    if controls:
        c16, c18 = controls['summary']['2016'], controls['summary']['2018']
        text.append(f"Under matched SA sampling budgets, full CAF performs better than CAF without injection in {c16['caf_wins']}/20 pairs on 2016. On 2018, the ablated variant performs better in all {c18['caf_losses']}/20 pairs. Both retain CAF's full-global objective verification and greedy sequential acceptance and improve their static initializations. This is component-level evidence that injection's additional benefit depends on the instance and heuristic backend; it is not evidence that static decomposition outperforms CAF or that the complete coordination architecture fails. The experiment does not identify a unique cause of the differing trajectories or independently isolate the effect of the shared acceptance mechanism.")
    text.append(r"In four independent frozen WK\_C180 batches, the two eight-asset candidates meet the declared improvement criterion, while the other proposals are rejected. The hardware distributions differ substantially from their ideal references; an exact uniform-sampling diagnostic shows that the two passes do not establish a QAOA sampling benefit. This supports candidate decoding and guarded acceptance from hardware output, not a complete hardware CAF outer loop.")
    text.append(r"The context and exact increment formulas support descent for accepted feasible updates, while the observed benefit depends on partition, fixed budgets, backend and effort. A large relative gain can coexist with a poor static starting point and does not establish a better final solution than a different decomposition. Heuristic stopping does not certify local optimality. The experiments establish refinement behavior, simulator compatibility and frozen hardware proposal/acceptance executability, not equal-effort superiority, quantum advantage, global optimality or out-of-sample financial performance. Equal-runtime comparisons, budget transfer, repeated hardware sampling and cloud-compilation diagnostics remain future work.")
    text.append(r"\FloatBarrier\appendices")
    text.extend(study_text)
    text.append(r"\FloatBarrier\section{Unrestricted Newman Reference}")
    natural_path = ROOT/'docs/natural-newman-revision-results.json'
    natural = json.loads(natural_path.read_text())
    ref2016 = natural['main']
    ref2018 = next(iter(natural['groups']['panel2018'].values()))
    output['natural_reference'] = dict(file=natural_path.name,
        sha256=hashlib.sha256(natural_path.read_bytes()).hexdigest(),
        main_reference_sha256=hashlib.sha256((ROOT/'docs/library-igraph_spectral-results.json').read_bytes()).hexdigest(),
        raw_archive='natural-newman-raw-results.zip',
        raw_archive_sha256=hashlib.sha256((ROOT/'docs/natural-newman-raw-results.zip').read_bytes()).hexdigest())
    text.append(r"These SA-only appendix references do not compare Gurobi to unrestricted Gurobi. The archived unrestricted studies use the same input matrices and twenty trials per panel, with original source fingerprints preserved. Natural Newman yields 2016 sizes 199,153,132 and 2018 sizes 405,356,343,293; the capacity-100 variants give 15 and 25 communities. These references change fixed feasible budgets, local dimensions and aggregate sampling effort and are not a one-factor or equal-effort comparison.")
    text.append(r"\begin{table}[t]\caption{SA-only descriptive unrestricted reference versus capacity 100. Lower objectives are better; each entry has 20 runs.}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{llcccc}\toprule Panel & Partition & Blocks & Static & CAF & Gain (\%)\\\midrule")
    for year, label, count, stat in [(2016,'Natural',3,ref2016),(2016,'Capacity100',15,main_stats),(2018,'Natural',4,ref2018)]:
        text.append(f"{year} & {label} & {count} & {stat['static']['objective_mean']:.6f} & {stat['caf']['objective_mean']:.6f} & {stat['objective_relative_improvement_mean_pct']:.3f}"+r"\\")
    if len(groups['panel2018']) == 20:
        stat=next(iter(output['groups']['panel2018'].values()))
        text.append(f"2018 & Capacity100 & 25 & {stat['static']['objective_mean']:.6f} & {stat['caf']['objective_mean']:.6f} & {stat['objective_relative_improvement_mean_pct']:.3f}"+r"\\")
    text.append(r"\bottomrule\end{tabular}}\end{table}")
    text.append(r"Unrestricted 2018 accepts no improving SA move in twenty trials; that observation is not a local-optimality certificate. The 2016 capacity variant has a larger relative CAF gain but almost the same final objective as unrestricted Newman. The 2018 capacity variant improves its own baseline but has a worse final objective than unrestricted Newman. Thus additional communities do not by themselves establish better financial solutions. The capacity study evaluates coordination when bounded local subproblems are required, while retaining the decomposition cost in the absolute objectives.")
    text.extend(hardware_appendix)
    text.append(r"\FloatBarrier\section{Historical Hardware Report}")
    text.append(r"Version 1 reported an $n=5,K=3$ frozen warm-started QAOA update on Origin Wukong 180 (\texttt{WK\_C180}), with reported global change $-1.0536\times10^{-4}$. That target came from the historical partition. Its original raw hardware count archive was not included with the source provided for this revision. It is recorded here as v1 history and has not been independently revalidated. The new four-community records validate a separate corrected-partition frozen proposal/acceptance experiment; they do not retrospectively validate the old five-asset target or establish complete hardware execution of CAF.")
    manuscript = ROOT/"manuscript"
    notice = (r"\noindent\textcolor{red}{\textbf{WORKING DRAFT: experimental reruns incomplete.}}\par"
              if incomplete else "")
    (manuscript/"revision_numbers.tex").write_text(
        "\\newcommand{\\RevisionDraftNotice}{"+notice+"}\n")
    if len(groups["panel2018"]) == 20:
        panel = next(iter(output["groups"]["panel2018"].values()))
        if panel["objective_improvement_runs"] == 0:
            panel_sentence = r"No improving update is accepted on the additional 2018 panel ($N=1397$, 20 runs)."
        else:
            panel_sentence = f"On the additional 2018 panel ($N=1397$, capacity 100), mean paired gain is ${panel['objective_relative_improvement_mean_pct']:.2f}\\%$, with {panel['objective_improvement_runs']} of 20 trials improving."
    else:
        panel_sentence = "The additional 2018 panel reruns are in progress."
    macro_values = dict(MainGain=f"{main_stats['objective_relative_improvement_mean_pct']:.2f}",
        MainWins=str(main_stats['objective_improvement_runs']), MainTies=str(main_stats['objective_tied_runs']),
        MainCILow=f"{main_ci[0]:.2f}", MainCIHigh=f"{main_ci[1]:.2f}", MainBlockCount=str(len(main_sizes)))
    macro_values['GurobiRevisionSentence'] = f"The primary classical experiment uses Gurobi for both initialization and local refinement, with relative MIP-gap tolerance $10^{{-4}}$. On the fixed 2016 ($N=484$, 15 communities) and 2018 ($N=1397$, 25 communities) panels, CAF improves the objective by {g16['gain_pct']:.2f}\\% and {g18['gain_pct']:.2f}\\%, respectively. Repeated unshifted local solves leave both initial portfolios unchanged. Each panel contributes one fixed-configuration pipeline run, without statistical replication."
    macro_values['HardwareRevisionSentence'] = 'Four independent WK\\_C180 batches test frozen local proposals. Two eight-asset candidates meet the improvement criterion, but a uniform-sampling diagnostic shows that these passes do not establish a QAOA sampling benefit.'
    if controls:
        c16, c18 = controls['summary']['2016'], controls['summary']['2018']
        macro_values['ControlRevisionSentence'] = f"An injection ablation retains CAF's global verification and greedy sequential acceptance. Under matched SA sampling budgets, full CAF performs better in all {c16['caf_wins']} pairs on 2016, while CAF without injection performs better in all {c18['caf_losses']} pairs on 2018."
    else:
        macro_values['ControlRevisionSentence'] = 'Sampling-matched proposal controls are in progress.'
    with (manuscript/"revision_numbers.tex").open("a") as f:
        f.write("\\newcommand{\\PanelRevisionSentence}{"+panel_sentence+"}\n")
        for name,value in macro_values.items():
            f.write("\\newcommand{\\"+name+"}{"+value+"}\n")
    abstract = (manuscript/'main.tex').read_text().split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    for name, value in dict(macro_values, PanelRevisionSentence=panel_sentence).items():
        abstract = abstract.replace('\\'+name+'{}', value)
    assert not re.search(r'\\(?:Main\w+|PanelRevisionSentence|GurobiRevisionSentence)', abstract)
    abstract_path = manuscript/('arxiv_abstract_working.txt' if incomplete else 'arxiv_abstract.txt')
    abstract_path.write_text(' '.join(abstract.split())+'\n')
    (manuscript/"revision_results.tex").write_text("\n\n".join(text)+"\n")
    (ROOT/"docs/revision-results.json").write_text(json.dumps(output,indent=2)+"\n")
    plot_gurobi_main(gurobi_panels, manuscript/"figures")
    plot_flow(manuscript/"figures")
    archive = ROOT/"docs/revision-raw-results.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.name)
        z.write(main_path,"main_2016.json")
        for p in args.results.glob("*.log"):
            z.write(p,"logs/"+p.name)
        for p in args.results.glob("manifest*.json"):
            z.write(p,p.name)
        if args.sqa_results:
            for p in args.sqa_results.glob("manifest*.json"):
                z.write(p,p.name)
        z.write(args.frozen,"frozen.json")
        if controls:
            z.write(args.sampling_control,'sampling-control-results.json')
            z.write(ROOT/'manuscript/run_sampling_control.py','controls/run_sampling_control.py')
    report = ["# arXiv v2：容量约束 Newman 完整重跑", "",
        f"状态：{output['status']}；{len(paths)}/{len(tasks())} 个任务。经典容量 100，量子模拟容量 16。", "",
        "主实验已切换为 Gurobi：两面板各一次固定配置流程；初始化、CAF和无上下文重复求解均用Gurobi。原345项SA/模拟/诊断/规模任务和40项SA对照完整保留，不改写执行指纹。", "",
        "| Gurobi主面板 | 静态目标 | CAF目标 | 改进 | 无上下文对照 |", "|---|---:|---:|---:|---:|"]
    for year, stat in gurobi['summary'].items():
        report.append(f"| {year} | {stat['static_objective']:.9f} | {stat['caf_objective']:.9f} | {stat['gain_pct']:.6f}% | {stat['control_objective']:.9f} |")
    report.extend(["", f"全部{total_g}次局部求解达到指定MIPGap=1e-4，无超时；不构成完整投资组合的全局最优证书。独立完整矩阵重放全部候选、接受决策和历史，详见[Gurobi审计](gurobi-main-validation.json)及[原始记录](gurobi-main-raw-results.zip)。", "", "## SA敏感性实验（保留）", "",
        "| 面板 | 社区数 | 静态目标均值 | CAF 目标均值 | 平均配对改进 | 改进次数 |",
        "|---|---:|---:|---:|---:|---:|"])
    report.append(f"| 2016 | {len(main_sizes)} | {main_stats['static']['objective_mean']:.9f} | {main_stats['caf']['objective_mean']:.9f} | {main_stats['objective_relative_improvement_mean_pct']:.6f}% | {main_stats['objective_improvement_runs']}/20 |")
    if len(groups['panel2018']) == 20:
        stat=next(iter(output['groups']['panel2018'].values()))
        report.append(f"| 2018 | 25 | {stat['static']['objective_mean']:.9f} | {stat['caf']['objective_mean']:.9f} | {stat['objective_relative_improvement_mean_pct']:.6f}% | {stat['objective_improvement_runs']}/20 |")
    report.extend(['', '## 规模与稀疏持仓', '',
        '| 后端 | 设置 | 静态目标 | CAF 目标 | 平均配对改进 | 次数 |',
        '|---|---|---:|---:|---:|---:|'])
    for group, label in [('scaling_sa','SA'), ('sparsity_sa','SA'), ('gurobi','Gurobi')]:
        for key, stat in output['groups'][group].items():
            report.append(f"| {label} | {key} | {stat['static']['objective_mean']:.9f} | {stat['caf']['objective_mean']:.9f} | {stat['objective_relative_improvement_mean_pct']:.6f}% | {stat['total_runs']} |")
    if controls:
        report.extend(['', '## CAF 内部的上下文注入消融', '',
            '| 面板 | 无上下文目标 | CAF 目标 | CAF 额外改进（百分点） | 95% bootstrap 区间 | CAF 更优/持平/更差 |',
            '|---|---:|---:|---:|---|---|'])
        for year, stat in controls['summary'].items():
            lo, hi = stat['caf_advantage_ci95_percentage_points']
            report.append(f"| {year} | {stat['no_context_mean']:.9f} | {stat['caf_mean']:.9f} | {stat['caf_advantage_mean_percentage_points']:.6f} | [{lo:.6f}, {hi:.6f}] | {stat['caf_wins']}/{stat['ties']}/{stat['caf_losses']} |")
        report.extend(['', '对照是关闭上下文注入的CAF消融变体：保留CAF的全局目标验证和贪心顺序接受。保持相同初始组合、轮数、调用种子及退火采样预算。2018结果限定的是注入组件在该SA配置下的额外作用，不是静态分治胜过CAF，也不否定整个协调架构。共同接受组件的独立作用尚未消融。额外改进以共同静态目标为分母，不代表完全相同的运行时间。'])
    report.extend(['', '## q 扫描', '', '| q | 平均配对改进 | 改进/持平 |', '|---|---:|---|'])
    for key, stat in output['groups']['q_sweep'].items():
        report.append(f"| {key} | {stat['objective_relative_improvement_mean_pct']:.6f}% | {stat['objective_improvement_runs']}/{stat['objective_tied_runs']} |")
    report.extend(['', '各 q 共享本次 q=0.5 的初始静态持仓；CAF 随机种子与主面板不同。', '',
        '## 模拟兼容性', '', '| 后端 | 平均目标变化 | 标准差 | 改进/持平 |', '|---|---:|---:|---|'])
    for key in ('qaoa2','qaoa3','sqa'):
        stat=output['groups'].get(key)
        if stat:
            report.append(f"| {key} | {stat['absolute_delta_mean']:.9g} | {stat['absolute_delta_std']:.9g} | {stat['objective_improvement_runs']}/{stat['objective_tied_runs']} |")
    report.extend(['', 'QAOA 使用无噪声精确期望进行参数训练，4000 次采样用于最终解码；SQA 显式有限横场。均不构成量子优势证据。', '',
        '## 冻结消融与核验', '',
        f"冻结 n={frozen['n_local']}、K={frozen['k_local']}，无注入精确变化 {selected['dO_noinject']:.9g}，有注入 {selected['dO_inject']:.9g}，warm QAOA {selected['dO_sim']:.9g}，Hamming={selected['hamming']}。该例为 simulator-only。"])
    validation_path = ROOT/'docs/revision-validation.json'
    if validation_path.exists():
        check = json.loads(validation_path.read_text())
        if check.get('status') == 'passed' and check.get('validated_source_files') == output['source_files']:
            report.extend(['', '独立完整矩阵核验通过：`'+json.dumps(check['checks'],ensure_ascii=False)+'`。'])
    test_path = ROOT/'docs/arxiv-v2-tests.json'
    if test_path.exists():
        test = json.loads(test_path.read_text())
        if test['source_sha256'] == main_data['source_sha256']:
            report.append(f"自动检查：{test['passed']} 项通过，{test['expected_failures']} 项预期失败。")
    report.extend(["", "目标越低越好。完整的规模、稀疏、风险系数、Gurobi、SQA 与 QAOA 数据见机器可读汇总。", "",
        "自然 Newman 结果作为附录对照保留：2016 的最终目标接近容量方案，2018 自然划分的绝对目标更低。跨配置固定预算与采样工作量不同。相对收益不能替代绝对目标比较；100 资产单社区案例不存在跨社区协调。", "",
        "[完整汇总](revision-results.json) · [独立验证](revision-validation.json) · [原始记录及日志](revision-raw-results.zip) · [修订说明](ARXIV_REVISION.md)", "",
        "论文草稿和投稿材料由作者本地保管；本次代码发布不包含稿件TeX或PDF。新增真机记录见[独立核验报告](HARDWARE_RESULTS.md)。", "",
        "[冻结代码快照](CAF_v2_code_snapshot.zip) · [代码指纹](code-snapshot-sha256.json) · [采样对照原始结果](sampling-control-results.json)", "",
        "最终上传状态以执行记录为准；本地构建不代表已上传 arXiv。"])
    (ROOT/'docs/REVISION_RESULTS.md').write_text("\n".join(report)+"\n")
    write_report()
    with (ROOT/'docs/REVISION_RESULTS.md').open('a') as f:
        f.write('\n修订范围：主实验聚焦独立静态初始化与CAF精炼；BCD作为技术来源，最新上下文顺序更新定位为补充更新时序消融。正文简述24/24终值持平，完整容量/输入子集和计时结果移至附录，详见[完整补充报告](COORDINATION_STUDY.md)。未将按结果搜索的合成例子加入主论文，本轮不新增实验。\n')
    print(f"{len(paths)}/{len(tasks())} tasks; manuscript status: {output['status']}")


if __name__ == "__main__":
    main()
