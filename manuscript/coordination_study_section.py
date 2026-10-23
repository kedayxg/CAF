"""Render the validated update-timing ablation and sensitivity appendix."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load_study():
    path=ROOT/'docs/coordination-study-results.json'
    validation_path=ROOT/'docs/coordination-study-validation.json'
    d=json.loads(path.read_text());v=json.loads(validation_path.read_text())
    assert d['status']=='complete' and v['status']=='passed'
    assert d['source_files']==v['validated_source_files']
    assert d['validation_sha256']==hashlib.sha256(validation_path.read_bytes()).hexdigest()
    assert v['summary_sha256']==hashlib.sha256(json.dumps(d['cases'],sort_keys=True).encode()).hexdigest()
    return d,v


def outcome_counts(rows):
    delta=np.array([r['bcd_minus_caf_objective'] for r in rows])
    return int(np.sum(delta < -1e-9)),int(np.sum(np.abs(delta)<=1e-9)),int(np.sum(delta > 1e-9))


def render_section():
    d,v=load_study();rows=d['cases'];failures=d['partition_failures']
    bcd_better,tied,caf_better=outcome_counts(rows)
    parallel_equal=sum(abs(r['parallel_serial_objective_delta'])<=1e-9 for r in rows)
    bcd_fewer_rounds=sum(r['methods']['sequential_bcd']['first_final_round']<r['methods']['caf_serial']['first_final_round'] for r in rows)
    bcd_equal_rounds=sum(r['methods']['sequential_bcd']['first_final_round']==r['methods']['caf_serial']['first_final_round'] for r in rows)
    parallel_faster=sum(r['methods']['caf_parallel4']['wall_seconds']<r['methods']['caf_serial']['wall_seconds'] for r in rows)
    # Identical input and community memberships across capacities are duplicate
    # optimization configurations, not independent examples of robustness.
    import zipfile
    distinct=set()
    with zipfile.ZipFile(ROOT/'docs/coordination-study-raw-results.zip') as z:
        for r in rows:
            from run_coordination_study import case_id
            raw=json.loads(z.read(case_id(r['case'])+'.json'))
            distinct.add((r['case']['year'],tuple(raw['input_indices']),
                          tuple(map(tuple,raw['communities'])),tuple(raw['cards'])))
    text=[r"\section{Update-Timing Ablation and Capacity Sensitivity}\label{app:update-timing}",
        r"BCD provides one of CAF's technical foundations. This supplementary update-timing ablation refreshes the external context before each community solve, giving cyclic discrete block-coordinate descent (BCD), while retaining the same partition, fixed budgets, Gurobi static portfolio, local solver parameters, cached exact increment and greedy acceptance threshold. CAF instead freezes all proposals within a sweep. The variant isolates context-refresh timing within the shared coordination implementation. Both receive six complete sweeps with patience disabled, giving the same number of native local solves. We additionally run CAF with four concurrent one-thread solver environments; BCD and serial CAF use one. This examines scheduling under explicit resource allocations, not equal peak CPU resources.",
        r"Before the full batch we fixed capacities 75, 100 and 150 on both full panels and on three 300-asset subsets per panel (subset seeds 1729, 2718 and 3141), totaling 24 declared configurations. Subsets overlap and are not independent market samples. Cases run serially and method order rotates by case index, giving one fixed order per capacity; each timing is one measurement, not a counterbalanced timing study. A preflight led only to aligned timers including method setup and quieter console logging, with no change to cases, budgets or solver settings. All unsplittable Newman failures are retained without forced splitting or replacement.",
        f"The batch has {len(rows)} feasible configurations and {len(failures)} partition failures, with {len(distinct)} distinct input/partition/budget configurations. At the end of six sweeps, BCD has a lower objective in {bcd_better}, matches CAF within $10^{{-9}}$ in {tied}, and CAF has a lower objective in {caf_better}. Four-worker and serial CAF match final objectives in {parallel_equal}/{len(rows)} feasible configurations. Equality is an observation on these instances, not a theorem about the two search trajectories.",
        r"\begin{table}[t]\caption{Full-panel capacity sensitivity, six matched sweeps. Dashes denote an unsplittable partition. Lower objectives are better.}\label{tab:capacity-bcd}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrr}\toprule Panel & Cap & Blocks & Static & CAF & BCD\\\midrule"]
    for year in (2016,2018):
        for cap in (75,100,150):
            r=next((r for r in rows if r['case']['year']==year and r['case']['subset_seed'] is None and r['case']['capacity']==cap),None)
            if r:
                text.append(f"{year} & {cap} & {r['community_count']} & {r['static_objective']:.6f} & {r['methods']['caf_serial']['objective']:.6f} & {r['methods']['sequential_bcd']['objective']:.6f}"+r"\\")
            else:text.append(f"{year} & {cap} & -- & -- & -- & --"+r"\\")
    text.extend([r"\bottomrule\end{tabular}}\end{table}",
        r"\begin{table}[t]\caption{300-asset subset sensitivity. Each entry uses three fixed subset seeds; mean gains are descriptive. Counts compare final BCD objective to CAF (better/tied/worse).}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrrl}\toprule Panel & Cap & Solved & CAF gain (\%) & BCD gain (\%) & BCD comparison\\\midrule"])
    for year in (2016,2018):
        for cap in (75,100,150):
            selected=[r for r in rows if r['case']['year']==year and r['case']['subset_seed'] is not None and r['case']['capacity']==cap]
            if selected:
                a,b,c=outcome_counts(selected)
                text.append(f"{year} & {cap} & {len(selected)}/3 & {np.mean([r['methods']['caf_serial']['gain_pct'] for r in selected]):.3f} & {np.mean([r['methods']['sequential_bcd']['gain_pct'] for r in selected]):.3f} & {a}/{b}/{c}"+r"\\")
            else:text.append(f"{year} & {cap} & 0/3 & -- & -- & --"+r"\\")
    text.extend([r"\bottomrule\end{tabular}}\end{table}",
        r"Changing capacity changes memberships, budgets and static initialization, so cross-capacity percentages are not improvements over a common feasible optimization problem. Full objectives are reported alongside gains, and repeated partitions are counted explicitly. A larger relative gain need not mean a better final objective.",
        r"\begin{table}[t]\caption{Full-panel capacity-100 resource comparison. One timing per method, excluding shared partitioning/static initialization. Round numbers indicate first reaching each method's own six-sweep endpoint, not an optimality certificate.}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{llrrr}\toprule Panel & Method & Calls & Time (s) & Endpoint round\\\midrule"])
    for year in (2016,2018):
        r=next((r for r in rows if r['case']['year']==year and r['case']['subset_seed'] is None and r['case']['capacity']==100),None)
        if r:
            for method,label in [('caf_serial','CAF, 1 worker'),('sequential_bcd','BCD, 1 worker'),('caf_parallel4','CAF, 4 workers')]:
                m=r['methods'][method]
                text.append(f"{year} & {label} & {m['calls']} & {m['wall_seconds']:.3f} & {m['first_final_round']}"+r"\\")
    text.extend([r"\bottomrule\end{tabular}}\end{table}",
        f"BCD reaches its six-sweep endpoint in fewer rounds than CAF in {bcd_fewer_rounds}/{len(rows)} configurations and the same number in {bcd_equal_rounds}/{len(rows)}. Four-worker CAF is faster than serial CAF in {parallel_faster}/{len(rows)} single measurements and is not faster in the remaining {len(rows)-parallel_faster}; this batch does not show a consistent local CPU speedup. Timings include model construction and coordinator overhead. Potential QPU/remote-backend latency savings would require a separate measurement; local CPU timings do not establish hardware queueing benefits.",
        f"Independent full-matrix replay verifies {v['checks'].get('verified_local_solves',0)} local solves, {v['checks'].get('verified_proposals',0)} proposals and {v['checks'].get('verified_histories',0)} histories. Local status counts are {v['checks'].get('optimal_local_solves',0)} optimal within requested tolerance and {v['checks'].get('time_limited_local_solves',0)} time-limited. These remain local-model certificates, not full-portfolio optimality certificates."])
    return text,dict(results_sha256=hashlib.sha256((ROOT/'docs/coordination-study-results.json').read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256((ROOT/'docs/coordination-study-validation.json').read_bytes()).hexdigest(),
        complete_cases=len(rows),partition_failures=len(failures),distinct_configurations=len(distinct),
        bcd_better=bcd_better,tied=tied,caf_better=caf_better,parallel_objective_matches=parallel_equal,
        bcd_fewer_endpoint_rounds=bcd_fewer_rounds,bcd_equal_endpoint_rounds=bcd_equal_rounds,
        parallel_faster_single_timings=parallel_faster)


def write_report():
    d,v=load_study();rows=d['cases'];counts=outcome_counts(rows)
    bcd_fewer=sum(r['methods']['sequential_bcd']['first_final_round']<r['methods']['caf_serial']['first_final_round'] for r in rows)
    faster=sum(r['methods']['caf_parallel4']['wall_seconds']<r['methods']['caf_serial']['wall_seconds'] for r in rows)
    lines=['# CAF 更新时序消融与容量/输入敏感性补充实验','',
        f"24个预先声明配置；{len(rows)}个可行、{len(d['partition_failures'])}个划分失败。核心源码未改动。",'',
        '每配置共享Gurobi静态初始化、社区和固定预算；六轮完整扫块，关闭patience，固定求解器参数。BCD每次更新前使用最新上下文；CAF在轮内冻结提案。串行CAF/BCD各一个求解工作线程，另测四工作线程CAF。每设置一次时间测量，不构成速度统计检验。', '',
        f"六轮终值比较：BCD更优/持平/CAF更优 = {counts[0]}/{counts[1]}/{counts[2]}（容差1e-9）。",'',
        f"BCD在{bcd_fewer}/24个配置中用更少轮数达到终值，其余配置轮数相同。四工作线程CAF仅在{faster}/24次单次测量中更快，本轮没有显示稳定的本地CPU加速；不预设冻结提案优于顺序更新。",'',
        '| 面板/输入 | 容量 | 社区数 | 静态目标 | CAF目标 | BCD目标 | CAF改进 | BCD改进 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        c=r['case'];sample='全面板' if c['subset_seed'] is None else f"N300/seed{c['subset_seed']}"
        a,b=r['methods']['caf_serial'],r['methods']['sequential_bcd']
        lines.append(f"| {c['year']}/{sample} | {c['capacity']} | {r['community_count']} | {r['static_objective']:.9f} | {a['objective']:.9f} | {b['objective']:.9f} | {a['gain_pct']:.4f}% | {b['gain_pct']:.4f}% |")
    lines+=['','## 所有实际成本','',
        '| 配置 | CAF串行秒 | BCD串行秒 | CAF四工作线程秒 | 每种方法调用数 | CAF/BCD达到各自终值轮数 |',
        '|---|---:|---:|---:|---:|---|']
    from run_coordination_study import case_id
    for r in rows:
        a,b,c=[r['methods'][k] for k in ['caf_serial','sequential_bcd','caf_parallel4']]
        lines.append(f"| {case_id(r['case'])} | {a['wall_seconds']:.4f} | {b['wall_seconds']:.4f} | {c['wall_seconds']:.4f} | {a['calls']} | {a['first_final_round']}/{b['first_final_round']} |")
    lines+=['','时间含各方法自身初始化、缓存、求解与协调，排除共同RMT/划分及静态初始化。四工作线程使用更多并发求解资源。每配置仅一次时间测量，方法顺序按配置轮换、在同一容量内固定，并非平衡重复的时间研究；不能据此泛化CPU或QPU加速比。', '',
        '容量可能产生相同分区，也可能改变预算可行域；各容量必须联合阅读绝对目标和改进率，不能把更大的相对改进当作更好的最终组合。随机子集有重叠，是固定历史面板内的敏感性，不是独立市场样本。', '',
        '## 划分失败（保留）','']
    if d['partition_failures']:
        for f in d['partition_failures']:lines.append(f"- {case_id(f['case'])}: {f['reason']}")
    else:lines.append('无；全部声明配置成功满足容量。')
    lines+=['','## 独立验证','',json.dumps(v['checks'],ensure_ascii=False),'',
        '[协议](coordination-study-protocol.json) · [完整汇总](coordination-study-results.json) · [独立验证](coordination-study-validation.json) · [原始记录与日志](coordination-study-raw-results.zip)', '',
        '最新上下文顺序更新沿用CAF的划分、预算、局部求解器和接受组件；在论文中定位为更新时序消融，详细结果列于附录，不作为主实验的独立竞争算法排名。BCD仍作为技术来源之一引用。完整结果、计时限制和SA注入消融均保留。']
    (ROOT/'docs/COORDINATION_STUDY.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    write_report()
