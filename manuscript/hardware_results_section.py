"""Render the independently audited frozen hardware study; no new executions."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_hardware():
    path=ROOT/'docs/hardware-results.json';validation_path=ROOT/'docs/hardware-validation.json'
    result=json.loads(path.read_text());validation=json.loads(validation_path.read_text())
    assert result['status']=='complete' and validation['status']=='passed'
    assert result['validation_sha256']==sha(validation_path)
    assert result['raw_archive_sha256']==sha(ROOT/'docs/hardware-raw-results.zip')
    assert result['qpu_records']==validation['qpu_records']
    assert validation['validator_sha256']==sha(ROOT/'manuscript/validate_hardware_results.py')
    with zipfile.ZipFile(ROOT/'docs/hardware-raw-results.zip') as archive:
        assert archive.testzip() is None
        for name,digest in validation['validated_source_files'].items():
            assert hashlib.sha256(archive.read('results/hardware/'+name)).hexdigest()==digest
    return result,validation


def render_hardware():
    d,v=load_hardware();rows=d['qpu_records']
    main=[r"\subsection{Frozen Local Updates on WK\_C180}",
        r"We additionally execute the four corrected communities as independent proposals from the same frozen $N=40,K=20$ iteration-0 portfolio. Each supplied record contains one WK\_C180 batch of 4,000 raw shots, with readout amendment disabled. Warm-start QAOA uses $p=2$, six restarts and at most 80 COBYLA evaluations, followed by the same recorded parameters on hardware. Local and global $q=0.5$. The hardware driver and its input hashes are archived separately from the earlier simulator batch. This is not a sequential hardware CAF outer loop.",
        r"The declared criterion requires accepted simulator and hardware candidates and $\Delta\mathcal O_{\rm hw}/\Delta\mathcal O_{\rm sim}\ge0.7$. We retain it without retuning and report all four communities. Communities 2 and 3 meet it; community 0's worsening proposal is rejected, while community 1 has no improving simulator or hardware update. The two accepted deltas are measured against the same frozen state and are not added as an outer-loop gain.",
        r"\begin{table}[t]\caption{Independent frozen hardware proposals, one 4,000-shot batch per community. Deltas are in units of $10^{-4}$. The original criterion requires improvement; community 1 is a no-improvement case.}\label{tab:hardware}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{crrrrl}\toprule Community & $n/K$ & $\Delta_{\rm sim}$ & $\Delta_{\rm hw}$ & Hamming & Criterion\\\midrule"]
    for row in rows:
        label='Pass' if row['criterion_pass'] else 'No gain' if not row['accepted_sim'] else 'Fail'
        delta_sim=0. if abs(row['dO_sim'])<1e-12 else row['dO_sim']*1e4
        delta_hw=0. if abs(row['dO_hw'])<1e-12 else row['dO_hw']*1e4
        main.append(f"{row['community']} & {row['n']}/{row['K']} & {delta_sim:.3f} & {delta_hw:.3f} & {row['candidate_hamming']} & {label}"+r"\\")
    main.extend([r"\bottomrule\end{tabular}}\end{table}",
        r"Best-of-sample agreement does not establish that the hardware preserves the intended QAOA distribution. Direct reconstruction from saved parameters gives ideal candidate probabilities 0.2218, approximately 1, 0.7772 and 0.4534; the hardware observes these candidates 0, 2, 21 and 14 times, respectively. As a post-hoc diagnostic, uniform sampling of 4,000 eight-bit strings hits any specified string with probability $1-(255/256)^{4000}\approx0.999999841$. The simulated candidates in the two eight-asset cases are conditional optima, so even this uninformed sampler almost always meets the same criterion. Thus these batches demonstrate feasible-candidate decoding and guarded acceptance from hardware output, without demonstrating a QAOA sampling benefit. Appendix~\ref{app:hardware-diagnostics} gives the complete distribution audit and its limits."])
    appendix=[r"\FloatBarrier\section{Frozen Hardware Distribution Audit}\label{app:hardware-diagnostics}",
        r"An independent replay checks all eight simulator/CPUQVM and QPU records, input matrix hashes, community membership, fixed budgets, penalized Ising coefficients, raw platform counts and full-matrix financial increments. All four archived QPU receipts report FINISHED, with 4,000 raw counts each. The ideal distributions are reconstructed directly from saved $\theta^\star$ and warm-start angles, without reoptimizing. Replaying those same parameters in the actual pyqpanda3 0.5.0 CPUQVM agrees with the Qiskit expectations within $10^{-11}$; this checks the pre-cloud circuit, not the submitted cloud compilation.",
        r"\begin{table}[t]\caption{Saved-parameter distribution audit. Hits count the recorded ideal candidate in 4,000 hardware shots; TV is total variation distance between empirical hardware and ideal probabilities. $P_{\rm unif}$ is the exact probability that uniform best-of-4,000 sampling meets the original improvement criterion.}\label{tab:hardware-distribution}\centering\resizebox{\columnwidth}{!}{\begin{tabular}{crrrr}\toprule Community & $p_{\rm ideal}$ & HW hits & TV & $P_{\rm unif}$\\\midrule"]
    for row in rows:
        appendix.append(f"{row['community']} & {row['ideal_candidate_probability']:.4f} & {row['observed_ideal_candidate_hits']} & {row['tv_hardware_ideal']:.4f} & {row['uniform_criterion_pass_probability']:.7f}"+r"\\")
    appendix.extend([r"\bottomrule\end{tabular}}\end{table}",
        r"At community 0 the ideal candidate would have about 887 hits in 4,000 ideal shots, versus zero observed. Finite sampling of that ideal distribution alone is therefore an inadequate explanation. The hardware distributions differ substantially from the ideal distributions in all four cases. Uniform controls are diagnostic and post hoc; a 2,000/2,000 success count in the supplied Monte Carlo diagnostic is not a probability-one guarantee. We use exact finite-shot event probabilities above.",
        r"Only one QPU batch per community is available. The archived platform metadata has null QProg entries and no submitted circuit/mapping trace, so these data do not isolate hardware noise from cloud compilation or mapping effects. They also do not establish full-distribution equality with a uniform sampler. Task status is checked against archived receipts, not a live account query. The accepted candidates improve the stated financial objective, but no sampling advantage, ideal-distribution transfer, repeated-run reliability or complete hardware CAF execution is established."])
    binding=dict(results_sha256=sha(ROOT/'docs/hardware-results.json'),validation_sha256=sha(ROOT/'docs/hardware-validation.json'),
        raw_archive_sha256=sha(ROOT/'docs/hardware-raw-results.zip'),source_sha256=v['source_sha256'],
        qpu_batches=4,raw_shots=16000,criterion_passes=v['criterion_passes'],sdk_saved_theta_replayed=v['sdk_saved_theta_replayed'])
    return main,appendix,binding


def write_report():
    validation_path=ROOT/'docs/hardware-validation.json';v=json.loads(validation_path.read_text())
    assert v['status']=='passed'
    assert v['sdk_saved_theta_replayed'], 'Final hardware report requires the full --sdk-local audit'
    d=dict(status='complete',validation_sha256=sha(validation_path),
        raw_archive_sha256=sha(ROOT/'docs/hardware-raw-results.zip'),source_sha256=v['source_sha256'],
        protocol=v['protocol'],qpu_records=v['qpu_records'],local_records=v['local_records'],
        scope=v['scope'],limitations=v['limitations'])
    (ROOT/'docs/hardware-results.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    lines=['# 新版冻结局部更新：WK_C180 真机结果与独立核验','',
        '四个社区的原始计数、候选、全局目标增量和接受判据均通过独立重放。社区2、3通过既定判据；社区0的变差提案被拒绝，社区1没有改善。结果支持候选解码和接受链路的可执行性，尚未显示QAOA采样优势。','',
        '| 社区 | n/K | 模拟ΔO | 硬件候选ΔO | 接受后ΔO | Hamming | 判据 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in v['qpu_records']:
        label='通过' if r['criterion_pass'] else '无改善' if not r['accepted_sim'] else '未通过，候选拒绝'
        lines.append(f"| {r['community']} | {r['n']}/{r['K']} | {r['dO_sim']:.12f} | {r['dO_hw']:.12f} | {r['post_acceptance_delta']:.12f} | {r['candidate_hamming']} | {label} |")
    lines+=['','四个目标来自同一冻结初态的独立实验；两个接受增量不能相加作为一轮CAF的收益。每个社区仅一批4000 shots，不是重复可靠性估计。','',
        '## 分布与判据的区分度','',
        '| 社区 | 理想候选概率 | 理想期望次数 | 硬件次数 | TV(硬件,理想) | 均匀采样通过原判据的精确概率 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in v['qpu_records']:
        lines.append(f"| {r['community']} | {r['ideal_candidate_probability']:.9f} | {r['expected_ideal_candidate_hits']:.3f} | {r['observed_ideal_candidate_hits']} | {r['tv_hardware_ideal']:.6f} | {r['uniform_criterion_pass_probability']:.9f} |")
    lines+=['',
        '理想分布从每条真机记录保存的theta和warm-start角度直接重建，未重新训练或优化。8比特均匀采样4000次命中指定比特串的概率为1−(255/256)^4000≈0.999999841，而不是严格100%。两个8资产模拟候选都是对应固定上下文下的精确条件最优，因此此判据对无信息采样也几乎必过。原始2000/2000蒙特卡洛结果仅是有限次诊断，不能称为必然。','',
        '社区0理想期望命中约887次、实际0次，不能仅归因为shots不足。四个硬件分布均明显偏离相应理想分布；它们与均匀分布也并不相同。实际SDK的CPUQVM按保存参数重放能量与Qiskit一致，但云端QProg为空，缺少提交后的编译电路和映射轨迹，因此不能唯一归因于硬件噪声或排除云端编译/映射问题。','',
        '均匀对照是事后分布诊断，不是事先注册的算法排名，也不是运行在芯片上的额外实验。未增加QPU任务、调整shots或改变既定接受判据。','',
        '## 核验范围','',
        f"- 核对{v['manifest_verified_files']}个公开原始文件指纹，以及4个QPU与4个CPUQVM完整记录。",
        f"- 平台原始probCount、任务回执与规范化计数逐项一致，合计16000 QPU shots。",
        f"- 从完整矩阵重算目标，并检查全部{v['conditional_feasible_states']}个条件可行状态的金融能量和Ising能量一致。",
        '- 重新构建社区和预算，检查原始样本可行性、最佳可行样本、接受决策、硬件能量和Hamming距离。',
        '- 使用实际pyqpanda3 0.5.0在CPUQVM重放四组保存参数，无云端请求。',
        '- 任务成功状态来自提供的原始平台回执；本次没有登录平台独立查询任务。','',
        '| 社区 | 完整 job_id |','|---|---|']
    for r in v['qpu_records']:lines.append(f"| {r['community']} | `{r['job_id']}` |")
    lines+=['','## 复现审计','',
        '```bash','unzip docs/hardware-raw-results.zip',
        'python manuscript/validate_hardware_results.py --results results/hardware --sdk-local',
        'python manuscript/hardware_results_section.py','```','',
        '不加--sdk-local可在没有pyqpanda3时重放数学、计数和理想分布检查；发布时的完整核验包含该选项。源码指纹b12e321...对应服务器运行版本，与此前经典/模拟批次9fc43f0...分别保留。','',
        '原上传压缩包及作者的论文笔记留在本地。公开归档保留原始数值、日志及服务器诊断脚本，排除作者稿件笔记和未修订的叙述报告，另生成公开文件指纹清单。','',
        '[汇总](hardware-results.json) · [独立核验](hardware-validation.json) · [原始记录](hardware-raw-results.zip) · [真机操作指南](HARDWARE_RUNBOOK.md)']
    (ROOT/'docs/HARDWARE_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    write_report()
