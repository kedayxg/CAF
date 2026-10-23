"""Report the selected mechanism examples and the entire exploratory search."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    load=lambda n:json.loads((ROOT/'docs'/n).read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    d=load('factor-portfolio-search-results.json');v=load('factor-portfolio-validation.json')
    examples=load('factor-portfolio-examples.json')
    assert v['status']=='passed' and sha(ROOT/'docs/factor-portfolio-search-results.json')==v['results_sha256']
    assert d['counts']==v['counts']
    a=examples['caf_example'];e=a['example']
    gap=e['bcd_objective']-e['optimum'];headroom=e['initial_objective']-e['optimum']
    groups=[list(range(i,i+4)) for i in (0,4,8)]
    sigma=np.array(e['covariance']);mu=np.array(e['returns'])
    obj=lambda x:float(.5*x@sigma@x-mu@x)
    bcd=np.array(a['results']['gurobi']['bcd']['metrics']['weights'])
    caf=np.array(a['results']['gurobi']['caf']['metrics']['weights'])
    barriers=[]
    for g in groups:
        trial=bcd.copy();trial[g]=caf[g];barriers.append(obj(trial)-obj(bcd))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['caf_example','bcd_example'],
                           ['Selected CAF-favorable example','Selected reverse example']):
        case=examples[key];seed=case['seed'];example=case['example']
        for method,label,color in [('caf','CAF, frozen proposals','#146b8c'),
                                    ('bcd','BCD, latest context','#bd682b')]:
            trace=case['trace']['gurobi_'+method][:6]
            values=[example['initial_objective']]+[row['objective'] for row in trace]
            ax.plot(range(7),values,marker='o',label=label,color=color,linewidth=1.7)
        ax.axhline(example['optimum'],color='#444444',linestyle=':',label='Enumerated global minimum')
        ax.set(title=f'{title} (seed {seed})',xlabel='Local proposal evaluations',
               ylabel='Global objective (lower is better)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Synthetic factor portfolios: identical initialization, order and exact local solvers',fontsize=11)
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(ROOT/f'docs/factor-portfolio-trajectories.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# 合成因子组合任务：CAF与顺序BCD的不同吸引域','',
        '**这是按结果挑选的机制示例，不是替代现有真实面板基准的性能证据。**', '',
        '## 任务与控制条件','',
        '12个候选资产分成3个预先给定的投资组，每组必须持有2个资产。协方差由三个风险因子及正的特异风险生成：', '',
        '`Σ = 0.04 × (L Lᵀ / 3 + 0.2 I)`，`L`为12×3标准正态载荷；`μ_i`从[0.01,0.05]均匀生成。', '',
        '协方差严格正定（最小特征值约0.008）。两种方法共用独立局部最优静态初态、组顺序1→2→3、q=0.5及固定预算。各局部问题仅6个可行解，全球固定组预算下只有216个可行组合，可全部枚举。投资组是外部给定的约束，不声称由Newman发现。', '',
        '分别比较精确局部求解下的冻结提案CAF和最新上下文顺序BCD；两者都使用真实全局目标的贪心接受。搜索追踪到完整无改进轮，并另核对相同六轮预算，排除提早停止造成的假差异。', '',
        '## CAF较好的实例（固定规则选出的第一个，seed=31）','',
        '| 状态 | 全局目标 | 距枚举全局最优的差距 |','|---|---:|---:|',
        f"| 共同初态 | {e['initial_objective']:.9f} | {headroom:.9f} |",
        f"| 顺序BCD终态 | {e['bcd_objective']:.9f} | {gap:.9f} |",
        f"| CAF终态 | {e['caf_objective']:.9f} | 0 |",'',
        f"BCD留下约{gap/headroom*100:.2f}%的初始可改善空间。双方都在第一轮后不再接受更新，并且本地枚举验证各自终态的所有单组变化均不能改善；CAF到达全局最优，BCD到达另一个块局部最小。", '',
        '第一轮的分歧非常具体：', '',
        '| 步骤 | CAF目标 | BCD目标 |','|---|---:|---:|',
        f"| 初态 | {e['initial_objective']:.9f} | {e['initial_objective']:.9f} |"]
    for ci in range(3):
        x=a['trace']['gurobi_caf'][ci]['objective'];y=a['trace']['gurobi_bcd'][ci]['objective']
        lines.append(f'| 检查组{ci+1}后 | {x:.9f} | {y:.9f} |')
    lines+=['',
        '两者先接受相同的组1更新。BCD随后在组2取得约0.000968的小改善，却进入另一个局部盆地；CAF的冻结提案保留组2原选择，随后接受组3约0.011390的改善并到达全局最优。没有任何一步接受全局目标变差。', '',
        f"在BCD终态，单独把组2换成CAF终态的选择，目标变化为{barriers[1]:+.9f}；单独换组3为{barriers[2]:+.9f}；同时换组2和组3则变化为{-gap:+.9f}。这体现了组间对冲/协同：单组贪心停住，但两个组共同改变会更好。CAF通过不同的先行路径避开该盆地。", '',
        '## 完整搜索与反向实例','',
        '搜索固定5000个种子0–4999，保留全部结果。例子选择规则事先写入本次搜索协议：取第一个CAF到达全局最优、BCD残差至少0.001且至少占初始可改善空间20%、两者六轮内稳定的实例；同时按对称规则保留BCD更好的例子。搜索是为了发现机制，因此示例有明确的结果选择，不能当作随机基准子集。', '',
        f"- CAF更好：{d['counts']['caf_better']}/5000。",
        f"- BCD更好：{d['counts']['bcd_better']}/5000。",
        f"- 持平：{d['counts']['tied']}/5000。",
        f"- 到达枚举全局最优：CAF {d['counts']['caf_global']}/5000；BCD {d['counts']['bcd_global']}/5000。", '']
    reverse=examples['bcd_example']['example']
    lines += [f"对称规则找到的反向例子为seed48：CAF终值{reverse['caf_objective']:.9f}，BCD及全局最优{reverse['bcd_objective']:.9f}。因此不存在由这些例子支持的普遍优劣次序。", '',
        '## 独立核验','',
        '全部5000个实例都重新用12维完整矩阵评估候选、独立重放两种动态；共10000条动态和1080000个可行状态。两个挑选例子另外调用真实CAF核心、实际顺序BCD实现，并分别使用局部穷举与Gurobi求解，验证78次局部穷举及78次原生Gurobi求解。Gurobi使用MIPGap=0、MIPGapAbs=1e-12，保留所有状态、界和日志。', '',
        '## 使用边界','',
        '它证明：即使协方差正定、局部精确求解、同一初态及顺序，冻结提案也可能避开顺序BCD的一个局部盆地；反方向同样可能。它不能证明实际市场面板、所有任务或平均性能上的CAF优势。原有24个真实面板/子集配置终值持平的结果完整保留，当前主论文未因这个挑选例子改写。', '',
        '[轨迹图PDF](factor-portfolio-trajectories.pdf) · [轨迹图PNG](factor-portfolio-trajectories.png) · [完整搜索](factor-portfolio-search-results.json) · [例子/矩阵/轨迹](factor-portfolio-examples.json) · [验证](factor-portfolio-validation.json) · [原始记录和源码](factor-portfolio-raw-results.zip)']
    report=ROOT/'docs/FACTOR_PORTFOLIO_EXAMPLE.md';report.write_text('\n'.join(lines)+'\n')
    archive=ROOT/'docs/factor-portfolio-raw-results.zip'
    with zipfile.ZipFile(archive) as z:entries={n:z.read(n) for n in z.namelist()}
    entries['scripts/report_factor_portfolio.py']=Path(__file__).read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in entries.items():z.writestr(n,b)
    files=[report]+[ROOT/'docs'/name for name in ['factor-portfolio-search-results.json','factor-portfolio-examples.json','factor-portfolio-validation.json','factor-portfolio-raw-results.zip','factor-portfolio-trajectories.pdf','factor-portfolio-trajectories.png']]
    metadata=dict(files={p.name:sha(p) for p in files},main_paper_unchanged=True,
        classification='Postselected synthetic mechanism examples; all 5000 outcomes and reverse case retained.')
    (ROOT/'docs/factor-portfolio-artifact-sha256.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
