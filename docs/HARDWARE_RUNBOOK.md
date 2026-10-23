# 在服务器补充真机实验

`main` 是维护后的代码。原论文代码原样保留在 [`archive/arxiv-v1`](https://github.com/kedayxg/CAF/tree/archive/arxiv-v1)，提交为 `356c1852c65a185b05aed040052c5c3f750991bd`。已完成的经典与模拟实验对应提交 [`17253da`](https://github.com/kedayxg/CAF/tree/17253dab077d6f7cc77520d09908d0a3ddb90195)，源码指纹为 `9fc43f0fe1de39fcefb2dd021cdd99efab882ca98d37df1055a2669d305fe9d8`。后续真机入口维护不改写这些实验记录。

当前代码发布不包含待定的 v2 论文 TeX、PDF、投稿元数据或内部审稿记录。`manuscript/` 中发布的 Python 文件是实验、验证和生成工具；其中论文生成工具还需要作者另行提供的稿件源文件。

## 安装与检查

使用 Python 3.11，从仓库根目录运行。这里用独立虚拟环境安装，避免改动服务器已有环境：

```bash
git clone --branch main https://github.com/kedayxg/CAF.git
cd CAF
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-hardware.txt pytest
python -m pytest -q
```

pyqpanda3 0.5.0 已用于本次实际 CPUQVM 检查；真实芯片的连接、拓扑、权限和当前校准仍需在你的服务器核验。脚本默认不提交任务，只有 `--submit` 会发起真机执行。`--local` 与 `--submit` 互斥。

## 先离线确认社区和参数

以下两步不加载 pyqpanda3，也不需要云端凭据：

```bash
python -m experiments.quantum.run_hardware_qaoa \
  --list-communities --out-dir results/hardware/list

python -m experiments.quantum.run_hardware_qaoa \
  --community 3 --prepare-only --out-dir results/hardware/prepare
```

默认协议为 2016 数据中的 40 资产子集，subset seed=1234，static SA seed=42、200 reads，igraph Newman 容量 16，全局与局部 q=0.5，关闭风险重标定。社区依次为 **14、10、8、8**，预算为 **7、5、4、4**。索引从零开始；社区 **3** 对应修订稿现有冻结模拟例子的 8 资产目标。社区 2 是另一个 8 资产问题。修改种子、数据或划分后，应重新查看列表，不能沿用历史索引。

默认 warm-start QAOA 使用 p=2、6 次重启、每次最多 80 次 COBYLA 评估；优化阶段使用精确状态向量期望，最后以 4000 shots 解码。每次执行都会重新产生该次模拟参考；该次运行保存的参数会原样用于同次硬件电路。它不自动加载以前的准备 JSON。

可选 `--local-risk-rebalancing` 会改用历史局部代理目标，不属于默认对齐协议。`--max-community-size 0` 表示自然划分；所选目标仍必须满足本驱动的 16 资产状态向量参考上限。

## 用实际硬件 SDK 做本地检查

```bash
python -m experiments.quantum.run_hardware_qaoa \
  --community 3 --local --out-dir results/hardware/local
```

这一步使用 pyqpanda3 的 CPUQVM，不消耗真机额度。它检查相同参数下的硬件侧电路与 Qiskit 参考。输出中的 `mode=local`、`machine=CPUQVM(local)` 明确表示本地模拟，不能当作真机结果。

## 配置凭据、预检与提交

凭据只在服务器设置，不写入仓库。Bash 中可以隐藏输入：

```bash
read -rsp 'OriginQ API token: ' ORIGINQ_API_TOKEN
export ORIGINQ_API_TOKEN
export ORIGINQ_MACHINE=WK_C180
```

也支持旧的本地 `experiments/config.py`；该文件受 Git 忽略。环境变量令牌优先。未指定 `--machine` 时使用配置中的机器名或环境默认值。

先进行云端编译预检（会访问拓扑／配置，不提交量子任务）：

```bash
python -m experiments.quantum.run_hardware_qaoa \
  --community 3 --machine WK_C180 --out-dir results/hardware/dryrun
```

再明确提交：

```bash
python -m experiments.quantum.run_hardware_qaoa \
  --community 3 --machine WK_C180 --submit \
  --shots 4000 --poll-interval 5 --max-polls 1440 \
  --out-dir results/hardware/qpu
```

如果需要限定物理比特，使用服务器上确认可用的 `--physical-qubits` 列表；列表至少包含目标所需的不同非负编号，可结合 `--layout-method fidelity`。不要直接复用旧论文的物理映射。

## 保留和检查结果

每次运行生成一个带微秒时间戳的 JSON。准备完成时先落盘；任务返回编号后立即更新 `job_receipt.job_id`，轮询期间持续保存状态。超时不取消任务，也不会自动重新提交；保留编号并先在平台确认旧任务状态，避免重复计费。SDK 任务结果读取或验证失败时，准备信息与已有任务编号仍保留。

保存内容包括输入矩阵及哈希、社区与预算、初始持仓、局部模型、变量顺序、warm-start 角度、theta、软件版本、Git 提交及源码指纹、模拟候选，以及硬件原始计数、状态、全局目标变化和接受判定。

量子执行要求任务成功结束，并从 `probCount` 通道取得非负整数计数、总数等于请求 shots。关闭读出修正；不以平台概率替代原始计数，不把零计数或修复后的未采样向量当成真机候选。没有可行样本时，候选和目标变化记为 null。保留失败或未通过阈值的记录。

`criterion_pass` 检查模拟和硬件候选是否都满足全局接受阈值，并且改善比至少为 `--tau`（默认 0.7）。它是单个冻结局部更新的检验，不代表完整 CAF 管线真机运行、量子优势或全局最优。

建议带回完整 JSON 和命令行日志，随后核验并将真实结果加入论文；当前代码发布没有新 QPU 结果。
