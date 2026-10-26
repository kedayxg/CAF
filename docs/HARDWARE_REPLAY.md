# 冻结参数重放与历史指标重算

维护入口为 `python -m experiments.quantum.replay_hardware_qaoa`，位于 `v2-development`。它直接读取原硬件驱动生成的完整 JSON，保持初始组合、局部模型和 warm-start 不变，不重新优化 QAOA。原 `run_hardware_qaoa.py` 的优化流程及默认设置保持原样；`main` 继续保留 v1。

这是对服务器临时重放脚本的正式维护：修复 p=1 参数截断、统一候选计数位序，保存实际运行源码，并提供不改写历史记录的分析入口。研究范围仍为固定社区配额。

## 参数含义

原 p=2 的参数顺序是 `[gamma0, gamma1, beta0, beta1]`。选择层数时先按**原始深度**分开两半：

| 选择 | 实际演化参数 |
| --- | --- |
| p=0 | 空；只制备 warm-start 初态 |
| p=1 | `[gamma0, beta0]` |
| p=2 | 原完整向量 |

超过保存深度、负层数、非有限或长度错误的参数在初始化 SDK 前拒绝。p=1 是截断原电路的诊断，不是重新优化后的 p=1 算法。其理想分布也重新计算，不沿用 p=2 理想概率。保存的原模拟候选继续作为冻结的金融目标/接受判据参考；截断电路可能对这个候选赋予不同概率。

当前社区 3 的该候选理想概率分别约为 p=0：43.6033%、p=1：43.6904%、p=2：45.3430%。这些是理想计算，不是真机结果。p=0 是对照选项，没有被替换成正式实验默认值。

## 查看计划与本地验证

在已安装 `requirements-hardware.txt` 的 Python 3.11 环境、仓库根目录运行。输入文件来自原始冻结硬件任务；示例使用服务器已有的社区 3 记录：

```bash
CAF_REPLAY_RECORD=results/hardware/qpu/HW_qaoa_submit_20261023_091934_007188Z.json

python -m experiments.quantum.replay_hardware_qaoa \
  --record "$CAF_REPLAY_RECORD" --p 1

python -m experiments.quantum.replay_hardware_qaoa \
  --record "$CAF_REPLAY_RECORD" --p 1 --local
```

不加执行标志只生成离线计划，不加载硬件 SDK、不访问云端。`--local` 使用实际 pyqpanda3 CPUQVM 和构造的零校准全连接芯片，检查编译及序列化脉冲的理想演化；本地编号固定为 `0..n-1`，不能用于验证真实芯片噪声或云端位序。输入上限为 16 个局部资产。

## 真实芯片预检与提交

沿用原手册的服务器凭据配置。提交前先完成[通道筛查](HARDWARE_SCREEN.md)，设置 `CAF_SCREEN_BLOCK` 和 `CAF_SCREEN_DIR`。不再将历史异常的 110/111 通道作为推荐块。缺少同机、同物理集合、最近四小时内通过的真机筛查记录时，维护入口会拒绝 `--submit`。

```bash
: "${CAF_SCREEN_BLOCK:?请先设置已筛查的物理编号列表}"
: "${CAF_SCREEN_DIR:?请先设置真机筛查目录}"

python -m experiments.quantum.replay_hardware_qaoa \
  --record "$CAF_REPLAY_RECORD" --p 1 --compile-only \
  --physical-qubits "$CAF_SCREEN_BLOCK"

python -m experiments.quantum.replay_hardware_qaoa \
  --record "$CAF_REPLAY_RECORD" --p 1 --submit --shots 4000 \
  --physical-qubits "$CAF_SCREEN_BLOCK" --screen-dir "$CAF_SCREEN_DIR"
```

`--compile-only` 读取真实配置而不提交任务；只有 `--submit` 会发起一个硬件任务。不自动运行 p=0/1/2 全套，也不自动重试或恢复提交。失败/超时先查询已保存的 job ID。

默认优化级别为 2，可显式选择 `--optimization-level 0/1/2`。固定 X1/RZ/CZ 路径在转译阶段使用零校准角，再由指令转换器应用一次真实校准。cloud mapping/optimization 及读出修正均关闭；这些设置与原正式驱动可能不同，结果应作为独立批次报告。

提交前验证未校准电路与独立 Qiskit 理想分布一致，并检查实际测量列表及物理比特范围。两阶段校准与元数据记录沿用[诊断说明](HARDWARE_DIAGNOSTICS.md)。SDK 指令回读不用于自动修正测量位序。

## 产物与可追溯性

每次运行在 `results/hardware_replay/<UTC时间戳>/` 保存：

- `source_record.json`：输入记录的原始字节及 SHA256；实际参数与原始参数分开记录。
- `source_code/`：重放、诊断、筛查、原驱动实际 Python 文件及两份 requirements；不复制本地凭据配置。
- `screen_evidence/`：实际用于提交前检查的真机筛查证据副本。
- `chip_backend.json`：当前芯片配置、校准角、拓扑及可用保真度。
- 逻辑/编译 OriginIR、实际提交 instruction 原文与哈希、实际测量映射、CZ 数和深度。
- `record.json`：运行模式、版本、job ID、状态轨迹、原始/规范化 counts、可获取的平台元数据和重新计算的指标。
- `analysis.json`：按原始金融矩阵与计数重算的结果、输入哈希和分析器版本。
- `SHA256.json`：本批产物哈希。

位序约定明确为右侧 c0 对应源记录的局部位置 0。因此原模拟候选 `[1,0,1,1,0,0,0,1]` 的候选键是 `10001101`。只从真实非零计数中选可行解；没有可行样本时不伪造候选。每次全局目标增量仍按真实金融目标计算，沿用 `1e-5` 接受阈值及默认比例阈值 `tau=0.7`。

## 离线重算历史结果

新入口自己的批次已包含原始输入：

```bash
python -m experiments.quantum.replay_hardware_qaoa \
  --analyze results/hardware_replay/实际时间戳
```

对服务器旧脚本的结果目录，显式提供对应原始源记录：

```bash
python -m experiments.quantum.replay_hardware_qaoa \
  --analyze results/hardware_diagnostics/c3_rerun_20261025T085328_606800Z \
  --record "$CAF_REPLAY_RECORD"
```

分析先核对源记录哈希、完成状态、原始计数和总 shots，再计算候选命中、完整分布 TV、最好可行样本及全局接受。仅写新的 `analysis.json`，不覆盖旧 `record.json`、旧 metrics、counts 或历史哈希清单。旧 metrics 与重算不符时列出差异。无法确定实际截断参数的旧 p=1 记录会被拒绝，不会悄悄按修正后的参数解释旧实验。

新增/修改文件会改变当前源码指纹。历史任务保留其真实版本和原始文件；本次维护不把旧任务重新标为新代码运行，也没有新增真机任务。[维护验证](hardware-replay-validation.json)区分了本地 CPUQVM 检查与历史真机文件重算。

原四社区强源码绑定审计 `manuscript/validate_hardware_results.py` 要求检出与记录相符的历史源码，例如提交 `1f3dcfc08915e5287045b0f324ef96334b7a3953`；它不会把新增诊断文件后的当前源码指纹当作旧运行指纹。新的 `--analyze` 提供源记录哈希和数值重算，不能替代那项完整历史源码审计，也不改写其既有验证证书。
