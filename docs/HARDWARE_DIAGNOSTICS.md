# 两比特真机诊断

入口：`python -m experiments.quantum.diagnose_hardware`。使用现有 Python 3.11 / pyqpanda3 0.5.0 环境，在仓库根目录运行。这里不重新优化 QAOA，不改变四社区冻结实验，也不修改其结果或接受判据。

已保存多比特参数的重放及 p=0/1/2 层数诊断统一使用[冻结参数重放](HARDWARE_REPLAY.md)。共享 helper 现支持明确的计数宽度和 `optimization_level=0/1/2`，两比特入口默认值保持不变。服务器历史修改版本仍应保留，不能用当前 helper 代替旧任务的源码归档。

**建议先反馈两比特结果，再决定是否重跑大电路。** 默认命令只生成计划，不导入 SDK、不访问云端。只有 `--submit` 提交任务；`--list-pairs` 和 `--compile-only` 读取真实芯片配置，不提交量子任务。脚本启动时打印电路数量、每电路 shots 和总 shots；计划超过 `--max-jobs`（默认 24）时直接停止。

## 本次确认的 SDK 行为

实际 pyqpanda3 0.5.0 离线测试发现：

- `Transpiler` 可以根据非零 `compensate_angle_map` 自行加入 CZ 相位补偿；因此原脚本在后续 `to_instruction(..., False)` 中关闭补偿，**不能单独判为代码错误**。仍需核对真实芯片与平台约定。
- `to_instruction` 的 `compensate_pulse=True` 不只是校准开关，还涉及 virtual-Z / 脉冲转换。直接将旧参数改为 True，会拒绝某些通用 RPhi 分解产生的角度。新入口用下面两条明确的路径进行诊断，而非直接翻转旧参数。
- SDK 的 `instruction_to_qprog` 回读可能改变测量到经典位的对应关系。两比特最小例子中，原程序的 `01` 变成回读程序的 `10`。现有提交脚本并未使用该回读器，这不能证明云端也采用相同顺序。新入口保存回读程序作证据，不用它自动修正真机 counts。

`--compiler-basis fixed`（诊断默认）：先在零校准配置上编译到 X1/RZ/CZ，再由指令转换器进行脉冲转换；`--compensation on/off` 决定这一阶段采用真实校准角还是零角。两组的 converter flag 均为 True，只有校准角不同。

`--compiler-basis backend`：使用芯片公布的门基，由转译器应用校准角，指令转换器 flag 保持 False。`--compensation off` 将转译阶段校准角置零；`on` 保留真实角。使用 on 并同时启用下面的两个 cloud 选项，接近原实验的提交流程。

两条路径都不会修改获取到的原始 chip 配置；**补偿只在一处应用**。在构造的非零校准、物理编号 12/13 的离线测试中，40 次准备检查通过，两条路径的脉冲演化分布最大差约 `3.1e-16`。这不是 WK_C180 实测结论。

## 1. 本地检查与查看计划

```bash
python -m experiments.quantum.diagnose_hardware \
  --local --suite all --compensation both --shots 1024

python -m experiments.quantum.diagnose_hardware \
  --plan-only --suite readout --shots 1024
```

第一个命令在构造的零校准两比特芯片上，检查转译、序列化脉冲和显式测量映射，使用 CPUQVM 执行序列化脉冲的理想酉变换；不消耗真机额度。它不能验证平台的 counts 位序、真实 CZ 校准或硬件噪声。

## 2. 获取可用相邻物理比特

沿用原运行手册，在服务器设置 `ORIGINQ_API_TOKEN`。凭据不通过命令行传入，也不写入诊断产物。

```bash
python -m experiments.quantum.diagnose_hardware \
  --list-pairs --machine WK_C180
```

输出目录的 `pairs.json` / `chip_backend.json` 保存 SDK 返回的连接与配置。选一条当前可用的相邻边；`--physical-qubits` 填 SDK 原始编号，不自行加一。指令文件中的 +1 偏移由脚本处理并记录。

下面的 `A,B` 是占位符，需要替换为查到的两个实际整数。限定同一对比特，便于分批比较。

## 3. 先编译，再提交基础测试

先按原实验的门基和 cloud 选项核查位序与基础门：

```bash
python -m experiments.quantum.diagnose_hardware \
  --compile-only --suite readout --physical-qubits A,B \
  --compiler-basis backend --compensation on \
  --cloud-mapping on --cloud-optimization on

python -m experiments.quantum.diagnose_hardware \
  --submit --suite readout --physical-qubits A,B --shots 1024 \
  --compiler-basis backend --compensation on \
  --cloud-mapping on --cloud-optimization on
```

基础测试共 **7 个任务 / 7168 shots**：

| 电路 | 理想分布，字符串为 q1q0 | 用途 |
| --- | --- | --- |
| basis_00 | 00 | 初态、读出基线 |
| basis_01 | 01 | 单独翻转 q0，核查位序 |
| basis_10 | 10 | 单独翻转 q1，核查位序 |
| basis_11 | 11 | 两位翻转与读出 |
| h_q0 | 00/01 各半 | q0 叠加与位序 |
| h_q1 | 00/10 各半 | q1 叠加与位序 |
| bell | 00/11 各半 | 含 CZ 的关联测试 |

Bell 的 Z 基计数只检查关联，不独立证明纠缠。脚本不根据单批结果自动宣布硬件通过，也不自动反转 counts。

`--submit` 会重新编译并保存该次真实配置；不加载之前编译目录的指令。每一批均先编译、验证并保存全部电路，再提交第一个任务。两比特逻辑电路和未校准编译电路应具有相同理想分布；不满足则在任何提交前停止。

如需对照干净的固定脉冲路径，可在相同物理比特上使用 `--compiler-basis fixed --cloud-mapping off --cloud-optimization off`，仍先 `--compile-only`。诊断默认关闭两个 cloud 选项，因为指令已经完成物理映射；这与原实验的 on/on 不同，所有选项均单独记录。当前资料不足以断言平台是否对 run_instruction 重新应用这些选项。

## 4. CZ 相位敏感测试

基础测试核查后，再执行 CZ 对照。两组使用同一物理对、相同理想逻辑电路与 shots，分别使用真实与零校准角：

```bash
python -m experiments.quantum.diagnose_hardware \
  --compile-only --suite cz --compensation both \
  --physical-qubits A,B --compiler-basis fixed

python -m experiments.quantum.diagnose_hardware \
  --submit --suite cz --compensation both --shots 1024 \
  --physical-qubits A,B --compiler-basis fixed
```

共 **12 个任务 / 12288 shots**：分别对 q0、q1 作 Ramsey 目标，另一位置 1；相位取 0、π/2。包括 4 个无 CZ 参考电路和 8 个含 CZ 的补偿开/关电路。对含 CZ 电路，理想目标位取 1 的概率为 `(1-cos(phi+pi))/2`；无 CZ 参考为 `(1-cos(phi))/2`。这是简单的相位敏感探针，不是完整芯片校准或门保真度测量。

如果芯片校准角均为零，on/off 指令可能完全相同，此时这组对照无法测出补偿差异。当前是一组有限样本，任务按计划顺序提交，没有随机交错与重复轮次，不能排除时间漂移。

## 5. 返回产物

每次运行创建 `results/hardware_diagnostics/<UTC时间戳>/`，包含：

- `manifest.json`：模式、任务预算、软件版本、Git 提交、脚本/原驱动哈希、全部明确选项与状态。
- `chip_backend.json`：拓扑、门基、校准角、时序及可用保真度字段；非字符串键以 key/value 列表保留。
- 各测试目录：逻辑、未校准编译、实际编译、SDK 回读 OriginIR；提交指令原文与 SHA256；脉冲角度、测量映射、原始 counts、任务编号、轮询轨迹及可获取的平台映射信息。
- `summary.json` / `SUMMARY.md`：按 q0 在右以及字符串反转两种假设报告总变差（TV），不覆盖原始 counts、不推断哪种平台约定一定正确。
- `SHA256.json`：该批所有产物的哈希。

先保存任务编号，再轮询；先保存原始 counts，再验证总数及请求附加信息。附加映射字段不受平台支持时，保留已有 counts 并记录异常类型，不重提任务。超时、编译失败或其他异常停止整批；不自动重试、取消或恢复提交。看到失败或超时，先查询已记录的任务编号，避免重复提交。

需要离线重建汇总时：

```bash
python -m experiments.quantum.diagnose_hardware \
  --analyze results/hardware_diagnostics/实际时间戳

tar -czf caf-hardware-diagnostics.tar.gz results/hardware_diagnostics
```

请反馈整个诊断目录的压缩包。基础测试结果已经足以开始排查，不必先重跑四社区大电路。

新增诊断入口和测试使当前源码指纹变化；历史实验、原始硬件文件及其源码指纹保持原样，不能将旧结果标记为新代码运行。
