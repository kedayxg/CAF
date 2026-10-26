# 通道筛查与冻结回放提交前检查

维护入口为 `python -m experiments.quantum.screen_hardware`。它针对制备、控制和读出的组合表现筛查 1—16 个物理通道，不修改 CAF 算法或自动纠正历史计数。`main` 仍保留 v1。

此前 110 在名义制备 0 后约 97% 输出为 1，111 在 X 后仅约 21% 输出为 1；标称保真度不足以识别当时的异常。服务器临时脚本中的 X·X 又被 level 2 完全消除。因此新入口同时检查原始计数与实际提交脉冲。

## 测试设计和费用范围

对于 n 个通道，固定运行 n+3 种线路：

- idle_zero：全 0，依赖平台初态/复位约定，不声称是主动复位。
- active_zero：每位四个原生 X1（四个 +π/2 旋转），理想返回 0。使用优化级别 0，并严格检查序列化指令仍含每位四个 +π/2 脉冲。
- onehot_i：逐位准备单个 1，其余为 0，可区分全 0/全 1 无法发现的位序交换。
- all_one：全 1。

每种线路默认 1024 shots。8 位块总共 **11 个任务、11264 shots**；单通道为 4 个任务、4096 shots。所有线路编译和脉冲检查完成后才开始提交；提交顺序按记录的种子打乱。没有自动重试或自动扩大 shots。失败/超时先核对已保存的 job ID。

## 离线计划与本地验证

```bash
# 默认只写计划，不加载 SDK、不读取账户、不提交任务。
python -m experiments.quantum.screen_hardware --physical-qubits 110

# 本地合成芯片，仅验证编译、脉冲与计数路径。
python -m experiments.quantum.screen_hardware --local --physical-qubits 0,1
```

本地全部通过也不能用作真机筛查凭据。物理编号采用 SDK 索引，程序统一负责序列化的 +1 偏移，用户不用手动加 1。

## 真实配置编译与显式提交

先根据平台当前可用拓扑设置 `CAF_SCREEN_BLOCK` 为逗号分隔的物理编号列表。不要把文档中的历史 110/111 异常通道直接当作健康块示例。筛查不要求连通，因为线路没有 CZ；用于 QAOA 的块还需满足路由与连通要求。

```bash
: "${CAF_SCREEN_BLOCK:?请先设置待筛查的物理编号列表}"

python -m experiments.quantum.screen_hardware \
  --physical-qubits "$CAF_SCREEN_BLOCK" --compile-only

# 只有这一命令明确提交 n+3 个任务。
python -m experiments.quantum.screen_hardware \
  --physical-qubits "$CAF_SCREEN_BLOCK" --submit --shots 1024
```

读取完整芯片配置后，编译器被约束到指定探针位置，程序检查实际测量映射、物理范围和提交列表。单通道不会使用无边的单节点 backend 子图，避免 SDK 无法放置的问题。

每种模式、每个通道都要求正确位概率的 **95% Wilson 下界至少 0.90**；这是固定工程筛查门槛，不是孤立读出保真度或整体设备保真度。真机资格至少需要每模式 1024 shots。

## 在冻结回放中使用

设置 `CAF_SCREEN_DIR` 为刚完成的 `results/hardware_screen/<时间戳>`，并沿用原冻结输入：

```bash
: "${CAF_SCREEN_DIR:?请先设置本次真机筛查目录}"
: "${CAF_REPLAY_RECORD:?请先设置冻结源 JSON 路径}"

python -m experiments.quantum.replay_hardware_qaoa \
  --record "$CAF_REPLAY_RECORD" --p 1 --submit --shots 4000 \
  --physical-qubits "$CAF_SCREEN_BLOCK" --screen-dir "$CAF_SCREEN_DIR"
```

该回放入口现在拒绝缺少有效筛查记录的 `--submit`。检查覆盖：

- 同一 machine、相同物理比特集合；集合可以有不同输入顺序。
- 所有模式完成，平台 FINISHED 回执、job ID、原始计数和保存计数一致。
- 文件哈希、实际脉冲及逐位指标重算；不信任旧 `passed=true` 标签。
- 最早任务开始时间距提交不超过 **4 小时**，不接受未来时间异常。
- 本地模拟、失败、缺模式、shots 不足或任一通道未达门槛的记录均不能用于提交。

检查在加载 SDK 前执行一次，编译完成、正式提交前再执行一次。回放批次保存完整筛查证据副本及其指纹。原 CAF 求解器、QAOA 方程、参数优化和历史正式驱动保持原样；新增保护覆盖维护的冻结回放入口。准备新局部问题时可先用原驱动 `--prepare-only` 生成完整源记录，然后走上述筛查及维护回放流程。不要用服务器旧临时脚本绕过新流程。

## 证明范围与归档

通过仅说明这些基态模式下的组合制备/控制/读出达到门槛，不证明相干控制、CZ、任意混合态或深线路质量。探针按传入物理列表的顺序测量；QAOA 路由可能改变最终测量顺序。回放额外记录 `screen_probe_readout_order_matches_replay`，**通道筛查不能代替对不同路由后输出顺序的独立标定**。若需要直接测试某个路由排列，可按该排列的 c0…cn 物理列表运行探针。

每批保存实际源码、requirements、版本、芯片快照、所有逻辑/编译 OriginIR、提交 instruction、实际测量映射、平台回执、原始计数、analysis.json 和 SHA256.json。原始记录保持真实版本，不会将历史失败任务改标为已通过新检查。

```bash
# 仅重算；写 reanalysis.json，保留原始文件与历史哈希清单。
python -m experiments.quantum.screen_hardware \
  --analyze "$CAF_SCREEN_DIR"
```

若两个名义制备产生几乎相同输出，应先解决设备/配置基线。不要通过匹配已知目标位、翻转结果位或直接求逆不稳定的校准矩阵来宣称恢复线路。

[维护验证记录](hardware-screen-validation.json)包含完整回归测试、实际 SDK 本地模拟与历史拓扑离线编译结果；没有新增 QPU 任务。
