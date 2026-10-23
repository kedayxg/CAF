# Current maintenance scope

## 后续收到的真机记录

服务器补充了四个独立冻结社区的WK_C180原始批次，已完成文件指纹、平台原始计数、输入、模型系数、完整矩阵增量和保存参数分布的独立重放。社区2、3通过既定判据；社区0变差提案拒绝，社区1无改善。8比特均匀best-of-4000采样也几乎必然通过相同判据，所以结果限定为候选解码与接受链路的可执行性，不支持QAOA采样优势。云端编译电路和映射没有被保存，不能唯一归因于硬件噪声。详情见[真机独立核验](HARDWARE_RESULTS.md)。此前“没有新QPU记录”的描述属于本次服务器数据到来前的执行阶段，原历史证书不改写。main保留v1，本轮仅更新v2-development的复现材料，论文草稿仍由作者本地保管。

The corrected reproduction default is pinned python-igraph 0.11.9 Newman
leading-eigenvector modularity splitting. With the default capacity zero it
infers a natural community count. Positive `max_community_size` uses the same
induced-subgraph recursion as the capacity studies, stopping when blocks fit;
an unsplittable oversized block raises instead of silently violating the limit.
Neither path adds custom forced splits. The absolute
RMT-filtered correlation matrix, including its diagonal, defines the weighted
graph; graph loops have weight A_ii/2 so matrix and graph degrees agree.

The original upstream spectral file and runtime selection names have been
removed. Only `newman` is supported by `Clustering`; old `legacy_spectral`,
`louvain` and `kmeans` selections raise `ValueError`. Historical reproduction
requires checkout of paper commit `356c1852c65a185b05aed040052c5c3f750991bd`.
See the [spectral review](SPECTRAL_IMPLEMENTATION_REVIEW.md)
and [provenance](../third_party/dcmppln.VENDORED.md) for the specific defects.
CAF accepts external communities and does not depend mathematically on a
particular partitioner.

## Retained fixes

- Validate the covariance, disjoint partition, community budgets and supplied
  initial portfolio. Without an initial portfolio, construct a feasible state
  before refinement. Skip backend calls for zero/full-budget blocks.
- Keep context extraction and acceptance consistent with the true global
  objective; distinguish raw feasible backend candidates from repaired vectors.
- Forward the pipeline risk-rescaling switch and compute rescaling from actual
  asset groups. Paper-facing static initializations explicitly use False,
  matching the historical actual behavior. Frozen local rescaling is corrected
  and its q_eff can change even though initialization stays unscaled.
- Preserve explicit zero budgets and risk factors, pass exact cardinalities,
  and select a feasible static SA sample or raise if none exists.
- Use the direct target-Pauli sign convention in simulator and hardware cost
  gates, retaining the plain-QAOA equivalence regression.
- Require successful hardware tasks and integer raw count batches totaling the
  requested shots; exclude zero-count candidates and probability fallback.
- Explicitly pass a finite transverse-field sequence to the path-integral sampler;
  its omitted field defaults to zero. See [SQA protocol review](SQA_PROTOCOL_REVIEW.md).
- Align local/global q in the frozen control by default, retaining norm rescaling
  as an explicit optional surrogate protocol.
- Fix the frozen data path and keep no-improvement fallback targets marked
  simulator-only even when n<=6; seed suppression sampling and exclude tied pairs
  from sign tests while retaining them in average gains and bootstrap intervals.

## Verification and results

The current primary backend is **Gurobi** for both static initialization and
CAF local refinement. The new two-panel diagnostic uses the same capacity-100
memberships and budgets. On 2016 the objective decreases from 1.637959598 to
1.335466020 (18.467707%); on 2018 from 54.504183500 to 51.649098601 (5.238286%).
Both repeated unshifted controls retain exactly the static weights. All 440
local solves reach the requested MIPGap=1e-4, with no time limits; an independent
full-matrix replay verifies 400 proposals, four histories and six portfolios.
This is one fixed-configuration flow per panel, not 20 statistical trials.
Times exclude shared RMT/partition preprocessing. The primary emphasis was
selected after observing the SA controls; all those results remain below and
in the paper, including the 2018 injection-ablation result. See
[gurobi-main-results.json](gurobi-main-results.json) and its
[validation](gurobi-main-validation.json).

The retired source's nine expected-failure cases were removed with that
implementation. The current suite has **96 passes, zero expected failures and
zero unexpected failures**. Graph validation, modularity conventions, recursive
membership and capacity handling, shared initialization, feasibility and solver
adapter regressions all run normally. See [the test record](arxiv-v2-tests.json).

The retained SA/simulation/scaling protocol has **345 fresh task records**, a fresh
frozen-state example and **40 sampling-matched SA controls**, all tied to core
source SHA256 `9fc43f0fe1de39fcefb2dd021cdd99efab882ca98d37df1055a2669d305fe9d8`.
Classical capacity is 100, quantum-simulator capacity is 16; the library's
unrestricted capacity-zero option remains available. Independent validation
recomputed 730 portfolios (610 fresh paired-run portfolios, 80 archived natural
references and 40 controls), 305 histories and 1,582 accepted-update deltas.
All 178 Gurobi local solves attained their requested 1e-4 tolerance; no solve
hit the time limit. This does not certify a full-portfolio global optimum.

| SA panel | Communities | Static mean | CAF mean | Mean paired gain |
|---|---:|---:|---:|---:|
| 2016, capacity 100 | 15 | 1.729826754 | 1.512529816 | 12.544363% |
| 2018, capacity 100 | 25 | 62.994889282 | 62.047109038 | 1.503586% |

Both SA panels improve in all 20 pairs against their own static initialization.
Under matching sampling budgets, full CAF performs better than CAF without
injection in all 20 pairs on 2016; the no-injection ablation performs better in
all 20 pairs on 2018 (objectives 60.638537516 versus 62.047109038). Both variants
retain CAF's full-global objective verification and greedy sequential
acceptance, and both improve static initialization. This is an injection
component ablation; it does not show static decomposition superior to CAF or
independently measure the shared acceptance mechanism. The contribution of
injection depends on the backend and instance; no unique cause of the 2018
difference has been established. Concurrent times do not establish a speed ranking.

Full protocols, results and independent checks are in
[ARXIV_REVISION.md](ARXIV_REVISION.md), [REVISION_RESULTS.md](REVISION_RESULTS.md)
and [revision-validation.json](revision-validation.json). The reviewed
PDF（作者本地材料，未随代码发布） and matching
source ZIP（作者本地材料，未随代码发布） compiled successfully from a
clean extraction. [The code snapshot](CAF_v2_code_snapshot.zip) preserves the
executed code; its manifest includes dependency files and input hashes.

The archived natural studies retain their original source fingerprints in
[natural-newman-revision-results.json](natural-newman-revision-results.json).
They are descriptive appendix references, not equivalent capacity protocols.
The earlier [forced-nine study](FIXED_NINE_STUDY.md),
[three-library comparison](LIBRARY_CLUSTERING_STUDY.md) and
[custom corrected spectral study](CORRECTED_SPECTRAL_STUDY.md) remain historical
sensitivity analyses. The prior maintenance stage is preserved in
[maintenance-before-capacity-v2.json](maintenance-before-capacity-v2.json).

## Mathematical and publication limits

The reviewed context-bias and global increment formulas are consistent under
symmetric covariance, feasible initialization and fixed community budgets.
Monotonic acceptance does not optimize transfers of budget across communities.
Iteration caps and heuristic patience do not certify local optimality. A full
no-improvement sweep using exact consistent local solves is needed even for a
tolerance-based block-local certificate.

Local shift extraction and acceptance cost O(|C|^2), accepted covariance-cache
updates O(N|C|), and the current global history metrics O(N^2) per outer round.
CAF timing measures refinement after initialization; total pipeline time must
include initialization. Equal per-solve caps do not imply equal aggregate
sampling effort. Paired gains over one-shot static solves do not by themselves
isolate context injection from extra search effort.

Plain-QAOA cost-sign conventions admit an equivalent parameterization under
the standard initial state and X mixer; a corrected direct Pauli sign is used
for clarity. That equivalence does not imply identical finite optimizer paths
or arbitrary warm-start distributions. No fresh QPU job, release, push or arXiv
submission has been performed. Historical hardware data must be identified as
such and cannot validate newly partitioned quantum instances.

## BCD、容量与子集补充实验（2026-10-23完成）

24个预先声明配置全部成功，覆盖两个全面板、容量75/100/150及每面板3个300资产
子集（种子1729/2718/3141）。其中22个不同的输入/划分/预算配置；重复分区不当作
独立市场证据。三种方法共用初始化和预算，固定六轮，关闭patience。

CAF与最新上下文顺序BCD的最终目标24/24一致；BCD在9个配置中更少轮达到终值，
15个轮数相同。四工作线程CAF保持相同终值，但仅在6/24次单次时间测量中更快，
未显示稳定的本地CPU加速；该测量不构成速度统计检验，也不代表QPU排队延迟。
所有5320次局部求解达到指定容差，独立核对5094个候选、72条历史和96个持仓记录。

容量75/100在2016全面板产生相同15社区；容量150产生10社区，绝对终值更低。
2018全面板社区数65/25/16，CAF终值54.945746/51.649099/51.303060；相对改进
8.8871%/5.2383%/3.6624%。这些容量改变预算可行域与初始解，不是同一问题上的
算法排名。完整结果见[补充报告](COORDINATION_STUDY.md)、[协议](coordination-study-protocol.json)
和[独立核验](coordination-study-validation.json)。

已收窄Benders/Dantzig–Wolfe相关工作表述，不再泛称其不适合离散量子硬件。
补充对照填补了原审稿提出的缺口，但没有证明CAF优于标准BCD的解质量或稳定CPU
速度；主张继续限定为面向局部求解器的冻结批量提案和全局协调架构。未上传arXiv。

## 当前稿件范围收敛

按作者确认，v2聚焦实现修复与原研究重新验证。主实验继续比较独立静态初始化和CAF精炼；BCD保留为技术来源，最新上下文顺序更新定位为更新时序消融。正文简述24/24终值一致，完整容量、子集、轮次和计时结果移至附录；摘要和结论不再把它作为主要算法排名。所有结果及限制保留，SA注入消融不删，按结果选择的合成例子不进入主论文。本轮没有新增实验或修改核心代码。前述审阅和执行结果为保留的历史记录；当前呈现以修订范围（作者本地材料，未随代码发布）、最新PDF及交付校验为准。未上传arXiv。
