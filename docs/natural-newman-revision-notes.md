# igraph Newman revision

Current code update: the original spectral implementation and its runtime
selection names have been removed. `Clustering` now supports only igraph Newman,
with optional positive-capacity induced-subgraph recursion. This record's
325-task rerun protocol and the generated PDF below still describe the natural
partition experiments. The supplementary capacity-100 studies for
[2016](CAPACITY_2016_CAF_STUDY.md) and [2018](CAPACITY_2018_CAF_STUDY.md) are
separate completed studies; the PDF has not yet incorporated them.

Completion: all 325 required offline task records are assembled and validated.
Independent full-matrix recomputation passes for 610 stored portfolios; frozen
exact moves also match independent enumeration. See [final results](REVISION_RESULTS.md),
[review PDF](../manuscript/CAF_arxiv_revision.pdf) and
[LaTeX source package](../manuscript/CAF_arxiv_source.zip).

The revision adopts python-igraph 0.11.9 Newman's leading-eigenvector method,
partition seed 42 and ARPACK maxiter=10000, tol=1e-10. It retains matrix diagonal
weights using half-weight graph self-loops. It infers natural communities;
there is no imposed nine-community constraint or maximum block size.

The 2016 default has blocks [199,153,132]. Every asset assignment and the solver
order match the independently completed twenty-run public-library experiment.
That experiment's original fingerprints and records stay in
[library-igraph_spectral-results.json](library-igraph_spectral-results.json).
It reports static 1.5409447432, CAF 1.5129375531, mean paired gain 1.802886%,
16 wins and four ties, replacing the historical nine-block headline for this
configuration. It is not necessary to execute identical twenty-run trials again
solely because the same partitioner is moved into a shared production adapter.

## Fixed rerun protocol

Run from the repository root with the pinned Python environment:

```bash
python -m experiments.run_revision --workers 4 --out-dir revision_results
python -m experiments.quantum.run_frozen_ablation --n-assets 40 --qaoa-p 2 --qaoa-restarts 6 --qaoa-maxiter 80 --warmstart --out-dir revision_frozen
```

The first command schedules 325 independent offline tasks, with one immutable
JSON checkpoint per task. Each subprocess uses one BLAS thread and one CAF
proposal worker to avoid CPU oversubscription across independent trials.
Failures do not discard completed checkpoints. Existing checkpoints are reused
only when task parameters and recorded source fingerprint agree.

| Group | Trials | Protocol |
| --- | --- | --- |
| Main 2016 | 20, already completed independently | N=484,K=242,q=0.5,seeds42--61; exact membership/order integration checked |
| Additional 2018 panel | 20 | N=1397,K=698,q=0.5,seeds42--61 |
| SA scaling | 80 | N=100,200,300,484; K=floor(N/2); 20 per size |
| SA sparsity | 20 | N=200,K=50; K=100 already covered by scaling |
| Gurobi | 5 | Four size settings plus N=200,K=50; one deterministic trial per setting, MIPGap=1e-4,Threads=1 |
| Shared-initialization risk sweep | 100 | q=0.1,0.3,0.5,0.7,1.0; twenty main q=0.5 static portfolios reused at each q |
| Suppression | 40 | Ten sampled subsets per N=100,200,300,484; SA100 reads |
| Plain QAOA | 40 | N=40,K=20; p=2 and3; 20 each; restarts6,maxiter80,shots4000,4CAFiterations,patience2 |
| SQA | 20 | Same N=40 subset; 1000sweeps,20static reads,size-adaptive CAF reads,4iterations,patience2 |
| Frozen warm-start example | One aligned-q scan plus archived rescaled control | N=40 subset1234,staticseed42,p=2,restarts6,maxiter80,shots4000 |

All static SA initialization and CAF updates use aligned q without risk
rescaling. The frozen control also aligns local/global q=0.5; its earlier
norm-rescaled q_eff≈0.664 scan remains archived separately. Main/2018/
scaling/risk-sweep CAF runs use six iterations and patience two. Scaling subsets
use seed123+N and solver seeds42+1000N+10K+i. The suppression subsets use
seed123+100N+i. Quantum subsets use seed1234; QAOA seeds1042--1061 and SQA
seeds31042--31061. Partition seed remains42 independently of solver seed.

Runtime measurements from concurrent trials are not hardware-independent speed
claims. The portable main-study archive omits CAF wall times, so the revised
main table reports objective, risk and return rather than mixing old/new timing.
Equal local caps do not equate aggregate effort; the revision does not present
static-versus-CAF paired gains as an effort-controlled context ablation.

## Source and execution changes

The initial batch encountered Gurobi error10010 on a 212-variable dense
quadratic model in one sampled suppression subset. A narrow diagnostic patch
now records size-limited presolve as unavailable, retaining all suppression
observations. Unavailable counts and reductions are null and excluded from
presolve means. Other Gurobi failures still propagate. Two regression checks
cover that distinction and JSON-safe missing-data aggregation.

This patch does not alter partitions, static candidates, CAF objectives or
acceptance. A subsequent frozen-control CLI change makes its default local q
equal the global q; `--local-risk-rebalancing` explicitly restores the historical
local surrogate. The 325-task batch does not execute this separate frozen CLI. The completed batch retains per-task source fingerprints, including
tasks begun before the diagnostic patch. The failed diagnostic task is rerun
with the same subset and solver seed; its original failure log is retained.
Thus multiple source fingerprints are deliberate and must not be relabeled as
a single-source execution. A fresh reproduction uses the final source for all
tasks. The Gurobi size sweep's largest communities are71,90,109,199 and fit the
current local license; the auxiliary presolve restriction is disclosed separately.

## Manuscript changes

- Replace inherited spectral execution with the pinned public Newman method;
  explain graph convention and actual natural community counts.
- Regenerate affected numbers from complete per-seed results and remove the
  old claim that the static pipeline actually applied risk rescaling.
- State finite accepted descent without claiming heuristic stopping certifies
  a fixed point. State accepted cache-update and history-evaluation costs.
- Separate refinement timing from end-to-end timing and local per-solve caps
  from total sampling effort. Exclude ties from sign tests.
- Use the direct Pauli cost sign; preserve the standard plain-QAOA equivalence
  qualification and distinguish arbitrary warm starts.
- Preserve historical n=5 hardware execution as a v1 fixed-instance report,
  not corrected-partition hardware validation. The new frozen selector gives
  n=8,K=4 and is simulator-only; no improving n<=6 target met the declared rule.

The supplied source is in [manuscript](../manuscript/README.md). Generated
[revision-results.json](revision-results.json) records completion and missing
tasks; incomplete manuscript builds contain a visible draft notice and cannot
be packaged by `package_revision.py`. Final publication artifacts are prepared
only after the full rerun evidence is assembled. No arXiv upload, proceedings
replacement, repository push or release is performed by these scripts.

## SQA protocol correction

Inspection of dwave-samplers1.7.0 also showed the wrapper's omitted transverse
field defaulted to zero. A separate complete twenty-seed rerun now explicitly
uses Hd_field=linspace(1,0,1000),Gamma=1, with all other settings retained.
The initial zero-field records are archived and excluded from the revised SQA
column. See [SQA_PROTOCOL_REVIEW.md](SQA_PROTOCOL_REVIEW.md) for the exact
default, scope and regression. This change affects SQA only; the plain-QAOA
classes and all classical CAF calculations are unchanged.

After implementation changes, use a new output directory for a fresh
reproduction. Checkpoint reuse deliberately requires a matching full source
fingerprint. The maintenance batch records intermediate source fingerprints:
license-diagnostic handling, the separate frozen protocol option, the SQA driver
correction, labels and the external-override guard are disclosed changes. The
classical objective/coordinator and plain-QAOA mathematics did not change during
this batch. The complete finite-field SQA group was independently rerun under
one fixed corrected configuration.
