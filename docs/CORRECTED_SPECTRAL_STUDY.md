> Subsequent protocol review found that the archived SQA wrapper omitted the transverse field, which defaults to zero in the pinned sampler. Those archived numbers are zero-field results, not finite-field quantum-model evidence; see [SQA_PROTOCOL_REVIEW.md](SQA_PROTOCOL_REVIEW.md). A new complete finite-field rerun is reported separately.

> Historical study snapshot: the revision now adopts igraph Newman as its default. Current scope and reruns are recorded in [ARXIV_REVISION.md](ARXIV_REVISION.md).

> Archived experiment: spectral corrections were subsequently deferred at the user's request. This document and its 60-test result describe the earlier corrected-clustering snapshot, not the current default. See [current maintenance notes](MAINTENANCE.md).

# Maintained implementation and historical reproduction

The historical paper implementation is commit
`356c1852c65a185b05aed040052c5c3f750991bd`.
That immutable commit remains the reference for the published numerical tables.
The maintained implementation corrects decomposition and solver adapters; its
outputs are new experiment results and must be identified separately.

## Changes

- Recursive spectral clustering maps child positions back to global asset indices.
  It now uses Newman's generalized modularity matrix with its diagonal row-sum
  correction and accepts only positive-modularity-gain splits. Labels are
  contiguous, and optional size bounds also terminate on edgeless graphs.
- `Pipeline.run(risk_rebalancing=True)` now honors the requested setting.
  Rescaling accepts partition labels or actual index groups and validates that
  groups partition the assets. The benchmark and frozen initializations now
  explicitly request False, reflecting their historical *actual* behavior.
  Frozen local rescaling uses the corrected index groups and can differ from
  historical q_eff values even though initialization remains unscaled.
- `run_caf_pipeline` constructs a feasible portfolio before refinement when no
  initial portfolio is given. Supplied initial indices, partitions and budgets
  are validated. Zero/full-budget blocks do not invoke a backend.
- Proposal diagnostics distinguish `raw_candidate_feasible`,
  `candidate_repaired`, `backend_called`, and normalized `candidate_feasible`.
  Initialization repairs are recorded in `initialization_diagnostics`.
  A repaired vector is not an actually sampled feasible backend result.
- Zero model budgets and q=0 are preserved. Scaling passes an exact cardinality
  instead of rounding it to a divisor. Static SA uses the best feasible sample
  in the returned batch and raises if none exists, rather than returning zeros.
- The frozen-ablation default data path resolves within this checkout.
  If no improving n<=6 community exists, it can report a simulator-only target;
  it does not silently claim a new hardware result.
- Suppression experiments explicitly seed their SA sampler.
- Paired sign tests exclude tied runs rather than counting them as losses.
  Ties remain in average gains and bootstrap intervals. Existing maintained
  summaries can be reaggregated without solver execution using
  `--refresh-statistics --summary-file path/to/benchmark_summary.json`.
  This statistic does not isolate context injection from extra search effort.
- Hardware decoding requires a finished task and a raw integer-count batch
  totaling the requested shots. Zero-count keys are excluded from candidates.
  Platform probabilities are not a substitute for raw measurements.
  Readout amendment is disabled in this raw-count protocol.
  Importing the module and requesting --help do not require private config;
  real hardware initialization still needs config.py and backend credentials.

## QAOA convention

For dimod's spin s=2x-1 and computational-basis Pauli Z eigenvalue z=1-2x,
the cost operator is

```text
H = offset - sum_i h_i Z_i + sum_ij J_ij Z_i Z_j.
```

The simulator and hardware cost layers now use `RZ(-2*gamma*h_i)` and
`RZZ(2*gamma*J_ij)`. Pauli expectation diagnostics apply the linear sign
conversion exactly once. Classical BQM scoring and bit decoding keep the
original x convention.

The historical opposite-field convention is not by itself evidence of an
incorrect final solution: with a |+> initial state and standard X mixers, a
global X conjugation is absorbed by `beta_last += pi/2`. With unrestricted
angles, both plain-QAOA state families have the same expressiveness.
Finite optimization trajectories can differ, and this equivalence is not
automatic for general warm starts. At the user's request, the maintained
implementation uses the direct target-operator convention for clarity.
Historical optimized angles must not be reused unchanged after this update.

## Guarantees and scope

Monotonic acceptance requires symmetric finite covariance, disjoint communities
covering every asset, and a feasible initial portfolio. It preserves the fixed
per-community budgets. It does not optimize how the total budget is allocated
between communities.

Stopping at the iteration limit or heuristic patience does not prove local
optimality. Even with exact local solves, a local-optimality statement requires
a completed no-improvement scan under the same target objective and is subject
to the acceptance tolerance.

Context extraction and the objective-increment calculation cost O(|C|^2).
An accepted cache update additionally costs O(N|C|), and the current implementation
recomputes global history metrics in O(N^2) per outer iteration.
Reported CAF timing is refinement after initialization, not the complete
static-initialization-plus-refinement runtime.

## Validation

Use Python 3.11 with the pinned requirements and pytest:

```bash
python -m pytest -q
python -m experiments.classical.run_benchmark --runs 20 --skip-q-sweep --out-dir reproduce_outputs/maintained_2016
python -m experiments.classical.run_scaling --sizes 100,200,300,484 --runs 1 --solvers gurobi --out-dir reproduce_outputs/maintained_gurobi
python -m experiments.quantum.run_sqa --n-assets 40 --runs 20 --include-sqa-reference --skip-qaoa --max-workers 1 --out-dir reproduce_outputs/maintained_sqa
```

The regression tests exercise recursion and permutation/scaling invariance on
known graph structure, exact local solves, feasibility failures, actual cost-layer
phases against BQM energies, plain-QAOA reparameterization, and offline hardware
failure/count handling. They do not submit jobs to a quantum backend.

Validation completed with Python 3.11.17 and the pinned pip requirements.
All **60 tests passed**. Machine-readable aggregates, source/data fingerprints
and per-seed 2016 objectives are in [corrected-spectral-results.json](corrected-spectral-results.json).

| Check | Maintained result | Coverage |
| --- | --- | --- |
| 2016 N=484 SA | Static 1.540945±0.022058; CAF 1.512938±0.014130 | 20 runs, seeds 42–61, K=242 |
| 2016 relative improvement | 1.802886%; 16 improving / 4 tied / 0 worse; bootstrap CI [1.2433%, 2.3938%] | Historical result was 6.589791%, 20/20 improving |
| 2016 partitions | Sizes 199, 153, 132 | Historical code produced 9 communities, maximum size 101 |
| Cached/full evaluation | Identical final objective in all 20 runs; all portfolios feasible and histories nonincreasing | This is numerical agreement, not a local-optimality certificate |
| Gurobi N=100/200/300/484 | Objective gains 20.0798% / 11.4103% / 8.3978% / 8.7034% | One run per setting, default MIPGap=1e-4 |
| SQA N=40 | 19/20 improving; mean Delta O=-8.39654e-4±5.21971e-4 | 20 runs, 288 raw feasible proposals, zero repaired proposals |
| QAOA N=40,p=2 | Completed one run; objective improved | Functional check only: 1 restart, 12 optimizer evaluations, 2 CAF iterations |
| Frozen ablation | New target n=8,K=4; injected exact and warm-start simulator Delta O=-1.6235756e-4, Hamming=0; no-injection Delta O=0 | Simulator-only; historical n=5 hardware evidence is not transferred |
| Custom cardinality | Both static and CAF hold exactly 12 assets for N=40,K/N=0.3 | Checks the former divisor-rounding mismatch |
| Suppression | Seeded runs and figure/JSON generation completed | One run each at N=100,484; not a replacement for the paper's 40-run aggregate |
| 2018 N=1397 | One run completed with no accepted improvement | K=698; does not validate the paper's twenty-run statistics |
| Hardware | Cost-angle, Pauli-energy, failure and raw-count paths tested offline; --help works without private config | No real QPU task or native SDK execution |

The 2016 and 2018 jobs started before the tied-pair statistics correction.
After completion, their stored summaries were refreshed from the unchanged
per-run objectives using --refresh-statistics. The 2016 p-value excluding
four ties is 3.0517578125e-5. Original console logs used the older tie treatment;
the maintained summaries and the result file above use the corrected method.

### Attribution of the 2016 SA change

A single-factor control restored only the historical spectral function imported
by `Clustering`, retaining all other maintenance changes. The static baseline
and cached CAF were rerun from data for all 20 original seeds (42–61), with
q=0.5, risk rescaling disabled, 200 baseline reads, six CAF iterations and
patience two. The redundant no-cache run and unrelated benchmark references
were omitted in this control.

| Implementation | Spectral partition | Mean per-run improvement | Improving / tied |
| --- | --- | --- | --- |
| Historical | Historical, 9 communities | 6.589791% | 20 / 0 |
| Maintained control | Historical, 9 communities | 6.589791% | 20 / 0 |
| Maintained | Corrected, 3 communities | 1.802886% | 16 / 4 |

The control reproduces every historical baseline and final CAF portfolio
exactly, with zero objective difference and matching objective histories to
1e-12. Reverse spot checks use the historical pipeline, objective builder,
SA adapter and CAF with corrected spectral clustering at seeds 42 and 44
(one improving run and one tie). Both reproduce the maintained portfolios
and objective histories. Per-seed comparisons and portfolio hashes are in
[spectral-attribution-results.json](spectral-attribution-results.json).

For this benchmark, these controls attribute the numerical change to the
spectral implementation changes as a bundle. They do not isolate the global
index repair, generalized modularity matrix and split acceptance from each
other. The QAOA sign convention is not involved in this classical SA run.

The partition change also changes community budgets, local problem dimensions
and adaptive sampling totals: the unchanged CAF rule requests 1,060 local SA
reads per outer iteration across the historical nine blocks, versus 600 across
the corrected three blocks. This is not an equal-total-read comparison.
On the absolute denoised correlation graph used by clustering, modularity
increases from 0.024676 to 0.167819 and the cross-block weight fraction falls
from 80.74% to 48.86%. These are partition diagnostics; they do not separately
establish why CAF improves less. Fewer blocks alone do not imply better global
portfolios or invalidate the context-injection mechanism.

These results retain positive coordination gains on the tested 2016 SA,
Gurobi and SQA settings, while the magnitude and community structures change.
Any manuscript update must report these differences rather than carry the
historical tables forward unchanged.

Full QAOA p=2/p=3 stochastic studies, the full 2018 twenty-run comparison and
fresh real-hardware validation are separate experimental work before updating
those publication claims. Recheck local dimension limits and target selection
after any decomposition change; historical community indices are not stable.
