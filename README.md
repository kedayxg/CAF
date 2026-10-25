# CAF: Context-Aware Folding

This is the [`v2-development`](https://github.com/kedayxg/CAF/tree/v2-development)
branch. `main` retains the v1 implementation until v2 is ready to merge.
The original paper code is also preserved unchanged on
[`archive/arxiv-v1`](https://github.com/kedayxg/CAF/tree/archive/arxiv-v1).
For server installation and the corrected hardware workflow, see the
[hardware runbook](docs/HARDWARE_RUNBOOK.md). This code release includes
reproduction tools and evidence; the pending v2 manuscript remains with the authors.
For small hardware probes before repeating QAOA, use the
[two-qubit diagnostics guide](docs/HARDWARE_DIAGNOSTICS.md): it records submitted
instructions, calibration, measurement maps and raw counts for feedback.
The [frozen-parameter replay guide](docs/HARDWARE_REPLAY.md) covers p=0 controls,
correct saved-layer truncation and offline recalculation of historical metrics.

The maintained reproduction path uses python-igraph 0.11.9 Newman's leading-
eigenvector method (`clustering_method="newman"`). It replaces the upstream
modified spectral routine, whose recursive indexing and gain-handling defects
are documented in [the implementation review](docs/SPECTRAL_IMPLEMENTATION_REVIEW.md).
The historical routine and its `legacy_spectral` / `louvain` selection names
have been removed. Unsupported method names raise an error.
Use commit [`356c1852c65a185b05aed040052c5c3f750991bd`](https://github.com/kedayxg/CAF/tree/356c1852c65a185b05aed040052c5c3f750991bd)
for the immutable historical implementation.

Use `Clustering(max_community_size=100)` for capacity-oriented induced-subgraph
Newman recursion, or the default `max_community_size=0` for natural communities.
Both paths call igraph; if an oversized block cannot be split, the capacity
path raises an error instead of returning a block over the limit.

```python
from dcmppln.clustering import Clustering

labels, process_seconds, wall_seconds = Clustering(max_community_size=100)(correlation)
```

The capacity-100 studies cover both [2016](docs/CAPACITY_2016_CAF_STUDY.md)
and [2018](docs/CAPACITY_2018_CAF_STUDY.md). Reproduce them with
`python -m experiments.classical.run_capacity_benchmark --year 2016 --capacity 100 --out-dir results/capacity100-2016`
(use `--year 2018` and a different output directory for that panel).

See [maintenance notes](docs/MAINTENANCE.md) and [the arXiv revision record](docs/ARXIV_REVISION.md)
for protocol changes and rerun coverage; the [revision results](docs/REVISION_RESULTS.md)
record the numerical evidence. The corrected
2016 natural partition has sizes 199, 153 and 132. Twenty paired SA runs yield
mean objectives 1.540945 (static) and 1.512938 (CAF), a 1.8029% mean paired gain
with 16 improvements and four ties. These are unrestricted reference results; the arXiv v2 primary experiments
use the declared capacity constraints below.

The [public-library comparison](docs/LIBRARY_CLUSTERING_STUDY.md) also evaluates
NetworkX fixed-nine CNM and actual Louvain; `pip install -e ".[clustering]"`
adds the optional comparison libraries. The [fixed-nine spectral study](docs/FIXED_NINE_STUDY.md)
uses a separate explicit forced-split constraint; it is not the default Newman method.

**Context-Aware Folding (CAF)** is a hybrid quantum-classical coordination layer for decomposed portfolio optimization. Static decomposition omits cross-cluster covariance during local solves; CAF re-injects this missing global information as a state-dependent linear bias and commits updates under a monotonic acceptance rule.

This repository is the public code release accompanying the CAF paper. It contains:

- the CAF engine in `caf_core/`
- the paper-facing reproduction scripts in `experiments/`
- the bundled benchmark inputs in `data/russell3000_subset/`
- a vendored copy of JPMorgan's `dcmppln` dependency used by the default reproduction path

## Repository layout

```text
CAF/
├── caf_core/
├── data/russell3000_subset/
├── dcmppln/
├── experiments/
│   ├── _common.py
│   ├── config.example.py
│   ├── classical/
│   │   ├── run_benchmark.py
│   │   ├── run_scaling.py
│   │   ├── run_capacity_benchmark.py
│   │   ├── run_sparsity.py
│   │   └── run_suppression.py
│   └── quantum/
│       ├── run_qaoa.py
│       ├── run_sqa.py
│       ├── run_hardware_qaoa.py
│       └── run_frozen_ablation.py
├── tests/test_smoke.py
├── third_party/
├── environment.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

The preferred reproduction path uses the pinned conda environment:

```bash
conda env create -f environment.yml
conda activate caf
```

If conda is not available, install the pinned pip dependencies instead:

```bash
pip install -r requirements.txt
```

For an editable install:

```bash
pip install -e .
```

To include the quantum dependencies:

```bash
pip install -e ".[quantum]"
```

## Quick self-check

Run the lightweight smoke test from the repository root:

```bash
conda run -n caf python3 tests/test_smoke.py
```

This check is intended to catch broken imports or packaging issues quickly before rerunning the paper-scale experiments.

## Using CAF with your own solver or data

The paper scripts are only one use case. The core entry point `caf_core.run_caf_pipeline(...)` is designed so that users can plug in:

- a custom covariance matrix and expected-return vector
- a custom partition of assets into communities
- a custom local model builder
- a custom local solver

At minimum, you need to provide:

- `covariance_matrix`: full covariance matrix
- `expected_returns`: expected-return vector
- `communities`: a list of index arrays, one per local subproblem
- `subproblem_cardinalities`: the required number of selected assets in each community
- `get_model_func(...)`: builds the local model consumed by your solver
- `optimizer_factory(...)`: returns a callable local solver

### Minimal example

```python
from types import SimpleNamespace

import numpy as np

from caf_core import run_caf_pipeline

covariance = np.array(
    [
        [1.0, 0.1, 0.2, 0.0],
        [0.1, 0.8, 0.0, 0.1],
        [0.2, 0.0, 0.9, 0.2],
        [0.0, 0.1, 0.2, 0.7],
    ],
    dtype=float,
)
expected_returns = np.array([1.2, 0.3, 1.0, 0.2], dtype=float)

communities = [np.array([0, 1]), np.array([2, 3])]
subproblem_cardinalities = [1, 1]


def get_model_func(adjusted_returns, sub_covariance, budget, risk_factor):
    model = SimpleNamespace()
    weights = [SimpleNamespace(name=f"x_{idx}") for idx in range(len(adjusted_returns))]
    model.adjusted_returns = np.asarray(adjusted_returns, dtype=float)
    model.sub_covariance = np.asarray(sub_covariance, dtype=float)
    model.budget = int(budget)
    model.risk_factor = float(risk_factor)
    return model, weights


def optimizer_factory(num_reads=None, seed=None):
    del num_reads, seed

    def optimizer(model, weights):
        del weights
        scores = np.asarray(model._caf_adjusted_returns, dtype=float)
        budget = int(model._caf_budget)
        chosen = np.argsort(-scores, kind="mergesort")[:budget]
        return chosen.tolist(), 0.0, 0.0

    return optimizer


result = run_caf_pipeline(
    covariance_matrix=covariance,
    expected_returns=expected_returns,
    communities=communities,
    subproblem_cardinalities=subproblem_cardinalities,
    get_model_func=get_model_func,
    optimizer_factory=optimizer_factory,
    risk_factor=0.5,
    iterations=4,
    patience=2,
    use_incremental=True,
    seed=7,
)

print(result["selected_indices"])
print(result["metrics"])
```

### Adapter requirements

- `get_model_func(...)` should return `(model, weights)` for one local subproblem.
- `optimizer_factory(...)` should return a callable `optimizer(model, weights)`.
- The local solver output may be either:
  - a binary vector of local length, or
  - a list of selected local indices.
- CAF normalizes the local output to an exact-cardinality binary decision before the global acceptance step.
- If no initial portfolio is supplied, CAF first solves the independent local
  problems to establish feasibility. A supplied initial portfolio must satisfy
  every community budget; invalid partitions and portfolios raise `ValueError`.
- With `return_diagnostics=True`, proposal records distinguish raw backend
  feasibility from any cardinality repair. A repaired vector is not evidence
  that a backend actually sampled it.

### Changing the upstream decomposition

CAF does not require the bundled `dcmppln` pipeline. If you already have your own preprocessing or clustering workflow, you can pass its output directly through `communities` and `subproblem_cardinalities`. The vendored `dcmppln` code is included only because it is the default path used in the paper reproduction.

## Reproducing the paper

The canonical arXiv v2 protocol uses classical community capacity 100 and
quantum-simulation capacity 16. These are declared benchmark resource scenarios,
not measured hardware limits. The complete batch has 345 task records, including
20 fresh SA runs for each historical panel. The current primary classical
experiment adds one Gurobi pipeline per panel, using Gurobi for both static
initialization and refinement:

The complete batch requires the quantum and Gurobi extras
(`pip install -e ".[quantum,gurobi]"`) and a usable local Gurobi license.

```bash
python -m experiments.run_revision --workers 4 --out-dir revision_results_v2
python -m experiments.quantum.run_frozen_ablation \
  --n-assets 40 --max-community-size 16 --qaoa-p 2 \
  --qaoa-restarts 6 --qaoa-maxiter 80 --warmstart --out-dir revision_frozen_v2
python manuscript/run_sampling_control.py --results revision_results_v2 \
  --out-dir sampling_controls_v2 --workers 4
python manuscript/run_gurobi_main.py --results revision_results_v2 --out-dir gurobi_main_v2
python manuscript/validate_gurobi_main.py --results gurobi_main_v2 --sa-results revision_results_v2
python manuscript/run_coordination_study.py --out-dir coordination_study_v2
python manuscript/validate_coordination_study.py --results coordination_study_v2
```

The [release provenance](docs/CODE_RELEASE.md) identifies the frozen experimental
code and retained independent validation. The experiment snapshot passed 96
tests; the maintained server release adds hardware and Gurobi-adapter regression
checks. Pending manuscript sources and PDFs are held locally by the authors and
are not included in this code release. No arXiv submission is performed here.

The primary Gurobi flows improve the objective by 18.47% on 2016 and 5.24% on
2018. Repeated unshifted solves leave both initial portfolios unchanged. All
440 local solves reach their requested gap; full-matrix replay validates every
proposal and commit. Local certificates do not prove full-portfolio optimality.
This primary emphasis was selected after examining the SA controls; their full
evidence remains in the manuscript.

In the retained SA experiments, mean paired objective gains are 12.54% on
2016 and 1.50% on 2018. The additional 40 sampling-matched controls qualify this
result: Full CAF outperforms its no-injection ablation in all 20 pairs on 2016;
the ablation outperforms full CAF in all 20 pairs on 2018. Both retain CAF's
full-global objective verification and greedy sequential acceptance. Both
improve static initialization. This qualifies the incremental benefit of
injection in SA, without showing static decomposition superior to CAF or
independently isolating the shared acceptance mechanism.

The supplementary update-timing ablation and capacity/subset follow-up is complete: 24 configurations
(22 distinct input/partition/budget configurations) match final CAF/BCD
objectives after six sweeps. BCD reaches the same endpoint in fewer rounds in
9 cases and equal rounds in 15. Four-worker CAF is faster in only 6 of 24
single timings, so no consistent local CPU speedup is established. All 5,320
local solves and 5,094 proposals pass independent replay. See
[the full follow-up](docs/COORDINATION_STUDY.md).

The v2 scope is implementation correction and revalidation. BCD remains a
technical foundation; the latest-context variant is an update-timing ablation.
The body summarizes the result and the appendix retains all detailed tables.
See [code release provenance](docs/CODE_RELEASE.md).

The standalone scripts below support additional configurations and exploration;
exact manuscript numbers come from the canonical batch and its manifests.

### Standalone large-scale SA benchmark

```bash
python -m experiments.classical.run_benchmark \
  --data-dir data/russell3000_subset \
  --runs 20 \
  --skip-q-sweep \
  --out-dir reproduce_outputs/benchmark_2016
```

### Standalone scaling and deterministic Gurobi cross-check

```bash
python -m experiments.classical.run_scaling \
  --data-dir data/russell3000_subset \
  --runs 20 \
  --solvers sa,gurobi \
  --out-dir reproduce_outputs/scaling_main
```

### Standalone quantum compatibility on folded subproblems

```bash
python -m experiments.quantum.run_sqa \
  --n-assets 40 \
  --runs 20 \
  --include-sqa-reference \
  --skip-qaoa \
  --out-dir reproduce_outputs/sqa

python -m experiments.quantum.run_qaoa \
  --n-assets 40 \
  --runs 20 \
  --qaoa-p 3 \
  --qaoa-restarts 6 \
  --qaoa-maxiter 80 \
  --iterations 4 \
  --patience 2 \
  --out-dir reproduce_outputs/qaoa_p3
```

### Frozen simulator study and optional hardware interface

Start with the offline community listing and simulator preparation:

```bash
python -m experiments.quantum.run_hardware_qaoa --list-communities
python -m experiments.quantum.run_hardware_qaoa --community 3 --prepare-only
```

The hardware driver now uses the corrected capacity-16 partition and aligned
local/global q=0.5 by default. `--local` checks the real pyqpanda3 CPUQVM;
`--submit` explicitly submits a QPU job. Cloud credentials come from
`ORIGINQ_API_TOKEN` or an optional ignored `experiments/config.py`. Job IDs and
preparation data are saved before polling. Follow the complete
[server hardware guide](docs/HARDWARE_RUNBOOK.md), including raw-count validation
and timeout handling. The subsequent [four-community hardware audit](docs/HARDWARE_RESULTS.md)
includes raw WK_C180 batches, saved-parameter distribution reconstruction and
exact uniform controls. Two best-of-sample passes do not establish a QAOA
sampling benefit or full hardware execution of CAF.

## Reproducibility notes

- The default reproduction path uses the vendored `dcmppln/` tree committed in this repository.
- The classical and simulator-based scripts are seed-controlled.
- The hardware study is protocol-reproducible, but raw counts are not expected to be bitwise identical across reruns because hardware calibration and noise fluctuate over time.
- Wall-clock times are environment-dependent and should be treated as indicative only.

## Data and dependency provenance

- `data/russell3000_subset/` contains the bundled benchmark inputs used in the paper.
- `third_party/dcmppln.VENDORED.md` records the provenance of the vendored `dcmppln` code and the local compatibility fixes applied in this release.

## Acknowledgement

CAF builds on JPMorgan Chase & Co.'s open-source `dcmppln` decomposition pipeline for RMT denoising and community detection. The bundled benchmark instance is derived from the `dcmppln` test data release.

External `--jpm-repo` overrides for revised scripts must support the `newman`
method. Unpatched historical upstream silently falls back to its old spectral
routine for unknown method names; the maintained loader rejects that override.
Use the immutable historical commit for exact legacy reproduction.
