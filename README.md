# CAF: Context-Aware Folding

**Context-Aware Folding (CAF)** is a hybrid quantum-classical coordination layer for decomposed portfolio optimization. Static decomposition omits cross-cluster covariance during local solves; CAF re-injects this missing global information as a state-dependent linear bias and commits updates under a monotonic acceptance rule.

This repository is the public code release accompanying the CAF paper. It contains:

- the CAF engine in `caf_core/`
- the paper-facing reproduction scripts in `experiments/`
- the bundled benchmark inputs in `data/russell3000_subset/`
- a vendored copy of JPMorgan's `dcmppln` dependency used by the default reproduction path

## Repository layout

```text
codes_official/
├── caf_core/
├── data/russell3000_subset/
├── dcmppln/
├── experiments/
│   ├── _common.py
│   ├── config.example.py
│   ├── classical/
│   │   ├── run_benchmark.py
│   │   ├── run_scaling.py
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

### Changing the upstream decomposition

CAF does not require the bundled `dcmppln` pipeline. If you already have your own preprocessing or clustering workflow, you can pass its output directly through `communities` and `subproblem_cardinalities`. The vendored `dcmppln` code is included only because it is the default path used in the paper reproduction.

## Reproducing the paper

### Table I: large-scale SA benchmark

```bash
python -m experiments.classical.run_benchmark \
  --data-dir data/russell3000_subset \
  --runs 20 \
  --skip-q-sweep \
  --out-dir reproduce_outputs/benchmark_2016
```

### Table II: scaling and deterministic Gurobi cross-check

```bash
python -m experiments.classical.run_scaling \
  --data-dir data/russell3000_subset \
  --runs 20 \
  --solvers sa,gurobi \
  --out-dir reproduce_outputs/scaling_main
```

### Table III: quantum compatibility on folded subproblems

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

### Table IV: frozen hardware executability study

`run_hardware_qaoa.py` requires a local `experiments/config.py`. Create it by copying `experiments/config.example.py`, then provide the hardware token through `ORIGINQ_API_TOKEN`.

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
