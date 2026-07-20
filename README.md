# CAF: Context-Aware Folding

**Context-Aware Folding (CAF)** is a hybrid quantum-classical coordination layer for decomposed portfolio optimization. Static decomposition (e.g., JPMorgan's RMT + community detection) severs cross-cluster covariance; CAF re-injects it as a state-dependent linear bias into each local subproblem and commits updates sequentially under a monotonic non-divergence guarantee. See the paper for the full formulation.

This repository contains the CAF engine (`caf_core/`) and the benchmark reproduction scripts used in the paper.

## Repository layout

```
codes/
├── caf_core/                      # the CAF engine
│   ├── __init__.py                #   public API exports (run_caf_pipeline, evaluate_*, normalize_local_solution, ...)
│   └── solver.py                  #   context injection, monotonic acceptance, incremental v-vector update
├── experiments/
│   ├── _common.py                 # shared helpers (data load, sign_test, bootstrap_ci, to_jsonable, cardinality split, ...)
│   ├── config.py                  # local Origin Quantum API token + MACHINE_NAME  ← GITIGNORED (create from template; never commit)
│   ├── config.example.py          # template — shipped; users `cp config.example.py config.py` and fill token
│   ├── classical/                 # all large-scale CLASSICAL experiments + their figures
│   │   ├── run_benchmark.py       #   Table I (2016, N=484, 6.59%) + 2018 panel (N=1397) + q-sweep + cached/non-cached ablation
│   │   ├── run_scaling.py         #   scaling sweep (N=100/200/300/484) + Gurobi Table II
│   │   ├── run_sparsity.py        #   sparsity K-sweep (N=200, K=50/100; SA + Gurobi)
│   │   ├── run_suppression.py     #   suppression / presolve analysis (§III-E, the 88pp result)
│   │   ├── plot_objective_convergence.py  # Fig: objective convergence        (reads benchmark_summary.json)
│   │   ├── plot_risk_return.py            # Fig: risk-return trajectory       (reads benchmark_summary.json)
│   │   ├── plot_ablation.py               # Fig: ablation summary             (reads benchmark_summary.json)
│   │   ├── plot_qsweep.py                 # Fig: q-sweep tradeoff             (reads benchmark_summary.json)
│   │   ├── plot_runtime.py                # Fig: Gurobi runtime overhead      (reads scaling_summary.json)
│   │   └── plot_suppression.py            # Fig: suppression (sparsity_presolve) (reads sparsity_summary.json)
│   └── quantum/                   # all QUANTUM experiments (simulator + hardware)
│       ├── run_qaoa.py            #   Table III QAOA (qiskit exact statevector, p=2/p=3, n≤16)
│       ├── run_sqa.py             #   Table III SQA (dwave PathIntegralAnnealingSampler) + depth sweep
│       ├── run_hardware_qaoa.py   #   Table IV hardware (Origin Wukong 180 / WK_C180, warm-start, n=5)  [needs token]
│       ├── run_frozen_ablation.py #   §III-E frozen inject/no-inject ablation
│       └── _investigation/        #   archived debugging probes (counts vs taskResult, bit-order, qubit-layout) — NOT in paper
├── tests/
│   └── test_smoke.py              # correctness guard: incremental == non-incremental (bit-identical)
├── data/russell3000_subset/       # 2016 + 2018 input panels (Σ, correlation, μ .npy)
├── dcmppln/                       # vendored JPMorgan dcmppln (with the 2 compatibility patches)
├── third_party/dcmppln.VENDORED.md
├── environment.yml                # pinned conda env (the reproducible "caf" env, Python 3.11)
├── pyproject.toml / requirements.txt
└── README.md
```

### Conventions
- **`run_*.py`** runs an experiment and writes a result JSON under the configured `--out-dir`. **`plot_*.py`** reads a saved result JSON and writes a figure. Compute and plotting are decoupled (re-plot without re-running).
- **`classical/` vs `quantum/`** mirrors the paper's two perspectives: large-scale classical validation vs small-scale quantum compatibility/executability.
- **`experiments/config.py`** is a local gitignored file created from `config.example.py`; it should read the Origin Quantum API token from `ORIGINQ_API_TOKEN`. Only the hardware script (`run_hardware_qaoa.py`) needs it; classical and simulator scripts do not.
- **`quantum/_investigation/`** archives the camera-ready debugging probes — not part of the paper's reproducibility.

### Refactor status
The directory refactor below has been applied; the legacy `benchmark/`, `scaling/`, `analysis/`, and `hardware/` experiment entry points have been consolidated into `classical/` and `quantum/`.

| legacy path | current path |
|---|---|
| `benchmark/reproduce_table1.py` | `classical/run_benchmark.py` (split its `generate_plots` into the `plot_*.py` files) |
| `scaling/run_scaling.py` | `classical/run_scaling.py` |
| (sparsity K-sweep, run via scaling args) | `classical/run_sparsity.py` |
| `analysis/analyze_sparsity_presolve.py` | `classical/run_suppression.py` (+ extract `plot_suppression.py`) |
| `analysis/plot_computational_overhead.py` | `classical/plot_runtime.py` |
| (plotting inside `reproduce_table1.generate_plots`) | `classical/plot_objective_convergence.py`, `plot_risk_return.py`, `plot_ablation.py`, `plot_qsweep.py` |
| `quantum/run_qaoa_feasibility.py` | `quantum/run_qaoa.py` |
| `quantum/run_backend_supplement.py` | `quantum/run_sqa.py` |
| `hardware/validate_warmstart_qaoa.py` | `quantum/run_hardware_qaoa.py` |
| `hardware/validate_frozen_state_control.py` | `quantum/run_frozen_ablation.py` |
| `hardware/config/{config.py,config.example.py}` | `experiments/config.py` + `experiments/config.example.py` (delete the `config/` folder) |
| `hardware/_investigation/` | `quantum/_investigation/` |
| (after moves) `benchmark/ scaling/ analysis/ hardware/` | removed from the source tree after consolidation; this table remains only as a provenance map |

The active commands and imports in this README already use the current paths.

### Reproduce flow
1. `conda env create -f environment.yml && conda activate caf`
2. run a `run_*.py` (e.g. `python -m experiments.classical.run_benchmark ...`) → writes result JSON to the selected `--out-dir`
3. run the matching `plot_*.py` → writes the figure
4. paper numbers/figures match the saved JSON summaries + the figure outputs.
- `third_party/dcmppln.VENDORED.md` — provenance, upstream commit marker, and the two local compatibility patches applied to the vendored `dcmppln`.

## Installation (reproduces the paper environment)

CAF was run on Linux-64 with the pinned `caf` conda environment:

```bash
conda env create -f environment.yml   # creates an env named "caf"
conda activate caf
```

The repository already vendors the JPMorgan `dcmppln` package (RMT denoising + clustering + the SA backend), so a separate clone is not required for the default reproduction path.

`environment.yml` is the authoritative paper-aligned environment. If you cannot use conda, install the pinned Python dependencies from `requirements.txt`; they mirror the paper-facing package versions used for the regenerated camera-ready results.

```bash
pip install -r requirements.txt
```

For editable installs:

```bash
pip install -e .
```

Editable installs now use the same pinned Python package versions declared in `pyproject.toml`, but the conda environment remains the preferred route for Table I / Table II reproduction because it also fixes the compiled stack and solver builds.

If you want to run the quantum scripts through an editable install, include the optional extras:

```bash
pip install -e ".[quantum]"
```

If you also need the local Gurobi validation path or pytest helpers:

```bash
pip install -e ".[quantum,gurobi,dev]"
```

If you prefer to override the vendored copy during local development, the main scripts still accept an optional `--jpm-repo /path/to/other/dcmppln` argument.

## Minimum Self-Check

Before rerunning the paper-scale experiments, run this lightweight smoke check from the repository root with the pinned `caf` environment:

```bash
conda run -n caf python3 tests/test_smoke.py
```

This smoke check verifies that:

- the public `caf_core` API is wired correctly;
- the bundled 2016 benchmark data loads successfully;
- the vendored `dcmppln` dependency imports through the default in-repo path;
- the cached and non-cached CAF paths agree on a tiny deterministic toy instance.

It is meant to catch broken imports or packaging regressions quickly; it does not replace the full benchmark reproduction commands below.

If you already installed the optional dev dependencies, the same file also runs under `pytest`.

## Vendored `dcmppln` patches

The vendored `dcmppln` copy includes two compatibility fixes relative to the upstream archive:

1. `dcmppln/denoiser.py`, inside `Denoiser.denoise`:
   ```diff
   - C_1, C_2, C_3 = split_covariance_matrices(C, q=self.q, q_fit=self.q_fit)
   + C_1, C_2, C_3 = split_covariance_matrices(C, beta=self.q, q_fit=self.q_fit)
   ```
   The function parameter is named `beta` (the Marchenko–Pastur ratio), which is CAF's `q`.

2. `dcmppln/optimizer.py`, near the top:
   ```diff
   - #from dwave.samplers import SimulatedAnnealingSampler
   + import dimod
   + from neal import SimulatedAnnealingSampler
   ```
   `SimulatedAnnealingSampler` lives in `neal` (not `dwave.samplers`), and `dimod` is used by `SimulatedAnnealingDwave`.

## Reproducing the paper

**Table I — large-scale SA benchmark (N=484):**
```bash
python -m experiments.classical.run_benchmark \
  --data-dir data/russell3000_subset \
  --runs 20 --skip-q-sweep --out-dir reproduce_outputs/benchmark_2016
```
Drop `--skip-q-sweep` to also regenerate the q-sweep and convergence figures.

**Table III — quantum compatibility (N=40, folded subproblems n≤16):**
```bash
# SQA (annealing-style, deterministic):
python -m experiments.quantum.run_sqa \
  --n-assets 40 --runs 20 \
  --include-sqa-reference --skip-qaoa --out-dir reproduce_outputs/sqa

# QAOA at depth p (gate-based statevector):
python -m experiments.quantum.run_qaoa \
  --n-assets 40 --runs 20 --qaoa-p 3 \
  --qaoa-restarts 6 --qaoa-maxiter 80 --iterations 4 --patience 2 \
  --out-dir reproduce_outputs/qaoa_p3
```

## Reproducibility notes

- **Determinism.** `run_caf_pipeline` propagates a per-(iteration, community) `seed` to the local solver whenever the solver factory accepts a `seed` argument (detected via `inspect`). With seeding, the cached (incremental) and non-cached CAF variants produce **identical** final portfolios, and the SA path is reproducible run-to-run. The quantum scripts use their own seeded optimizers and are deterministic.
- **Quantum (Table III) reproduces exactly.** SQA results are bit-for-bit reproducible (seeded `PathIntegralAnnealingSampler`); QAOA p=2 reproduces closely; QAOA p=3 reproduces the direction but its magnitudes are sensitive to the `scipy`/`qiskit` version (p=3 carries very high variance).
- **Dependency source is explicit.** The default reproduction path uses the vendored `dcmppln/` tree committed in this repository, not a floating upstream checkout. The upstream archive marker and local compatibility patches are recorded in `third_party/dcmppln.VENDORED.md`.
- **Classical SA (Table I) is version-sensitive.** Absolute objectives depend on the `neal`/`dimod` version (default annealing schedule and the LP→CQM→BQM penalty path). Use `environment.yml` first; if you need a pip-only setup, use the pinned `requirements.txt` / `pyproject.toml` versions rather than floating upgrades. The camera-ready numbers were regenerated under this pinned stack. The CAF *mechanism* — monotonic improvement over the static baseline, lower risk, wins on every run — is robust across versions.

## Reference numbers (this pinned environment)

With `environment.yml` + the seeded scripts, the 20-run N=484 2016 benchmark (q=0.5) reproduces deterministically to:

| Method | Objective | Risk | Exp. Return |
|---|---|---|---|
| Static baseline (JPM) | 1.5453 ± 0.0212 | 3.3666 | 0.1380 |
| CAF (incremental, cached) | 1.4433 ± 0.0148 | 3.1214 | 0.1174 |
| CAF (non-cached ablation) | 1.4433 ± 0.0148 | 3.1214 | 0.1174 |

CAF improvement: **6.59%** (95% bootstrap CI [5.99%, 7.22%]), 20/20 runs improved (two-sided sign-test p = 1.91e-6). Cached and non-cached CAF reach identical portfolios.

These are the numbers reported in the paper's Table I — the camera-ready was regenerated under this pinned environment. The reviewed submission reported 2.40% under an earlier, unpinned environment; the pinned regeneration is stronger but directionally consistent (CAF < static, 20/20 wins, lower risk, cached == non-cached).

## Acknowledgement
CAF builds on JPMorgan Chase & Co.'s open-source `dcmppln` decomposition pipeline for RMT denoising and community detection. The bundled `data/russell3000_subset/` instance is taken from the `dcmppln` test data.
