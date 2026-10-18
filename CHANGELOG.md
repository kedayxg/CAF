# Changelog

## 0.1.1

Code maintenance based on pre-maintenance commit
`356c1852c65a185b05aed040052c5c3f750991bd`. The paper and historical experiment
artifacts are preserved. Numerical results from changed protocols require new
validation; see [the protocol notes](docs/maintenance-0.1.1.md).

### Correctness

- Start CAF from a feasible exact-cardinality selection and validate supplied
  partitions, cardinalities, and warm starts.
- Fix local/global index handling in recursive community detection.
- Preserve explicit zero budget and zero risk factor, honor risk-rebalancing
  options, and interpret community inputs as asset groups.
- Use exact K for custom-cardinality baselines and CAF comparisons.
- Exclude ties from paired sign tests and recompute each q-sweep baseline at its
  own risk factor.
- Supply an explicit transverse-field schedule for SQA instead of inheriting a
  zero-field sampler default.

### Reproducibility

- Expose and record QAOA decoding modes and explicit annealing seeds.
- Fix the frozen-ablation default path to the bundled data.
- Add regression tests for feasibility, decomposition, comparison constraints,
  statistics, annealing schedules, and ordinary-QAOA parameter equivalence.
- Retain ordinary QAOA's RZ convention and document its unrestricted-parameter
  equivalence. The warm-start circuit is preserved for separate evaluation.

## 0.1.0

Initial package version. No retrospective claim is made that a particular local
commit or decoding mode generated the published paper's tables.
