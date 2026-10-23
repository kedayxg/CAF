# Changelog

## Unreleased maintenance

- Scope arXiv v2 as an implementation correction and revalidation. Keep BCD as
  a technical foundation and latest-context updates as an appendix ablation;
  retain all capacity/subset and resource results without adding experiments.
- Complete 24 predeclared CAF/BCD capacity/subset configurations: matched final
  objectives in all cases, fewer BCD endpoint rounds in nine, no consistent
  four-worker CPU speedup. Validate 5,320 local solves and replay 5,094 proposals.
  Preserve all results and narrow Benders/Dantzig–Wolfe wording in the paper.

- Add independently audited Gurobi primary experiments on the full 2016/2018
  panels, using Gurobi for initialization and refinement. Gains are 18.47% and
  5.24%; repeated unshifted controls remain unchanged. Verify all 440 local
  solves and replay 400 proposals. Retain complete SA evidence and explicitly
  document that the backend emphasis changed after seeing the SA controls.

- Complete the arXiv v2 capacity-100/16 protocol: 345 fresh task records, a
  frozen-state example and 40 sampling-matched controls. Independently verify
  730 portfolios, 305 histories and 1,582 accepted-update deltas; all 96 tests
  pass without expected failures. Qualify the empirical claim using the 2018
  control result, where CAF without injection outperforms full CAF in all 20 pairs.
  Prepare the reviewed PDF and matching source ZIP; no arXiv upload performed.
- Adopt pinned igraph Newman leading-eigenvector clustering as the corrected
  reproduction default; remove the upstream spectral source and legacy method
  selection names. Historical source remains available in the paper commit.
  Keep graph diagonals with half-weight loops to preserve matrix modularity.
- Share capacity-oriented igraph recursion between `Clustering(max_community_size=...)`
  and the two-panel capacity experiment; fail explicitly on unsplittable
  oversized blocks. Replace retired legacy expected failures with active
  igraph validation, index-mapping and capacity regression checks.
- Add checkpointed revision reruns and incorporate the supplied manuscript source.
- Isolate each Gurobi solve's temporary LP file for concurrent experiments.
- Add an independent corrected spectral experiment with exactly nine communities,
  an explicit balanced continuation policy, per-seed SA/CAF results and checks.
- Add optional pinned igraph/NetworkX dependencies and a three-method library
  benchmark with checked self-loop, weighted-degree and modularity conventions.
- Honor risk rescaling and interpret partition labels as communities.
- Establish and validate CAF feasibility before monotonic refinement; distinguish
  repaired proposals from raw feasible samples.
- Preserve zero budgets and zero risk factors, pass exact scaling cardinalities,
  and require a feasible static SA sample.
- Align simulator and hardware cost gates with the target Pauli convention,
  retaining an explicit plain-QAOA equivalence regression.
- Require successful raw hardware count batches and ignore unsampled zero-count
  bitstrings; remove probability fallback from this protocol.
- Fix frozen data paths, simulator-only target fallback, and suppression seeds.
- Align frozen local/global q by default; retain norm rescaling as an explicit option.
- Pass a nonzero transverse-field schedule to the path-integral sampler; the
  omitted field previously defaulted to zero. Rerun the complete SQA comparison.
- Keep a no-improvement frozen fallback marked simulator-only even when its
  local dimension is small enough for hardware.
- Exclude tied pairs from sign tests and allow statistics refresh from recorded
  per-run objectives without rerunning expensive solvers.
- Document the immutable paper commit and separate maintained numerical results.

See [maintenance notes](docs/MAINTENANCE.md) for behavior changes and validation.
