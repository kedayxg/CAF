# Maintenance protocol for 0.1.1

This maintenance version updates the code and its validation. It does not modify
the published paper or overwrite archived experiments. The repository state
before maintenance is commit `356c1852c65a185b05aed040052c5c3f750991bd`.
The repository alone does not establish the exact artifact used for the paper;
retain the original manuscript, commands, parameter files, raw outputs, and
hardware job records together when reproducing its results.

## Ordinary QAOA convention

The ordinary QAOA implementation retains its positive linear RZ angles. For the
dimod convention `s = 2x - 1`, the direct target Hamiltonian in the computational
basis has negative linear Z terms. Let `P = X` on every qubit. The current cost
Hamiltonian is `P H_target P`. The ordinary all-Hadamard initial state is invariant
under P, and its X mixer commutes with P. Shifting the last mixer parameter beta
by pi/2 therefore gives the same output probability distribution as the direct
target convention, up to global phase. This holds at every depth when beta is
unrestricted, as it is in the current classical optimizer.

Comparing the two conventions at unchanged parameters does not demonstrate an
ordinary-QAOA optimization defect. Their complete variational state families are
equivalent. This does not guarantee identical fixed-seed, finite-budget optimizer
trajectories after a sign change: the initial points would also have to be mapped.
Changing the RZ sign silently would compromise historical reproducibility.

The initial state and tilted mixer of a fixed warm-start problem do not generally
have this symmetry. This release preserves that circuit as well; the ordinary
equivalence proof is not a validation of the warm-start cost convention.
Simulator/hardware executability comparisons must state the actual circuit,
initialization, shared parameters, and decoding protocol. A separate controlled
comparison is required before changing this protocol or interpreting its effect
on an archived hardware result.

## Protocols that can change results

- Feasible CAF initialization replaces an unconstrained all-zero starting point
  when no explicit initial selection is provided. Invalid partitions and warm
  starts are rejected instead of being accepted silently.
- Community recursion maps local indices back to global asset indices.
- Custom cardinality experiments solve the exact same K in both comparison arms.
- Paired sign tests omit tied objective differences. Q sweeps reoptimize the
  static baseline at each q before comparing it with CAF.
- Risk-rebalancing options are honored and operate on actual community index
  groups. Explicit zero budget and zero risk factor retain their meaning.
- SQA uses an explicit nonzero transverse-field schedule. This is a protocol
  update from the previous sampler defaults; historical zero-field outputs must
  not be relabeled as results of the new schedule.
- QAOA experiment commands expose the decoding mode. `shots` uses the best
  feasible sampled state; `argmax_prob` selects the most probable feasible state.
  Choose the mode matching the original recorded protocol for an archived run;
  the presence of a legacy comment alone does not identify a published protocol.

The SQA reference schedule is `linear_normalized`, version 1. With local Ising
scale `L = max_i(|h_i| + sum_j |J_ij|)` (or 1 for an all-zero BQM), its beta-scaled
fields are `Hp = 5*s/L` and `Hd = 5*(1-s)` by default, for a linear sweep from
`s=0` to `s=1`. The API exposes positive `inverse_temperature` and
`transverse_field` settings; at least two sweeps are required to include both a
nonzero starting transverse field and a zero-field endpoint. Experiment summaries
record the schedule law/version and decoding mode. This reference schedule is
explicit and reproducible; it is not presented as a tuned optimum or as a
reconstruction of an undocumented historical schedule.

Record the code commit, command, seeds, solver configuration, and newly generated
summary files for any comparison. Rerun the affected experiments before asserting
that a corrected protocol reproduces a published quantitative claim. A code-only
maintenance release neither establishes an error in the paper nor establishes
that every reported result is unchanged.
