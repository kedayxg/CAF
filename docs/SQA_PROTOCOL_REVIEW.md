# Path-integral transverse-field configuration

The maintained reproduction environment pins dwave-samplers1.7.0. Its
`PathIntegralAnnealingSampler.sample` describes `Hd_field` as the sequence of
inverse-temperature-scaled transverse fields, and its implementation sets
`Hd_field=np.zeros(num_betas)` when it is omitted. `Gamma=1` alone does not
provide a nonzero transverse schedule in that branch. The C++ backend confirms
this: `localPIMC::run` assigns `invTempGamma=HdField[schedI]` at every
schedule step, overriding its initialization value. See
[localPIMC.cpp](https://github.com/dwavesystems/dwave-samplers/blob/1.7.0/dwave/samplers/sqa/src/localPIMC.cpp#L127)
and [recorded source hashes](sqa-library-default.json).

The original `SQAOptimizer` passed only reads, sweeps and seed. Thus this public
execution path uses the zero-transverse-field limit, even though the sampler
class is capable of quantum-model path-integral simulation. It should not be
presented as evidence for a finite-transverse-field SQA protocol. This diagnosis
concerns the checked pinned library and wrapper; it does not reconstruct an
unprovided original machine environment from the publication alone.

The revision explicitly passes `Hd_field=np.linspace(1,0,1000)` and `Gamma=1`
with 1,000 sweeps, retaining the pinned library's default geometric longitudinal
schedule and model-dependent automatic beta range. These are the sampler's
inverse-temperature-scaled field values. The schedule was fixed before rerunning
the corrected twenty-seed SQA comparison; it was not selected by trying schedules
and maximizing CAF gains. Zero-field checkpoints are preserved separately and
are not included in the revised finite-field table. No quantum advantage or
schedule optimality is claimed.

The regression in `tests/test_sqa_protocol.py` checks that the actual sampler
call receives a nonzero driver sequence, ends at zero and preserves the seed.
`manuscript/build_revision.py` also rejects old SQA checkpoints lacking the
recorded finite-field protocol. The complete corrected SQA repeats use the same
N=40 subset, static and CAF seeds, budgets and four-iteration limit as the other
quantum compatibility checks.

Reference: [dwave-samplers1.7.0 sampler source](https://github.com/dwavesystems/dwave-samplers/blob/1.7.0/dwave/samplers/sqa/sampler.py).
