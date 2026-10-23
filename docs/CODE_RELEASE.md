# Maintained code release

- `main`: restored v1 implementation while v2 matures separately.
- [`v2-development`](https://github.com/kedayxg/CAF/tree/v2-development): corrected implementation, reproduction material and server hardware workflow; merge into main only when ready.
- [`archive/arxiv-v1`](https://github.com/kedayxg/CAF/tree/archive/arxiv-v1): unchanged original main, commit `356c1852c65a185b05aed040052c5c3f750991bd`.
- [`17253da`](https://github.com/kedayxg/CAF/tree/17253dab077d6f7cc77520d09908d0a3ddb90195): corrected implementation corresponding to the completed classical/simulator experiments, before the final server hardware adapter maintenance.

The completed experiment records retain source fingerprint `9fc43f0fe1de39fcefb2dd021cdd99efab882ca98d37df1055a2669d305fe9d8`. The frozen [code archive](CAF_v2_code_snapshot.zip) and [manifest](code-snapshot-sha256.json) retain that experiment snapshot. Historical result fingerprints and raw records are not relabeled as runs of the newer hardware driver.

Main was restored using a new commit, without rewriting published history. The development branch descends from that restoration and reapplies the v2 tree, so a later ordinary merge can bring in the complete v2 changes. Existing server checkouts retain their code; switch explicitly to `v2-development` when updating v2.

After that snapshot, maintenance changes the hardware entry point, the optional Gurobi adapter, and their regression tests. CAF's coordination equations, proposal/commit engine, Newman partitioner, SA backend and simulation optimizer are unchanged. The Gurobi adapter now honors explicit caller parameters, preserves requested variable order, rejects absent incumbents and imports Gurobi on demand. Existing default numerical experiment settings remain unchanged.

The [publication checks](code-release-validation.json) record tests, exact source fingerprints, frozen-snapshot differences and actual SDK CPUQVM checks. CPUQVM evidence is local simulation, not QPU execution. New server experiments must keep their own current source fingerprint and runtime versions.

This release includes code, dependencies, bundled inputs, tests, reproduction companions and archived numerical evidence. Pending manuscript TeX/PDF, submission metadata and internal peer-review notes are retained locally. The `manuscript/` Python tools are provided for reproduction; document compilation additionally requires the authors' separately supplied manuscript sources. Links mentioned in historical local-delivery notes may refer to author-held artifacts.

See the [hardware runbook](HARDWARE_RUNBOOK.md) for installation, preparation, local SDK validation, explicit QPU submission and job receipts. The initial code-release validation above predates the server QPU runs and remains an unchanged historical record. The subsequently supplied four frozen WK_C180 batches are independently audited in [HARDWARE_RESULTS.md](HARDWARE_RESULTS.md), including their distribution discrepancy and exact uniform controls. They do not establish a sampling advantage or complete hardware execution of CAF.
