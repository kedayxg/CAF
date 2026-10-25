"""Replay saved hardware parameters without reoptimizing QAOA.

Default: write an offline plan. --local uses a synthetic zero-calibration chip.
--compile-only reads chip configuration; only --submit sends a quantum job.
--p truncates the original gamma and beta halves; p=0 is state preparation only.
--analyze writes a new analysis.json without changing historical raw records.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.quantum import diagnose_hardware as diagnostic

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_parameters(source, p=None):
    """Use [gamma[:p], beta[:p]], with halves defined by the SOURCE depth."""
    original_p = source["p"]
    if isinstance(original_p, bool) or not isinstance(original_p, int) or original_p < 0:
        raise ValueError("Source p must be a nonnegative integer")
    p = original_p if p is None else p
    if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p <= original_p:
        raise ValueError(f"p must be in [0,{original_p}]; new layers are not synthesized")
    theta = list(source["theta_star"])
    if len(theta) != 2*original_p or not all(
            not isinstance(x, bool) and isinstance(x, (float, int)) and math.isfinite(x) for x in theta):
        raise ValueError("Saved theta must contain 2*source_p finite numbers")
    return p, theta[:p] + theta[original_p:original_p+p]


def validate_source(source):
    frozen_parameters(source)
    n, k = source["n"], source["K"]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (n, k)) or not 1 <= n <= 16 or not 0 <= k <= n:
        raise ValueError("Source must have 1..16 local assets and a valid integer K")
    labels = source["var_labels"]
    if len(labels) != n or len(set(labels)) != n or set(source["ising_linear"]) - set(labels):
        raise ValueError("Invalid variable ordering or Ising labels")
    numbers = list(source["ising_linear"].values())
    for a, b, value in source["ising_quadratic"]:
        if a not in labels or b not in labels:
            raise ValueError("Unknown Ising interaction label")
        numbers.append(value)
    if source["warmstart_angles"] is not None:
        if len(source["warmstart_angles"]) != n:
            raise ValueError("Warm-start angles must match n")
        numbers.extend(source["warmstart_angles"])
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("Circuit coefficients must be finite")
    covariance = np.asarray(source["covariance"], float)
    returns = np.asarray(source["expected_returns"], float)
    state = np.asarray(source["static_state"])
    group = np.asarray(source["community_indices"])
    if returns.ndim != 1 or covariance.shape != (len(returns), len(returns)) or state.shape != returns.shape:
        raise ValueError("Invalid full financial matrix or state dimensions")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(returns)) or not np.allclose(covariance, covariance.T):
        raise ValueError("Covariance must be finite and symmetric; returns must be finite")
    if not np.all(np.isin(state, [0, 1])) or group.shape != (n,) or not np.issubdtype(group.dtype, np.integer):
        raise ValueError("Invalid binary state or local asset indices")
    if len(set(group)) != n or np.any(group < 0) or np.any(group >= len(state)) or state[group].sum() != k:
        raise ValueError("Invalid community or initial local cardinality")
    for key, shape in (("local_covariance", (n, n)), ("adjusted_returns", (n,)), ("sim_candidate", (n,))):
        values = np.asarray(source[key])
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid {key}")
    candidate = np.asarray(source["sim_candidate"])
    if not np.all(np.isin(candidate, [0, 1])) or candidate.sum() != k:
        raise ValueError("Source reference candidate must be feasible")
    if not math.isfinite(source["risk_factor"]) or source["risk_factor"] < 0:
        raise ValueError("Invalid global risk factor")


def ideal_probabilities(source, p=None):
    p, theta = frozen_parameters(source, p)
    n = source["n"]
    positions = {label: i for i, label in enumerate(source["var_labels"])}
    angles = source["warmstart_angles"]
    circuit = QuantumCircuit(n)
    for i in range(n):
        circuit.ry(float(angles[i]), i) if angles is not None else circuit.h(i)
    for layer in range(p):
        gamma, beta = theta[layer], theta[p+layer]
        for label, value in source["ising_linear"].items():
            circuit.rz(-2*gamma*float(value), positions[label])
        for a, b, value in source["ising_quadratic"]:
            circuit.rzz(2*gamma*float(value), positions[a], positions[b])
        for i in range(n):
            if angles is None:
                circuit.rx(2*beta, i)
            else:
                circuit.ry(-float(angles[i]), i)
                circuit.rz(2*beta, i)
                circuit.ry(float(angles[i]), i)
    return {format(i, f"0{n}b"): float(value) for i, value in
            enumerate(Statevector.from_instruction(circuit).probabilities())}


def metrics(source, counts, ideal, tau=.7):
    """Rightmost c0 maps to local position 0; always evaluate actual samples."""
    n, k = source["n"], source["K"]
    shots = sum(counts.values())
    counts = diagnostic.counts_checked(counts, shots, n)
    covariance, returns = np.asarray(source["covariance"]), np.asarray(source["expected_returns"])
    state, group = np.asarray(source["static_state"], int), np.asarray(source["community_indices"], int)
    q, q_eff = source["risk_factor"], source.get("q_eff", source["risk_factor"])
    block = covariance[np.ix_(group, group)]
    cov_state = covariance @ state
    local_cov, adjusted = np.asarray(source["local_covariance"]), np.asarray(source["adjusted_returns"])
    def delta(bits):
        change = bits-state[group]
        return float(q*(change @ block @ change + 2*cov_state[group] @ change)-returns[group] @ change)
    sim = np.asarray(source["sim_candidate"], int)
    sim_key = "".join(str(int(x)) for x in sim[::-1])
    d_sim = delta(sim)
    if not np.isclose(d_sim, source["dO_sim"], rtol=1e-9, atol=1e-10):
        raise ValueError("Source reference delta differs from full financial objective")
    best_local = best_global = None
    for key, count in counts.items():
        bits = np.asarray([int(x) for x in key[::-1]])
        if bits.sum() != k:
            continue
        entry = dict(count_key=key, bits_q0_to_qn=bits.tolist(), frequency=count,
                     dO=delta(bits), local_energy=float(q_eff*bits @ local_cov @ bits-adjusted @ bits))
        if best_local is None or entry["local_energy"] < best_local["local_energy"]:
            best_local = entry
        if best_global is None or entry["dO"] < best_global["dO"]:
            best_global = entry
    accepted_sim = d_sim <= -1e-5
    accepted_hw = bool(best_local and best_local["dO"] <= -1e-5)
    ratio = best_local["dO"]/d_sim if best_local and d_sim != 0 else None
    empirical = np.array([counts.get(key, 0)/shots for key in ideal])
    probability = np.array(list(ideal.values()))
    return dict(shots=shots, tv_hw_ideal=float(abs(empirical-probability).sum()/2),
                tv_hw_uniform=float(abs(empirical-1/(2**n)).sum()/2),
                sim_candidate_count_key=sim_key, sim_candidate_frequency=counts.get(sim_key, 0),
                sim_candidate_ideal_probability=ideal[sim_key], sim_candidate_expected_hits=shots*ideal[sim_key],
                best_feasible_by_local_energy=best_local, best_feasible_by_global_dO=best_global,
                best_feasible_dO=None if best_local is None else best_local["dO"],
                dO_sim=d_sim, accepted_sim=bool(accepted_sim), accepted_hw=accepted_hw, ratio_dO=ratio,
                criterion_pass=bool(accepted_hw and accepted_sim and ratio is not None and ratio >= tau),
                criterion_reference="Original frozen candidate; truncated layers are not reoptimized")


def analyze(directory, source_path=None):
    """Write analysis.json; do not replace historical record.json or its metrics."""
    path = directory / "record.json"
    record = json.loads(path.read_text())
    source_path = source_path or directory / "source_record.json"
    if sha(source_path) != record["source_sha256"]:
        raise ValueError("Frozen source SHA256 mismatch")
    source = json.loads(source_path.read_text())
    validate_source(source)
    p, theta = frozen_parameters(source, record.get("p"))
    if p and record.get("theta_star") != theta:
        raise ValueError("Recorded truncated theta is ambiguous or differs from saved-layer parameters")
    if record["status"] != "completed":
        raise ValueError("Only completed count batches can be analyzed")
    if record.get("mode", "qpu") != "local" and record.get("status_trace", [None])[-1] != "FINISHED":
        raise ValueError("A QPU batch needs a FINISHED receipt")
    shots = record["shots"]
    raw = record.get("raw_counts", record.get("counts_full", record.get("counts", {})))
    counts = diagnostic.counts_checked(raw, shots, source["n"])
    if record.get("counts") is not None and counts != diagnostic.counts_checked(record["counts"], shots, source["n"]):
        raise ValueError("Raw counts and stored normalized counts disagree")
    result = metrics(source, counts, ideal_probabilities(source, p), record.get("tau", .7))
    old = record.get("metrics", {})
    differences = {}
    for key in ("sim_candidate_frequency", "sim_candidate_ideal_probability", "criterion_pass"):
        if key not in old:
            continue
        equal = np.isclose(old[key], result[key], rtol=1e-10, atol=1e-12) if key.endswith("probability") else old[key] == result[key]
        if not equal:
            differences[key] = {"stored": old[key], "recomputed": result[key]}
    analysis = dict(record_sha256=sha(path), source_sha256=sha(source_path), p=p, theta_used=theta,
                    analyzer_sha256=sha(Path(__file__)), metrics=result, stored_metric_differences=differences)
    diagnostic.save(directory / "analysis.json", analysis)
    return analysis


def compile_replay(source, p, core, transpilation, chip, physical, level, directory):
    from experiments.quantum import run_hardware_qaoa as hardware
    _, theta = frozen_parameters(source, p)
    h, interactions = source["ising_linear"], {(a, b): c for a, b, c in source["ising_quadratic"]}
    original = hardware.build_qaoa_prog(h, interactions, theta, p, source["warmstart_angles"], source["n"], source["var_labels"])
    original << core.measure(list(range(source["n"])), list(range(source["n"])))
    target = diagnostic.copy_chip(transpilation, chip)
    target.compensate_angle_map = {}
    target.basic_gates = ["X1", "RZ", "CZ"]
    compiled = transpilation.Transpiler().transpile(original, target, dict(enumerate(physical)), level, "default")
    if set(int(q) for q in compiled.qubits()) - set(physical):
        raise ValueError("Compiler used qubits outside the requested block")
    expected = ideal_probabilities(source, p)
    actual = diagnostic.ideal_compiled_distribution(core, compiled, physical)
    tv = sum(abs(actual[k]-expected[k]) for k in expected)/2
    if tv > 1e-9:
        raise ValueError(f"Uncalibrated compiled circuit differs from ideal: TV={tv}")
    instruction = compiled.to_instruction(chip, 1, False, True)
    ops = json.loads(instruction)
    mapping = diagnostic.measure_map(compiled)
    if [x["Measure"][0] for x in ops if "Measure" in x] != [[q+1 for q, c in sorted(mapping, key=lambda x: -x[1])]]:
        raise ValueError("Unexpected serialized measurement order")
    (directory / "logical.originir").write_text(original.originir(17))
    (directory / "compiled.originir").write_text(compiled.originir(17))
    (directory / "instruction.json").write_text(instruction)
    replay = diagnostic.pulse_program(core, instruction, physical, mapping)
    details = dict(measurement_map=mapping, compiled_operations=compiled.count_ops(), compiled_depth=compiled.depth(),
                   instruction_cz_count=sum("CZ" in op for op in ops), instruction_sha256=sha(directory / "instruction.json"),
                   tv_compiled_vs_qiskit=tv, calibration_stage="instruction_converter_only",
                   compiler_basis="fixed_X1_RZ_CZ", converter_compensate_pulse_argument=True)
    return instruction, replay, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--analyze", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--p", type=int)
    parser.add_argument("--physical-qubits", help="Exactly n distinct SDK physical indices, comma-separated")
    parser.add_argument("--machine", default="WK_C180")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--optimization-level", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--max-polls", type=int, default=240)
    parser.add_argument("--tau", type=float, default=.7)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "hardware_replay")
    args = parser.parse_args(argv)
    if args.analyze:
        if args.p is not None:
            parser.error("Analysis uses the recorded depth; omit --p")
        print(json.dumps(analyze(args.analyze, args.record), indent=2))
        return
    if args.record is None:
        parser.error("--record is required")
    source_bytes = args.record.read_bytes()
    source = json.loads(source_bytes)
    try:
        validate_source(source)
        p, theta = frozen_parameters(source, args.p)
        if args.shots < 1 or args.max_polls < 1 or not math.isfinite(args.poll_interval) or not 0 <= args.poll_interval <= 60:
            raise ValueError("Positive shots/polls and poll interval in [0,60] required")
        if not math.isfinite(args.tau) or not 0 <= args.tau <= 1:
            raise ValueError("tau must be in [0,1]")
        physical = list(range(source["n"])) if args.physical_qubits is None else [int(x) for x in args.physical_qubits.split(",")]
        if len(physical) != source["n"] or len(set(physical)) != len(physical) or min(physical) < 0:
            raise ValueError("Physical block must contain n distinct nonnegative indices")
        if (args.submit or args.compile_only) and args.physical_qubits is None:
            raise ValueError("Specify --physical-qubits for real chip compilation/submission")
        if args.local and physical != list(range(source["n"])):
            raise ValueError("--local uses synthetic physical indices 0..n-1")
        # Validate the financial reference before SDK loading or submission.
        metrics(source, {"".join(str(int(x)) for x in source["sim_candidate"][::-1]): 1}, ideal_probabilities(source, p), args.tau)
    except ValueError as exc:
        parser.error(str(exc))
    directory = args.out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "source_record.json").write_bytes(source_bytes)
    code = directory / "source_code"
    code.mkdir()
    code_hashes = {}
    for path in [Path(__file__), Path(diagnostic.__file__), Path(__file__).with_name("run_hardware_qaoa.py"), ROOT/"requirements.txt", ROOT/"requirements-hardware.txt"]:
        (code/path.name).write_bytes(path.read_bytes())
        code_hashes[path.name] = sha(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    execution_mode = "qpu" if args.submit else "compile_only" if args.compile_only else "local" if args.local else "plan_only"
    record = dict(schema_version=1, status="planned", mode=execution_mode, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  p=p, source_p=source["p"], theta_star=theta, source_theta_star=source["theta_star"],
                  parameter_policy="prefix_of_saved_gamma_beta_halves_no_reoptimization",
                  warmstart_angles=source["warmstart_angles"], n=source["n"], K=source["K"],
                  physical_qubits=physical, shots=args.shots, tau=args.tau, optimization_level=args.optimization_level,
                  machine="CPUQVM(synthetic)" if args.local else args.machine,
                  cloud_mapping=False, cloud_optimization=False, readout_amend=False,
                  git_commit=commit, code_sha256=code_hashes, python_version=platform.python_version())
    diagnostic.save(directory / "record.json", record)
    print(f"Mode={execution_mode}; p={p}/{source['p']}; one circuit, {args.shots} shots. Output: {directory}", flush=True)
    try:
        if execution_mode == "plan_only":
            return
        from experiments.quantum import run_hardware_qaoa as hardware
        hardware.setup_pyqpanda(args.machine, local=args.local, physical_qubits=None if args.local else physical)
        core, transpilation = hardware.core, hardware.transpilation
        record["sdk_version"] = importlib.metadata.version("pyqpanda3")
        record["versions"] = hardware._runtime_versions()
        if args.local:
            chip = transpilation.ChipBackend()
            chip.chip_topology_edges = [[i, j] for i in physical for j in physical if i < j]
            chip.basic_gates = ["RPhi", "CZ"]
            chip.single_gate_timing, chip.double_gate_timing = 40, 40
        else:
            chip = hardware.CHIP_BACKEND
        record["status"] = "compiling"
        diagnostic.save(directory / "record.json", record)
        diagnostic.save(directory / "chip_backend.json", {k: getattr(chip, k) for k in diagnostic.CHIP_FIELDS})
        instruction, replay, details = compile_replay(source, p, core, transpilation, chip, physical, args.optimization_level, directory)
        record.update(details, status="compiled")
        diagnostic.save(directory / "record.json", record)
        if args.compile_only:
            return
        if args.local:
            hardware.BACKEND.run(replay, args.shots)
            record["raw_counts"] = hardware.BACKEND.result().get_counts()
            record["counts"] = diagnostic.counts_checked(record["raw_counts"], args.shots, source["n"])
            record["status"] = "completed"
        else:
            options = hardware.qcloud.QCloudOptions()
            options.set_mapping(False)
            options.set_optimization(False)
            options.set_amend(False)
            record["status"] = "submission_attempted"
            diagnostic.save(directory / "record.json", record)
            job = hardware.BACKEND.run_instruction([instruction], args.shots, options)
            diagnostic.collect_job(job, record, directory, args.shots, args.poll_interval, args.max_polls, source["n"])
        record["metrics"] = metrics(source, record["counts"], ideal_probabilities(source, p), args.tau)
        diagnostic.save(directory / "record.json", record)
        analyze(directory)
        print(json.dumps(record["metrics"], indent=2))
    except BaseException as exc:
        record.update(execution_status_before_error=record["status"], error_type=type(exc).__name__,
                      status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    finally:
        diagnostic.save(directory / "record.json", record)
        diagnostic.save(directory / "SHA256.json", {p.relative_to(directory).as_posix(): sha(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SHA256.json"})


if __name__ == "__main__":
    main()
