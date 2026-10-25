"""Small, auditable two-qubit probes; no cloud access unless explicitly selected.

Run with --help. The default only writes a plan. --compile-only reads chip
configuration, --local uses CPUQVM, and only --submit submits quantum jobs.
These probes do not change the frozen QAOA experiment or its acceptance rule.
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
import time


ROOT = Path(__file__).resolve().parents[2]
RESULT_FIELDS = ["probCount", "mappingQprog", "mappingQubit", "measureQubits",
                 "srcQubits", "targetCbits", "convertQProg", "QProg"]
CHIP_FIELDS = ["chip_topology_edges", "basic_gates", "compensate_angle_map",
               "patterns", "single_gate_timing", "double_gate_timing",
               "echo_gate_timing", "barrier_gate_timing", "single_gate_fidelity",
               "double_gate_fidelity", "readout_fidelity", "gate_fidelity"]


def packed(value):
    """Preserve calibration maps with tuple keys without serializing SDK objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [packed(x) for x in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: packed(v) for k, v in value.items()}
        return [{"key": packed(k), "value": packed(v)} for k, v in value.items()]
    raise TypeError(f"Unsupported diagnostic value type: {type(value).__name__}")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(packed(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def probes(suite="readout", compensation="on"):
    """Logical bit q0 is the rightmost bit in the declared ideal distribution."""
    specifications = []
    if suite in ("readout", "all"):
        for index in range(4):
            bits = f"{index:02b}"
            gates = [("X", q) for q in range(2) if index >> q & 1]
            specifications.append((f"basis_{bits}", gates, {bits: 1.0}))
        specifications.extend([
            ("h_q0", [("H", 0)], {"00": .5, "01": .5}),
            ("h_q1", [("H", 1)], {"00": .5, "10": .5}),
            ("bell", [("H", 0), ("H", 1), ("CZ", 0, 1), ("H", 1)],
             {"00": .5, "11": .5}),
        ])
    if suite in ("cz", "all"):
        for target in range(2):
            control = 1 - target
            for degrees in (0, 90):
                for use_cz in (False, True):
                    gates = [("X", control), ("H", target),
                             ("RZ", target, math.radians(degrees))]
                    if use_cz:
                        gates.append(("CZ", control, target))
                    gates.append(("H", target))
                    # P(target=1)=(1-cos(phi + pi*control*use_cz))/2.
                    p1 = (1 - math.cos(math.radians(degrees) + math.pi*use_cz))/2
                    base = 1 << control
                    ideal = {f"{base:02b}": 1-p1, f"{base | (1 << target):02b}": p1}
                    name = f"ramsey_q{target}_{'cz' if use_cz else 'reference'}_{degrees}deg"
                    specifications.append((name, gates, ideal))
    result = []
    for name, gates, ideal in specifications:
        has_cz = any(g[0] == "CZ" for g in gates)
        modes = [True, False] if compensation == "both" and has_cz else [compensation != "off"]
        for enabled in modes:
            result.append(dict(name=f"{name}_comp_{'on' if enabled else 'off'}", gates=gates,
                               status="planned", expected_probabilities=ideal, cz_phase_compensation=enabled))
    return result


def counts_checked(raw, shots, width=2):
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("Shots must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Count width must be a positive integer")
    normalized = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Count key must be a binary/hex string")
        if key.lower().startswith("0x"):
            index = int(key, 16)
        elif set(key) <= {"0", "1"}:
            index = int(key, 2)
        else:
            raise ValueError("Count key must be a binary/hex string")
        if not 0 <= index < 2**width or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Invalid raw count or bit width")
        bits = format(index, f"0{width}b")
        if value:
            normalized[bits] = normalized.get(bits, 0) + value
    if sum(normalized.values()) != shots:
        raise ValueError("Raw count total differs from requested shots")
    return normalized


def distribution_comparison(counts, ideal):
    shots = sum(counts.values())
    if shots < 1:
        raise ValueError("Empty counts")
    empirical = {f"{i:02b}": counts.get(f"{i:02b}", 0)/shots for i in range(4)}
    tv = lambda p: sum(abs(p.get(k, 0)-ideal.get(k, 0)) for k in empirical)/2
    return dict(empirical_probabilities=empirical,
                tv_assuming_q0_rightmost=tv(empirical),
                tv_after_string_reversal=tv({k[::-1]: v for k, v in empirical.items()}))


def measure_map(prog):
    return [[int(q), int(c)] for q, c in prog.get_measure_qubits_cbits()]


def build_program(core, gates):
    prog = core.QProg()
    # Ensure both qubits exist even for |00>, without needing non-native idle pulses.
    prog << core.I(0) << core.I(1)
    for name, *args in gates:
        prog << getattr(core, name)(*args)
    prog << core.measure([0, 1], [0, 1])
    return prog


def ideal_compiled_distribution(core, compiled, pair):
    """Check ideal gate evolution before device-specific pulse compensation."""
    compact = {q: i for i, q in enumerate(pair)}
    prog = core.QProg()
    n = len(pair)
    for q in range(n):
        prog << core.I(q)
    for gate in compiled.gate_operations():
        prog << gate.remap(compact)
    vm = core.CPUQVM()
    vm.run(prog, 1)
    probabilities = vm.result().get_prob_list(list(range(n)))
    result = {format(i, f"0{n}b"): 0.0 for i in range(2**n)}
    mapping = measure_map(compiled)
    if len(mapping) != n or sorted(c for _, c in mapping) != list(range(n)) or len({q for q, _ in mapping}) != n:
        raise ValueError("Expected one distinct measurement per logical bit")
    for index, probability in enumerate(probabilities):
        output = sum(((index >> compact[q]) & 1) << c for q, c in mapping)
        result[format(output, f"0{n}b")] += probability
    return result


def pulse_program(core, instruction, pair, original_measurements):
    """Replay pulse unitaries with an explicitly declared readout hypothesis.

    This does not model CZ hardware phase errors. It deliberately avoids the
    SDK instruction reader's classical-bit allocation and does not validate
    the server's bit ordering.
    """
    compact = {q: i for i, q in enumerate(pair)}
    prog = core.QProg()
    for q in range(len(pair)):
        prog << core.I(q)
    for operation in json.loads(instruction):
        name, args = next(iter(operation.items()))
        if name == "RPhi":
            q, phi, theta, *_ = args
            q = compact[q-1]
            phi, theta = math.radians(phi), math.radians(theta)
            prog << core.RZ(q, -phi) << core.RX(q, theta) << core.RZ(q, phi)
        elif name == "CZ":
            a, b, *_ = args
            prog << core.CZ(compact[a-1], compact[b-1])
        elif name != "Measure":
            raise ValueError(f"Unexpected instruction in probe: {name}")
    for q, c in original_measurements:
        prog << core.measure(compact[q], c)
    return prog


def copy_chip(transpilation, chip):
    clone = transpilation.ChipBackend()
    for name in CHIP_FIELDS:
        setattr(clone, name, getattr(chip, name))
    return clone


def prepare(core, transpilation, chip, pair, case, directory, compiler_basis="fixed", optimization_level=2):
    if isinstance(optimization_level, bool) or not isinstance(optimization_level, int) or optimization_level not in (0, 1, 2):
        raise ValueError("optimization_level must be 0, 1 or 2")
    original = build_program(core, case["gates"])
    target = copy_chip(transpilation, chip)
    # The SDK transpiler can itself apply calibration when this map is nonzero.
    # First compile an explicitly uncalibrated reference in either pipeline.
    target.compensate_angle_map = {}
    if compiler_basis == "fixed":
        # The SDK virtual-Z converter accepts X1/RZ/CZ, but rejects arbitrary
        # RPhi rotation angles produced by its generic RPhi-basis optimizer.
        target.basic_gates = ["X1", "RZ", "CZ"]
    compiled = transpilation.Transpiler().transpile(original, target, {0: pair[0], 1: pair[1]}, optimization_level, "default")
    # Retain the SDK's actual placement; don't assume an init_mapping was honored.
    if set(int(q) for q in compiled.qubits()) - set(pair):
        raise ValueError("Compiler used qubits outside the requested pair")
    probability = ideal_compiled_distribution(core, compiled, pair)
    ideal = case["expected_probabilities"]
    tv = sum(abs(probability[k]-ideal.get(k, 0)) for k in probability)/2
    if tv > 1e-9:
        raise ValueError(f"Compiled ideal distribution differs from reference: TV={tv}")
    (directory / "uncalibrated_compiled.originir").write_text(compiled.originir(17), encoding="utf-8")
    if compiler_basis == "backend" and case["cz_phase_compensation"]:
        target.compensate_angle_map = chip.compensate_angle_map
        compiled = transpilation.Transpiler().transpile(original, target, {0: pair[0], 1: pair[1]}, optimization_level, "default")
    pulse_backend = copy_chip(transpilation, chip)
    if compiler_basis == "fixed" and not case["cz_phase_compensation"]:
        pulse_backend.compensate_angle_map = {k: (0., 0.) for k in chip.compensate_angle_map}
    # In the fixed path, keep virtual-Z lowering enabled in both arms; zero only
    # calibration angles in the off arm. No mutation of the fetched chip object.
    converter_flag = compiler_basis == "fixed"
    instruction = compiled.to_instruction(pulse_backend, 1, False, converter_flag)
    # Preserve exactly the string passed to run_instruction, not a reformatted copy.
    (directory / "instruction.json").write_text(instruction, encoding="utf-8")
    (directory / "logical.originir").write_text(original.originir(17), encoding="utf-8")
    (directory / "compiled.originir").write_text(compiled.originir(17), encoding="utf-8")
    instruction_data = json.loads(instruction)
    # Check serialization against the measured program, without claiming the
    # server interprets this list in the same direction as the formatter.
    expected_measure = [q+1 for q, _ in sorted(measure_map(compiled), key=lambda x: -x[1])]
    if [x["Measure"][0] for x in instruction_data if "Measure" in x] != [expected_measure]:
        raise ValueError("Unexpected serialized measurement list; inspect SDK convention")
    replay = pulse_program(core, instruction, pair, measure_map(compiled))
    if not case["cz_phase_compensation"] or not any(any(v) for v in chip.compensate_angle_map.values()):
        replay_p = ideal_compiled_distribution(core, replay, [0, 1])
        pulse_tv = sum(abs(replay_p[k]-ideal.get(k, 0)) for k in replay_p)/2
        if pulse_tv > 1e-9:
            raise ValueError(f"Zero-calibration pulse replay differs from reference: TV={pulse_tv}")
        case["zero_calibration_pulse_tv"] = pulse_tv
    angles = [op["RPhi"][2] for op in instruction_data if "RPhi" in op]
    case.update(status="compiled", compiler_basis=compiler_basis, optimization_level=optimization_level,
                converter_compensate_pulse_argument=converter_flag,
                calibration_stage=("instruction_converter" if compiler_basis == "fixed" else "transpiler") if case["cz_phase_compensation"] else "disabled",
                transpiler_calibration_map_enabled=compiler_basis == "backend" and case["cz_phase_compensation"],
                converter_calibration_map_enabled=compiler_basis == "fixed" and case["cz_phase_compensation"],
                instruction_rphi_angles_degrees=sorted(set(round(a, 10) for a in angles)),
                instruction_non_quarter_turn_rphi_count=sum(abs(a/90-round(a/90)) > 1e-8 for a in angles),
                instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                compiled_measurement_map=measure_map(compiled),
                instruction_measurement=[x for x in instruction_data if "Measure" in x],
                compiled_operations=compiled.count_ops(), compiled_depth=compiled.depth(),
                ideal_compiled_probabilities=probability, ideal_compiled_tv=tv,
                ideal_check_scope="Before pulse compensation; not a calibrated device model")
    # Diagnostic evidence only: this SDK round trip may change classical-bit assignments.
    try:
        parsed = core.instruction_to_qprog(instruction, -1, False)
        (directory / "sdk_roundtrip.originir").write_text(parsed.originir(17), encoding="utf-8")
        case["sdk_roundtrip_measurement_map"] = measure_map(parsed)
    except Exception as exc:
        case["sdk_roundtrip_error_type"] = type(exc).__name__
    save(directory / "record.json", case)
    return compiled, instruction, replay


def field(obj, name):
    try:
        return getattr(obj, name)()
    except Exception:
        return None


def collect_job(job, case, directory, shots, interval, polls, width=2):
    path = directory / "record.json"
    case.update(status="submitted", job_id=job.job_id(), status_trace=[])
    save(path, case)
    for attempt in range(polls):
        status = job.status()
        status = getattr(status, "name", str(status))
        case["status_trace"].append(status)
        save(path, case)
        if status == "FINISHED":
            break
        if status in {"FAILED", "CANCELLED", "CANCELED", "ERROR"}:
            raise RuntimeError(f"Diagnostic job {case['job_id']} ended with {status}")
        if attempt + 1 < polls and interval:
            time.sleep(interval)
    else:
        raise TimeoutError(f"Inspect existing job {case['job_id']} before submitting again")
    # Raw counts are fetched/saved first, independently of optional mapping fields.
    result = job.result(keys=["probCount"])
    counts_list = field(result, "get_counts_list")
    raw = counts_list[0] if counts_list else field(result, "get_counts")
    case.update(raw_counts=raw, origin_data=field(result, "origin_data"),
                backend_error=field(result, "error_message"))
    save(path, case)
    if case["backend_error"]:
        raise RuntimeError("Platform reported an error; raw receipt retained")
    case["counts"] = counts_checked(raw or {}, shots, width)
    save(path, case)
    try:
        extra = job.result(keys=RESULT_FIELDS)
        case["extra_origin_data"] = field(extra, "origin_data")
        case["platform_metadata"] = {k: field(extra, k) for k in
            ("instructions", "mapping_qprog", "mapping_qubit", "measure_qubits",
             "src_qubits", "target_cbits", "timing_info")}
    except Exception as exc:
        case["optional_metadata_error_type"] = type(exc).__name__
    case["status"] = "completed"
    save(path, case)


def summarize(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    for spec in manifest["cases"]:
        path = directory / spec["name"] / "record.json"
        record = json.loads(path.read_text()) if path.exists() else spec
        row = {k: record.get(k) for k in ("name", "status", "job_id", "cz_phase_compensation", "error_type")}
        if record.get("counts"):
            counts = counts_checked(record["counts"], manifest["shots"])
            row.update(distribution_comparison(counts, record["expected_probabilities"]))
        rows.append(row)
    summary = dict(mode=manifest["mode"], rows=rows,
                   interpretation="TV values compare two decoding hypotheses; no automatic bit-order correction or hardware pass claim.")
    save(directory / "summary.json", summary)
    lines = ["# Two-qubit hardware diagnostics", "", f"Mode: {manifest['mode']}", "",
             "TV compares raw counts with the declared ideal distribution. Smaller is closer.",
             "Reversal is a hypothesis, not a correction or a pass criterion.", "",
             "| Probe | Status | TV (q0 rightmost) | TV (reversed) | Job ID |",
             "| --- | --- | ---: | ---: | --- |"]
    for row in rows:
        a = f"{row['tv_assuming_q0_rightmost']:.6f}" if "tv_assuming_q0_rightmost" in row else "—"
        b = f"{row['tv_after_string_reversal']:.6f}" if "tv_after_string_reversal" in row else "—"
        lines.append(f"| {row['name']} | {row['status']} | {a} | {b} | {row['job_id'] or '—'} |")
    (directory / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    # A failed/interrupted batch still gets a manifest of everything saved so far.
    files = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SHA256.json"}
    save(directory / "SHA256.json", files)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Default: no SDK or network")
    mode.add_argument("--local", action="store_true", help="CPUQVM with a synthetic two-qubit chip")
    mode.add_argument("--compile-only", action="store_true", help="Read real chip configuration, compile, no job submission")
    mode.add_argument("--list-pairs", action="store_true", help="Read/save real chip topology; no job submission")
    mode.add_argument("--submit", action="store_true", help="Submit the printed number of QPU jobs")
    mode.add_argument("--analyze", type=Path, help="Rebuild summary/hash manifest offline from an existing batch")
    parser.add_argument("--suite", choices=["readout", "cz", "all"], default="readout")
    parser.add_argument("--compensation", choices=["on", "off", "both"], default="on")
    parser.add_argument("--compiler-basis", choices=["fixed", "backend"], default="fixed",
                        help="fixed=X1/RZ/CZ and converter calibration; backend=legacy chip basis and transpiler calibration")
    parser.add_argument("--physical-qubits", help="Exactly two distinct, connected physical indices from SDK topology")
    parser.add_argument("--machine", default="WK_C180")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-jobs", type=int, default=24)
    parser.add_argument("--cloud-mapping", choices=["on", "off"], default="off")
    parser.add_argument("--cloud-optimization", choices=["on", "off"], default="off")
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--max-polls", type=int, default=240)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "hardware_diagnostics")
    args = parser.parse_args(argv)
    if args.analyze:
        summarize(args.analyze)
        print(args.analyze / "SUMMARY.md")
        return
    if min(args.shots, args.max_polls, args.max_jobs) < 1 or not math.isfinite(args.poll_interval) or not 0 <= args.poll_interval <= 60:
        parser.error("Positive shots/polls/job limit and a finite poll interval in [0,60] are required")
    pair = [0, 1]
    if args.physical_qubits:
        try:
            pair = [int(x.strip()) for x in args.physical_qubits.split(",")]
        except ValueError:
            parser.error("--physical-qubits must contain two comma-separated integers")
        if len(pair) != 2 or len(set(pair)) != 2 or min(pair) < 0:
            parser.error("--physical-qubits requires two distinct nonnegative indices")
    if (args.submit or args.compile_only) and not args.physical_qubits:
        parser.error("Specify a connected --physical-qubits pair from the SDK topology")
    if args.local and pair != [0, 1]:
        parser.error("--local uses the synthetic pair 0,1; use the real pair only for cloud modes")
    cases = probes(args.suite, args.compensation)
    if len(cases) > args.max_jobs:
        parser.error(f"Plan has {len(cases)} jobs, exceeding --max-jobs {args.max_jobs}")
    execution_mode = "qpu" if args.submit else "list_pairs" if args.list_pairs else "compile_only" if args.compile_only else "local" if args.local else "plan_only"
    if args.list_pairs:
        cases = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / stamp
    directory.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(schema_version=1, mode=execution_mode, status="planned", machine=args.machine,
                    shots=args.shots, planned_jobs=len(cases), planned_total_shots=len(cases)*args.shots,
                    physical_qubits=pair, git_commit=commit,
                    diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    hardware_driver_sha256=hashlib.sha256(Path(__file__).with_name("run_hardware_qaoa.py").read_bytes()).hexdigest(),
                    python_version=platform.python_version(),
                    options=dict(mapping=args.cloud_mapping == "on", optimization=args.cloud_optimization == "on",
                                 readout_amend=False, instruction_offset=1, scheduling=False),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, cases=cases)
    save(directory / "manifest.json", manifest)
    print(f"Mode={execution_mode}; {len(cases)} circuits, {args.shots} shots each, {len(cases)*args.shots} shots total.", flush=True)
    print(f"Output: {directory}", flush=True)
    if execution_mode == "plan_only":
        summarize(directory)
        return
    current = None
    try:
        from pyqpanda3 import core, transpilation, qcloud
        manifest["sdk_version"] = importlib.metadata.version("pyqpanda3")
        if args.local:
            chip = transpilation.ChipBackend()
            chip.chip_topology_edges = [[0, 1]]
            chip.basic_gates = ["RPhi", "CZ"]
            chip.single_gate_timing, chip.double_gate_timing = 20, 40
            backend, options = core.CPUQVM(), None
        else:
            # Reuse credential loading only; record and explicitly set diagnostic options.
            from experiments.quantum import run_hardware_qaoa as hardware
            hardware.setup_pyqpanda(args.machine, physical_qubits=pair if args.physical_qubits else None)
            chip, backend = hardware.CHIP_BACKEND, hardware.BACKEND
            options = qcloud.QCloudOptions()
            options.set_mapping(args.cloud_mapping == "on")
            options.set_optimization(args.cloud_optimization == "on")
            options.set_amend(False)
        save(directory / "chip_backend.json", {k: getattr(chip, k) for k in CHIP_FIELDS})
        if args.list_pairs:
            save(directory / "pairs.json", chip.chip_topology_edges)
            print("Connected physical pairs (SDK indices, no +1 adjustment):", flush=True)
            print(json.dumps(chip.chip_topology_edges), flush=True)
            manifest["status"] = "topology_saved"
            return
        if not any(set(edge) == set(pair) for edge in chip.chip_topology_edges):
            raise ValueError("Requested physical pair is not a connected chip edge")
        # Compile/check/persist the entire batch before the first paid submission.
        prepared = []
        for case in cases:
            current = case
            subdirectory = directory / case["name"]
            subdirectory.mkdir()
            case["status"] = "preparing"
            save(subdirectory / "record.json", case)
            compiled, instruction, replay = prepare(core, transpilation, chip, pair, case, subdirectory, args.compiler_basis)
            prepared.append((case, subdirectory, compiled, instruction, replay))
        manifest["status"] = "compiled"
        save(directory / "manifest.json", manifest)
        if args.compile_only:
            return
        for case, subdirectory, compiled, instruction, replay in prepared:
            current = case
            if args.local:
                backend.run(replay, args.shots)
                case["raw_counts"] = backend.result().get_counts()
                case["local_execution"] = "Serialized ideal pulse unitaries with original compiled measurement map; synthetic chip only"
                case["counts"] = counts_checked(case["raw_counts"], args.shots)
                case["status"] = "completed"
                save(subdirectory / "record.json", case)
            else:
                case["status"] = "submission_attempted"
                save(subdirectory / "record.json", case)
                job = backend.run_instruction([instruction], args.shots, options)
                collect_job(job, case, subdirectory, args.shots, args.poll_interval, args.max_polls)
            print(f"{case['name']}: completed; job={case.get('job_id', 'CPUQVM')}", flush=True)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        if current is not None:
            current.update(error_type=type(exc).__name__, execution_status_before_error=current["status"],
                           status=manifest["status"])
            save(directory / current["name"] / "record.json", current)
        raise
    finally:
        save(directory / "manifest.json", manifest)
        summarize(directory)


if __name__ == "__main__":
    main()
