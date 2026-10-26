"""Basis-state channel screening. Default: offline plan; only --submit runs QPU jobs.

This measures combined preparation/control/readout, not isolated gate fidelity.
An active-zero case retains four native X1 pulses per wire at optimization 0.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from experiments.quantum import diagnose_hardware as diagnostic

ROOT = Path(__file__).resolve().parents[2]
MIN_SHOTS = 1024
MIN_LOWER_SUCCESS = .90
MAX_AGE_SECONDS = 4*3600


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patterns(n):
    return ([dict(name="idle_zero", index=0, quarter_turns=[0]*n),
             dict(name="active_zero", index=0, quarter_turns=[4]*n)] +
            [dict(name=f"onehot_{q}", index=1 << q,
                  quarter_turns=[2 if j == q else 0 for j in range(n)]) for q in range(n)] +
            [dict(name="all_one", index=(1 << n)-1, quarter_turns=[2]*n)])


def wilson_lower(successes, shots):
    z = 1.959963984540054
    p = successes/shots
    return (p+z*z/(2*shots)-z*math.sqrt(p*(1-p)/shots+z*z/(4*shots*shots)))/(1+z*z/shots)


def check_instruction(instruction, physical, turns):
    """Reject eliminated controls, extra wires, CZ, or changed measurement order."""
    ops = json.loads(instruction)
    pulses = Counter()
    for op in ops:
        if len(op) != 1 or next(iter(op)) not in ("RPhi", "Measure"):
            raise ValueError("Basis screen must contain only RPhi and measurement")
        if "RPhi" in op:
            q, phi, theta, *_ = op["RPhi"]
            if q-1 not in physical or not all(math.isfinite(x) for x in (phi, theta)):
                raise ValueError("Invalid pulse or physical wire")
            if abs(theta-90.) > 1e-8:
                raise ValueError("Expected native +pi/2 pulses in screening control")
            pulses[q-1] += 1
    if [op["Measure"][0] for op in ops if "Measure" in op] != [[q+1 for q in physical[::-1]]]:
        raise ValueError("Screen measurement order changed")
    if any(pulses[q] != count for q, count in zip(physical, turns)):
        raise ValueError("Expected active pulses were removed or altered; do not submit")
    return dict(pulses)


def compile_case(core, transpilation, chip, physical, spec, directory):
    n = len(physical)
    prog = core.QProg()
    for q, turns in enumerate(spec["quarter_turns"]):
        prog << core.I(q)
        for _ in range(turns):
            prog << core.X1(q)
    prog << core.measure(list(range(n)), list(range(n)))
    target = diagnostic.copy_chip(transpilation, chip)
    target.compensate_angle_map = {}
    target.basic_gates = ["X1", "RZ", "CZ"]
    compiled = transpilation.Transpiler().transpile(prog, target, dict(enumerate(physical)), 0, "default")
    if diagnostic.measure_map(compiled) != [[q, i] for i, q in enumerate(physical)]:
        raise ValueError("Compiler changed the screening placement")
    if set(int(q) for q in compiled.qubits()) - set(physical):
        raise ValueError("Compiler used wires outside screening block")
    ideal = diagnostic.ideal_compiled_distribution(core, compiled, physical)
    expected_key = format(spec["index"], f"0{n}b")
    if abs(ideal[expected_key]-1) > 1e-9:
        raise ValueError("Compiled basis preparation differs from expected state")
    instruction = compiled.to_instruction(chip, 1, False, True)
    pulses = check_instruction(instruction, physical, spec["quarter_turns"])
    replay = diagnostic.pulse_program(core, instruction, physical, diagnostic.measure_map(compiled))
    actual = diagnostic.ideal_compiled_distribution(core, replay, list(range(n)))
    if abs(actual[expected_key]-1) > 1e-9:
        raise ValueError("Serialized native pulses differ from basis preparation")
    (directory/"logical.originir").write_text(prog.originir(17))
    (directory/"compiled.originir").write_text(compiled.originir(17))
    (directory/"instruction.json").write_text(instruction)
    return instruction, replay, dict(status="compiled", expected_key=expected_key,
        measurement_map=diagnostic.measure_map(compiled), optimization_level=0,
        compiled_operations=compiled.count_ops(), compiled_depth=compiled.depth(),
        instruction_sha256=sha(directory/"instruction.json"), native_pulses_by_physical=pulses)


def assess(directory, write=True):
    """Recompute from raw records; a local batch can never qualify for QPU use."""
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest.get("kind") != "basis_block_screen_v1":
        raise ValueError("Not a maintained basis screening batch")
    physical = manifest["physical_qubits"]
    if not 1 <= len(physical) <= 16 or len(set(physical)) != len(physical) or any(
            isinstance(q, bool) or not isinstance(q, int) or q < 0 for q in physical):
        raise ValueError("Invalid screening block")
    n, shots = len(physical), manifest["shots"]
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("Invalid screening shots")
    rows, jobs, start_times = [], [], []
    for spec in patterns(n):
        folder = directory/spec["name"]
        record = json.loads((folder/"record.json").read_text())
        if record["status"] != "completed":
            raise ValueError("All screening cases must be completed")
        expected = format(spec["index"], f"0{n}b")
        if record["expected_key"] != expected or record["measurement_map"] != [[q, i] for i, q in enumerate(physical)]:
            raise ValueError("Invalid screening state or measurement map")
        instruction = (folder/"instruction.json").read_text()
        if sha(folder/"instruction.json") != record["instruction_sha256"]:
            raise ValueError("Screen instruction hash mismatch")
        check_instruction(instruction, physical, spec["quarter_turns"])
        counts = diagnostic.counts_checked(record["raw_counts"], shots, n)
        if counts != diagnostic.counts_checked(record["counts"], shots, n):
            raise ValueError("Raw and normalized screening counts differ")
        if manifest["mode"] == "qpu":
            if record.get("backend_error") or record.get("status_trace", [None])[-1] != "FINISHED":
                raise ValueError("Screen needs a FINISHED receipt without backend error")
            payload = json.loads(record["origin_data"])
            obj = payload["obj"]
            if not payload.get("success") or obj["taskId"] != record["job_id"] or obj["taskState"] != "3" or obj["chipId"] != manifest["machine"]:
                raise ValueError("Screen receipt does not match machine/job")
            raw = json.loads(obj["probCount"][0])
            if len(raw["key"]) != len(set(raw["key"])) or len(raw["key"]) != len(raw["value"]):
                raise ValueError("Invalid platform count payload")
            if dict(zip(raw["key"], raw["value"])) != record["raw_counts"]:
                raise ValueError("Platform and saved raw counts differ")
            jobs.append(record["job_id"])
            start_times.append(obj["startTime"]/1000)
        for i, q in enumerate(physical):
            target = spec["index"] >> i & 1
            successes = sum(v for key, v in counts.items() if int(key, 2) >> i & 1 == target)
            lower = wilson_lower(successes, shots)
            rows.append(dict(case=spec["name"], physical_qubit=q, expected_bit=target,
                successes=successes, shots=shots, success_fraction=successes/shots,
                wilson_95_lower=lower, passed=lower >= MIN_LOWER_SUCCESS))
    if len(jobs) != len(set(jobs)):
        raise ValueError("Screen cases reuse a job ID")
    passed = all(row["passed"] for row in rows)
    result = dict(kind=manifest["kind"], mode=manifest["mode"], machine=manifest["machine"],
        physical_qubits=physical, shots_per_case=shots, rows=rows, basis_screen_passed=passed,
        qpu_eligible=manifest["mode"] == "qpu" and manifest["status"] == "completed" and shots >= MIN_SHOTS and passed,
        earliest_job_start_seconds=min(start_times) if start_times else None,
        latest_job_start_seconds=max(start_times) if start_times else None,
        threshold=dict(min_shots=MIN_SHOTS, per_pattern_per_channel_wilson_95_lower=MIN_LOWER_SUCCESS),
        scope="Combined basis preparation/control/readout only; not CZ, coherence or routed measurement-order certification")
    if write:
        diagnostic.save(directory/"analysis.json", result)
    return result


def require_screen(directory, machine, physical, now=None):
    """Verify archived evidence and re-evaluate raw counts, not a stored pass flag."""
    directory = Path(directory)
    hashes = json.loads((directory/"SHA256.json").read_text())
    required = {"manifest.json", "chip_backend.json"}
    for spec in patterns(len(physical)):
        required.update(f"{spec['name']}/{name}" for name in ("record.json", "instruction.json"))
    if not required <= hashes.keys():
        raise ValueError("Screen hash manifest is incomplete")
    for name, expected in hashes.items():
        path = directory/name
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError("Screen artifact SHA256 mismatch")
    result = assess(directory, write=False)
    if not result["qpu_eligible"]:
        raise ValueError("Screen failed, has insufficient shots, or is local simulation")
    if result["machine"] != machine or set(result["physical_qubits"]) != set(physical):
        raise ValueError("Screen machine or physical block differs from requested replay")
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    if now-result["earliest_job_start_seconds"] > MAX_AGE_SECONDS or result["latest_job_start_seconds"] > now+300:
        raise ValueError("Screen is older than four hours or has future-dated receipts")
    return dict(machine=machine, physical_qubits=result["physical_qubits"],
        manifest_sha256=sha(directory/"manifest.json"), hashes_sha256=sha(directory/"SHA256.json"),
        earliest_job_start_seconds=result["earliest_job_start_seconds"],
        worst_wilson_95_lower=min(row["wilson_95_lower"] for row in result["rows"]),
        scope=result["scope"])


def archive_screen(source, destination):
    destination.mkdir()
    for name in list(json.loads((source/"SHA256.json").read_text()))+["SHA256.json"]:
        path = source/name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Invalid evidence path")
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--analyze", type=Path)
    parser.add_argument("--physical-qubits", help="SDK indices in probe c0..cn order; 1..16 distinct wires")
    parser.add_argument("--machine", default="WK_C180")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--max-jobs", type=int, default=20)
    parser.add_argument("--order-seed", type=int, default=20261026)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--max-polls", type=int, default=240)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"results/hardware_screen")
    args = parser.parse_args(argv)
    if args.analyze:
        result = assess(args.analyze, write=False)
        diagnostic.save(args.analyze/"reanalysis.json", result)
        print(json.dumps(result, indent=2))
        return
    try:
        physical = [int(q.strip()) for q in (args.physical_qubits or "0,1").split(",")]
        if not 1 <= len(physical) <= 16 or len(set(physical)) != len(physical) or min(physical) < 0:
            raise ValueError("Specify 1..16 distinct nonnegative physical indices")
        if (args.submit or args.compile_only) and not args.physical_qubits:
            raise ValueError("Cloud modes require --physical-qubits")
        if args.local and physical != list(range(len(physical))):
            raise ValueError("Local mode uses synthetic indices 0..n-1")
        if min(args.shots, args.max_polls, args.max_jobs) < 1 or not math.isfinite(args.poll_interval) or not 0 <= args.poll_interval <= 60:
            raise ValueError("Positive shots/polls/job limit and poll interval in [0,60] required")
        if args.submit and args.shots < MIN_SHOTS:
            raise ValueError("QPU basis screening requires at least 1024 shots per pattern")
        cases = patterns(len(physical))
        if len(cases) > args.max_jobs:
            raise ValueError("Screen exceeds --max-jobs")
    except ValueError as exc:
        parser.error(str(exc))
    execution_mode = "qpu" if args.submit else "compile_only" if args.compile_only else "local" if args.local else "plan_only"
    directory = args.out_dir/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True)
    code = directory/"source_code"
    code.mkdir()
    for path in (Path(__file__), Path(diagnostic.__file__), Path(__file__).with_name("run_hardware_qaoa.py"), ROOT/"requirements.txt", ROOT/"requirements-hardware.txt"):
        shutil.copyfile(path, code/path.name)
    order = list(range(len(cases)))
    random.Random(args.order_seed).shuffle(order)
    manifest = dict(kind="basis_block_screen_v1", mode=execution_mode, status="planned", machine=args.machine,
        physical_qubits=physical, shots=args.shots, cases=cases, submission_order=order, order_seed=args.order_seed,
        planned_jobs=len(cases), planned_total_shots=len(cases)*args.shots, optimization_level=0,
        cloud_mapping=False, cloud_optimization=False, readout_amend=False)
    diagnostic.save(directory/"manifest.json", manifest)
    print(f"Mode={execution_mode}; {len(cases)} jobs, {args.shots} shots each. Output: {directory}", flush=True)
    current = None
    try:
        if execution_mode == "plan_only":
            return
        from experiments.quantum import run_hardware_qaoa as hardware
        # A singleton backend subgraph has no edges; the SDK cannot place it.
        # Fetch full topology, then constrain and verify the actual probe wires.
        hardware.setup_pyqpanda(args.machine, local=args.local)
        core, transpilation = hardware.core, hardware.transpilation
        manifest["versions"] = hardware._runtime_versions()
        if args.local:
            chip = transpilation.ChipBackend()
            size = max(2, len(physical))
            chip.chip_topology_edges = [[i, j] for i in range(size) for j in range(i+1, size)]
            chip.basic_gates = ["X1", "RZ", "CZ"]
            chip.single_gate_timing, chip.double_gate_timing = 40, 40
        else:
            chip = hardware.CHIP_BACKEND
        diagnostic.save(directory/"chip_backend.json", {k: getattr(chip, k) for k in diagnostic.CHIP_FIELDS})
        prepared = []
        for spec in cases:
            folder = directory/spec["name"]
            folder.mkdir()
            instruction, replay, current = compile_case(core, transpilation, chip, physical, spec, folder)
            current["name"] = spec["name"]
            diagnostic.save(folder/"record.json", current)
            prepared.append((folder, instruction, replay, current))
        manifest["status"] = "compiled"
        diagnostic.save(directory/"manifest.json", manifest)
        if args.compile_only:
            return
        options = None
        if args.submit:
            options = hardware.qcloud.QCloudOptions()
            options.set_mapping(False)
            options.set_optimization(False)
            options.set_amend(False)
        # Every case must compile and retain its active pulses before the first job.
        for index in order:
            folder, instruction, replay, current = prepared[index]
            if args.local:
                hardware.BACKEND.run(replay, args.shots)
                current["raw_counts"] = hardware.BACKEND.result().get_counts()
                current["counts"] = diagnostic.counts_checked(current["raw_counts"], args.shots, len(physical))
                current["status"] = "completed"
            else:
                current["status"] = "submission_attempted"
                diagnostic.save(folder/"record.json", current)
                job = hardware.BACKEND.run_instruction([instruction], args.shots, options)
                diagnostic.collect_job(job, current, folder, args.shots, args.poll_interval, args.max_polls, len(physical))
            diagnostic.save(folder/"record.json", current)
        manifest["status"] = "completed"
        diagnostic.save(directory/"manifest.json", manifest)
        result = assess(directory)
        print(f"Basis screen passed={result['basis_screen_passed']}; QPU eligible={result['qpu_eligible']}")
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        diagnostic.save(directory/"manifest.json", manifest)
        diagnostic.save(directory/"SHA256.json", {p.relative_to(directory).as_posix(): sha(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SHA256.json"})


if __name__ == "__main__":
    main()
