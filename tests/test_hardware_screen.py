"""Reject biased, stale or synthetic hardware evidence before any QPU replay."""
import builtins
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.quantum import diagnose_hardware as diagnostic
from experiments.quantum import screen_hardware as screen


def rehash(directory):
    diagnostic.save(directory/"SHA256.json", {p.relative_to(directory).as_posix(): screen.sha(p)
        for p in directory.rglob("*") if p.is_file() and p.name != "SHA256.json"})


@pytest.fixture
def evidence(tmp_path):
    directory = tmp_path/"screen"
    directory.mkdir()
    physical, shots = [12, 13], 1024
    manifest = dict(kind="basis_block_screen_v1", mode="qpu", status="completed",
                    physical_qubits=physical, machine="WK_C180", shots=shots)
    diagnostic.save(directory/"manifest.json", manifest)
    diagnostic.save(directory/"chip_backend.json", {})
    now = datetime.now(timezone.utc).timestamp()
    for spec in screen.patterns(2):
        folder = directory/spec["name"]
        folder.mkdir()
        ops = [{"RPhi": [physical[q]+1, 0., 90., 40*i]} for q, count in enumerate(spec["quarter_turns"]) for i in range(count)]
        ops.append({"Measure": [[14, 13], 200]})
        diagnostic.save(folder/"instruction.json", ops)
        key = format(spec["index"], "02b")
        raw = {hex(spec["index"]): shots}
        job = "job-"+spec["name"]
        payload = dict(success=True, code=10000, obj=dict(taskId=job, chipId="WK_C180", taskState="3",
                       startTime=int(now*1000), probCount=[json.dumps(dict(key=list(raw), value=list(raw.values())))]))
        record = dict(name=spec["name"], status="completed", expected_key=key,
                      measurement_map=[[12, 0], [13, 1]], raw_counts=raw, counts={key: shots},
                      instruction_sha256=screen.sha(folder/"instruction.json"), job_id=job,
                      status_trace=["FINISHED"], backend_error="", origin_data=json.dumps(payload))
        diagnostic.save(folder/"record.json", record)
    rehash(directory)
    return directory


def edit_record(directory, name, change):
    path = directory/name/"record.json"
    value = json.loads(path.read_text())
    change(value)
    diagnostic.save(path, value)
    rehash(directory)


def test_default_screen_is_offline_and_declares_all_jobs(monkeypatch, tmp_path):
    real_import = builtins.__import__
    def guarded(name, *a, **kw):
        assert not name.startswith("pyqpanda3") and name != "experiments.quantum.run_hardware_qaoa"
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", guarded)
    screen.main(["--physical-qubits", "110,119", "--out-dir", str(tmp_path)])
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["mode"] == "plan_only" and manifest["planned_jobs"] == 5
    assert manifest["planned_total_shots"] == 5120


@pytest.mark.parametrize("arguments", [["--submit"], ["--submit", "--physical-qubits", "110", "--shots", "64"],
    ["--physical-qubits", "1,1"], ["--local", "--physical-qubits", "12,13"], ["--shots", "0"],
    ["--physical-qubits", "1,2", "--max-jobs", "4"], ["--poll-interval", "nan"]])
def test_invalid_screen_never_reaches_sdk(arguments, monkeypatch, tmp_path):
    from experiments.quantum import run_hardware_qaoa as hardware
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *a, **k: pytest.fail("SDK initialized"))
    with pytest.raises(SystemExit) as exc:
        screen.main(arguments+["--out-dir", str(tmp_path)])
    assert exc.value.code == 2 and not list(tmp_path.iterdir())


def test_passing_raw_evidence_is_read_only_and_archivable(evidence, tmp_path):
    before = {p.relative_to(evidence): p.read_bytes() for p in evidence.rglob("*") if p.is_file()}
    result = screen.require_screen(evidence, "WK_C180", [13, 12])
    assert result["worst_wilson_95_lower"] > .99
    assert before == {p.relative_to(evidence): p.read_bytes() for p in evidence.rglob("*") if p.is_file()}
    target = tmp_path/"copy"
    screen.archive_screen(evidence, target)
    assert screen.require_screen(target, "WK_C180", [12, 13]) == result


def test_stored_pass_flag_does_not_hide_stuck_at_one(evidence):
    diagnostic.save(evidence/"analysis.json", dict(qpu_eligible=True, basis_screen_passed=True))
    def corrupt(record):
        record["raw_counts"] = {"0x1": 1024}
        record["counts"] = {"01": 1024}
        payload = json.loads(record["origin_data"])
        payload["obj"]["probCount"] = [json.dumps(dict(key=["0x1"], value=[1024]))]
        record["origin_data"] = json.dumps(payload)
    edit_record(evidence, "idle_zero", corrupt)
    with pytest.raises(ValueError, match="Screen failed"):
        screen.require_screen(evidence, "WK_C180", [12, 13])


def test_onehot_cases_detect_swap_that_zero_and_one_patterns_cannot(evidence):
    for name, wrong in (("onehot_0", 2), ("onehot_1", 1)):
        def swap(record):
            record["raw_counts"] = {hex(wrong): 1024}
            record["counts"] = {format(wrong, "02b"): 1024}
            payload = json.loads(record["origin_data"])
            payload["obj"]["probCount"] = [json.dumps(dict(key=[hex(wrong)], value=[1024]))]
            record["origin_data"] = json.dumps(payload)
        edit_record(evidence, name, swap)
    with pytest.raises(ValueError, match="Screen failed"):
        screen.require_screen(evidence, "WK_C180", [12, 13])


@pytest.mark.parametrize("machine,physical", [("other-machine", [12, 13]), ("WK_C180", [12, 14])])
def test_other_machine_or_block_is_rejected(evidence, machine, physical):
    with pytest.raises(ValueError, match="machine or physical block"):
        screen.require_screen(evidence, machine, physical)


def test_stale_and_future_receipts_are_rejected(evidence):
    now = datetime.now(timezone.utc).timestamp()
    for clock in (now+screen.MAX_AGE_SECONDS+1, now-600):
        with pytest.raises(ValueError, match="four hours|future"):
            screen.require_screen(evidence, "WK_C180", [12, 13], now=clock)


def test_local_counts_cannot_authorize_qpu(evidence):
    path = evidence/"manifest.json"
    record = json.loads(path.read_text())
    record["mode"] = "local"
    diagnostic.save(path, record)
    rehash(evidence)
    with pytest.raises(ValueError, match="local simulation"):
        screen.require_screen(evidence, "WK_C180", [12, 13])


def test_changed_raw_artifact_and_missing_pattern_are_rejected(evidence):
    path = evidence/"onehot_0/record.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="SHA256"):
        screen.require_screen(evidence, "WK_C180", [12, 13])
    path.unlink()
    rehash(evidence)
    with pytest.raises(ValueError, match="incomplete"):
        screen.require_screen(evidence, "WK_C180", [12, 13])


def test_removed_active_control_and_changed_readout_list_are_rejected():
    instruction = json.dumps([dict(Measure=[[14, 13], 0])])
    with pytest.raises(ValueError, match="removed"):
        screen.check_instruction(instruction, [12, 13], [4, 4])
    with pytest.raises(ValueError, match="measurement order"):
        screen.check_instruction(instruction, [13, 12], [0, 0])


def test_platform_receipt_and_saved_counts_must_agree(evidence):
    edit_record(evidence, "idle_zero", lambda r: r.update(raw_counts={"0x1": 1024}, counts={"01": 1024}))
    with pytest.raises(ValueError, match="Platform and saved"):
        screen.require_screen(evidence, "WK_C180", [12, 13])


def test_incomplete_batch_never_authorizes_qpu(evidence):
    edit_record(evidence, "active_zero", lambda r: r.update(status="submitted"))
    with pytest.raises(ValueError, match="completed"):
        screen.require_screen(evidence, "WK_C180", [12, 13])


def test_no_submission_when_any_screen_case_fails_to_compile(monkeypatch, tmp_path):
    from experiments.quantum import run_hardware_qaoa as hardware
    chip = SimpleNamespace(**{key: {} for key in diagnostic.CHIP_FIELDS})
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *a, **kw: None)
    monkeypatch.setattr(hardware, "CHIP_BACKEND", chip)
    monkeypatch.setattr(hardware, "BACKEND", SimpleNamespace(run_instruction=lambda *a: pytest.fail("Submitted")))
    calls = []
    def compile_case(*a):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("active pulses eliminated")
        return "[]", None, dict(status="compiled")
    monkeypatch.setattr(screen, "compile_case", compile_case)
    with pytest.raises(ValueError, match="eliminated"):
        screen.main(["--submit", "--physical-qubits", "12,13", "--out-dir", str(tmp_path)])
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["status"] == "failed" and len(calls) == 2
