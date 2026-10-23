"""Diagnostic safety, decoding evidence, and independent analytic probe checks."""
import builtins
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.quantum import diagnose_hardware as diagnostic


def test_default_plan_does_not_import_sdk_or_original_driver(monkeypatch, tmp_path):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith("pyqpanda3")
        assert name != "experiments.quantum.run_hardware_qaoa"
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    diagnostic.main(["--out-dir", str(tmp_path)])
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["mode"] == "plan_only"
    assert manifest["planned_jobs"] == 7
    assert manifest["planned_total_shots"] == 7*1024


@pytest.mark.parametrize("args", [["--submit"], ["--compile-only", "--physical-qubits", "1,1"],
    ["--local", "--submit"], ["--shots", "0"], ["--poll-interval", "nan"],
    ["--suite", "all", "--compensation", "both", "--max-jobs", "19"]])
def test_invalid_plans_fail_before_output_or_cloud(args, tmp_path):
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(args+["--out-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_all_declared_ideals_match_independent_statevector():
    # Four amplitudes in |q1 q0> order, independent of the hardware SDK.
    identity = np.eye(2)
    h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    x = np.array([[0, 1], [1, 0]])
    cases = diagnostic.probes("all", "both")
    assert len(cases) == 20 and len({c["name"] for c in cases}) == 20
    assert len(diagnostic.probes("cz", "both")) == 12
    for case in cases:
        state = np.array([1, 0, 0, 0], complex)
        for name, *args in case["gates"]:
            if name == "CZ":
                matrix = np.diag([1, 1, 1, -1])
            else:
                q = args[0]
                gate = h if name == "H" else x if name == "X" else np.diag(
                    [np.exp(-.5j*args[1]), np.exp(.5j*args[1])])
                matrix = np.kron(identity, gate) if q == 0 else np.kron(gate, identity)
            state = matrix @ state
        np.testing.assert_allclose(abs(state)**2,
            [case["expected_probabilities"].get(f"{i:02b}", 0) for i in range(4)], atol=1e-14)


def test_raw_count_normalization_keeps_only_actual_integer_shots():
    assert diagnostic.counts_checked({"0x1": 2, "01": 3, "0x0": 0}, 5) == {"01": 5}
    for counts in ({"01": 1.0}, {"00": True}, {"100": 1}, {"0x4": 1},
                   {"00": -1}, {"x": 1}, {"00": 2}):
        with pytest.raises(ValueError):
            diagnostic.counts_checked(counts, 1)


def test_bit_reversal_remains_a_reported_hypothesis():
    comparison = diagnostic.distribution_comparison({"10": 100}, {"01": 1.})
    assert comparison["tv_assuming_q0_rightmost"] == 1
    assert comparison["tv_after_string_reversal"] == 0
    assert comparison["empirical_probabilities"]["10"] == 1


def test_timeout_retains_job_id_and_never_fetches_result(tmp_path):
    def forbidden(**kwargs):
        pytest.fail("Unfinished job must not fetch results")
    job = SimpleNamespace(job_id=lambda: "pending-123", status=lambda: SimpleNamespace(name="RUNNING"),
                          result=forbidden)
    case = dict(name="probe")
    with pytest.raises(TimeoutError, match="pending-123"):
        diagnostic.collect_job(job, case, tmp_path, 10, 0, 2)
    record = json.loads((tmp_path/"record.json").read_text())
    assert record["job_id"] == "pending-123"
    assert record["status_trace"] == ["RUNNING", "RUNNING"]


def test_raw_counts_survive_optional_metadata_failure(tmp_path):
    result = SimpleNamespace(get_counts_list=lambda: [{"0x1": 100}],
                             origin_data=lambda: '{"taskId":"finished-123"}')
    def fetch(keys):
        if keys == ["probCount"]:
            return result
        raise RuntimeError("Optional mapping fields unavailable")
    job = SimpleNamespace(job_id=lambda: "finished-123", status=lambda: SimpleNamespace(name="FINISHED"), result=fetch)
    case = dict(name="probe")
    diagnostic.collect_job(job, case, tmp_path, 100, 0, 1)
    record = json.loads((tmp_path/"record.json").read_text())
    assert record["status"] == "completed"
    assert record["raw_counts"] == {"0x1": 100} and record["counts"] == {"01": 100}
    assert record["optional_metadata_error_type"] == "RuntimeError"


def test_invalid_count_batch_retains_unmodified_receipt(tmp_path):
    result = SimpleNamespace(get_counts_list=lambda: [{"0x1": 99}], origin_data=lambda: "raw")
    job = SimpleNamespace(job_id=lambda: "short-123", status=lambda: SimpleNamespace(name="FINISHED"),
                          result=lambda **kwargs: result)
    with pytest.raises(ValueError, match="total"):
        diagnostic.collect_job(job, {}, tmp_path, 100, 0, 1)
    saved = json.loads((tmp_path/"record.json").read_text())
    assert saved["raw_counts"] == {"0x1": 99} and saved["job_id"] == "short-123"


def test_tuple_calibration_keys_are_preserved():
    assert diagnostic.packed({(12, 13): (.2, .3)}) == [{"key": [12, 13], "value": [.2, .3]}]


def test_entire_batch_must_compile_before_any_paid_submission(monkeypatch, tmp_path):
    from experiments.quantum import run_hardware_qaoa as hardware
    def forbidden(*args, **kwargs):
        pytest.fail("A partially compiled batch must not submit any jobs")
    chip = SimpleNamespace(**{name: {} for name in diagnostic.CHIP_FIELDS})
    chip.chip_topology_edges = [[12, 13]]
    monkeypatch.setattr(hardware, "CHIP_BACKEND", chip)
    monkeypatch.setattr(hardware, "BACKEND", SimpleNamespace(run_instruction=forbidden))
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *a, **kw: None)
    options = SimpleNamespace(set_mapping=lambda x: None, set_optimization=lambda x: None, set_amend=lambda x: None)
    monkeypatch.setitem(sys.modules, "pyqpanda3", SimpleNamespace(core=None, transpilation=None,
                        qcloud=SimpleNamespace(QCloudOptions=lambda: options)))
    monkeypatch.setattr(diagnostic.importlib.metadata, "version", lambda name: "0.5.0")
    prepared = []
    def prepare(core, tr, config, pair, case, directory, basis):
        prepared.append(case["name"])
        if len(prepared) == 4:
            raise ValueError("Compiler rejected the fourth case")
        case["status"] = "compiled"
        diagnostic.save(directory/"record.json", case)
        return None, "instruction", None
    monkeypatch.setattr(diagnostic, "prepare", prepare)
    with pytest.raises(ValueError, match="fourth"):
        diagnostic.main(["--submit", "--physical-qubits", "12,13", "--out-dir", str(tmp_path)])
    directory = next(tmp_path.iterdir())
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["status"] == "failed" and len(prepared) == 4
    assert (directory/"SHA256.json").exists()
    record = json.loads((directory/prepared[-1]/"record.json").read_text())
    assert record["status"] == "failed" and record["error_type"] == "ValueError"
    assert not any(c.get("job_id") for c in manifest["cases"])
