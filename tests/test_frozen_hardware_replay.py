"""Frozen replay must preserve layer meanings, source bytes and sampled bits."""
import builtins
import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.quantum import diagnose_hardware as diagnostic
from experiments.quantum import replay_hardware_qaoa as replay


@pytest.fixture
def source():
    return dict(n=2, K=1, p=2, theta_star=[.2, .7, .4, .9], var_labels=["a", "b"],
                warmstart_angles=[.7, 1.2], ising_linear={"a": .2, "b": -.1},
                ising_quadratic=[["a", "b", .3]], covariance=(np.eye(3)*.1).tolist(),
                expected_returns=[.2, .4, .1], static_state=[1, 0, 0], community_indices=[0, 1],
                local_covariance=(np.eye(2)*.1).tolist(), adjusted_returns=[.2, .4],
                sim_candidate=[0, 1], risk_factor=.5, q_eff=.5, dO_sim=-.2)


@pytest.mark.parametrize("p,expected", [(0, []), (1, [.2, .4]), (2, [.2, .7, .4, .9]),
                                      (None, [.2, .7, .4, .9])])
def test_parameter_halves_are_split_at_original_depth(source, p, expected):
    _, theta = replay.frozen_parameters(source, p)
    assert theta == expected
    assert source["theta_star"] == [.2, .7, .4, .9]


@pytest.mark.parametrize("p", [-1, 3, True, 1.0])
def test_invalid_depth_is_rejected(source, p):
    with pytest.raises(ValueError):
        replay.frozen_parameters(source, p)


def test_invalid_source_parameter_vector_is_rejected(source):
    for theta in [[.2, .7], [.2, float("nan"), .4, .9], [True, .7, .4, .9]]:
        with pytest.raises(ValueError):
            replay.frozen_parameters(dict(source, theta_star=theta), 1)


def test_p1_reference_uses_saved_beta0_not_gamma1(source):
    circuit = QuantumCircuit(2)
    for i, angle in enumerate([.7, 1.2]):
        circuit.ry(angle, i)
    circuit.rz(-2*.2*.2, 0)
    circuit.rz(-2*.2*(-.1), 1)
    circuit.rzz(2*.2*.3, 0, 1)
    for i, angle in enumerate([.7, 1.2]):
        circuit.ry(-angle, i)
        circuit.rz(2*.4, i)
        circuit.ry(angle, i)
    np.testing.assert_allclose(list(replay.ideal_probabilities(source, 1).values()),
                               Statevector.from_instruction(circuit).probabilities(), atol=1e-14)


def test_default_plan_is_offline_and_keeps_source_bytes(source, monkeypatch, tmp_path):
    path = tmp_path/"frozen.json"
    path.write_text(json.dumps(source, indent=3)+"\n")
    source_bytes = path.read_bytes()
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith("pyqpanda3")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    replay.main(["--record", str(path), "--p", "1", "--out-dir", str(tmp_path/"out")])
    directory = next((tmp_path/"out").iterdir())
    record = json.loads((directory/"record.json").read_text())
    assert record["mode"] == "plan_only" and record["theta_star"] == [.2, .4]
    assert record["source_theta_star"] == source["theta_star"]
    assert (directory/"source_record.json").read_bytes() == source_bytes
    assert not record.get("job_id")
    hashes = json.loads((directory/"SHA256.json").read_text())
    assert "source_code/replay_hardware_qaoa.py" in hashes
    for relative, expected in hashes.items():
        assert replay.sha(directory/relative) == expected


def test_candidate_frequency_uses_c0_on_right_and_only_real_samples(source):
    result = replay.metrics(source, {"10": 3, "01": 7}, replay.ideal_probabilities(source))
    assert result["sim_candidate_count_key"] == "10" and result["sim_candidate_frequency"] == 3
    assert result["best_feasible_by_local_energy"]["count_key"] == "10"
    assert result["best_feasible_dO"] == pytest.approx(-.2)
    assert result["criterion_pass"]
    absent = replay.metrics(source, {"01": 10}, replay.ideal_probabilities(source))
    assert absent["sim_candidate_frequency"] == 0 and not absent["criterion_pass"]


def test_financial_delta_includes_cross_community_covariance(source):
    covariance = np.array([[.3, .02, .08], [.02, .4, -.03], [.08, -.03, .5]])
    initial, candidate = np.array([1, 0, 1]), np.array([0, 1, 1])
    returns = np.asarray(source["expected_returns"])
    objective = lambda x: .5*x @ covariance @ x-returns @ x
    expected = float(objective(candidate)-objective(initial))
    source.update(covariance=covariance.tolist(), static_state=initial.tolist(),
                  local_covariance=covariance[:2, :2].tolist(), adjusted_returns=[.12, .43], dO_sim=expected)
    result = replay.metrics(source, {"10": 3, "01": 7}, replay.ideal_probabilities(source))
    assert result["best_feasible_dO"] == pytest.approx(expected)


def test_multiqubit_counts_and_receipts_are_preserved(tmp_path):
    raw = {"0x8d": 3, "10110001": 7}
    assert diagnostic.counts_checked(raw, 10, 8) == {"10001101": 3, "10110001": 7}
    result = SimpleNamespace(get_counts_list=lambda: [raw])
    job = SimpleNamespace(job_id=lambda: "eight-bit-job", status=lambda: SimpleNamespace(name="FINISHED"),
                          result=lambda **kw: result)
    case = {}
    diagnostic.collect_job(job, case, tmp_path, 10, 0, 1, width=8)
    saved = json.loads((tmp_path/"record.json").read_text())
    assert saved["raw_counts"] == raw and saved["counts"]["10001101"] == 3


def legacy_batch(source, tmp_path):
    src = tmp_path/"source_record.json"
    src.write_text(json.dumps(source))
    record = dict(status="completed", p=2, theta_star=source["theta_star"], source_sha256=replay.sha(src),
                  status_trace=["FINISHED"], shots=10, raw_counts={"0x2": 3, "0x1": 7},
                  metrics=dict(sim_candidate_frequency=7, sim_candidate_ideal_probability=0))
    diagnostic.save(tmp_path/"record.json", record)
    return record


def test_analysis_recomputes_legacy_metrics_without_rewriting_record(source, tmp_path):
    legacy_batch(source, tmp_path)
    before = (tmp_path/"record.json").read_bytes()
    result = replay.analyze(tmp_path)
    assert result["metrics"]["sim_candidate_frequency"] == 3
    assert result["stored_metric_differences"]["sim_candidate_frequency"] == {"stored": 7, "recomputed": 3}
    assert (tmp_path/"record.json").read_bytes() == before


def test_analysis_refuses_changed_source(source, tmp_path):
    legacy_batch(source, tmp_path)
    (tmp_path/"source_record.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256"):
        replay.analyze(tmp_path)
    assert not (tmp_path/"analysis.json").exists()


def test_analysis_does_not_silently_reinterpret_incorrect_p1_parameters(source, tmp_path):
    record = legacy_batch(source, tmp_path)
    record.update(p=1, theta_star=[.2, .7])
    diagnostic.save(tmp_path/"record.json", record)
    with pytest.raises(ValueError, match="theta"):
        replay.analyze(tmp_path)


@pytest.mark.parametrize("arguments", [["--p", "3"], ["--p", "-1"], ["--submit"],
                                      ["--submit", "--physical-qubits", "12,13"],
                                      ["--shots", "0"], ["--poll-interval", "nan"]])
def test_invalid_replay_stops_before_sdk(source, monkeypatch, tmp_path, arguments):
    from experiments.quantum import run_hardware_qaoa as hardware
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *a, **kw: pytest.fail("SDK initialized"))
    path = tmp_path/"frozen.json"
    path.write_text(json.dumps(source))
    with pytest.raises(SystemExit) as exc:
        replay.main(["--record", str(path), "--out-dir", str(tmp_path/"out")]+arguments)
    assert exc.value.code == 2 and not (tmp_path/"out").exists()


def test_compile_failure_never_submits_and_preserves_preparation(source, monkeypatch, tmp_path):
    from experiments.quantum import run_hardware_qaoa as hardware
    path = tmp_path/"frozen.json"
    path.write_text(json.dumps(source))
    chip = SimpleNamespace(**{k: {} for k in diagnostic.CHIP_FIELDS})
    monkeypatch.setattr(hardware, "CHIP_BACKEND", chip)
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *a, **kw: None)
    monkeypatch.setattr(hardware, "BACKEND", SimpleNamespace(run_instruction=lambda *a: pytest.fail("Submitted")))
    monkeypatch.setattr(replay.importlib.metadata, "version", lambda name: "0.5.0")
    monkeypatch.setattr(replay.screening, "require_screen", lambda *a, **kw: {})
    monkeypatch.setattr(replay.screening, "archive_screen", lambda *a, **kw: None)
    def fail(*args, **kwargs):
        raise ValueError("Preflight failed")
    monkeypatch.setattr(replay, "compile_replay", fail)
    with pytest.raises(ValueError, match="Preflight"):
        replay.main(["--record", str(path), "--submit", "--screen-dir", str(tmp_path/"screen"),
                     "--physical-qubits", "12,13", "--out-dir", str(tmp_path/"out")])
    directory = next((tmp_path/"out").iterdir())
    record = json.loads((directory/"record.json").read_text())
    assert record["status"] == "failed" and not record.get("job_id")
    assert (directory/"source_record.json").exists() and (directory/"SHA256.json").exists()


def test_screen_is_checked_again_after_compilation_before_submission(source, monkeypatch, tmp_path):
    from experiments.quantum import run_hardware_qaoa as hardware
    path = tmp_path/"frozen.json"
    path.write_text(json.dumps(source))
    chip = SimpleNamespace(**{k: {} for k in diagnostic.CHIP_FIELDS})
    monkeypatch.setattr(hardware, "CHIP_BACKEND", chip)
    monkeypatch.setattr(hardware, "setup_pyqpanda", lambda *a, **kw: None)
    monkeypatch.setattr(hardware, "BACKEND", SimpleNamespace(run_instruction=lambda *a: pytest.fail("Submitted")))
    monkeypatch.setattr(replay.importlib.metadata, "version", lambda name: "0.5.0")
    calls = []
    def check(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise ValueError("Screen expired during compilation")
        return {}
    monkeypatch.setattr(replay.screening, "require_screen", check)
    monkeypatch.setattr(replay.screening, "archive_screen", lambda *a, **kw: None)
    monkeypatch.setattr(replay, "compile_replay", lambda *a: ("[]", None, dict(measurement_map=[[12, 0], [13, 1]])))
    with pytest.raises(ValueError, match="expired"):
        replay.main(["--record", str(path), "--submit", "--screen-dir", str(tmp_path/"screen"),
                     "--physical-qubits", "12,13", "--out-dir", str(tmp_path/"out")])
    directory = next((tmp_path/"out").iterdir())
    record = json.loads((directory/"record.json").read_text())
    assert len(calls) == 2 and calls[1] == directory/"screen_evidence"
    assert record["status"] == "failed" and not record.get("job_id")
