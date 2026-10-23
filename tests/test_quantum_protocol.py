"""Cost-operator and raw-shot checks; no hardware credentials or submissions."""
from types import SimpleNamespace

import dimod
import numpy as np
import pytest
pytest.importorskip("qiskit")
from qiskit.quantum_info import Operator

import experiments.quantum.run_qaoa as simulator
import experiments.quantum.run_hardware_qaoa as hardware


def fixed_circuit(monkeypatch, h, interactions, theta, n, p):
    circuits = []
    from_instruction = simulator.Statevector.from_instruction
    def capture(circuit):
        circuits.append(circuit.copy())
        return from_instruction(circuit)
    def minimize(fun, x0, method, options):
        return SimpleNamespace(fun=fun(theta), x=np.asarray(theta))
    with monkeypatch.context() as patch:
        patch.setattr(simulator, "minimize", minimize)
        patch.setattr(simulator.Statevector, "from_instruction", capture)
        simulator.QiskitQAOAOptimizer(
            simulator.QAOAConfig(p=p, restarts=1, maxiter=1)
        )._optimize_qaoa(h, interactions, 0., n, list(range(n)))
    return circuits[-1]


def test_simulator_cost_layer_matches_binary_bqm_energy(monkeypatch):
    bqm = dimod.BinaryQuadraticModel({0: -1.2, 1: .8}, {(0, 1): 1.7}, .3, dimod.BINARY)
    h, interactions, offset = bqm.to_ising()
    gamma = .37
    circuit = fixed_circuit(monkeypatch, h, interactions, [gamma, 0.], 2, 1)
    cost = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.operation.name in ("rz", "rzz"):
            cost.append(inst.operation, [circuit.find_bit(q).index for q in inst.qubits])
    energy = np.array([bqm.energy({q: (i >> q) & 1 for q in range(2)}) for i in range(4)])
    ratio = np.diag(Operator(cost).data) / np.exp(-1j * gamma * energy)
    np.testing.assert_allclose(ratio / ratio[0], np.ones(4), atol=1e-12)


def test_plain_qaoa_historical_sign_has_equivalent_parameterization(monkeypatch):
    h = {0: .4, 1: -.8, 2: 1.2}
    interactions = {(0, 1): .7, (1, 2): -.2}
    theta = np.array([.2, -.5, .3, .8])
    # Passing -h to the maintained method reproduces the historical cost gates.
    legacy = fixed_circuit(monkeypatch, {q: -v for q, v in h.items()}, interactions, theta, 3, 2)
    shifted = theta.copy()
    shifted[-1] += np.pi / 2
    maintained = fixed_circuit(monkeypatch, h, interactions, shifted, 3, 2)
    np.testing.assert_allclose(simulator.Statevector.from_instruction(legacy).probabilities(),
                               simulator.Statevector.from_instruction(maintained).probabilities(), atol=1e-12)


def test_hardware_cost_and_pauli_energy_use_same_linear_sign(monkeypatch):
    class Program:
        def __init__(self):
            self.ops = []
        def __lshift__(self, op):
            self.ops.append(op)
            return self
    monkeypatch.setattr(hardware, "core", SimpleNamespace(QProg=Program, RZ=lambda q, a: ("rz", q, a)))
    monkeypatch.setattr(hardware, "_h_via_x1", lambda prog, q: None)
    monkeypatch.setattr(hardware, "_u3_via_x1", lambda *args: None)
    monkeypatch.setattr(hardware, "_rzz_via_native", lambda prog, i, j, angle: prog.ops.append(("rzz", i, j, angle)))
    prog = hardware.build_qaoa_prog({"a": 1., "b": 2.}, {("a", "b"): .7},
                                    [.3, .2], 1, None, 2, ["a", "b"])
    assert prog.ops == [("rz", 0, -.6), ("rz", 1, -1.2), ("rzz", 0, 1, .42)]
    terms, _ = hardware.make_hamiltonian({"a": 1., "b": 2.}, {("a", "b"): .7}, ["a", "b"])
    assert terms == {"Z0": -1., "Z1": -2., "Z0 Z1": .7}


def test_expval_energy_does_not_flip_linear_sign_twice(monkeypatch):
    monkeypatch.setattr(hardware, "PauliOperator", lambda terms: terms)
    monkeypatch.setattr(hardware, "_expval", lambda prog, op: op["Z0"] * -1)
    assert hardware.hw_energy(None, {"a": 2.}, {}, .3, ["a"]) == pytest.approx(2.3)


RAW = {"distribution_kind": "raw_counts", "backend_mode": "local"}


@pytest.mark.parametrize("counts,meta", [
    ({"01": .9, "10": .1}, {"distribution_kind": "platform_probs"}),
    ({"01": -1, "10": 5}, RAW),
    ({"01": .5, "10": 3.5}, RAW),
    ({"01": 3}, RAW),
    ({"100": 4}, RAW),
    ({"01": 4}, {**RAW, "backend_mode": "qpu", "final_status": "FAILED"}),
])
def test_unusable_hardware_distributions_are_rejected(counts, meta):
    with pytest.raises(RuntimeError):
        hardware._validated_raw_counts(counts, meta, 4, 2)


def test_zero_count_optimum_is_never_returned_as_a_sample(monkeypatch):
    class Program:
        def __lshift__(self, other):
            return self
    monkeypatch.setattr(hardware, "core", SimpleNamespace(measure=lambda *args: None))
    monkeypatch.setattr(hardware, "_run_counts", lambda *args: ({"01": 4, "10": 0}, RAW))
    energy, candidate, counts, _ = hardware.hw_counts_energy_and_sample(
        Program(), 2, 1, 4, np.array([1., 2.]), np.zeros((2, 2)),
        .5, {}, {}, 0., ["a", "b"])
    assert candidate.tolist() == [1, 0]
    assert counts == {"01": 4}


@pytest.mark.parametrize("status", ["FAILED", "RUNNING"])
def test_failed_or_unfinished_jobs_do_not_fetch_results(monkeypatch, status):
    def forbidden(**kwargs):
        raise AssertionError("invalid tasks must not be decoded")
    job = SimpleNamespace(status=lambda: SimpleNamespace(name=status), result=forbidden)
    backend = SimpleNamespace(run_instruction=lambda *args: job)
    monkeypatch.setattr(hardware, "IS_LOCAL", False)
    monkeypatch.setattr(hardware, "CHIP_BACKEND", None)
    monkeypatch.setattr(hardware, "BACKEND", backend)
    with pytest.raises(RuntimeError, match="did not finish"):
        hardware._run_counts(SimpleNamespace(to_instruction=lambda: None), 4, 2, poll_interval_s=0, max_polls=1)


def test_missing_raw_channel_never_falls_back_to_platform_probabilities(monkeypatch):
    calls = []
    def result(**kwargs):
        calls.append(kwargs)
        raise TypeError("unsupported raw channel")
    job = SimpleNamespace(status=lambda: SimpleNamespace(name="FINISHED"), result=result)
    monkeypatch.setattr(hardware, "IS_LOCAL", False)
    monkeypatch.setattr(hardware, "CHIP_BACKEND", None)
    monkeypatch.setattr(hardware, "BACKEND", SimpleNamespace(run_instruction=lambda *args: job))
    with pytest.raises(RuntimeError, match="Raw probCount"):
        hardware._run_counts(SimpleNamespace(to_instruction=lambda: None), 4, 2, poll_interval_s=0)
    assert calls == [{"keys": ["probCount"]}]


def test_hardware_reference_does_not_repair_an_unsampled_candidate():
    def model_builder(*args, **kwargs):
        return SimpleNamespace(), [SimpleNamespace(name="weights_0"), SimpleNamespace(name="weights_1")]
    # A positive budget with an all-zero backend fallback has no feasible sample.
    # Exercise the protocol boundary without constructing a cloud backend.
    from unittest.mock import patch
    with patch.object(simulator.QiskitQAOAOptimizer, "__call__", return_value=(np.zeros(2), 0., 0.)):
        with pytest.raises(RuntimeError, match="actually sampled feasible"):
            hardware.run_sim(model_builder, None, np.ones(2), np.eye(2), 1, .5,
                             simulator.QAOAConfig(warmstart=False), 2)


def test_zero_budget_fallback_is_not_mistaken_for_a_measured_sample(monkeypatch):
    from dcmppln.utils.objective_after_clustering import get_model
    monkeypatch.setattr(simulator.QiskitQAOAOptimizer, "_select_candidate", lambda *args: None)
    with pytest.raises(RuntimeError, match="actually sampled feasible"):
        hardware.run_sim(get_model, simulator.QiskitQAOAOptimizer._docplex_model_to_cqm,
                         np.ones(2), np.eye(2), 0, .5,
                         simulator.QAOAConfig(p=1, restarts=1, maxiter=5), 2)
