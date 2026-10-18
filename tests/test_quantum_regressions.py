"""Regression checks for quantum backend conventions and sampling semantics."""
from types import SimpleNamespace

import dimod
import numpy as np
import pytest

pytest.importorskip("qiskit")
from docplex.mp.model import Model
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from experiments.quantum import run_qaoa


def _fixed_parameters(monkeypatch, theta):
    """Evaluate the real circuit at known parameters without optimizer noise."""
    theta = np.asarray(theta, dtype=float)
    monkeypatch.setattr(
        run_qaoa, "minimize",
        lambda objective, x0, **kwargs: SimpleNamespace(fun=objective(theta), x=theta.copy()),
    )


def _direct_cost_probabilities(bqm, labels, p, theta):
    """Reference exp(-i*gamma*E(x)) using direct BQM energies, not RZ signs."""
    energies = np.array([
        bqm.energy({label: (index >> pos) & 1 for pos, label in enumerate(labels)})
        for index in range(1 << len(labels))
    ])
    initial = QuantumCircuit(len(labels))
    initial.h(range(len(labels)))
    state = Statevector.from_instruction(initial)
    for layer in range(p):
        state = Statevector(state.data * np.exp(-1j * theta[layer] * energies))
        mixer = QuantumCircuit(len(labels))
        mixer.rx(2 * theta[p + layer], range(len(labels)))
        state = state.evolve(mixer)
    return state.probabilities()


@pytest.mark.parametrize("p", [1, 2, 3])
def test_ordinary_qaoa_last_beta_shift_matches_direct_bqm_cost(monkeypatch, p):
    # Ordinary H initialization and X mixers only: fixed warm-starts do not
    # have this symmetry, and this test makes no claim about their convention.
    rng = np.random.default_rng(418 + p)
    labels = ["z", "a", "m", "b"]
    h = {label: float(rng.normal()) for label in labels}
    J = {(left, right): float(rng.normal()) for i, left in enumerate(labels) for right in labels[i + 1:]}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset=0.37)
    bqm.change_vartype(dimod.BINARY)
    theta = rng.normal(size=2 * p)
    expected = _direct_cost_probabilities(bqm, labels, p, theta)
    shifted = theta.copy()
    shifted[-1] += np.pi / 2
    _fixed_parameters(monkeypatch, shifted)
    opt = run_qaoa.QiskitQAOAOptimizer(run_qaoa.QAOAConfig(p=p, restarts=1))
    _, actual, _, _ = opt._optimize_qaoa(h, J, 0.37, len(labels), labels)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def _constrained_model():
    model = Model(name="quantum_regression")
    weights = model.binary_var_list(4, name="x")
    covariance = np.array([[1.3, .2, .05, .1], [.2, .9, .07, .03], [.05, .07, 1.8, .13], [.1, .03, .13, .8]])
    returns = [.12, .31, .18, .23]
    model.minimize(
        .5 * model.sum(float(covariance[i, j]) * weights[i] * weights[j] for i in range(4) for j in range(4))
        - model.sum(float(returns[i]) * weights[i] for i in range(4))
    )
    model.add_constraint(model.sum(weights) == 2)
    return model, weights


def test_qaoa_energies_and_decoders_use_actual_feasible_bqm_states(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    model, _ = _constrained_model()
    cqm = run_qaoa.QiskitQAOAOptimizer._docplex_model_to_cqm(model)
    bqm, _ = dimod.cqm_to_bqm(cqm)
    labels = list(bqm.variables)
    h, J, offset = bqm.to_ising()
    _fixed_parameters(monkeypatch, [.43, .31])
    opt = run_qaoa.QiskitQAOAOptimizer(run_qaoa.QAOAConfig(p=1, restarts=1, shots=4000, seed=123))
    _, probabilities, _, energies = opt._optimize_qaoa(h, J, offset, len(labels), labels)
    samples = [opt._decode_state(index, labels) for index in range(1 << len(labels))]
    np.testing.assert_allclose(energies, [bqm.energy(sample) for sample in samples], atol=2e-13, rtol=0)
    feasible = [index for index, sample in enumerate(samples) if cqm.check_feasible(sample)]
    np.testing.assert_allclose(
        energies[feasible], [cqm.objective.energy(samples[index]) for index in feasible], atol=2e-13, rtol=0,
    )
    sampled = np.random.default_rng(130).choice(len(probabilities), size=4000, p=probabilities)
    expected_energy = min(energies[index] for index in set(sampled.tolist()) if index in feasible)
    candidate = opt._select_candidate(probabilities, energies, labels, cqm)
    assert cqm.check_feasible(candidate)
    assert bqm.energy(candidate) == pytest.approx(expected_energy, abs=2e-13)
    opt = run_qaoa.QiskitQAOAOptimizer(run_qaoa.QAOAConfig(decode="argmax_prob"))
    candidate = opt._select_candidate(probabilities, energies, labels, cqm)
    assert candidate == samples[max(feasible, key=lambda index: probabilities[index])]


def test_sqa_transverse_schedule_reaches_real_sampler_and_is_reproducible(monkeypatch, tmp_path):
    # Capture the Cython boundary rather than merely asserting wrapper kwargs.
    import dwave.samplers.sqa.sampler as sampler_module

    monkeypatch.chdir(tmp_path)
    model, weights = _constrained_model()
    original = sampler_module.path_annealing
    schedules = []

    def capture(*args, **kwargs):
        schedules.append((np.array(args[6]), np.array(args[7])))
        return original(*args, **kwargs)

    monkeypatch.setattr(sampler_module, "path_annealing", capture)
    opt = run_qaoa.SQAOptimizer(num_reads=40, num_sweeps=40, seed=123)
    first, wall_time, cpu_time = opt(model, weights)
    second, _, _ = opt(model, weights)
    assert first.sum() == 2
    np.testing.assert_array_equal(first, second)
    assert wall_time >= 0 and cpu_time >= 0
    hp, hd = schedules[0]
    assert len(hp) == len(hd) == 40
    assert hp[0] == 0 and hp[-1] > 0 and np.all(np.diff(hp) >= 0)
    assert hd[0] > 0 and hd[-1] == 0 and np.all(hd[:-1] > 0)
    assert np.all(np.diff(hd) <= 0)
    assert hd[0] == opt.inverse_temperature * opt.transverse_field
    np.testing.assert_array_equal(hp, opt.last_schedule["Hp_field"])
    np.testing.assert_array_equal(hd, opt.last_schedule["Hd_field"])
    assert opt.params["schedule"] == "linear_normalized"


def test_sqa_requires_two_schedule_points_and_supports_minimal_anneal(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="num_sweeps>=2"):
        run_qaoa.SQAOptimizer(num_sweeps=1)
    monkeypatch.chdir(tmp_path)
    model, weights = _constrained_model()
    opt = run_qaoa.SQAOptimizer(num_reads=40, num_sweeps=2, seed=123)
    solution, _, _ = opt(model, weights)
    assert solution.sum() == 2
    assert opt.last_schedule["Hd_field"] == [5.0, 0.0]
