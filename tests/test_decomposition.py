"""Regression checks for the maintained decomposition and model adapters."""
import numpy as np
import pytest
import dimod

from dcmppln.utils.risk_rebalance import rebalancing_risk_factor
from dcmppln.utils.objective_after_clustering import get_model
import dcmppln.pipeline as pipeline_module
import dcmppln.optimizer as optimizer_module


def test_risk_rescaling_groups_labels_and_relabeling_agree():
    covariance = np.array([[2., .2, .3], [.2, 1., .1], [.3, .1, 3.]])
    returns = np.array([1., 2., 4.])
    q = rebalancing_risk_factor(.5, covariance, returns, [0, 0, 1])
    assert q == pytest.approx(rebalancing_risk_factor(.5, covariance, returns, [9, 9, 2]))
    assert q == pytest.approx(rebalancing_risk_factor(.5, covariance, returns, [[0, 1], [2]]))
    assert q == pytest.approx(rebalancing_risk_factor(.5, covariance, returns, {"a": [0, 1], "b": [2]}))
    assert q != pytest.approx(.5)


def test_risk_rescaling_handles_zero_returns():
    assert rebalancing_risk_factor(.5, np.eye(2), np.zeros(2), [0, 1]) == .5
    with pytest.raises(ValueError, match="partition"):
        rebalancing_risk_factor(.5, np.eye(2), np.ones(2), [[0], [0]])


@pytest.mark.parametrize("enabled", [True, False])
def test_pipeline_rescaling_switch_controls_local_models(monkeypatch, enabled):
    seen = {}
    def optimize(*args, **kwargs):
        seen.update(kwargs)
        return 0., np.array([1, 0, 0]), {0: 0.}, {0: 0.}
    monkeypatch.setattr(pipeline_module, "optimize_and_score", optimize)
    class Denoise:
        params = {}
        def __call__(self, matrix):
            return matrix, 0., 0.
    class Cluster:
        params = {}
        def __call__(self, matrix):
            return np.array([7, 7, 4]), 0., 0.
    pipe = pipeline_module.Pipeline(np.eye(3), np.diag([2., 1., 3.]), np.array([1., 2., 4.]),
                                    denoiser=Denoise(), cluster=Cluster())
    pipe.run(run_optimizer=True, input_risk_factor=.5, risk_rebalancing=enabled)
    assert (seen["rebalanced_risk_factor"] is not None) == enabled
    assert pipe.__state__["key_parameters"]["problem_parameters"]["risk_rebalancing"] == enabled


def test_zero_budget_and_zero_risk_are_preserved():
    model, weights = get_model(np.array([1., 2.]), np.eye(2), budget=0, risk_factor=0)
    constraint = next(model.iter_constraints())
    assert constraint.rhs.constant == 0
    assert model.objective_expr.get_coef(weights[0]) == -1
    assert not model.objective_expr.is_quad_expr()


def test_explicit_cardinality_is_not_rounded_to_a_divisor(monkeypatch):
    seen = {}
    def optimize(*args, **kwargs):
        seen.update(kwargs)
        return 0., np.zeros(10), {0: 0.}, {0: 0.}
    monkeypatch.setattr(pipeline_module, "optimize_and_score", optimize)
    pipe = pipeline_module.Pipeline(np.eye(10), np.eye(10), np.ones(10))
    pipe.optimize_and_score(np.eye(10), np.eye(10), np.zeros(10, dtype=int), .5,
                            "test", False, True, False, 0., 0., 0., 0., cardinality=3)
    assert seen["card_"] == 3


def test_sa_uses_best_feasible_sample_and_preserves_variable_order(monkeypatch):
    model, weights = get_model(np.ones(2), np.eye(2), budget=1)
    samples = dimod.SampleSet.from_samples(
        [{"weights_0": 0, "weights_1": 0}, {"weights_0": 0, "weights_1": 1}],
        vartype=dimod.BINARY, energy=[0., 1.],
    )
    class Sampler:
        def sample(self, *args, **kwargs):
            return samples
    monkeypatch.setattr(optimizer_module, "SimulatedAnnealingSampler", Sampler)
    assert optimizer_module.SimulatedAnnealingDwave()(model, weights)[0] == [0, 1]


def test_sa_rejects_no_feasible_samples(monkeypatch):
    model, weights = get_model(np.ones(2), np.eye(2), budget=1)
    samples = dimod.SampleSet.from_samples({"weights_0": 0, "weights_1": 0},
                                         vartype=dimod.BINARY, energy=[0.])
    class Sampler:
        def sample(self, *args, **kwargs):
            return samples
    monkeypatch.setattr(optimizer_module, "SimulatedAnnealingSampler", Sampler)
    with pytest.raises(RuntimeError, match="no feasible"):
        optimizer_module.SimulatedAnnealingDwave()(model, weights)
