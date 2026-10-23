"""A sampler class name alone must not silently imply a nonzero transverse field."""
import dimod
import numpy as np

from dcmppln.utils.objective_after_clustering import get_model
from experiments.quantum.run_qaoa import SQAOptimizer


def test_sqa_passes_nonzero_driver_and_ends_at_classical_target(monkeypatch):
    import dwave.samplers
    seen = {}
    class Sampler:
        def sample(self, bqm, **kwargs):
            seen.update(kwargs)
            return dimod.SampleSet.from_samples({"weights_0":1,"weights_1":0},
                vartype=dimod.BINARY, energy=[0.])
    monkeypatch.setattr(dwave.samplers,"PathIntegralAnnealingSampler",Sampler)
    model, weights = get_model(np.ones(2),np.eye(2),budget=1,risk_factor=.5)
    result, _, _ = SQAOptimizer(num_reads=20,num_sweeps=1000,seed=31042)(model,weights)
    np.testing.assert_array_equal(result,[1,0])
    np.testing.assert_allclose(seen["Hd_field"],np.linspace(1.,0.,1000))
    assert np.any(seen["Hd_field"] > 0) and seen["Hd_field"][-1] == 0
    assert seen["Gamma"] == 1. and seen["seed"] == 31042
