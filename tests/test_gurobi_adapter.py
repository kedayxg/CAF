"""Preserve caller settings and variable order at the optional solver boundary."""
import importlib.abc
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dcmppln.optimizer import GurobiOptimizer


def test_sa_import_does_not_require_optional_gurobi():
    code = """
import importlib.abc, sys
class BlockGurobi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gurobipy' or fullname.startswith('gurobipy.'):
            raise ModuleNotFoundError('gurobipy unavailable')
sys.meta_path.insert(0, BlockGurobi())
from experiments._common import load_upstream_components
assert load_upstream_components(None)[2].__name__ == 'SimulatedAnnealingDwave'
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gurobi_adapter_honors_parameters_and_requested_variable_order(monkeypatch):
    settings = {}
    values = {"custom_a": 1., "custom_b": 0.}
    model = SimpleNamespace(SolCount=1, Status=2, NumVars=2,
        setParam=lambda name, value: settings.update({name:value}), optimize=lambda: None,
        getVarByName=lambda name: SimpleNamespace(X=values[name]))
    monkeypatch.setitem(sys.modules, "gurobipy", SimpleNamespace(read=lambda path: model))
    backend = GurobiOptimizer(TimeLimit=9, MIPGap=.03, OutputFlag=0)
    result, _, _ = backend(SimpleNamespace(lp_string="test"),
                           [SimpleNamespace(name="custom_b"), SimpleNamespace(name="custom_a")])
    assert settings == {"TimeLimit":9, "MIPGap":.03, "Threads":1, "OutputFlag":0}
    np.testing.assert_array_equal(result, [0,1])


def test_gurobi_without_incumbent_is_not_decoded(monkeypatch):
    model = SimpleNamespace(SolCount=0, Status=9, setParam=lambda *args: None,
                            optimize=lambda: None)
    monkeypatch.setitem(sys.modules, "gurobipy", SimpleNamespace(read=lambda path: model))
    with pytest.raises(RuntimeError, match="no feasible incumbent"):
        GurobiOptimizer()(SimpleNamespace(lp_string="test"))
