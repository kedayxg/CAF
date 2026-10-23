"""License limits must not erase the independent landscape diagnostic."""
import json
from types import SimpleNamespace

import gurobipy as gp
import pytest

from experiments.classical.run_suppression import get_active_vars, summarize


def test_size_limited_presolve_is_unavailable_but_other_errors_propagate(monkeypatch):
    def fail():
        raise gp.GurobiError(10010, "Size-limited license")
    model = SimpleNamespace(setParam=lambda *a:None, presolve=fail)
    monkeypatch.setattr(gp, "read", lambda *a:model)
    source = SimpleNamespace(export_as_lp_string=lambda:"test")
    assert get_active_vars(source) is None
    def unrelated():
        raise gp.GurobiError(10001, "Unrelated error")
    model.presolve = unrelated
    with pytest.raises(gp.GurobiError):
        get_active_vars(source)


def test_summary_excludes_unavailable_presolve_and_keeps_all_suppression():
    row = dict(N=300, community_size=212, active_static=None, active_caf=None,
        reduction_vs_size_pct=None, reduction_vs_static_pct=None, suppressed_delta_pct=80.)
    stats = summarize([row], [300])
    assert stats["overall"]["presolve_available_samples"] == 0
    assert stats["overall"]["suppressed_delta_mean_pct"] == 80.
    assert stats["overall"]["reduction_vs_size_mean_pct"] is None
    json.dumps(stats, allow_nan=False)
