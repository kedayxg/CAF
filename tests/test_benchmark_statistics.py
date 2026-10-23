import numpy as np
import pytest
import json
import sys

from experiments.classical.run_benchmark import paired_objective_statistics
from experiments.classical import run_benchmark


def test_all_ties_are_not_significant_losses():
    result = paired_objective_statistics(np.ones(20), np.ones(20))
    assert result["objective_sign_test_pvalue"] == 1.0
    assert result["objective_tied_runs"] == 20
    assert result["objective_improvement_runs"] == result["objective_worsened_runs"] == 0


def test_ties_are_excluded_only_from_sign_test():
    result = paired_objective_statistics([2, 2, 2, 2, 2], [1, 1, 1, 2, 2])
    assert result["objective_sign_test_pvalue"] == .25
    assert result["objective_improvement_runs"] == 3
    assert result["objective_tied_runs"] == 2
    assert result["objective_relative_improvement_mean_pct"] == 30.


def test_balanced_improvements_and_losses_have_unit_pvalue():
    result = paired_objective_statistics([2, 2, 2, 2], [1, 1, 3, 3])
    assert result["objective_sign_test_pvalue"] == 1.
    assert result["objective_tied_runs"] == 0
    assert result["objective_worsened_runs"] == 2


@pytest.mark.parametrize("left,right", [([], []), ([1, 2], [1]), ([0], [1]), ([np.nan], [1])])
def test_invalid_pairs_are_rejected(left, right):
    with pytest.raises(ValueError):
        paired_objective_statistics(left, right)


def test_statistics_refresh_uses_saved_pairs_without_running_solvers(tmp_path, monkeypatch):
    records = [{"jpm": {"metrics": {"objective": 2.}},
                "caf": {"metrics": {"objective": 2.}}} for _ in range(20)]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"runs": 20, "per_run_records": records}))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--refresh-statistics", "--summary-file", str(path)])
    def forbidden(*args, **kwargs):
        raise AssertionError("refresh must not run solvers")
    monkeypatch.setattr(run_benchmark, "load_upstream_components", forbidden)
    run_benchmark.main()
    result = json.loads(path.read_text())
    assert result["objective_sign_test_pvalue"] == 1.
    assert result["objective_tied_runs"] == 20
    assert result["per_run_records"] == records
