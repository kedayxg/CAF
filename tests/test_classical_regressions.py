import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from caf_core import evaluate_portfolio
from experiments.classical import run_benchmark, run_scaling


def test_paired_sign_test_excludes_ties_and_uses_symmetric_tolerance():
    baseline = np.ones(6)
    candidate = np.array([0.8, 0.9, 0.7, 1.1, 1.0, 1.0 + 1e-11])
    result = run_benchmark.paired_objective_sign_test(baseline, candidate)
    assert result == {"wins": 3, "losses": 1, "ties": 2, "non_tied_runs": 4, "pvalue": 0.625}
    reverse = run_benchmark.paired_objective_sign_test(candidate, baseline)
    assert reverse["wins"] == result["losses"]
    assert reverse["ties"] == result["ties"]
    assert reverse["pvalue"] == result["pvalue"]


def test_paired_sign_test_all_ties_has_no_evidence_of_difference():
    result = run_benchmark.paired_objective_sign_test(np.ones(20), np.ones(20))
    assert result["pvalue"] == 1.0
    assert result["ties"] == 20
    assert result["non_tied_runs"] == 0


def test_scaling_rejects_mismatched_constraints_before_aggregation():
    covariance = np.eye(20)
    returns = np.ones(20)
    record = {
        "run_idx": 0,
        "static": evaluate_portfolio(list(range(10)), covariance, returns),
        "caf": evaluate_portfolio(list(range(8)), covariance, returns),
        "static_time_sec": 0.0,
        "caf_time_sec": 0.0,
    }
    with pytest.raises(ValueError, match="expected K=8, got 10"):
        run_scaling.aggregate([record], cardinality=8)


def test_comparison_rejects_nonbinary_portfolio():
    with pytest.raises(ValueError, match="non-binary"):
        run_benchmark.validate_portfolio_cardinality([1.0, 0.5, 0.0], 1, "baseline")


class _UnusedComponent:
    def __init__(self, **kwargs):
        self.params = kwargs


def test_scaling_passes_exact_custom_cardinality(monkeypatch, tmp_path):
    n_assets = 20
    covariance = pd.DataFrame(np.eye(n_assets))
    returns = pd.Series(np.ones(n_assets))
    optimize_calls = []

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.__state__ = {"result": {"partitions": np.zeros(n_assets, dtype=int)}}

        def run(self, **kwargs):
            assert not kwargs["run_optimizer"]
            return covariance.values, self.__state__["result"]["partitions"]

        def optimize_and_score(self, **kwargs):
            optimize_calls.append(kwargs)
            weights = np.zeros(n_assets)
            weights[:kwargs["cardinality"]] = 1
            return {"recombined_solution": weights}

    monkeypatch.setattr(run_scaling, "parse_args", lambda: SimpleNamespace(
        sizes="20", solvers="sa", k_ratios="0.4", jpm_repo=None,
        data_dir=tmp_path, out_dir=tmp_path, sa_num_reads=20, runs=1,
        risk_factor=0.5, iterations=2, patience=2, max_workers=1,
        subset_seed=123, base_seed=42, summary_file=None,
    ))
    monkeypatch.setattr(run_scaling, "load_upstream_components", lambda *args, **kwargs: (
        _UnusedComponent, _UnusedComponent, _UnusedComponent, _UnusedComponent,
        FakePipeline, None,
    ))
    monkeypatch.setattr(run_scaling, "load_local_jpm_data", lambda *args: (
        covariance, covariance, returns,
    ))
    monkeypatch.setattr(run_scaling, "generate_cardinality_plot", lambda *args: None)
    monkeypatch.setattr(run_scaling, "run_caf_pipeline", lambda *args, **kwargs: {
        "metrics": evaluate_portfolio(kwargs["initial_selected_indices"], covariance, returns),
    })
    run_scaling.main()
    assert optimize_calls[0]["cardinality"] == 8
    assert "cardinality_divider" not in optimize_calls[0]
    summary = json.loads((tmp_path / "scaling_summary.json").read_text())
    assert summary["config"]["caf_version"] == run_scaling.caf_version
    record = summary["rows"][0]["run_records"][0]
    assert record["static_cardinality"] == record["caf_cardinality"] == 8


def test_q_sweep_reoptimizes_same_partition_at_each_q(monkeypatch, tmp_path):
    covariance = pd.DataFrame(np.eye(4))
    returns = pd.Series([2.0, 1.0, 0.5, 0.2])
    partitions = np.array([0, 0, 1, 1])
    optimize_calls = []
    caf_calls = []
    optimizers = []

    class FakeOptimizer(_UnusedComponent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            optimizers.append(kwargs)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            self.__state__ = {"result": {"partitions": partitions.copy()}}

        def run(self, **kwargs):
            assert kwargs["risk_rebalancing"] is False
            return {"recombined_solution": [1, 0, 1, 0]}

        def optimize_and_score(self, **kwargs):
            optimize_calls.append(kwargs)
            return {"recombined_solution": [0, 1, 0, 1]}

    def fake_caf(*args, **kwargs):
        caf_calls.append(kwargs)
        metrics = evaluate_portfolio(
            kwargs["initial_selected_indices"], covariance, returns, kwargs["risk_factor"]
        )
        return {
            "metrics": metrics,
            "timing": {"total_seconds": 0.0},
            "history": {key: [metrics[key]] for key in ("objective", "risk", "expected_return")},
        }

    monkeypatch.setattr(run_benchmark, "parse_args", lambda: SimpleNamespace(
        summary_file=None, out_dir=tmp_path, plot_only=False, jpm_repo=None,
        data_dir=tmp_path, data_prefix="test", risk_factor=0.5, q_values="0.9",
        runs=1, base_seed=42, skip_q_sweep=False,
    ))
    monkeypatch.setattr(run_benchmark, "load_upstream_components", lambda *args: (
        _UnusedComponent, _UnusedComponent, FakeOptimizer, FakePipeline, None, None,
    ))
    monkeypatch.setattr(run_benchmark, "load_local_jpm_data", lambda *args, **kwargs: (
        covariance, covariance, returns,
    ))
    monkeypatch.setattr(run_benchmark, "run_caf_pipeline", fake_caf)
    monkeypatch.setattr(run_benchmark, "generate_plots", lambda *args: None)
    run_benchmark.main()
    assert len(optimize_calls) == 1
    assert optimize_calls[0]["risk_factor"] == 0.9
    assert optimize_calls[0]["cardinality"] == 2
    np.testing.assert_array_equal(optimize_calls[0]["partitions"], partitions)
    assert caf_calls[-1]["risk_factor"] == caf_calls[-1]["effective_risk_factor"] == 0.9
    assert caf_calls[-1]["initial_selected_indices"] == [1, 3]
    assert optimizers == [{"num_reads": 200, "seed": 42}, {"num_reads": 200, "seed": 10042}]
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert summary["config"]["caf_version"] == run_benchmark.caf_version
    assert summary["objective_tied_runs"] == 1
    assert summary["objective_sign_test_pairs"] == 0
    assert summary["objective_sign_test_pvalue"] == 1.0
    assert summary["q_sweep_rows"][0]["run_records"][0]["seed"] == 10042
