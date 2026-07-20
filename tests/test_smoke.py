import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import caf_core
from caf_core import evaluate_portfolio, run_caf_pipeline
from caf_core import solver as solver_module
from experiments._common import (
    BUNDLED_DATA_DIR,
    extract_communities_from_partitions,
    load_local_jpm_data,
    load_upstream_components,
    split_cardinality_constraint,
)


def test_public_api_aliases_exposed():
    assert caf_core.normalize_local_solution is solver_module._normalize_local_solution
    assert caf_core.attach_local_problem_metadata is solver_module._attach_local_problem_metadata


def test_common_helpers_with_bundled_data():
    cov_df, corr_df, ret_s = load_local_jpm_data(BUNDLED_DATA_DIR)

    assert cov_df.shape[0] == cov_df.shape[1]
    assert corr_df.shape == cov_df.shape
    assert ret_s.shape[0] == cov_df.shape[0]

    partitions = np.array([1, 1, 0, 2, 2, 2])
    communities = extract_communities_from_partitions(partitions)
    cards = split_cardinality_constraint([len(c) for c in communities], cardinality=3)

    assert [len(c) for c in communities] == [3, 2, 1]
    assert int(cards.sum()) == 3


def test_vendored_dcmppln_import_path_smoke():
    (
        Clustering,
        Denoiser,
        SimulatedAnnealingDwave,
        Pipeline,
        get_model,
        rebalancing_risk_factor,
    ) = load_upstream_components(None)

    assert Clustering.__name__ == "Clustering"
    assert Denoiser.__name__ == "Denoiser"
    assert SimulatedAnnealingDwave.__name__ == "SimulatedAnnealingDwave"
    assert Pipeline.__name__ == "Pipeline"
    assert callable(get_model)
    assert callable(rebalancing_risk_factor)


def test_run_caf_pipeline_cached_matches_original_on_tiny_problem():
    covariance = np.array(
        [
            [1.0, 0.1, 0.2, 0.0],
            [0.1, 0.8, 0.0, 0.1],
            [0.2, 0.0, 0.9, 0.2],
            [0.0, 0.1, 0.2, 0.7],
        ],
        dtype=float,
    )
    expected_returns = np.array([1.2, 0.3, 1.0, 0.2], dtype=float)
    communities = [np.array([0, 1]), np.array([2, 3])]
    subproblem_cardinalities = [1, 1]

    def get_model_func(adjusted_returns, sub_covariance, budget, risk_factor):
        model = SimpleNamespace()
        weights = [SimpleNamespace(name=f"x_{idx}") for idx in range(len(adjusted_returns))]
        model.adjusted_returns = np.asarray(adjusted_returns, dtype=float)
        model.sub_covariance = np.asarray(sub_covariance, dtype=float)
        model.budget = int(budget)
        model.risk_factor = float(risk_factor)
        return model, weights

    def optimizer_factory(num_reads=None, seed=None):
        del num_reads, seed

        def optimizer(model, weights):
            del weights
            scores = np.asarray(model._caf_adjusted_returns, dtype=float)
            budget = int(model._caf_budget)
            chosen = np.argsort(-scores, kind="mergesort")[:budget]
            return chosen.tolist(), 0.0, 0.0

        return optimizer

    cached = run_caf_pipeline(
        covariance,
        expected_returns,
        communities,
        subproblem_cardinalities,
        get_model_func=get_model_func,
        optimizer_factory=optimizer_factory,
        risk_factor=0.5,
        iterations=2,
        patience=2,
        use_incremental=True,
        seed=7,
    )
    original = run_caf_pipeline(
        covariance,
        expected_returns,
        communities,
        subproblem_cardinalities,
        get_model_func=get_model_func,
        optimizer_factory=optimizer_factory,
        risk_factor=0.5,
        iterations=2,
        patience=2,
        use_incremental=False,
        seed=7,
    )

    assert cached["selected_indices"] == original["selected_indices"] == [0, 2]
    assert np.isclose(
        cached["metrics"]["objective"],
        evaluate_portfolio([0, 2], covariance, expected_returns, risk_factor=0.5)["objective"],
    )
    assert np.isclose(cached["metrics"]["objective"], original["metrics"]["objective"])


def main():
    test_public_api_aliases_exposed()
    test_common_helpers_with_bundled_data()
    test_vendored_dcmppln_import_path_smoke()
    test_run_caf_pipeline_cached_matches_original_on_tiny_problem()
    print("Smoke check passed.")


if __name__ == "__main__":
    main()
