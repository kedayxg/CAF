import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
CAF_ROOT = EXPERIMENTS_ROOT.parent
BUNDLED_DATA_DIR = CAF_ROOT / "data" / "russell3000_subset"

if str(CAF_ROOT) not in sys.path:
    sys.path.insert(0, str(CAF_ROOT))


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_upstream_components(jpm_repo: Path | None, include_gurobi: bool = False):
    if jpm_repo is not None:
        if not jpm_repo.exists():
            raise FileNotFoundError(f"JPM repo path does not exist: {jpm_repo}")
        if str(jpm_repo) not in sys.path:
            sys.path.insert(0, str(jpm_repo))

    try:
        from dcmppln.clustering import Clustering
        from dcmppln.denoiser import Denoiser
        from dcmppln.optimizer import SimulatedAnnealingDwave
        from dcmppln.pipeline import Pipeline
        from dcmppln.utils.objective_after_clustering import get_model
    except ImportError as exc:
        raise ImportError(
            "Failed to import dcmppln. The default path uses the vendored copy in this "
            "repository; alternatively provide --jpm-repo pointing to another checkout."
        ) from exc

    if include_gurobi:
        from dcmppln.optimizer import GurobiOptimizer

        return (
            Clustering,
            Denoiser,
            SimulatedAnnealingDwave,
            GurobiOptimizer,
            Pipeline,
            get_model,
        )

    try:
        from dcmppln.utils.risk_rebalance import rebalancing_risk_factor
    except ImportError as exc:
        raise ImportError(
            "Failed to import dcmppln risk rebalancing utilities from the vendored copy."
        ) from exc

    return (
        Clustering,
        Denoiser,
        SimulatedAnnealingDwave,
        Pipeline,
        get_model,
        rebalancing_risk_factor,
    )


def load_local_jpm_data(base_path: Path, data_prefix: str = "1_2016-01-01"):
    covariance = np.load(base_path / f"{data_prefix}_covariance.npy")
    correlation = np.load(base_path / f"{data_prefix}_correlation.npy")
    expected_returns = np.load(base_path / f"{data_prefix}_returns.npy")
    tickers = [f"asset_{i:03d}" for i in range(covariance.shape[0])]
    return (
        pd.DataFrame(covariance, index=tickers, columns=tickers),
        pd.DataFrame(correlation, index=tickers, columns=tickers),
        pd.Series(expected_returns, index=tickers),
    )


def subset_problem(cov_df, corr_df, ret_s, n_assets, seed):
    total_assets = len(ret_s)
    if n_assets > total_assets:
        raise ValueError(f"Requested N={n_assets} > total available {total_assets}")
    if n_assets == total_assets:
        return cov_df.copy(), corr_df.copy(), ret_s.copy()

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total_assets, size=n_assets, replace=False))
    idx = cov_df.index[selected]
    return cov_df.loc[idx, idx], corr_df.loc[idx, idx], ret_s.loc[idx]


def split_cardinality_constraint(community_sizes, cardinality):
    continuous_solution = np.array(community_sizes) * cardinality / np.sum(community_sizes)
    integer_solution = np.floor(continuous_solution).astype(int)
    increment = cardinality - np.sum(integer_solution)
    order = np.argsort(community_sizes)[::-1]
    cursor = 0
    while increment > 0:
        integer_solution[order[cursor]] += 1
        increment -= 1
        cursor += 1
    return integer_solution


def extract_communities_from_partitions(partitions):
    labels = np.unique(partitions)
    communities = [np.argwhere(partitions == label).reshape(-1) for label in labels]
    return sorted(communities, key=lambda community: community.shape[0], reverse=True)
