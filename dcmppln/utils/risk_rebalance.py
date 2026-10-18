###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
from collections.abc import Mapping

import numpy as np


def rebalancing_risk_factor(risk_factor, full_covariance, full_avg_return, communities):
    """Balance risk using sums of the norms of the actual community blocks.

    ``communities`` is either one label per asset or a mapping from arbitrary
    community labels to disjoint asset-index sequences covering all assets.
    Labels identify groups; they are never interpreted as asset indices.
    """
    covariance = np.asarray(full_covariance, dtype=float)
    returns = np.asarray(full_avg_return, dtype=float)
    if returns.ndim != 1 or covariance.shape != (len(returns), len(returns)):
        raise ValueError("Covariance and returns must describe the same assets")

    if isinstance(communities, Mapping):
        groups = []
        for indices in communities.values():
            indices = np.asarray(indices)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Community indices must be one-dimensional integers")
            groups.append(indices.astype(int, copy=False))
        all_indices = np.concatenate(groups) if groups else np.array([], dtype=int)
        if not np.array_equal(np.sort(all_indices), np.arange(len(returns))):
            raise ValueError("Communities must cover every asset exactly once")
    else:
        labels = np.asarray(communities)
        if labels.ndim != 1 or len(labels) != len(returns):
            raise ValueError("Provide one community label per asset")
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]

    cov_norm = np.linalg.norm(covariance)
    avg_return_norm = np.linalg.norm(returns)
    # A zero risk coefficient, risk matrix, or return vector needs no rescaling.
    if risk_factor == 0 or cov_norm == 0 or avg_return_norm == 0:
        return risk_factor

    community_cov_norm = sum(
        np.linalg.norm(covariance[np.ix_(indices, indices)]) for indices in groups
    )
    community_avg_return_norm = sum(np.linalg.norm(returns[indices]) for indices in groups)
    if community_cov_norm == 0:
        raise ValueError("Cannot rebalance risk with zero within-community covariance")

    return risk_factor * (cov_norm / community_cov_norm) * (
        community_avg_return_norm / avg_return_norm
    )
