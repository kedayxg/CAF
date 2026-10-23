###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2024: Amazon Web Services, Inc. - Contributions from JPMC
#
###############################################################################
import numpy as np


def rebalancing_risk_factor(risk_factor, full_covariance, full_avg_return, communities):
    """Rescale using actual communities, supplied as labels, a dict, or groups.

    Degenerate zero-risk or zero-return data retain the input risk factor.
    """
    covariance = np.asarray(full_covariance, dtype=float)
    returns = np.asarray(full_avg_return, dtype=float)
    n = len(returns)
    if returns.ndim != 1 or covariance.shape != (n, n):
        raise ValueError("covariance and returns dimensions must agree")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(returns)):
        raise ValueError("covariance and returns must be finite")
    if not np.isfinite(risk_factor) or risk_factor < 0:
        raise ValueError("risk_factor must be finite and non-negative")
    if isinstance(communities, dict):
        groups = list(communities.values())
    else:
        items = list(communities)
        if len(items) == n and all(np.ndim(item) == 0 for item in items):
            labels = np.asarray(items)
            groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        else:
            groups = items
    indices = []
    for group in groups:
        values = np.asarray(group)
        if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("communities must contain nonempty integer index groups")
        indices.append(values.astype(int))
    covered = np.concatenate(indices) if indices else np.array([], dtype=int)
    if not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError("communities must partition all assets exactly once")
    cov_norm, return_norm = np.linalg.norm(covariance), np.linalg.norm(returns)
    if cov_norm == 0 or return_norm == 0 or risk_factor == 0:
        return float(risk_factor)
    block_cov_norm = sum(np.linalg.norm(covariance[np.ix_(c, c)]) for c in indices)
    block_return_norm = sum(np.linalg.norm(returns[c]) for c in indices)
    if block_cov_norm == 0:
        raise ValueError("cannot rescale a zero within-community covariance norm")
    return float(risk_factor * cov_norm / block_cov_norm * block_return_norm / return_norm)
