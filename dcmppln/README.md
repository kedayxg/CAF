# CAF's maintained decomposition adapters

This directory contains the RMT denoiser, portfolio models, local-solver adapters
and decomposition pipeline used by CAF. Attribution and historical upstream
source identifiers are recorded in [the vendoring record](../third_party/dcmppln.VENDORED.md).

Community detection now uses only pinned python-igraph 0.11.9 Newman's
leading-eigenvector method, implemented by the adapter in `newman.py`.
The original modified spectral implementation and its runtime selection names
have been removed; historical source is available at CAF paper commit
`356c1852c65a185b05aed040052c5c3f750991bd`.

```python
from dcmppln.clustering import Clustering

# Infer natural communities.
labels, process_seconds, wall_seconds = Clustering()(correlation)

# Restart igraph on oversized induced subgraphs until each block fits.
labels, process_seconds, wall_seconds = Clustering(max_community_size=100)(correlation)
```

A positive capacity stops when a block fits, including an initial graph already
within the limit. If igraph cannot split an oversized block, the adapter raises
`RuntimeError`; it does not force a split or return an oversized block silently.
Capacity recursion recomputes the local null model and can lower full-graph
modularity. Matrix diagonal weights are represented by half-weight undirected
loops so weighted graph degrees equal the matrix row sums.

Use the repository's [installation and reproduction instructions](../README.md)
and pinned environment. The optional public-library comparison scripts are
separate research experiments and do not change the `Clustering` adapter.
