# Vendored `dcmppln` provenance

- Upstream repository: `https://github.com/JPMorganChase/dcmppln`
- Upstream archive comment / commit marker: `454f151e02c3a652b1a56162999a844b4ec9699f`
- Upstream spectral source verification (2026-10-22): the [pinned upstream
  file](https://github.com/JPMorganChase/dcmppln/blob/454f151e02c3a652b1a56162999a844b4ec9699f/dcmppln/modularity_maximization/modularity_spectral_optimization.py)
  is byte-identical to the file in both CAF's initial commit `d215144` and
  historical paper commit `356c1852c65a185b05aed040052c5c3f750991bd`.
  SHA-256: `219d4761bd0c87669321dccade3c38b41d2222c7b3a9044c5c0f57708c29a4b1`.
  The `main` and `master` raw URLs fetched on that date returned the same file.
  Thus the recursive local/global index defects already exist in the cited
  upstream snapshot: lines 54–55 write local eigenvector positions into global
  partition labels, and lines 63/68 use those positions to index the full graph.
  The attribution concerns this specific implementation, not other upstream
  algorithms or the validity of spectral clustering as a method.
- License: Apache-2.0; full text copied to `third_party/licenses/dcmppln.Apache-2.0.txt`.
- Local patches applied after extraction:
  1. `dcmppln/denoiser.py`: call `split_covariance_matrices(..., beta=self.q, q_fit=self.q_fit)` instead of the broken `q=` keyword.
  2. `dcmppln/optimizer.py`: import `dimod` and `neal.SimulatedAnnealingSampler` so `SimulatedAnnealingDwave` imports correctly.
  3. Maintenance: forward the pipeline risk-rescaling switch and compute norm
     rescaling from community index groups rather than interpreting labels as
     asset indices. Zero-risk/zero-return data retain the supplied risk factor.
  4. Maintenance: preserve explicit zero budgets/risk factors, support exact
     cardinalities in the scaling adapter, and fail when SA has no feasible
     sample instead of returning an invalid all-zero portfolio.
- The maintained `Clustering` adapter now uses only `dcmppln/newman.py` and
  pinned python-igraph 0.11.9. The old spectral source has been removed from
  this working tree, and `legacy_spectral`, `louvain` and `kmeans` selection
  names are rejected. The upstream byte-identity evidence above describes the
  historical snapshots, not a file still shipped by the current package.
- Natural Newman is used with `max_community_size=0`. A positive capacity
  restarts igraph on oversized induced subgraphs, stopping when blocks fit.
  The local null model is recomputed, so this is a separate capacity-oriented
  protocol rather than uninterrupted optimization of full-graph modularity.
  No custom forced split is used; an unsplittable oversized block raises.
- The retired source's nine expected-failure test cases have been removed
  with that implementation. Active igraph tests cover graph validity,
  relabeling, global asset mapping, modularity and capacity failure handling.
- Historical paper code is preserved at commit
  `356c1852c65a185b05aed040052c5c3f750991bd`; corrected partitions and outputs
  must not be presented as identical reproductions of that snapshot.
- This vendored copy is the default dependency for CAF reproducibility; CLI `--jpm-repo` flags remain optional overrides, but revised scripts reject
  checkouts that do not honor the `newman` method.
