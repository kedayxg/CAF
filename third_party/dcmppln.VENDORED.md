# Vendored `dcmppln` provenance

- Upstream repository: `https://github.com/JPMorganChase/dcmppln`
- Upstream archive comment / commit marker: `454f151e02c3a652b1a56162999a844b4ec9699f`
- License: Apache-2.0; full text copied to `third_party/licenses/dcmppln.Apache-2.0.txt`.
- Local patches applied after extraction:
  1. `dcmppln/denoiser.py`: call `split_covariance_matrices(..., beta=self.q, q_fit=self.q_fit)` instead of the broken `q=` keyword.
  2. `dcmppln/optimizer.py`: import `dimod` and `neal.SimulatedAnnealingSampler` so `SimulatedAnnealingDwave` imports correctly.
  3. Maintenance 0.1.1: fix recursive community indices and unique labels in
     `modularity_spectral_optimization.py`; preserve explicit zero model
     parameters; support exact cardinality and forward risk-rebalancing options
     in the pipeline; interpret risk-rebalancing inputs as actual community
     groups while retaining the sum-of-community-norms formula.
- This vendored copy is the default dependency for CAF reproducibility; CLI `--jpm-repo` flags remain only as optional overrides.
