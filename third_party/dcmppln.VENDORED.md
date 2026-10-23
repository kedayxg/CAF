# Vendored `dcmppln` provenance

- Upstream repository: `https://github.com/JPMorganChase/dcmppln`
- Upstream archive comment / commit marker: `454f151e02c3a652b1a56162999a844b4ec9699f`
- License: Apache-2.0; full text copied to `third_party/licenses/dcmppln.Apache-2.0.txt`.
- Local patches applied after extraction:
  1. `dcmppln/denoiser.py`: call `split_covariance_matrices(..., beta=self.q, q_fit=self.q_fit)` instead of the broken `q=` keyword.
  2. `dcmppln/optimizer.py`: import `dimod` and `neal.SimulatedAnnealingSampler` so `SimulatedAnnealingDwave` imports correctly.
- This vendored copy is the default dependency for CAF reproducibility; CLI `--jpm-repo` flags remain only as optional overrides.
