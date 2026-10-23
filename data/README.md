This directory contains the bundled benchmark datasets kept in the `CAF_20260714` review snapshot.

Layout
- `russell3000_subset/`
  - Primary benchmark dataset used by the paper and by the default experiment scripts.
  - Contains the retained `2016` benchmark files and the kept `2018` full-panel files.
  - This is the only data directory referenced by default in the main benchmark, scaling, quantum, and hardware validation scripts.

Provenance
- Source repository: JPMorgan Chase `dcmppln`
- Source location: `tests/data/`
- Original data type:
  - Asset-level return panels stored as NumPy arrays, with rows corresponding to time points and columns corresponding to assets.

Files kept in `russell3000_subset/`
- `1_2016-01-01_correlation.npy`
- `1_2016-01-01_covariance.npy`
- `1_2016-01-01_returns.npy`
- `1_2018-01-01_correlation.npy`
- `1_2018-01-01_covariance.npy`
- `1_2018-01-01_returns.npy`

Processing
- For each retained panel, the per-asset expected return vector is computed as the column-wise mean of the return matrix.
- The covariance matrix is computed from the same panel with `np.cov(X, rowvar=False)`.
- The correlation matrix is then derived from that covariance matrix.
- In the CAF codebase, these three arrays are used as the standard portfolio inputs: expected returns, covariance, and correlation.

Panels kept here
- `1_2016-01-01_*`
  - Main benchmark panel used by the default CAF experiments and by the paper's primary large-scale results.
- `1_2018-01-01_*`
  - Retained out-of-sample full-panel dataset used for the 2018 validation in this review snapshot.

Practical rule
- If you are reproducing the main paper results, start from `russell3000_subset/`.

Notes
- These files are included only to make the published CAF benchmark directly reproducible.
- Please retain the original provenance when redistributing this repository.
