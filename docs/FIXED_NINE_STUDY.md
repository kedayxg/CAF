> Historical study snapshot: the revision now adopts igraph Newman as its default. Current scope and reruns are recorded in [ARXIV_REVISION.md](ARXIV_REVISION.md).

# Corrected spectral clustering with exactly nine communities

This optional experiment leaves the historical vendored partitioner unchanged.
The count constraint and split rule were fixed before observing solver outcomes.

The graph is the absolute RMT-filtered 2016 correlation matrix. Corrected Newman
spectral modularity splitting first produces its natural three communities.
To reach exactly nine, the largest remaining community is split next. The
generalized modularity matrix always uses original graph degrees and total
weight. Its constant mode is removed with an orthonormal basis, and the leading
remaining eigenvector is cut at its median to give balanced children. Six such
forced cuts are applied. Forced cuts may lower modularity; this is not the
unconstrained modularity optimum or a certified best nine-way partition.

The fixed sizes are [76, 66, 66, 50, 50, 50, 49, 39, 38]; budgets are
[39, 33, 33, 25, 25, 25, 24, 19, 19]. Natural modularity is 0.167819;
fixed-nine modularity is 0.121573. Cross-community
absolute denoised correlation weight is
73.83%.

All settings use the same N=484 data, K=242, q=0.5, seeds 42–61, unscaled static
initialization, 200 SA reads per static block, up to six CAF iterations and
patience two. CAF retains the original size-dependent read rule. This requests
900 reads per full CAF scan here, versus 1,060 for historical nine-way blocks
and 600 for natural three-way blocks. Different partitions also change the
fixed community budgets and local search spaces; this is not a pure count-only
or equal-computation comparison. Within each setting CAF is compared with its
own static initialization.

| Configuration | Static objective | CAF objective | Mean per-run improvement | Improving / tied |
| --- | --- | --- | --- | --- |
| historical_9 | 1.545345 ± 0.021181 | 1.443284 ± 0.014796 | 6.5898% | 20 / 0 |
| corrected_natural_3 | 1.540945 ± 0.022058 | 1.512938 ± 0.014130 | 1.8029% | 16 / 4 |
| corrected_fixed_9 | 1.629618 ± 0.027288 | 1.478698 ± 0.015256 | 9.2387% | 20 / 0 |

The fixed-nine bootstrap 95% interval is [8.5257%, 9.9641%], and the paired
sign test p-value excluding ties is 1.9073486e-06.
Every final portfolio meets all block budgets and K=242; every recorded CAF
history is nonincreasing. There are 1062 proposals,
0 repaired proposals, and 329 accepted proposals.
The extra no-cache run at seed 42 reproduces the cached final portfolio exactly,
with objective histories agreeing to 1e-12. All 59 active tests pass; nine strict
expected failures continue to record the retained upstream implementation's
known issues. Raw objectives, complete partitions, split gains, configurations
and source/data fingerprints are in [fixed-nine-results.json](fixed-nine-results.json).

This comparison distinguishes percentage gain from absolute solution quality:
the fixed-nine static baseline is different, so a larger percentage gain does
not by itself imply a better final portfolio than every other configuration.
The experiment also does not isolate context injection from additional search
effort, or prove that these results generalize to another partitioning policy.

Reproduce with:

```bash
python -m experiments.classical.run_fixed_community_benchmark --communities 9 --runs 20 --check-no-cache-first --out-dir reproduce_outputs/corrected_fixed_nine
```
