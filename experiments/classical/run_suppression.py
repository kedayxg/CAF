import argparse
import csv
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT.parent
SNAPSHOT_ROOT = CODE_ROOT.parent
CAF_ROOT = PROJECT_ROOT
for root in (CAF_ROOT,):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from experiments._common import (  # noqa: E402
    extract_communities_from_partitions,
    load_local_jpm_data,
    split_cardinality_constraint,
    subset_problem,
)
from dcmppln.clustering import Clustering  # noqa: E402
from dcmppln.denoiser import Denoiser  # noqa: E402
from dcmppln.optimizer import SimulatedAnnealingDwave  # noqa: E402
from dcmppln.pipeline import Pipeline  # noqa: E402
from dcmppln.utils.objective_after_clustering import get_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure context-induced presolve sparsity in CAF subproblems."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CAF_ROOT / "data" / "russell3000_subset",
        help="Directory containing 2016 benchmark npy files.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,200,300,484",
        help="Comma-separated list of N values.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per N.")
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=123,
        help="Base seed used for subset sampling and per-run randomization.",
    )
    parser.add_argument("--risk-factor", type=float, default=0.5, help="Risk factor q.")
    parser.add_argument("--sa-num-reads", type=int, default=100, help="SA num_reads.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SNAPSHOT_ROOT / "results" / "classical" / "outputs_sparsity_aligned",
        help="Output folder for csv/json artifacts.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=SNAPSHOT_ROOT / "paper" / "figures" / "sparsity_presolve.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def parse_sizes(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def get_active_vars(model):
    import gurobipy as gp

    thread_id = threading.get_ident()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=f"_{thread_id}.lp",
            prefix="caf_gurobi_",
            delete=False,
        ) as tmp:
            tmp.write(model.export_as_lp_string().replace("*", " * "))
            tmp_path = tmp.name
        m = gp.read(tmp_path)
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)
    m.setParam("OutputFlag", 0)
    m.setParam("Presolve", -1)
    p = m.presolve()
    if p is None:
        return 0
    return int(p.NumVars)


def run_one_setting(cov_df, corr_df, ret_s, n_assets, run_idx, args):
    cov, corr, ret = subset_problem(
        cov_df, corr_df, ret_s, n_assets, seed=args.subset_seed + n_assets * 100 + run_idx
    )
    cardinality = n_assets // 2
    pipeline = Pipeline(
        corr.values,
        cov.values,
        ret.values,
        denoiser=Denoiser(active=True, q=0.5, q_fit=True),
        cluster=Clustering(
            active=True, clustering_method="louvain", take_absolute_value=True
        ),
        optimize_func=SimulatedAnnealingDwave(num_reads=args.sa_num_reads),
    )
    result = pipeline.run(
        run_optimizer=True,
        # Keep the sparsity diagnostic aligned with the final "aligned q" setup.
        risk_rebalancing=False,
        cluster_on_correlation=True,
        optimize_on_correlation=False,
        input_risk_factor=args.risk_factor,
    )

    static_selected = np.where(
        np.asarray(result["recombined_solution"]).astype(np.int64) == 1
    )[0].tolist()
    partitions = pipeline.__state__["result"]["partitions"]
    communities = extract_communities_from_partitions(partitions)
    cards = split_cardinality_constraint([len(comm) for comm in communities], cardinality)
    eff_risk = args.risk_factor

    state = np.zeros(n_assets, dtype=int)
    state[static_selected] = 1
    cov_vec = cov.values @ state

    rows = []
    for community_idx, (community, card) in enumerate(zip(communities, cards)):
        if int(card) <= 0:
            continue
        community_size = int(len(community))
        sub_cov = cov.values[np.ix_(community, community)]

        model_static, _ = get_model(
            ret.values[community], sub_cov, budget=int(card), risk_factor=eff_risk
        )
        active_static = get_active_vars(model_static)

        local_state = state[community]
        shift = 2 * eff_risk * (cov_vec[community] - sub_cov @ local_state)
        adjusted_returns = ret.values[community] - shift
        model_caf, _ = get_model(
            adjusted_returns, sub_cov, budget=int(card), risk_factor=eff_risk
        )
        active_caf = get_active_vars(model_caf)

        suppressed_static = float(np.mean(ret.values[community] <= 0.0))
        suppressed_caf = float(np.mean(adjusted_returns <= 0.0))
        rows.append(
            {
                "N": int(n_assets),
                "K": int(cardinality),
                "run_idx": int(run_idx),
                "community_idx": int(community_idx),
                "community_size": int(community_size),
                "active_static": int(active_static),
                "active_caf": int(active_caf),
                "reduction_vs_size_pct": float(
                    100.0 * (community_size - active_caf) / max(community_size, 1)
                ),
                "reduction_vs_static_pct": float(
                    100.0 * (active_static - active_caf) / max(active_static, 1)
                ),
                "suppressed_static_pct": float(100.0 * suppressed_static),
                "suppressed_caf_pct": float(100.0 * suppressed_caf),
                "suppressed_delta_pct": float(100.0 * (suppressed_caf - suppressed_static)),
            }
        )
    return rows


def summarize(rows, sizes):
    summary = {"overall": {}, "by_size": {}}
    if not rows:
        return summary

    red_size = np.asarray([r["reduction_vs_size_pct"] for r in rows], dtype=float)
    red_static = np.asarray([r["reduction_vs_static_pct"] for r in rows], dtype=float)
    sup_delta = np.asarray([r["suppressed_delta_pct"] for r in rows], dtype=float)
    summary["overall"] = {
        "samples": int(len(rows)),
        "reduction_vs_size_mean_pct": float(red_size.mean()),
        "reduction_vs_size_std_pct": float(red_size.std()),
        "reduction_vs_size_q10_pct": float(np.quantile(red_size, 0.10)),
        "reduction_vs_size_q90_pct": float(np.quantile(red_size, 0.90)),
        "reduction_vs_static_mean_pct": float(red_static.mean()),
        "reduction_vs_static_std_pct": float(red_static.std()),
        "suppressed_delta_mean_pct": float(sup_delta.mean()),
        "suppressed_delta_std_pct": float(sup_delta.std()),
        "suppressed_delta_q10_pct": float(np.quantile(sup_delta, 0.10)),
        "suppressed_delta_q90_pct": float(np.quantile(sup_delta, 0.90)),
    }

    for n_assets in sizes:
        rows_n = [r for r in rows if r["N"] == n_assets]
        if not rows_n:
            continue
        a_size = np.asarray([r["community_size"] for r in rows_n], dtype=float)
        a_static = np.asarray([r["active_static"] for r in rows_n], dtype=float)
        a_caf = np.asarray([r["active_caf"] for r in rows_n], dtype=float)
        r_size = np.asarray([r["reduction_vs_size_pct"] for r in rows_n], dtype=float)
        s_delta = np.asarray([r["suppressed_delta_pct"] for r in rows_n], dtype=float)
        summary["by_size"][str(n_assets)] = {
            "samples": int(len(rows_n)),
            "community_size_mean": float(a_size.mean()),
            "active_static_mean": float(a_static.mean()),
            "active_caf_mean": float(a_caf.mean()),
            "reduction_vs_size_mean_pct": float(r_size.mean()),
            "reduction_vs_size_std_pct": float(r_size.std()),
            "suppressed_delta_mean_pct": float(s_delta.mean()),
            "suppressed_delta_std_pct": float(s_delta.std()),
        }
    return summary


def save_plot(rows, sizes, figure_path):
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bar_labels = [str(n) for n in sizes]
    static_means = []
    caf_means = []
    suppression_delta_means = []
    for n_assets in sizes:
        rows_n = [r for r in rows if r["N"] == n_assets]
        if rows_n:
            static_means.append(np.mean([r["active_static"] for r in rows_n]))
            caf_means.append(np.mean([r["active_caf"] for r in rows_n]))
            suppression_delta_means.append(
                np.mean([r["suppressed_delta_pct"] for r in rows_n])
            )
        else:
            static_means.append(0.0)
            caf_means.append(0.0)
            suppression_delta_means.append(0.0)

    x = np.arange(len(sizes))
    width = 0.36
    axes[0].bar(x - width / 2, static_means, width, label="JPM Pipeline Presolve Active")
    axes[0].bar(x + width / 2, caf_means, width, label="CAF Presolve Active")
    axes[0].set_xticks(x, bar_labels)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Active variables per subproblem (mean)")
    axes[0].set_title("Presolve Active Variables")
    axes[0].legend()

    axes[1].plot(sizes, suppression_delta_means, marker="o", linewidth=2)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Suppressed asset ratio increase (pp)")
    axes[1].set_title("Context-Induced Return Suppression")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cov_df, corr_df, ret_s = load_local_jpm_data(args.data_dir)

    rows = []
    print("Running sparsity measurement...")
    print("N\tRuns\tSamples\tMean(active_static)\tMean(active_caf)\tSuppressed Delta")
    for n_assets in sizes:
        rows_n = []
        for run_idx in range(args.runs):
            rows_n.extend(run_one_setting(cov_df, corr_df, ret_s, n_assets, run_idx, args))
        rows.extend(rows_n)
        if rows_n:
            mean_static = np.mean([r["active_static"] for r in rows_n])
            mean_caf = np.mean([r["active_caf"] for r in rows_n])
            mean_supp_delta = np.mean([r["suppressed_delta_pct"] for r in rows_n])
            print(
                f"{n_assets}\t{args.runs}\t{len(rows_n)}\t{mean_static:.2f}\t\t\t"
                f"{mean_caf:.2f}\t\t\t{mean_supp_delta:.2f}pp"
            )

    summary = summarize(rows, sizes)
    csv_path = args.out_dir / "sparsity_samples.csv"
    json_path = args.out_dir / "sparsity_summary.json"

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(rows, sizes, args.figure_path)

    print(f"\nSaved sample rows: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(f"Saved figure: {args.figure_path}")
    print("\nOverall summary:")
    print(json.dumps(summary.get("overall", {}), indent=2))


if __name__ == "__main__":
    main()
