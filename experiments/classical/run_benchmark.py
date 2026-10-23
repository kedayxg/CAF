import argparse
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

current_dir = Path(__file__).resolve().parent
caf_root = current_dir.parent.parent
if str(caf_root) not in sys.path:
    sys.path.insert(0, str(caf_root))

from caf_core import evaluate_portfolio, run_caf_pipeline
from experiments._common import (
    BUNDLED_DATA_DIR as bundled_data_dir,
    extract_communities_from_partitions,
    load_local_jpm_data,
    load_upstream_components,
    split_cardinality_constraint,
    to_jsonable,
)

def print_result_block(name, result, baseline_objective):
    metrics = result["metrics"]
    print(name)
    print(f"  -> Objective       : {metrics['objective']:.6f}")
    print(f"  -> Risk            : {metrics['risk']:.6f}")
    print(f"  -> Expected Return : {metrics['expected_return']:.6f}")
    print(f"  -> Relative vs Full-Matrix SA : {((metrics['objective'] - baseline_objective) / abs(baseline_objective)) * 100:.2f}%")

def aggregate_history_runs(history_list, total_iterations):
    metric_names = ["objective", "risk", "expected_return"]
    aggregated = {}
    target_len = total_iterations + 1
    for metric in metric_names:
        padded = []
        for history in history_list:
            values = np.asarray(history[metric], dtype=float)
            if values.size < target_len:
                pad_width = target_len - values.size
                values = np.pad(values, (0, pad_width), mode="edge")
            padded.append(values[:target_len])
        padded = np.vstack(padded)
        aggregated[metric] = np.mean(padded, axis=0)
        aggregated[f"{metric}_std"] = np.std(padded, axis=0)
    return aggregated


def aggregate_gain_runs(history_list, total_iterations):
    target_len = total_iterations + 1
    gains = []
    for history in history_list:
        values = np.asarray(history["objective"], dtype=float)
        if values.size < target_len:
            values = np.pad(values, (0, target_len - values.size), mode="edge")
        values = values[:target_len]
        gains.append(values[:-1] - values[1:])
    gains = np.vstack(gains)
    return np.mean(gains, axis=0), np.std(gains, axis=0)


def sign_test_two_sided(num_better: int, num_total: int) -> float:
    """Exact two-sided sign test p-value under H0: p=0.5."""
    if num_total <= 0:
        return 1.0
    k = min(num_better, num_total - num_better)
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(num_total, i)
    p = min(1.0, 2.0 * tail / (2.0 ** num_total))
    return float(p)


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42):
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = samples.mean(axis=1)
    lo = np.quantile(means, alpha / 2.0)
    hi = np.quantile(means, 1.0 - alpha / 2.0)
    return float(lo), float(hi)


def generate_plots(caf_res, jpm_res, static_res, q_sweep_rows, out_dir: Path, total_iterations: int):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except ModuleNotFoundError:
        plt.style.use("default")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else value
        for key, value in caf_res["history"].items()
    }
    iterations = np.arange(total_iterations + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history["objective"], marker="o", linewidth=2, color="b", label="CAF Mean Objective")
    plt.fill_between(
        iterations,
        history["objective"] - history["objective_std"],
        history["objective"] + history["objective_std"],
        color="b",
        alpha=0.15,
        label="CAF +/-1 std",
    )
    plt.axhline(
        y=jpm_res["metrics"]["objective"],
        color="r",
        linestyle="--",
        label="JPM Mean Baseline",
    )
    plt.title("CAF vs JPM: Objective Convergence over Iterations", fontsize=14, pad=15)
    plt.xlabel("Iteration (0 = Initialization from JPM)", fontsize=12)
    plt.ylabel("Objective Value (Lower is Better)", fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(iterations)
    plt.tight_layout()
    plt.savefig(out_dir / "objective_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        history["risk"],
        history["expected_return"],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="g",
        label="CAF Mean Trajectory",
    )
    highlight_indices = sorted(set([0, 1, len(history["risk"]) - 1]))
    for i in highlight_indices:
        plt.annotate(
            f"Iter {i}",
            (history["risk"][i], history["expected_return"][i]),
            textcoords="offset points",
            xytext=(0, 10 if i != len(history["risk"]) - 1 else -15),
            ha="center",
            fontsize=9,
        )
    plt.scatter(
        [history["risk"][0]],
        [history["expected_return"][0]],
        color="r",
        s=150,
        marker="*",
        zorder=5,
        label="JPM / Iter 0",
    )
    plt.title("Risk-Return Tradeoff Trajectory during CAF Folding", fontsize=14, pad=15)
    plt.xlabel("Portfolio Risk (Lower is Better)", fontsize=12)
    plt.ylabel("Expected Return (Higher is Better)", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "risk_return_trajectory.png", dpi=300)
    plt.close()

    # 3) Ablation summary
    labels = ["JPM Pipeline", "CAF Static (No Incremental)", "CAF Incremental"]
    objectives = [
        jpm_res["metrics"]["objective"],
        static_res["metrics"]["objective"],
        caf_res["metrics"]["objective"],
    ]
    risks = [
        jpm_res["metrics"]["risk"],
        static_res["metrics"]["risk"],
        caf_res["metrics"]["risk"],
    ]
    returns = [
        jpm_res["metrics"]["expected_return"],
        static_res["metrics"]["expected_return"],
        caf_res["metrics"]["expected_return"],
    ]
    objective_errs = [
        jpm_res["metrics"]["objective_std"],
        static_res["metrics"]["objective_std"],
        caf_res["metrics"]["objective_std"],
    ]
    risk_errs = [
        jpm_res["metrics"]["risk_std"],
        static_res["metrics"]["risk_std"],
        caf_res["metrics"]["risk_std"],
    ]
    return_errs = [
        jpm_res["metrics"]["expected_return_std"],
        static_res["metrics"]["expected_return_std"],
        caf_res["metrics"]["expected_return_std"],
    ]
    times = [
        jpm_res["timing"]["total_seconds"],
        static_res["timing"]["total_seconds"],
        caf_res["timing"]["total_seconds"],
    ]
    time_errs = [
        jpm_res["timing"]["total_seconds_std"],
        static_res["timing"]["total_seconds_std"],
        caf_res["timing"]["total_seconds_std"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(labels, objectives, yerr=objective_errs, capsize=4, color=["#d62728", "#ff7f0e", "#1f77b4"])
    axes[0, 0].set_title("Objective (Lower Better)")
    axes[0, 0].tick_params(axis="x", rotation=15)

    axes[0, 1].bar(labels, risks, yerr=risk_errs, capsize=4, color=["#d62728", "#ff7f0e", "#1f77b4"])
    axes[0, 1].set_title("Risk (Lower Better)")
    axes[0, 1].tick_params(axis="x", rotation=15)

    axes[1, 0].bar(labels, returns, yerr=return_errs, capsize=4, color=["#d62728", "#ff7f0e", "#1f77b4"])
    axes[1, 0].set_title("Expected Return (Higher Better)")
    axes[1, 0].tick_params(axis="x", rotation=15)

    axes[1, 1].bar(labels, times, yerr=time_errs, capsize=4, color=["#d62728", "#ff7f0e", "#1f77b4"])
    axes[1, 1].set_title("Wall Clock Time (s)")
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.suptitle("CAF Ablation Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_summary.png", dpi=300)
    plt.close(fig)

    # 4) q-sweep tradeoff
    q_vals = [row["q"] for row in q_sweep_rows]
    delta_obj_pct = [row["delta_obj_pct"] for row in q_sweep_rows]
    delta_risk_pct = [row["delta_risk_pct"] for row in q_sweep_rows]
    delta_ret_pct = [row["delta_ret_pct"] for row in q_sweep_rows]
    delta_obj_std = [row["delta_obj_std"] for row in q_sweep_rows]
    delta_risk_std = [row["delta_risk_std"] for row in q_sweep_rows]
    delta_ret_std = [row["delta_ret_std"] for row in q_sweep_rows]

    plt.figure(figsize=(10, 6))
    plt.errorbar(q_vals, delta_obj_pct, yerr=delta_obj_std, marker="o", linewidth=2, capsize=4, label="ΔObjective % (CAF vs JPM)")
    plt.errorbar(q_vals, delta_risk_pct, yerr=delta_risk_std, marker="s", linewidth=2, capsize=4, label="ΔRisk %")
    plt.errorbar(q_vals, delta_ret_pct, yerr=delta_ret_std, marker="^", linewidth=2, capsize=4, label="ΔReturn %")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("q-Sweep Trade-off (CAF vs JPM)")
    plt.xlabel("Risk Factor q")
    plt.ylabel("Relative Change (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "q_sweep_tradeoff.png", dpi=300)
    plt.close()

    # 5) iteration gain
    gains = history["gain_mean"]
    gain_stds = history["gain_std"]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(1, len(gains) + 1), gains, yerr=gain_stds, capsize=4, color="#2ca02c")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Objective Gain Per Iteration (CAF Mean +/-1 std)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Drop (positive is better)")
    plt.tight_layout()
    plt.savefig(out_dir / "iteration_gain.png", dpi=300)
    plt.close()

    # 6) overall workflow diagram (clean layout, minimal arrow crossing)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def add_box(x, y, w, h, text, face, edge, fontsize=11, bold=False):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=2.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
        )

    # Top pipeline blocks (uniform spacing)
    add_box(0.8, 6.0, 2.8, 1.2, "Market Data\nSigma, mu, Corr", "#E8F2FF", "#2E5AAC", fontsize=11, bold=True)
    add_box(4.0, 6.0, 2.8, 1.2, "Static Decomposition\nRMT + Newman", "#F2EDFF", "#6A3FB5", fontsize=11)
    add_box(7.2, 6.0, 2.8, 1.2, "JPM Initialization\nCommunities + x0", "#FFF1E6", "#C96A1B", fontsize=11)

    # CAF core loop container
    loop_container = FancyBboxPatch(
        (3.0, 1.15),
        7.8,
        4.2,
        boxstyle="round,pad=0.05,rounding_size=0.10",
        linewidth=2.0,
        edgecolor="#C0392B",
        facecolor="#FFF8F6",
        linestyle="--",
    )
    ax.add_patch(loop_container)
    ax.text(6.9, 5.1, "CAF Context-Aware Folding Loop", ha="center", va="center", fontsize=12, fontweight="bold", color="#A93226")

    # Internal loop: simple and direct
    add_box(3.6, 3.75, 2.2, 0.95, "Select Block Ck", "#FFFFFF", "#D35400")
    add_box(6.2, 3.75, 2.8, 0.95, "Inject Context\nshift = v_C - Sigma_CC x_C", "#FFFFFF", "#D35400", fontsize=10)
    add_box(3.6, 2.2, 2.2, 0.95, "Local Solve", "#FFFFFF", "#D35400")
    add_box(6.2, 2.2, 2.8, 0.95, "Monotonic Check\nDeltaO <= 0 ?", "#FFFFFF", "#D35400", bold=True)
    add_box(4.8, 0.95, 3.0, 0.8, "Update state x, v  |  Stop if no improvement", "#FFFFFF", "#D35400", fontsize=9.5)

    add_box(11.1, 3.0, 2.3, 1.5, "Final Portfolio\nObjective / Risk / Return", "#EAF9EA", "#2E7D32", fontsize=11, bold=True)

    def arrow(p1, p2, color="#37474F", lw=2.0, style="-|>", connection="arc3,rad=0.0"):
        arr = FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=connection,
        )
        ax.add_patch(arr)

    # Main stream
    arrow((3.6, 6.6), (4.0, 6.6))
    arrow((6.8, 6.6), (7.2, 6.6))
    arrow((8.6, 6.0), (8.6, 4.7), color="#C0392B")
    arrow((7.8, 1.35), (11.1, 3.75), color="#2E7D32", connection="arc3,rad=-0.05")

    # Internal flow
    arrow((5.8, 4.23), (6.2, 4.23), color="#C0392B")
    arrow((4.7, 3.75), (4.7, 3.15), color="#C0392B")
    arrow((5.8, 2.68), (6.2, 2.68), color="#C0392B")
    arrow((7.6, 2.2), (7.6, 1.75), color="#C0392B")
    ax.text(7.78, 1.92, "accept", fontsize=9, color="#7B241C")
    arrow((6.2, 1.35), (4.7, 3.75), color="#C0392B", connection="arc3,rad=0.25")
    ax.text(5.0, 1.55, "continue", fontsize=9, color="#7B241C")
    ax.text(8.65, 2.55, "reject", fontsize=9, color="#7B241C")

    ax.text(6.9, 0.45, "Repeat until no accepted update in the patience window", ha="center", fontsize=10.0, color="#4E342E")
    ax.set_title("Context-Aware Folding (CAF): End-to-End Workflow", fontsize=15, pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "caf_overall_flow.png", dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce CAF vs JPM benchmark.")
    parser.add_argument(
        "--jpm-repo",
        type=Path,
        default=None,
        help="Optional override path to another dcmppln checkout. Defaults to the vendored copy.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing benchmark npy files. Defaults to CAF/data/jpm_benchmark if present.",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="1_2016-01-01",
        help="Dataset prefix inside --data-dir, e.g. 1_2016-01-01 or 1_2018-01-01.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated plot files.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of random seed runs for statistical averaging.",
    )
    parser.add_argument(
        "--skip-q-sweep",
        action="store_true",
        help="Skip q-sweep robustness to speed up large-run significance experiments.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path to save machine-readable benchmark summary JSON.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed used to deterministically derive per-run random seeds.",
    )
    parser.add_argument(
        "--risk-factor",
        type=float,
        default=0.5,
        help="Risk-aversion factor q used for the main benchmark.",
    )
    parser.add_argument(
        "--q-values",
        type=str,
        default="0.1,0.3,0.5,0.7,1.0",
        help="Comma-separated q values used by the q-sweep robustness check.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from a saved benchmark summary without rerunning experiments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = args.summary_file if args.summary_file is not None else (args.out_dir / "benchmark_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found for --plot-only: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        total_iterations = int(summary["config"]["iterations"])
        generate_plots(summary["caf"], summary["jpm"], summary["caf_static"], summary["q_sweep_rows"], args.out_dir, total_iterations)
        print(f"Plots successfully regenerated from {summary_path}")
        return

    (
        Clustering,
        Denoiser,
        SimulatedAnnealingDwave,
        Pipeline,
        get_model,
        _rebalancing_risk_factor,
    ) = load_upstream_components(args.jpm_repo)

    data_dir = args.data_dir if args.data_dir is not None else bundled_data_dir
    if not data_dir.exists():
        raise FileNotFoundError(
            "Benchmark data directory not found. Provide --data-dir or place the JPM benchmark "
            f"files in {bundled_data_dir}."
        )

    covariance_matrix, correlation_matrix, expected_returns = load_local_jpm_data(
        data_dir, data_prefix=args.data_prefix
    )
    cardinality = len(expected_returns) // 2
    risk_factor = float(args.risk_factor)
    q_values = [float(v.strip()) for v in args.q_values.split(",") if v.strip()]
    
    # Hyperparameters for reproduction
    num_runs = args.runs
    num_reads = 200
    iterations = 6
    patience = 2
    
    print(f"--- CAF Benchmark Setup ---")
    print(f"Data directory: {data_dir}")
    print(f"Data prefix: {args.data_prefix}")
    print(f"Assets: {len(expected_returns)}, Cardinality K: {cardinality}")
    print(f"Risk Factor q: {risk_factor}")
    print(f"Q-sweep values: {q_values}")
    print(f"Annealing Reads: {num_reads}, CAF Max Iterations: {iterations}, Patience: {patience}")
    print(f"Number of statistical runs: {num_runs}\n")

    jpm_results_list = []
    caf_results_list = []
    caf_static_results_list = []
    caf_histories = []
    run_contexts = []
    per_run_records = []
    run_seeds = [int(args.base_seed + idx) for idx in range(num_runs)]

    for run_idx, run_seed in enumerate(run_seeds):
        np.random.seed(run_seed)
        random.seed(run_seed)
        print(f"\n>>> Starting Run {run_idx + 1}/{num_runs} (seed={run_seed}) <<<")
        # 1. Run JPM Pipeline to get initialization and communities
        t0_jpm = perf_counter()
        pipeline = Pipeline(
            correlation_matrix.values, covariance_matrix.values, expected_returns.values,
            denoiser=Denoiser(active=True, q=0.5, q_fit=True),
            cluster=Clustering(active=True, clustering_method="louvain", take_absolute_value=True),
            optimize_func=SimulatedAnnealingDwave(num_reads=num_reads),
        )
        pipeline_result = pipeline.run(run_optimizer=True, risk_rebalancing=True, cluster_on_correlation=True, optimize_on_correlation=False, input_risk_factor=risk_factor)
        
        communities = extract_communities_from_partitions(pipeline.__state__["result"]["partitions"])
        subproblem_cardinalities = split_cardinality_constraint([len(c) for c in communities], cardinality)
        effective_risk_factor = risk_factor
        jpm_selected = np.where(np.asarray(pipeline_result["recombined_solution"]).astype(np.int64) == 1)[0].tolist()
        
        jpm_res = {"metrics": evaluate_portfolio(jpm_selected, covariance_matrix, expected_returns, risk_factor), "timing": {"total_seconds": perf_counter() - t0_jpm}}
        jpm_results_list.append(jpm_res)

        # 2. Run CAF Pipeline using JPM's decomposition as initialization
        def optimizer_factory(num_reads, seed=None):
            return SimulatedAnnealingDwave(num_reads=num_reads, seed=seed)

        caf_res = run_caf_pipeline(
            covariance_matrix, expected_returns, communities, subproblem_cardinalities,
            get_model_func=get_model,
            optimizer_factory=optimizer_factory,
            risk_factor=risk_factor, effective_risk_factor=effective_risk_factor,
            initial_selected_indices=jpm_selected,
            iterations=iterations, patience=patience, use_incremental=True, seed=run_seed
        )
        caf_results_list.append(caf_res)

        # 3. CAF static ablation (same initialization, no incremental context caching)
        caf_static_res = run_caf_pipeline(
            covariance_matrix,
            expected_returns,
            communities,
            subproblem_cardinalities,
            get_model_func=get_model,
            optimizer_factory=optimizer_factory,
            risk_factor=risk_factor,
            effective_risk_factor=effective_risk_factor,
            initial_selected_indices=jpm_selected,
            iterations=iterations,
            patience=patience,
            use_incremental=False,
            seed=run_seed,
        )
        caf_static_results_list.append(caf_static_res)
        
        caf_histories.append(caf_res["history"])
        run_contexts.append(
            {
                "communities": communities,
                "subproblem_cardinalities": subproblem_cardinalities,
                "partitions": pipeline.__state__["result"]["partitions"],
                "jpm_selected": jpm_selected,
            }
        )
        per_run_records.append(
            {
                "run_idx": run_idx,
                "seed": run_seed,
                "community_sizes": [int(len(c)) for c in communities],
                "subproblem_cardinalities": [int(k) for k in subproblem_cardinalities],
                "jpm": jpm_res,
                "caf": {
                    "metrics": caf_res["metrics"],
                    "timing": caf_res["timing"],
                    "history": caf_res["history"],
                },
                "caf_static": {
                    "metrics": caf_static_res["metrics"],
                    "timing": caf_static_res["timing"],
                },
            }
        )

    # Calculate statistics
    def aggregate_results(res_list):
        objs = [r["metrics"]["objective"] for r in res_list]
        risks = [r["metrics"]["risk"] for r in res_list]
        rets = [r["metrics"]["expected_return"] for r in res_list]
        times = [r["timing"]["total_seconds"] for r in res_list]
        builds = [r["timing"].get("time_build", 0.0) for r in res_list]
        solves = [r["timing"].get("time_solve", 0.0) for r in res_list]
        accepts = [r["timing"].get("time_accept", 0.0) for r in res_list]
        return {
            "metrics": {
                "objective": np.mean(objs), "objective_std": np.std(objs),
                "risk": np.mean(risks), "risk_std": np.std(risks),
                "expected_return": np.mean(rets), "expected_return_std": np.std(rets)
            },
            "timing": {
                "total_seconds": np.mean(times), "total_seconds_std": np.std(times),
                "time_build": np.mean(builds), "time_build_std": np.std(builds),
                "time_solve": np.mean(solves), "time_solve_std": np.std(solves),
                "time_accept": np.mean(accepts), "time_accept_std": np.std(accepts),
            }
        }

    agg_jpm = aggregate_results(jpm_results_list)
    agg_caf = aggregate_results(caf_results_list)
    agg_static = aggregate_results(caf_static_results_list)

    aggregated_history = aggregate_history_runs(caf_histories, iterations)
    gain_mean, gain_std = aggregate_gain_runs(caf_histories, iterations)
    aggregated_history["gain_mean"] = gain_mean
    aggregated_history["gain_std"] = gain_std

    partial_summary_path = summary_path.with_name(f"{summary_path.stem}.partial{summary_path.suffix}")

    # 4. q-sweep robustness (averaged over multiple runs)
    q_sweep_rows = []
    if args.skip_q_sweep:
        print("\n--- Skipping q-sweep (--skip-q-sweep enabled) ---")
    else:
        print("\n--- Running q-sweep ---")
        for q_idx, q_val in enumerate(q_values):
            delta_obj_vals = []
            delta_risk_vals = []
            delta_ret_vals = []
            for run_idx in range(num_runs):
                q_seed = int(args.base_seed + 10000 + q_idx * 100 + run_idx)
                np.random.seed(q_seed)
                random.seed(q_seed)
                run_ctx = run_contexts[run_idx]
                run_jpm_selected = run_ctx["jpm_selected"]
                jpm_q_metrics = evaluate_portfolio(run_jpm_selected, covariance_matrix, expected_returns, q_val)
                caf_q_res = run_caf_pipeline(
                    covariance_matrix,
                    expected_returns,
                    run_ctx["communities"],
                    run_ctx["subproblem_cardinalities"],
                    get_model_func=get_model,
                    optimizer_factory=optimizer_factory,
                    risk_factor=q_val,
                    effective_risk_factor=q_val,
                    initial_selected_indices=run_jpm_selected,
                    iterations=iterations,
                    patience=patience,
                    use_incremental=True,
                    seed=q_seed,
                )
                caf_q = caf_q_res["metrics"]
                delta_obj_vals.append(((caf_q["objective"] - jpm_q_metrics["objective"]) / abs(jpm_q_metrics["objective"])) * 100)
                delta_risk_vals.append(((caf_q["risk"] - jpm_q_metrics["risk"]) / abs(jpm_q_metrics["risk"])) * 100)
                delta_ret_vals.append(((caf_q["expected_return"] - jpm_q_metrics["expected_return"]) / max(abs(jpm_q_metrics["expected_return"]), 1e-12)) * 100)
            q_sweep_rows.append(
                {
                    "q": q_val,
                    "delta_obj_pct": float(np.mean(delta_obj_vals)),
                    "delta_obj_std": float(np.std(delta_obj_vals)),
                    "delta_risk_pct": float(np.mean(delta_risk_vals)),
                    "delta_risk_std": float(np.std(delta_risk_vals)),
                    "delta_ret_pct": float(np.mean(delta_ret_vals)),
                    "delta_ret_std": float(np.std(delta_ret_vals)),
                }
            )
            partial_summary = {
                "config": {
                    "base_seed": int(args.base_seed),
                    "run_seeds": run_seeds,
                    "runs": int(num_runs),
                    "data_prefix": args.data_prefix,
                    "risk_factor": float(risk_factor),
                    "q_values": q_values,
                    "num_reads": int(num_reads),
                    "iterations": int(iterations),
                    "patience": int(patience),
                },
                "status": {
                    "main_benchmark_completed": True,
                    "q_sweep_completed_q_count": int(len(q_sweep_rows)),
                    "q_sweep_total_q_count": int(len(q_values)),
                    "last_completed_q": float(q_val),
                },
                "jpm": agg_jpm,
                "caf": {"metrics": agg_caf["metrics"], "timing": agg_caf["timing"], "history": aggregated_history},
                "caf_static": agg_static,
                "q_sweep_rows": q_sweep_rows,
                "per_run_records": per_run_records,
            }
            partial_summary_path.write_text(json.dumps(to_jsonable(partial_summary), indent=2), encoding="utf-8")
            print(f"Partial q-sweep checkpoint saved to {partial_summary_path}")

    # Output benchmark
    baseline_objective = agg_jpm["metrics"]["objective"]
    
    def print_agg_block(name, agg_res, base_obj):
        m = agg_res["metrics"]
        t = agg_res["timing"]
        print(name)
        print(f"  -> Objective       : {m['objective']:.6f} ± {m['objective_std']:.6f}")
        print(f"  -> Risk            : {m['risk']:.6f} ± {m['risk_std']:.6f}")
        print(f"  -> Expected Return : {m['expected_return']:.6f} ± {m['expected_return_std']:.6f}")
        print(f"  -> Total Time      : {t['total_seconds']:.6f}s ± {t['total_seconds_std']:.6f}s")
        print(f"  -> Rel vs JPM      : {((m['objective'] - base_obj) / abs(base_obj)) * 100:.2f}%\n")

    print("\n==================================================")
    print(f"BENCHMARK RESULTS (Averaged over {num_runs} runs)")
    print("==================================================")
    print_agg_block("JPM Official Pipeline", agg_jpm, baseline_objective)
    print_agg_block("CAF Pipeline (Ours)", agg_caf, baseline_objective)
    print_agg_block("CAF Static (No Incremental)", agg_static, baseline_objective)

    # Statistical evidence for objective improvement: paired sign test + bootstrap CI
    jpm_objs = np.asarray([r["metrics"]["objective"] for r in jpm_results_list], dtype=float)
    caf_objs = np.asarray([r["metrics"]["objective"] for r in caf_results_list], dtype=float)
    rel_impr = ((jpm_objs - caf_objs) / np.abs(jpm_objs)) * 100.0
    wins = int(np.sum(caf_objs < jpm_objs))
    p_sign = sign_test_two_sided(wins, num_runs)
    ci_lo, ci_hi = bootstrap_ci_mean(rel_impr, n_boot=5000, alpha=0.05, seed=42)
    print("Paired Statistical Check (CAF vs JPM, objective)")
    print(f"  -> Runs with improvement : {wins}/{num_runs}")
    print(f"  -> Sign test p-value     : {p_sign:.6f}")
    print(f"  -> Mean rel improvement  : {np.mean(rel_impr):.4f}%")
    print(f"  -> 95% bootstrap CI      : [{ci_lo:.4f}%, {ci_hi:.4f}%]\n")
    print("==================================================")
    
    summary = {
        "config": {
            "base_seed": int(args.base_seed),
            "run_seeds": run_seeds,
            "runs": int(num_runs),
            "risk_factor": float(risk_factor),
            "num_reads": int(num_reads),
            "iterations": int(iterations),
            "patience": int(patience),
        },
        "runs": int(num_runs),
        "jpm": agg_jpm,
        "caf": {"metrics": agg_caf["metrics"], "timing": agg_caf["timing"], "history": aggregated_history},
        "caf_static": agg_static,
        "objective_sign_test_pvalue": p_sign,
        "objective_improvement_runs": wins,
        "objective_relative_improvement_mean_pct": float(np.mean(rel_impr)),
        "objective_relative_improvement_std_pct": float(np.std(rel_impr)),
        "objective_relative_improvement_ci95_pct": [ci_lo, ci_hi],
        "q_sweep_rows": q_sweep_rows,
        "per_run_records": per_run_records,
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")

    if args.skip_q_sweep:
        print("\nSkipping plot generation because --skip-q-sweep is enabled.")
    else:
        print("\nGenerating visualization plots (fully aligned with aggregated statistics)...")
        agg_caf["history"] = aggregated_history
        generate_plots(agg_caf, agg_jpm, agg_static, q_sweep_rows, args.out_dir, iterations)
        print(f"Plots successfully saved to {args.out_dir}/")

if __name__ == "__main__":
    main()
