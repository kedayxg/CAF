import argparse
import json
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
    subset_problem,
    to_jsonable,
)


def choose_solver_factories(solver_name, sa_num_reads, SimulatedAnnealingDwave, GurobiOptimizer):
    solver_name = solver_name.lower().strip()
    if solver_name == "sa":
        def optimizer_factory(num_reads=None, seed=None):
            dynamic_reads = sa_num_reads if num_reads is None else int(num_reads)
            kwargs = {"num_reads": dynamic_reads}
            if seed is not None:
                kwargs["seed"] = int(seed)
            return SimulatedAnnealingDwave(**kwargs)

        return optimizer_factory, optimizer_factory

    if solver_name == "gurobi":
        def optimizer_factory(num_reads=None, seed=None):
            del num_reads, seed
            return GurobiOptimizer()

        return optimizer_factory, optimizer_factory

    raise ValueError(f"Unsupported solver: {solver_name}. Supported: sa, gurobi")


def aggregate(records):
    arr_obj_static = np.asarray([r["static"]["objective"] for r in records], dtype=float)
    arr_obj_caf = np.asarray([r["caf"]["objective"] for r in records], dtype=float)
    arr_risk_static = np.asarray([r["static"]["risk"] for r in records], dtype=float)
    arr_risk_caf = np.asarray([r["caf"]["risk"] for r in records], dtype=float)
    arr_ret_static = np.asarray([r["static"]["expected_return"] for r in records], dtype=float)
    arr_ret_caf = np.asarray([r["caf"]["expected_return"] for r in records], dtype=float)
    arr_time_static = np.asarray([r["static_time_sec"] for r in records], dtype=float)
    arr_time_caf = np.asarray([r["caf_time_sec"] for r in records], dtype=float)

    rel_impr = ((arr_obj_static - arr_obj_caf) / np.abs(arr_obj_static)) * 100.0
    risk_delta = ((arr_risk_caf - arr_risk_static) / np.abs(arr_risk_static)) * 100.0
    ret_delta = ((arr_ret_caf - arr_ret_static) / np.maximum(np.abs(arr_ret_static), 1e-12)) * 100.0

    return {
        "static": {
            "objective_mean": float(arr_obj_static.mean()),
            "objective_std": float(arr_obj_static.std()),
            "risk_mean": float(arr_risk_static.mean()),
            "risk_std": float(arr_risk_static.std()),
            "return_mean": float(arr_ret_static.mean()),
            "return_std": float(arr_ret_static.std()),
            "time_sec_mean": float(arr_time_static.mean()),
            "time_sec_std": float(arr_time_static.std()),
        },
        "caf": {
            "objective_mean": float(arr_obj_caf.mean()),
            "objective_std": float(arr_obj_caf.std()),
            "risk_mean": float(arr_risk_caf.mean()),
            "risk_std": float(arr_risk_caf.std()),
            "return_mean": float(arr_ret_caf.mean()),
            "return_std": float(arr_ret_caf.std()),
            "time_sec_mean": float(arr_time_caf.mean()),
            "time_sec_std": float(arr_time_caf.std()),
        },
        "delta_pct": {
            "objective_improvement_mean_pct": float(rel_impr.mean()),
            "objective_improvement_std_pct": float(rel_impr.std()),
            "risk_change_mean_pct": float(risk_delta.mean()),
            "risk_change_std_pct": float(risk_delta.std()),
            "return_change_mean_pct": float(ret_delta.mean()),
            "return_change_std_pct": float(ret_delta.std()),
        },
    }


def generate_scaling_plot(rows, out_dir):
    import matplotlib.pyplot as plt

    n_vals = [r["N"] for r in rows]
    obj_impr = [r["summary"]["delta_pct"]["objective_improvement_mean_pct"] for r in rows]
    obj_err = [r["summary"]["delta_pct"]["objective_improvement_std_pct"] for r in rows]
    static_t = [r["summary"]["static"]["time_sec_mean"] for r in rows]
    caf_t = [r["summary"]["caf"]["time_sec_mean"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(n_vals, obj_impr, yerr=obj_err, marker="o", linewidth=2, capsize=4)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("CAF Objective Improvement vs N")
    axes[0].set_xlabel("N (assets)")
    axes[0].set_ylabel("Improvement over JPM Pipeline (%)")

    axes[1].plot(n_vals, static_t, marker="s", linewidth=2, label="JPM Pipeline")
    axes[1].plot(n_vals, caf_t, marker="o", linewidth=2, label="CAF")
    axes[1].set_title("Runtime vs N")
    axes[1].set_xlabel("N (assets)")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "scaling_summary.png", dpi=300)
    plt.close(fig)


def generate_cardinality_plot(rows, out_dir, n_assets):
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda r: r["K"])
    x_ratio = [r["K"] / r["N"] for r in sorted_rows]
    obj_impr = [r["summary"]["delta_pct"]["objective_improvement_mean_pct"] for r in sorted_rows]
    obj_err = [r["summary"]["delta_pct"]["objective_improvement_std_pct"] for r in sorted_rows]
    static_t = [r["summary"]["static"]["time_sec_mean"] for r in sorted_rows]
    caf_t = [r["summary"]["caf"]["time_sec_mean"] for r in sorted_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].errorbar(x_ratio, obj_impr, yerr=obj_err, marker="o", linewidth=2, capsize=4)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(f"CAF Objective Improvement vs K/N (N={n_assets})")
    axes[0].set_xlabel("K/N")
    axes[0].set_ylabel("Improvement over JPM Pipeline (%)")

    axes[1].plot(x_ratio, static_t, marker="s", linewidth=2, label="JPM Pipeline")
    axes[1].plot(x_ratio, caf_t, marker="o", linewidth=2, label="CAF")
    axes[1].set_title(f"Runtime vs K/N (N={n_assets})")
    axes[1].set_xlabel("K/N")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "cardinality_summary.png", dpi=300)
    plt.close(fig)


def parse_sizes(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_solvers(text):
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run CAF scaling study on bundled 2016 benchmark.")
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
        help="Directory containing 2016 benchmark npy files. Defaults to CAF/data/russell3000_subset.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_scaling"), help="Output folder.")
    parser.add_argument("--runs", type=int, default=5, help="Number of stochastic runs per setting.")
    parser.add_argument("--sizes", type=str, default="100,200,300,484", help="Comma-separated N list.")
    parser.add_argument("--solvers", type=str, default="sa", help="Comma-separated solver list: sa,gurobi.")
    parser.add_argument(
        "--k-ratios",
        type=str,
        default=None,
        help="Optional comma-separated K/N ratios (e.g., 0.25,0.33,0.5). If set, cardinality study is run.",
    )
    parser.add_argument("--sa-num-reads", type=int, default=200, help="num_reads for SA solver.")
    parser.add_argument("--risk-factor", type=float, default=0.5, help="Risk factor q.")
    parser.add_argument("--iterations", type=int, default=6, help="CAF max iterations.")
    parser.add_argument("--patience", type=int, default=2, help="CAF patience.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum worker count passed to the CAF proposal stage.",
    )
    parser.add_argument("--subset-seed", type=int, default=123, help="Seed for N-subset selection.")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed used to deterministically derive per-run random seeds.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <out-dir>/scaling_summary.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = parse_sizes(args.sizes)
    solvers = parse_solvers(args.solvers)
    k_ratios = parse_float_list(args.k_ratios) if args.k_ratios else None

    (
        Clustering,
        Denoiser,
        SimulatedAnnealingDwave,
        GurobiOptimizer,
        Pipeline,
        get_model,
    ) = load_upstream_components(args.jpm_repo, include_gurobi=True)

    data_dir = args.data_dir if args.data_dir is not None else bundled_data_dir
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Benchmark data directory not found: {data_dir}. Provide --data-dir if needed."
        )

    cov_full, corr_full, ret_full = load_local_jpm_data(data_dir)
    total_assets = len(ret_full)
    if any(n > total_assets for n in sizes):
        raise ValueError(f"Requested sizes {sizes} exceed available N={total_assets}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for solver_name in solvers:
        print(f"\n=== Solver: {solver_name.upper()} ===")
        try:
            static_optimizer_factory, caf_optimizer_factory = choose_solver_factories(
                solver_name, args.sa_num_reads, SimulatedAnnealingDwave, GurobiOptimizer
            )
        except Exception as exc:
            print(f"Skip solver {solver_name}: {exc}")
            continue

        solver_rows = []
        for n_assets in sizes:
            cov, corr, ret = subset_problem(cov_full, corr_full, ret_full, n_assets, args.subset_seed + n_assets)
            cardinality_list = (
                sorted(set(max(1, min(n_assets - 1, int(round(r * n_assets)))) for r in k_ratios))
                if k_ratios
                else [len(ret) // 2]
            )

            for cardinality in cardinality_list:
                run_records = []
                print(f"\n-> N={n_assets}, K={cardinality}, runs={args.runs}")
                for run_idx in range(args.runs):
                    run_seed = int(args.base_seed + n_assets * 1000 + cardinality * 10 + run_idx)
                    np.random.seed(run_seed)
                    random.seed(run_seed)
                    t0 = perf_counter()
                    pipeline = Pipeline(
                        corr.values,
                        cov.values,
                        ret.values,
                        denoiser=Denoiser(active=True, q=0.5, q_fit=True),
                        cluster=Clustering(active=True, clustering_method="louvain", take_absolute_value=True),
                        optimize_func=static_optimizer_factory(seed=run_seed),
                    )
                    if k_ratios:
                        # Pipeline.run hardcodes cardinality_divider=2, so for custom K/N we call optimize_and_score directly.
                        C2, partitions = pipeline.run(
                            run_optimizer=False,
                            risk_rebalancing=False,
                            cluster_on_correlation=True,
                            optimize_on_correlation=False,
                            input_risk_factor=args.risk_factor,
                        )
                        cardinality_divider = max(1, int(round(n_assets / cardinality)))
                        pipeline_result = pipeline.optimize_and_score(
                            matrix_to_cluster=C2,
                            matrix_to_optimize=cov.values,
                            partitions=partitions,
                            risk_factor=args.risk_factor,
                            unique_identifier_base=f"scaling_{n_assets}_{cardinality}_{run_idx}",
                            log_best_feasible=False,
                            cluster_on_correlation=True,
                            optimize_on_correlation=False,
                            denoise_process_time=0.0,
                            clustering_process_time=0.0,
                            denoise_time=0.0,
                            clustering_time=0.0,
                            risk_rebalancing=False,
                            cardinality_divider=cardinality_divider,
                        )
                    else:
                        pipeline_result = pipeline.run(
                            run_optimizer=True,
                            risk_rebalancing=False,
                            cluster_on_correlation=True,
                            optimize_on_correlation=False,
                            input_risk_factor=args.risk_factor,
                        )
                    static_time = perf_counter() - t0

                    partitions = pipeline.__state__["result"]["partitions"]
                    communities = extract_communities_from_partitions(partitions)
                    subproblem_cardinalities = split_cardinality_constraint([len(c) for c in communities], cardinality)
                    eff_risk = args.risk_factor
                    static_selected = np.where(
                        np.asarray(pipeline_result["recombined_solution"]).astype(np.int64) == 1
                    )[0].tolist()
                    static_metrics = evaluate_portfolio(static_selected, cov, ret, args.risk_factor)

                    t1 = perf_counter()
                    caf_res = run_caf_pipeline(
                        cov,
                        ret,
                        communities,
                        subproblem_cardinalities,
                        get_model_func=get_model,
                        optimizer_factory=caf_optimizer_factory,
                        risk_factor=args.risk_factor,
                        effective_risk_factor=eff_risk,
                        initial_selected_indices=static_selected,
                        iterations=args.iterations,
                        patience=args.patience,
                        use_incremental=True,
                        max_workers=1 if solver_name == "gurobi" else args.max_workers,
                        seed=run_seed,
                    )
                    caf_time = perf_counter() - t1

                    run_records.append(
                        {
                            "run_idx": run_idx,
                            "seed": run_seed,
                            "static": static_metrics,
                            "caf": caf_res["metrics"],
                            "static_time_sec": static_time,
                            "caf_time_sec": caf_time,
                        }
                    )

                summary = aggregate(run_records)
                print(
                    f"   Objective improvement (CAF vs JPM Pipeline): "
                    f"{summary['delta_pct']['objective_improvement_mean_pct']:.3f}% "
                    f"+/- {summary['delta_pct']['objective_improvement_std_pct']:.3f}%"
                )
                solver_rows.append(
                    {
                        "solver": solver_name,
                        "N": n_assets,
                        "K": cardinality,
                        "K_over_N": float(cardinality / n_assets),
                        "summary": summary,
                        "run_records": run_records,
                    }
                )

        if solver_name == "sa" and solver_rows:
            if k_ratios and len(sizes) == 1:
                generate_cardinality_plot(solver_rows, out_dir, sizes[0])
            else:
                generate_scaling_plot(solver_rows, out_dir)
        all_rows.extend(solver_rows)

    if not all_rows:
        raise RuntimeError("No successful solver runs were completed.")

    summary = {
        "config": {
            "runs": args.runs,
            "sizes": sizes,
            "solvers": solvers,
            "risk_factor": args.risk_factor,
            "iterations": args.iterations,
            "patience": args.patience,
            "subset_seed": args.subset_seed,
            "base_seed": args.base_seed,
        },
        "rows": all_rows,
    }
    summary_path = args.summary_file if args.summary_file is not None else (out_dir / "scaling_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"\nScaling summary saved to: {summary_path}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
