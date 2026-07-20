import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

current_dir = Path(__file__).resolve().parent
caf_root = current_dir.parent.parent
if str(caf_root) not in sys.path:
    sys.path.insert(0, str(caf_root))

from experiments.quantum.run_qaoa import (
    SAOptimizer,
    QAOAConfig,
    QiskitQAOAOptimizer,
    SQAOptimizer,
    aggregate_results,
    make_denoiser_factory,
    run_reference_backend,
    run_one_setting,
)
from experiments._common import (
    BUNDLED_DATA_DIR as bundled_data_dir,
    load_local_jpm_data,
    load_upstream_components,
    subset_problem,
    to_jsonable,
)


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def generate_p_sweep_plot(rows, out_dir):
    import matplotlib.pyplot as plt

    p_values = [row["p"] for row in rows]
    win_rates = [100.0 * row["summary"]["wins"] / max(1, row["summary"]["total_runs"]) for row in rows]
    abs_delta = [row["summary"]["absolute_delta_mean"] for row in rows]
    abs_delta_err = [row["summary"]["absolute_delta_std"] for row in rows]
    risk_reduction = [row["summary"]["risk_reduction_pct_mean"] for row in rows]
    risk_reduction_err = [row["summary"]["risk_reduction_pct_std"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(p_values, win_rates, marker="o", linewidth=2)
    axes[0].set_title("QAOA Acceptance Rate vs Depth")
    axes[0].set_xlabel("QAOA depth p")
    axes[0].set_ylabel("Accepted improving runs (%)")
    axes[0].set_ylim(0, 100)

    axes[1].errorbar(p_values, abs_delta, yerr=abs_delta_err, marker="o", linewidth=2, capsize=4)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Objective Delta vs Depth")
    axes[1].set_xlabel("QAOA depth p")
    axes[1].set_ylabel("Mean objective delta")

    axes[2].errorbar(p_values, risk_reduction, yerr=risk_reduction_err, marker="o", linewidth=2, capsize=4)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_title("Risk Reduction vs Depth")
    axes[2].set_xlabel("QAOA depth p")
    axes[2].set_ylabel("Mean risk reduction (%)")

    fig.tight_layout()
    fig.savefig(out_dir / "qaoa_depth_sweep.png", dpi=300)
    plt.close(fig)


def generate_subproblem_size_plot(community_sizes, out_dir, max_gate_size=16):
    import matplotlib.pyplot as plt

    idx = np.arange(len(community_sizes))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(idx, community_sizes, color="#4C78A8")
    ax.axhline(max_gate_size, color="#E45756", linestyle="--", linewidth=1.5, label=f"Gate-model cutoff n={max_gate_size}")
    ax.set_title("Folded Community Sizes on the N=40 Quantum Instance")
    ax.set_xlabel("Community index")
    ax.set_ylabel("Community size")
    ax.set_xticks(idx)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "folded_subproblem_sizes.png", dpi=300)
    plt.close(fig)


def write_backend_csv(rows, csv_path):
    fieldnames = [
        "backend",
        "p",
        "optimizer",
        "runs",
        "wins",
        "win_rate_pct",
        "absolute_delta_mean",
        "absolute_delta_std",
        "risk_reduction_pct_mean",
        "risk_reduction_pct_std",
        "return_increase_pct_mean",
        "return_increase_pct_std",
        "objective_improvement_pct_mean",
        "objective_improvement_pct_std",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            summary = row["summary"]
            writer.writerow(
                {
                    "backend": row["backend"],
                    "p": row.get("p"),
                    "optimizer": row.get("optimizer"),
                    "runs": summary["total_runs"],
                    "wins": summary["wins"],
                    "win_rate_pct": 100.0 * summary["wins"] / max(1, summary["total_runs"]),
                    "absolute_delta_mean": summary["absolute_delta_mean"],
                    "absolute_delta_std": summary["absolute_delta_std"],
                    "risk_reduction_pct_mean": summary["risk_reduction_pct_mean"],
                    "risk_reduction_pct_std": summary["risk_reduction_pct_std"],
                    "return_increase_pct_mean": summary["return_increase_pct_mean"],
                    "return_increase_pct_std": summary["return_increase_pct_std"],
                    "objective_improvement_pct_mean": summary["objective_improvement_pct_mean"],
                    "objective_improvement_pct_std": summary["objective_improvement_pct_std"],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Supplementary quantum experiments for CAF.")
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
        help="Directory containing benchmark npy files. Defaults to CAF/data/russell3000_subset.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_quantum_supplement"), help="Output folder.")
    parser.add_argument("--summary-file", type=Path, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--runs", type=int, default=10, help="Number of stochastic runs per setting.")
    parser.add_argument("--n-assets", type=int, default=40, help="Quantum feasibility subinstance size.")
    parser.add_argument("--subset-seed", type=int, default=1234, help="Seed for subinstance extraction.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed for stochastic runs.")
    parser.add_argument("--risk-factor", type=float, default=0.5, help="Risk factor q.")
    parser.add_argument("--iterations", type=int, default=4, help="CAF max iterations.")
    parser.add_argument("--patience", type=int, default=2, help="CAF patience.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional cap on CAF proposal-stage worker threads. Use a small value to reduce oversubscription.",
    )
    parser.add_argument("--p-values", type=str, default="1,2", help="Comma-separated QAOA depth list.")
    parser.add_argument("--qaoa-restarts", type=int, default=6, help="Random restarts for QAOA parameter search.")
    parser.add_argument("--qaoa-maxiter", type=int, default=80, help="Max iterations per QAOA restart.")
    parser.add_argument(
        "--qaoa-optimizer",
        type=str,
        default="COBYLA",
        help="Classical optimizer for QAOA parameter search: COBYLA, NELDER-MEAD, POWELL.",
    )
    parser.add_argument(
        "--warmstart",
        action="store_true",
        help="Use Egger-style warm-start QAOA based on the continuous local relaxation.",
    )
    parser.add_argument("--include-sa-reference", action="store_true", help="Also run SA reference.")
    parser.add_argument("--include-sqa-reference", action="store_true", help="Also run SQA reference.")
    parser.add_argument(
        "--skip-qaoa",
        action="store_true",
        help="Skip QAOA depth sweep and run only the requested reference backends.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    p_values = parse_int_list(args.p_values)
    if not p_values:
        raise ValueError("At least one QAOA depth must be provided via --p-values.")

    (
        Clustering,
        Denoiser,
        _SimulatedAnnealingDwave,
        Pipeline,
        get_model,
        rebalancing_risk_factor,
    ) = load_upstream_components(args.jpm_repo)
    denoiser_factory = make_denoiser_factory(Denoiser)

    data_dir = args.data_dir if args.data_dir is not None else bundled_data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Benchmark data directory not found: {data_dir}")

    cov_full, corr_full, ret_full = load_local_jpm_data(data_dir)
    cov, corr, ret = subset_problem(cov_full, corr_full, ret_full, args.n_assets, args.subset_seed)
    cardinality = len(ret) // 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    qaoa_rows = []
    backend_rows = []
    if not args.skip_qaoa:
        for depth in p_values:
            qaoa_cfg = QAOAConfig(
                p=int(depth),
                restarts=int(args.qaoa_restarts),
                maxiter=int(args.qaoa_maxiter),
                seed=int(args.base_seed),
                optimizer=str(args.qaoa_optimizer),
                warmstart=bool(args.warmstart),
            )

            def qaoa_factory(seed, num_reads=None):
                del num_reads
                cfg = QAOAConfig(
                    p=qaoa_cfg.p,
                    restarts=qaoa_cfg.restarts,
                    maxiter=qaoa_cfg.maxiter,
                    seed=int(seed),
                    optimizer=qaoa_cfg.optimizer,
                    warmstart=qaoa_cfg.warmstart,
                )
                return QiskitQAOAOptimizer(cfg)

            print(f"\n=== Running QAOA depth sweep: p={depth} ===")
            records = run_one_setting(
                cov=cov,
                corr=corr,
                ret=ret,
                cardinality=cardinality,
                risk_factor=args.risk_factor,
                runs=args.runs,
                # Reuse the same stochastic seeds across depths so p-sweeps stay paired.
                base_seed=args.base_seed,
                iterations=args.iterations,
                patience=args.patience,
                optimizer_factory_static=lambda seed: qaoa_factory(seed),
                optimizer_factory_caf=lambda seed, num_reads: qaoa_factory(seed, num_reads=num_reads),
                Clustering=Clustering,
                denoiser_factory=denoiser_factory,
                Pipeline=Pipeline,
                get_model=get_model,
                rebalancing_risk_factor=rebalancing_risk_factor,
                max_workers=args.max_workers,
            )
            summary = aggregate_results(records)
            qaoa_rows.append(
                {
                    "backend": f"QAOA-p{depth}",
                    "p": int(depth),
                    "optimizer": qaoa_cfg.optimizer,
                    "summary": summary,
                    "records": records,
                }
            )
            backend_rows.append(
                {
                    "backend": f"QAOA-p{depth}",
                    "p": int(depth),
                    "optimizer": qaoa_cfg.optimizer,
                    "summary": summary,
                }
            )
            print(
                f"p={depth}: wins={summary['wins']}/{summary['total_runs']}, "
                f"delta={summary['absolute_delta_mean']:.6f}, "
                f"risk_reduction={summary['risk_reduction_pct_mean']:.3f}%"
            )

    community_sizes = qaoa_rows[0]["records"][0]["community_sizes"] if qaoa_rows and qaoa_rows[0]["records"] else []

    sa_reference = None
    if args.include_sa_reference:
        print("\n=== Running SA reference ===")
        sa_summary, sa_records = run_reference_backend(
            optimizer_cls=SAOptimizer,
            seed_offset=20000,
            cov=cov,
            corr=corr,
            ret=ret,
            cardinality=cardinality,
            risk_factor=args.risk_factor,
            runs=args.runs,
            base_seed=args.base_seed,
            iterations=args.iterations,
            patience=args.patience,
            Clustering=Clustering,
            denoiser_factory=denoiser_factory,
            Pipeline=Pipeline,
            get_model=get_model,
            rebalancing_risk_factor=rebalancing_risk_factor,
            max_workers=args.max_workers,
        )
        sa_reference = {"backend": "SA", "summary": sa_summary, "records": sa_records}
        backend_rows.append({"backend": "SA", "p": None, "optimizer": None, "summary": sa_summary})

    sqa_reference = None
    if args.include_sqa_reference:
        print("\n=== Running SQA reference ===")
        sqa_summary, sqa_records = run_reference_backend(
            optimizer_cls=SQAOptimizer,
            seed_offset=30000,
            cov=cov,
            corr=corr,
            ret=ret,
            cardinality=cardinality,
            risk_factor=args.risk_factor,
            runs=args.runs,
            base_seed=args.base_seed,
            iterations=args.iterations,
            patience=args.patience,
            Clustering=Clustering,
            denoiser_factory=denoiser_factory,
            Pipeline=Pipeline,
            get_model=get_model,
            rebalancing_risk_factor=rebalancing_risk_factor,
            max_workers=args.max_workers,
        )
        sqa_reference = {"backend": "SQA", "summary": sqa_summary, "records": sqa_records}
        backend_rows.append({"backend": "SQA", "p": None, "optimizer": None, "summary": sqa_summary})
        if not community_sizes and sqa_records:
            community_sizes = sqa_records[0]["community_sizes"]

    comparison_csv = out_dir / "quantum_backend_comparison.csv"
    write_backend_csv(backend_rows, comparison_csv)
    if community_sizes:
        generate_subproblem_size_plot(community_sizes, out_dir)
    if qaoa_rows:
        generate_p_sweep_plot(qaoa_rows, out_dir)

    summary = {
        "config": {
            "runs": int(args.runs),
            "n_assets": int(args.n_assets),
            "cardinality": int(cardinality),
            "risk_factor": float(args.risk_factor),
            "iterations": int(args.iterations),
            "patience": int(args.patience),
            "max_workers": None if args.max_workers is None else int(args.max_workers),
            "subset_seed": int(args.subset_seed),
            "base_seed": int(args.base_seed),
            "p_values": [int(p) for p in p_values],
            "qaoa_restarts": int(args.qaoa_restarts),
            "qaoa_maxiter": int(args.qaoa_maxiter),
            "qaoa_optimizer": str(args.qaoa_optimizer),
            "warmstart": bool(args.warmstart),
            "include_sa_reference": bool(args.include_sa_reference),
            "include_sqa_reference": bool(args.include_sqa_reference),
        },
        "community_sizes": [int(x) for x in community_sizes],
        "qaoa_depth_sweep": qaoa_rows,
        "sa_reference": sa_reference,
        "sqa_reference": sqa_reference,
        "comparison_csv": str(comparison_csv),
    }
    summary_path = args.summary_file if args.summary_file is not None else (out_dir / "quantum_supplement_summary.json")
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Backend comparison CSV: {comparison_csv}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
