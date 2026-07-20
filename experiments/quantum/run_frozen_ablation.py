#!/usr/bin/env python
"""CAF 真机验证 S0 —— 冻结 iteration-0 快照 + Exp A(注入/不注入) + Exp B 经典行(exact/sim_qaoa)。

关键设计（修正版）：exact 与 sim 都从【同一次 QAOA 运行】取，且所有候选都用 weights
钉在 C 序，消除变量顺序/跨方法/跨快照歧义：
  - exact   = argmin over 可行态 of bqm 能量（真·QUBO 最优）
  - sim_qaoa= shot 采样后 best-feasible（hw 同口径）
两者同一 QUBO、同一快照、同一变量序 → 可直接比较。

全离线，不调真机。
"""
import argparse
import itertools
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import dimod
from scipy.optimize import minimize

_HW = Path(__file__).resolve().parent
CODE_ROOT = _HW.parents[1]
SNAPSHOT_ROOT = CODE_ROOT.parent
CAF_ROOT = CODE_ROOT
HARDWARE_RESULTS_ROOT = SNAPSHOT_ROOT / "results" / "hardware"
for p in (str(CAF_ROOT),):
    import sys
    if p not in sys.path:
        sys.path.insert(0, p)

from caf_core import evaluate_state_vector, normalize_local_solution  # noqa: E402
from experiments._common import (  # noqa: E402
    extract_communities_from_partitions,
    load_local_jpm_data,
    load_upstream_components,
    split_cardinality_constraint,
    subset_problem,
)
from experiments.quantum.run_qaoa import (  # noqa: E402
    QAOAConfig,
    QiskitQAOAOptimizer,
)

EPS = 1e-5


def feedback(current_state, community, cand_Corder, cov_df, ret_s, risk_factor, base_obj):
    state = current_state.copy()
    state[community] = np.asarray(cand_Corder, dtype=int)
    m = evaluate_state_vector(state, cov_df, ret_s, risk_factor=risk_factor)
    dO = float(m["objective"] - base_obj)
    return dO, bool(dO <= -EPS)


def _relax(adjusted, sub_cov, K, q_eff):
    """局部 QUBO 的连续松弛 min q·xᵀΣx − rᵀx  s.t. 0≤x≤1, Σx=K（warm-start 用）。"""
    n = len(adjusted)
    x0 = np.full(n, K / n)
    res = minimize(lambda x: q_eff * (x @ sub_cov @ x) - adjusted @ x, x0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - K}],
                   options={"maxiter": 200, "ftol": 1e-9})
    return np.clip(res.x, 0.0, 1.0)


def solve_local(get_model, to_cqm, adjusted, sub_cov, K, q_eff, cfg):
    """对单个局部 QUBO 跑一次 QAOA，返回 (exact_cand, sim_cand, E_exact, E_sim)，均按 C 序。

    exact = bqm 能量最优可行态（穷举）；sim = QAOA shot 解码候选。同一 bqm、同一变量序。
    cfg.warmstart=True 时用 Egger warm-start（从连续松弛初态出发）。
    """
    model, weights = get_model(adjusted, sub_cov, budget=K, risk_factor=q_eff)
    n = len(weights)
    ws = None
    if getattr(cfg, "warmstart", False):
        c = _relax(adjusted, sub_cov, K, q_eff)
        label_of_Cpos = [weights[j].name for j in range(n)]
        ws = {label_of_Cpos[j]: float(c[j]) for j in range(n)}
    opt = QiskitQAOAOptimizer(cfg, warmstart_c=ws)
    sim_cand, _, _ = opt(model, weights)                 # 已按 weights(C) 序
    sim_cand = normalize_local_solution(sim_cand, n, K)  # 强制基数（C 序不变）

    cqm = to_cqm(model)
    bqm, _ = dimod.cqm_to_bqm(cqm)
    label_of_Cpos = [weights[j].name for j in range(n)]   # C 序 -> bqm label

    # exact：穷举可行态，bqm.energy，返回 C 序
    best_combo, best_e = None, None
    for combo in itertools.combinations(range(n), K):
        sample = {label_of_Cpos[j]: (1 if j in combo else 0) for j in range(n)}
        e = float(bqm.energy(sample))
        if best_e is None or e < best_e:
            best_e, best_combo = e, combo
    exact_cand = np.array([1 if j in best_combo else 0 for j in range(n)], dtype=int)

    sim_sample = {label_of_Cpos[j]: int(sim_cand[j]) for j in range(n)}
    E_sim = float(bqm.energy(sim_sample))
    return exact_cand, sim_cand, float(best_e), E_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jpm-repo",
        type=Path,
        default=None,
        help="Optional override path to another dcmppln checkout. Defaults to the vendored copy.",
    )
    ap.add_argument("--data-dir", type=Path, default=SNAPSHOT_ROOT / "data" / "russell3000_subset")
    ap.add_argument("--out-dir", type=Path, default=HARDWARE_RESULTS_ROOT)
    ap.add_argument("--n-assets", type=int, default=40)
    ap.add_argument("--subset-seed", type=int, default=1234)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--risk-factor", type=float, default=0.5)
    ap.add_argument("--qaoa-p", type=int, default=2)
    ap.add_argument("--qaoa-restarts", type=int, default=6)
    ap.add_argument("--qaoa-maxiter", type=int, default=80)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--warmstart", action="store_true", help="Egger warm-start QAOA（从连续松弛初态）")
    args = ap.parse_args()

    Clustering, Denoiser, SimulatedAnnealingDwave, Pipeline, get_model, rebal = load_upstream_components(args.jpm_repo)
    to_cqm = QiskitQAOAOptimizer._docplex_model_to_cqm

    cov_df, corr_df, ret_s = load_local_jpm_data(args.data_dir)
    cov_df, corr_df, ret_s = subset_problem(cov_df, corr_df, ret_s, args.n_assets, args.subset_seed)
    cardinality = len(ret_s) // 2
    q = args.risk_factor

    np.random.seed(args.base_seed); random.seed(args.base_seed)
    pipe = Pipeline(corr_df.values, cov_df.values, ret_s.values,
                    denoiser=Denoiser(active=True, q=0.5, q_fit=True),
                    cluster=Clustering(active=True, clustering_method="louvain", take_absolute_value=True),
                    optimize_func=SimulatedAnnealingDwave(num_reads=args.num_reads, seed=args.base_seed))
    pr = pipe.run(run_optimizer=True, risk_rebalancing=True, cluster_on_correlation=True,
                  optimize_on_correlation=False, input_risk_factor=q)
    communities = extract_communities_from_partitions(pipe.__state__["result"]["partitions"])
    subcard = split_cardinality_constraint([len(c) for c in communities], cardinality)
    q_eff = rebal(q, cov_df.values, ret_s.values, pipe.__state__["result"]["partitions"])
    static_sel = np.where(np.asarray(pr["recombined_solution"]).astype(np.int64) == 1)[0].tolist()

    cov = cov_df.values; ret = ret_s.values
    current_state = np.zeros(cov.shape[0], dtype=int); current_state[static_sel] = 1
    v = cov @ current_state
    base_obj = float(evaluate_state_vector(current_state, cov_df, ret_s, risk_factor=q)["objective"])

    sizes = [(i, int(len(c))) for i, c in enumerate(communities)]
    print("社区大小:", sizes)

    cfg = QAOAConfig(p=args.qaoa_p, restarts=args.qaoa_restarts, maxiter=args.qaoa_maxiter,
                     seed=args.base_seed, optimizer="COBYLA", shots=4000, decode="shots",
                     warmstart=args.warmstart)

    # ---- 扫描所有 n≤16 社区：注入/不注入 exact + sim_qaoa（同一次 QAOA）----
    scan = []
    for i, C_i in enumerate(communities):
        n_i = int(len(C_i)); K_i = int(subcard[i])
        if n_i > 16 or K_i <= 0:
            scan.append({"idx": i, "n": n_i, "K": K_i, "dO_inject": None, "dO_sim": None})
            continue
        sub_i = cov[np.ix_(C_i, C_i)]; x_i = current_state[C_i]
        sh_i = 2.0 * q_eff * (v[C_i] - sub_i @ x_i)
        adj_i = ret[C_i] - sh_i
        ex_i, sm_i, E_ex, E_sm = solve_local(get_model, to_cqm, adj_i, sub_i, K_i, q_eff, cfg)
        exn_i, _, _, _ = solve_local(get_model, to_cqm, ret[C_i], sub_i, K_i, q_eff, cfg)  # 不注入
        dO_i, acc_i = feedback(current_state, C_i, ex_i, cov_df, ret_s, q, base_obj)
        dO_n, acc_n = feedback(current_state, C_i, exn_i, cov_df, ret_s, q, base_obj)
        dO_s, acc_s = feedback(current_state, C_i, sm_i, cov_df, ret_s, q, base_obj)
        scan.append({"idx": i, "n": n_i, "K": K_i,
                     "dO_inject": dO_i, "acc_inject": acc_i,
                     "dO_noinject": dO_n, "acc_noinject": acc_n,
                     "dO_sim": dO_s, "acc_sim": acc_s,
                     "E_exact": E_ex, "E_sim": E_sm,
                     "approx": (E_sm / E_ex) if E_ex != 0 else None,
                     "hamming": int(np.sum(sm_i != ex_i))})
    scan.sort(key=lambda r: (r["dO_inject"] if r["dO_inject"] is not None else 0.0))
    print("\n---- 全社区扫描（按 ΔO_inject 升序；负=改进）----")
    print(f"{'comm':>5} {'n':>3} {'K':>3} {'dO_inject':>11} {'acc':>5} {'dO_noinj':>10} {'dO_sim':>10} {'accS':>5} {'approx':>7} {'hamm':>5}")
    for r in scan:
        def fmt(x, f="{:.6f}"): return "—" if x is None else (f.format(x) if isinstance(x, float) else str(x))
        print(f"{r['idx']:>5} {r['n']:>3} {r['K']:>3} {fmt(r['dO_inject']):>11} {fmt(r.get('acc_inject')):>5} "
              f"{fmt(r.get('dO_noinject')):>10} {fmt(r['dO_sim']):>10} {fmt(r.get('acc_sim')):>5} {fmt(r.get('approx'),'{:.3f}'):>7} {fmt(r.get('hamming')):>5}")

    # 选真机靶点：n≤6 且 exact 与 sim 都 ΔO<-EPS
    both = [r for r in scan if r["n"] <= 6 and r["dO_inject"] is not None and r["dO_inject"] < -EPS and r["dO_sim"] < -EPS]
    exact_only = [r for r in scan if r["n"] <= 6 and r["dO_inject"] is not None and r["dO_inject"] < -EPS]
    if both:
        ci = max(both, key=lambda r: -r["dO_sim"])["idx"]; why = "n≤6 且 exact 与 sim_qaoa 都 ΔO<0（真机靶点成立）"
    elif exact_only:
        ci = max(exact_only, key=lambda r: -r["dO_inject"])["idx"]; why = "⚠️ 仅 exact ΔO<0、sim 未命中（QAOA 在该 QUBO 上未集中到最优）"
    else:
        ci = min([r["idx"] for r in scan if r["n"] <= 6], key=lambda idx: len(communities[idx])); why = "⚠️ 无 n≤6 余量社区"
    C = communities[ci]; K_C = int(subcard[ci]); n = len(C)
    print(f"\n选定社区 #{ci}: n_local={n}, k_local={K_C}  （{why}）")

    # ---- 主表（选定社区，复用扫描结果）----
    sel = next(r for r in scan if r["idx"] == ci)
    rows = [
        {"solver": "noop", "injection": "—", "global_dO": 0.0, "accepted": False},
        {"solver": "exact", "injection": "off", "global_dO": sel["dO_noinject"], "accepted": sel["acc_noinject"]},
        {"solver": "exact", "injection": "on", "global_dO": sel["dO_inject"], "accepted": sel["acc_inject"]},
        {"solver": "sim_qaoa", "injection": "on", "global_dO": sel["dO_sim"], "accepted": sel["acc_sim"]},
        {"solver": "hw_qaoa", "injection": "on", "global_dO": None, "accepted": None},
    ]
    print("\n================ S0 主表（经典部分）================")
    print(f"{'solver':<10} {'inject':<6} {'global_dO':>12} {'accepted':>9}")
    for r in rows:
        dO = f"{r['global_dO']:.6f}" if r['global_dO'] is not None else "—(待真机)"
        ac = str(r['accepted']) if r['accepted'] is not None else "—"
        print(f"{r['solver']:<10} {r['injection']:<6} {dO:>12} {ac:>9}")
    print("====================================================")
    if sel.get("E_exact") is not None:
        print(f"exact 与 sim 是否同解: Hamming={sel['hamming']}  (0=sim 命中真·QUBO 最优)")
        print(f"通过门: exact ΔO<0 = {sel['dO_inject']<0}; 注入有效 = {sel['dO_inject']<sel['dO_noinject']}; sim 命中 = {sel['dO_sim']<-EPS}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = {"base_seed": args.base_seed, "n_assets": args.n_assets, "q_eff": float(q_eff),
            "community_sizes": sizes, "selected": int(ci), "n_local": n, "k_local": K_C,
            "current_state": current_state.tolist(), "base_objective": base_obj,
            "scan": scan, "rows": rows}
    out = args.out_dir / f"S0_freeze_control_{ts}.json"
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n快照+结果已保存: {out}")


if __name__ == "__main__":
    main()
