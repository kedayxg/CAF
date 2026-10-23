#!/usr/bin/env python
"""CAF hardware validation driver (S2-S4).

This script runs warm-start QAOA on WK_C180 and compares it against the ideal
simulator using the same optimized parameters `theta*`.

Workflow:
  1. Freeze the iteration-0 snapshot and select comm#2 (n<=6, with S0 showing exact=sim=ΔO<0).
  2. Run warm-start QAOA on the ideal simulator to obtain theta*, E_sim, the
     simulated candidate, and ΔO_sim.
  3. Reuse the same theta* and the same warm-start ansatz on WK_C180:
       - expval_pauli_operator(prog, H_C) -> E_hw
       - run_instruction sampling -> hardware candidate -> global ΔO_hw
  4. Check the pre-declared criterion from the protocol:
     accepted_hw == accepted_sim, ΔO_hw < 0, and ΔO_hw / ΔO_sim >= 0.7.

This script depends on pyqpanda3 and a valid API token. By default it runs in
dry-run mode, which only transpiles the circuit and prints gate statistics.
Use --submit to actually send the job.
"""
import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import dimod
from scipy.optimize import minimize

_HW = Path(__file__).resolve().parent
PROJECT_ROOT = _HW.parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT.parent
CAF_ROOT = PROJECT_ROOT
HARDWARE_RESULTS_ROOT = SNAPSHOT_ROOT / "results" / "hardware"
for p in (str(CAF_ROOT), str(_HW)):
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
from experiments.config import API_TOKEN, MACHINE_NAME  # noqa: E402
from experiments.quantum.run_qaoa import (  # noqa: E402
    QAOAConfig,
    QiskitQAOAOptimizer,
)

# Hardware-side objects are imported lazily and initialized only when needed.
SERVICE = BACKEND = CHIP_BACKEND = None
OPTIONS = None
SELECTED_PHYSICAL_QUBITS = None
LAYOUT_METHOD = "default"
IS_LOCAL = False
core = qcloud = transpilation = PauliOperator = None


def setup_pyqpanda(machine=None, local=False, physical_qubits=None, layout_method="default"):
    global SERVICE, BACKEND, CHIP_BACKEND, OPTIONS, SELECTED_PHYSICAL_QUBITS, LAYOUT_METHOD
    global IS_LOCAL, core, qcloud, transpilation, PauliOperator
    from pyqpanda3 import core as _core, qcloud as _qcloud, transpilation as _tr
    from pyqpanda3.hamiltonian import PauliOperator as _PO
    core, qcloud, transpilation, PauliOperator = _core, _qcloud, _tr, _PO
    IS_LOCAL = local
    SELECTED_PHYSICAL_QUBITS = sorted(set(int(q) for q in (physical_qubits or []))) or None
    LAYOUT_METHOD = str(layout_method or "default")
    if local:
        BACKEND = core.CPUQVM()          # Local full-amplitude simulator for validation.
        SERVICE = CHIP_BACKEND = None
        return "CPUQVM(local)"
    mach = machine or MACHINE_NAME
    SERVICE = qcloud.QCloudService(api_key=API_TOKEN)
    BACKEND = SERVICE.backend(mach)
    try:
        if SELECTED_PHYSICAL_QUBITS:
            CHIP_BACKEND = BACKEND.chip_backend(SELECTED_PHYSICAL_QUBITS)
        else:
            CHIP_BACKEND = BACKEND.chip_backend()
    except Exception:
        CHIP_BACKEND = None
    options = qcloud.QCloudOptions()
    options.set_mapping(True); options.set_optimization(True); options.set_amend(True)
    # NOTE: result-type (integer counts vs platform probs) is NO LONGER controlled by options
    # in current pyqpanda3 — set_is_prob_counts() is a no-op now. It is selected via
    # job.result(keys=["probCount"]) in _run_counts(). Kept amend=True (readout calibration).
    OPTIONS = options
    return mach


def _parse_qubit_list(text):
    if text is None:
        return None
    parts = [p.strip() for p in str(text).replace(";", ",").split(",")]
    vals = [int(p) for p in parts if p]
    return sorted(set(vals)) or None


# ============================ pyqpanda3 warm-start QAOA circuit ============================

def _rzz(prog, i, j, angle):
    """Apply RZZ(angle) or fall back to a CNOT-RZ-CNOT decomposition."""
    if hasattr(core, "RZZ"):
        prog << core.RZZ(i, j, angle)          # VERIFY: parameter order / angle convention if backend APIs change
    else:
        prog << core.CNOT(i, j) << core.RZ(j, angle) << core.CNOT(i, j)


def _u3_via_x1(prog, q, theta, phi, lam):
    """Decompose U3(theta, phi, lam) into the X1/RZ basis accepted by to_instruction().

    QPanda3's cloud path currently accepts X1/RZ/CZ at the QProg level. X1 is the
    fixed pi/2 X-rotation used by the backend instruction generator.
    """
    prog << core.RZ(q, float(phi + math.pi))
    prog << core.X1(q)
    prog << core.RZ(q, float(theta + math.pi))
    prog << core.X1(q)
    if abs(float(lam)) > 1e-12:
        prog << core.RZ(q, float(lam))


def _ry_via_x1(prog, q, theta):
    _u3_via_x1(prog, q, theta, 0.0, 0.0)


def _h_via_x1(prog, q):
    _u3_via_x1(prog, q, math.pi / 2.0, 0.0, math.pi)


def _cnot_via_cz(prog, control, target):
    _h_via_x1(prog, target)
    prog << core.CZ(control, target)
    _h_via_x1(prog, target)


def _rzz_via_native(prog, i, j, angle):
    _cnot_via_cz(prog, i, j)
    prog << core.RZ(j, float(angle))
    _cnot_via_cz(prog, i, j)


def build_qaoa_prog(h, J, theta, p, ws_theta, n, var_labels):
    """Build a warm-start ansatz aligned with QiskitQAOAOptimizer.

    The circuit uses an RY initialization, RZ/RZZ cost layers, and an RY-RZ-RY
    mixer. If `ws_theta` is None, it falls back to an H-initialized plain QAOA
    circuit. The keys of `h` and `J` are BQM variable labels and are mapped to
    qubit positions through `var_labels`.
    """
    l2p = {lbl: q for q, lbl in enumerate(var_labels)}
    prog = core.QProg()
    gammas, betas = theta[:p], theta[p:]
    for q in range(n):
        if ws_theta is not None:
            _ry_via_x1(prog, q, float(ws_theta[q]))
        else:
            _h_via_x1(prog, q)
    for layer in range(p):
        g, b = float(gammas[layer]), float(betas[layer])
        for i, c in h.items():
            prog << core.RZ(l2p[i], 2.0 * g * float(c))
        for (i, j), c in J.items():
            _rzz_via_native(prog, l2p[i], l2p[j], 2.0 * g * float(c))
        for q in range(n):
            if ws_theta is not None:
                _ry_via_x1(prog, q, -float(ws_theta[q]))
                prog << core.RZ(q, 2.0 * b)
                _ry_via_x1(prog, q, float(ws_theta[q]))
            else:
                _u3_via_x1(prog, q, 2.0 * b, -math.pi / 2.0, math.pi / 2.0)
    return prog


def make_hamiltonian(h, J, var_labels):
    """Build the PauliOperator for sum_i h_i Z_i + sum_ij J_ij Z_i Z_j."""
    l2p = {lbl: q for q, lbl in enumerate(var_labels)}
    terms = {}
    for i, c in h.items():
        terms[f"Z{l2p[i]}"] = float(c)
    for (i, j), c in J.items():
        terms[f"Z{l2p[i]} Z{l2p[j]}"] = float(c)
    try:
        return PauliOperator(terms), True
    except Exception:
        return terms, False  # Fallback: return a term dictionary for term-by-term expval accumulation.


def _expval(prog, op):
    """CPUQVM: expval_pauli_operator(prog, op)；QPU: expval_pauli_operator(prog, op, options)。"""
    try:
        return float(BACKEND.expval_pauli_operator(prog, op))
    except TypeError:
        return float(BACKEND.expval_pauli_operator(prog, op, OPTIONS))


def _status_name(status):
    return getattr(status, "name", str(status))


def _safe_call(obj, method_name, default=None):
    method = getattr(obj, method_name, None)
    if method is None:
        return default
    try:
        return method()
    except Exception:
        return default


def _parse_origin_payload(payload):
    if not payload:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return None
    return None


def _decode_taskresult_key(key, width):
    if isinstance(key, str) and key.startswith(("0x", "0X")):
        return format(int(key, 16), f"0{width}b")
    key = str(key)
    if set(key) <= {"0", "1"}:
        return key.rjust(width, "0")
    return None


def _normalize_count_keys(counts, n):
    """Normalize count keys to fixed-width-n binary strings.

    The QPU probCount channel returns HEX keys (e.g. '0x3'); the local CPUQVM and
    the taskResult fallback return binary strings. Route both through the hex-aware
    decoder so downstream width/decode logic is uniform.
    """
    out = {}
    for k, v in counts.items():
        b = _decode_taskresult_key(k, n)
        if b is not None:
            out[b] = out.get(b, 0) + v
    return out


def _taskresult_distribution(origin_data, width):
    payload = _parse_origin_payload(origin_data)
    obj = payload.get("obj", {}) if isinstance(payload, dict) else {}
    task_result = obj.get("taskResult", [])
    if not task_result:
        return {}, None
    entry = _parse_origin_payload(task_result[0])
    if not isinstance(entry, dict):
        return {}, None
    keys = entry.get("key", [])
    values = entry.get("value", [])
    dist = {}
    for key, value in zip(keys, values):
        bits = _decode_taskresult_key(key, width)
        if bits is None:
            continue
        dist[bits] = float(value)
    return dist, entry


def _transpile_prog_for_chip(prog):
    if CHIP_BACKEND is None:
        return prog, False
    init_mapping = {}
    use_transpile = bool(SELECTED_PHYSICAL_QUBITS) or LAYOUT_METHOD != "default"
    if not use_transpile:
        try:
            prog.to_instruction(CHIP_BACKEND, 1, False, False)
            return prog, False
        except Exception:
            use_transpile = True
    layout_method = LAYOUT_METHOD if LAYOUT_METHOD != "default" else "default"
    mapped = transpilation.Transpiler().transpile(prog, CHIP_BACKEND, init_mapping, 2, layout_method)
    return mapped, True


def _run_counts(prog, shots, n, poll_interval_s=3, max_polls=20):
    """Fetch counts together with backend-side diagnostic metadata."""
    if IS_LOCAL:
        BACKEND.run(prog, shots=shots)
        result = BACKEND.result()
        return result.get_counts(), {
            "backend_mode": "local",
            "counts_source": "result.get_counts",
            "decode_source": "counts",
            "distribution_kind": "raw_counts",
        }
    if CHIP_BACKEND is not None:
        try:
            compiled_prog, transpile_used = _transpile_prog_for_chip(prog)
            instr = compiled_prog.to_instruction(CHIP_BACKEND, 1, False, False)
        except Exception:
            mapped = transpilation.Transpiler().transpile(prog, CHIP_BACKEND, {}, 2, LAYOUT_METHOD)
            instr = mapped.to_instruction(CHIP_BACKEND, 1, False, False)
            transpile_used = True
    else:
        instr = prog.to_instruction()
        transpile_used = False
    job = BACKEND.run_instruction([instr], shots, OPTIONS)
    job_id = _safe_call(job, "job_id")
    status_trace = []
    final_status = None
    for _ in range(max_polls):
        final_status = _status_name(_safe_call(job, "status", "UNKNOWN"))
        status_trace.append(final_status)
        if final_status in {"FINISHED", "FAILED"}:
            break
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)
    try:
        res = job.result(keys=["probCount"])   # current pyqpanda3: result-type is selected HERE, not via options. "probCount" => integer per-shot counts.
    except Exception as _e:
        print(f"[warn] job.result(keys=['probCount']) unsupported on this backend ({type(_e).__name__}: {_e}); falling back to result() (will likely hit the taskResult/platform-prob fallback). If so, switch BACKEND.run_instruction([...]) to BACKEND.run(prog, shots, OPTIONS).")
        res = job.result()
    counts_list = _safe_call(res, "get_counts_list")
    counts = (counts_list or [_safe_call(res, "get_counts", {})])[0] or {}
    origin_data = _safe_call(res, "origin_data")
    taskresult_dist, taskresult_entry = ({}, None)
    decode_source = "counts"
    distribution_kind = "raw_counts"
    if not counts:
        taskresult_dist, taskresult_entry = _taskresult_distribution(origin_data, n)
        if taskresult_dist:
            counts = taskresult_dist
            decode_source = "taskResult"
            distribution_kind = "platform_probs"
    meta = {
        "backend_mode": "qpu",
        "job_id": job_id,
        "status_trace": status_trace,
        "final_status": final_status or _status_name(_safe_call(job, "status", "UNKNOWN")),
        "transpile_used": transpile_used,
        "timing_info": _safe_call(res, "timing_info"),
        "error_message": _safe_call(res, "error_message"),
        "origin_data": origin_data,
        "counts_source": "get_counts_list" if counts_list else "get_counts",
        "decode_source": decode_source,
        "distribution_kind": distribution_kind,
        "distribution_total": float(sum(counts.values())) if counts else 0.0,
        "taskResult_entry": taskresult_entry,
        "selected_physical_qubits": SELECTED_PHYSICAL_QUBITS,
        "layout_method": LAYOUT_METHOD,
    }
    return counts, meta


# ============================ simulator-side reference ============================

def run_sim(get_model, to_cqm, adjusted, sub, K, q_eff, cfg, n):
    model, weights = get_model(adjusted, sub, budget=K, risk_factor=q_eff)
    label_of_Cpos = [weights[j].name for j in range(n)]
    ws = None
    if cfg.warmstart:
        c = _relax(adjusted, sub, K, q_eff)
        ws = {label_of_Cpos[j]: float(c[j]) for j in range(n)}
    opt = QiskitQAOAOptimizer(cfg, warmstart_c=ws)
    sim_cand, _, _ = opt(model, weights)
    sim_cand = normalize_local_solution(sim_cand, n, K)
    bqm, _ = dimod.cqm_to_bqm(to_cqm(model))
    h, J, offset = bqm.to_ising()
    var_labels = list(bqm.variables)
    # Warm-start angles in var_labels order, reused by the hardware circuit.
    ws_theta_hw = None
    if ws is not None:
        ws_theta_hw = np.array([
            2.0 * np.arcsin(np.sqrt(min(1.0, max(0.0, float(ws.get(lbl, 0.5))))))
            for lbl in var_labels
        ])
    E_sim = float(np.dot(opt.last_probs, opt.last_energies))
    return dict(h=h, J=J, offset=offset, var_labels=var_labels, theta=opt.last_theta,
                ws_theta_hw=ws_theta_hw, sim_cand=sim_cand, E_sim=E_sim,
                p=cfg.p, n=n, model=model, weights=weights, bqm=bqm)


def _relax(adjusted, sub, K, q_eff):
    n = len(adjusted)
    res = minimize(lambda x: q_eff * (x @ sub @ x) - adjusted @ x, np.full(n, K / n),
                   method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - K}],
                   options={"maxiter": 200, "ftol": 1e-9})
    return np.clip(res.x, 0.0, 1.0)


# ============================ hardware-side evaluation ============================

def hw_energy(prog, h, J, offset, var_labels):
    # pyqpanda's Pauli-Z expval uses |0> -> +1, |1> -> -1, while dimod's
    # Ising convention here uses s = 2x - 1. Flip the linear terms so the
    # expval-based diagnostic energy matches the simulator-side Ising scale.
    h_fixed = {label: -float(coeff) for label, coeff in h.items()}
    op, ok = make_hamiltonian(h_fixed, J, var_labels)
    if ok:
        return float(offset) + _expval(prog, op)
    val = float(offset)  # Fallback: accumulate expvals term by term.
    for spec, c in op.items():
        val += c * _expval(prog, PauliOperator({spec: 1.0}))
    return val


def _decode_counts_bitstring(bits, n, width):
    bits = bits.rjust(width, "0")
    return np.array([int(bits[width - 1 - q]) for q in range(n)], dtype=int)


def _ising_energy_from_bits(x_bits, h, J, offset, var_labels):
    label_to_pos = {label: pos for pos, label in enumerate(var_labels)}
    # Keep the bit-to-spin convention aligned with the simulator-side QAOA path.
    spins = 2 * np.asarray(x_bits, dtype=int) - 1
    val = float(offset)
    for i, coeff in h.items():
        val += float(coeff) * float(spins[label_to_pos[i]])
    for (i, j), coeff in J.items():
        val += float(coeff) * float(spins[label_to_pos[i]] * spins[label_to_pos[j]])
    return val


def hw_counts_energy_and_sample(prog, n, K, shots, adjusted, sub, q_eff, h, J, offset, var_labels):
    """Single measurement pass for diagonal cost Hamiltonians.

    The local cost used here is diagonal in the computational basis (Z/ZZ only), so
    one batch of computational-basis counts is sufficient to:
    1) estimate the hardware energy E_hw by Monte Carlo averaging, and
    2) choose the best feasible sampled candidate for the CAF feedback step.
    """
    prog << core.measure(list(range(n)), list(range(n)))
    counts, meta = _run_counts(prog, shots, n)
    counts = _normalize_count_keys(counts, n)
    width = max((len(k) for k in counts), default=n)
    total = sum(counts.values()) or 1

    best_x, best_local_e = None, None
    energy_acc = 0.0
    for bits, freq in counts.items():
        x_bits = _decode_counts_bitstring(bits, n, width)
        energy_acc += float(freq) * _ising_energy_from_bits(x_bits, h, J, offset, var_labels)
        if int(x_bits.sum()) != K:
            continue
        local_e = q_eff * (x_bits @ sub @ x_bits) - adjusted @ x_bits
        if best_local_e is None or local_e < best_local_e:
            best_local_e, best_x = float(local_e), x_bits

    return energy_acc / float(total), best_x, counts, meta


def hw_sample(prog, n, K, shots, adjusted, sub, q_eff):
    """Select the best feasible candidate from one batch of sampled counts.

    The bit ordering follows the CPUQVM-confirmed convention used in this code:
    the k-th bit from the right corresponds to qubit k. Measurement gates are
    appended here because counts require them whereas expval evaluation does not.
    """
    prog << core.measure(list(range(n)), list(range(n)))   # Add measurements for count-based sampling.
    counts, meta = _run_counts(prog, shots, n)
    counts = _normalize_count_keys(counts, n)
    L = max(len(k) for k in counts) if counts else n
    best_x, best_e = None, None
    for bits, _ in counts.items():
        bits = bits.rjust(L, "0")                      # Left-pad to a fixed width.
        b = np.array([int(bits[L - 1 - q]) for q in range(n)], dtype=int)  # Rightmost bit q maps to qubit q.
        if int(b.sum()) != K:                          # Keep only feasible samples with the required cardinality.
            continue
        e = q_eff * (b @ sub @ b) - adjusted @ b
        if best_e is None or e < best_e:
            best_e, best_x = e, b
    return best_x, counts, meta


# ============================ driver ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jpm-repo",
        type=Path,
        default=None,
        help="Optional override path to another dcmppln checkout. Defaults to the vendored copy.",
    )
    ap.add_argument("--data-dir", type=Path, default=CAF_ROOT / "data" / "russell3000_subset")
    ap.add_argument("--out-dir", type=Path, default=HARDWARE_RESULTS_ROOT)
    ap.add_argument("--n-assets", type=int, default=40, help="Asset subset size used when freezing the snapshot.")
    ap.add_argument("--subset-seed", type=int, default=1234, help="Random seed for asset subset selection during snapshot freezing.")
    ap.add_argument("--community", type=int, default=2, help="Community index to evaluate (S0 uses #2).")
    ap.add_argument("--qaoa-p", type=int, default=2)
    ap.add_argument("--qaoa-restarts", type=int, default=6)
    ap.add_argument("--qaoa-maxiter", type=int, default=80)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--submit", action="store_true", help="Actually submit the job to hardware. By default the script stays in dry-run mode.")
    ap.add_argument("--machine", default=None, help="Optional backend override, e.g. WK_C180 or full_amplitude. Defaults to config.MACHINE_NAME.")
    ap.add_argument("--local", action="store_true", help="Run on local CPUQVM for validation without consuming hardware quota.")
    ap.add_argument("--physical-qubits", type=str, default=None, help="Comma-separated physical-qubit subset, for example 12,13,14,15,16,17,18.")
    ap.add_argument("--layout-method", choices=["default", "fidelity"], default="default",
                    help="Transpilation layout strategy. With a physical-qubit subset, 'fidelity' performs fidelity-aware placement within that subset.")
    ap.add_argument("--tau", type=float, default=0.7, help="Threshold for the pre-declared ΔO_hw/ΔO_sim criterion.")
    ap.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Disable Egger-style warm-start and use plain QAOA instead.",
    )
    args = ap.parse_args()
    physical_qubits = _parse_qubit_list(args.physical_qubits)
    mach = setup_pyqpanda(args.machine, local=args.local, physical_qubits=physical_qubits, layout_method=args.layout_method)
    print(f"Backend: {mach}")
    if physical_qubits:
        print(f"Physical-qubit subset: {physical_qubits}  [layout={args.layout_method}]")

    Clustering, Denoiser, SimulatedAnnealingDwave, Pipeline, get_model, rebal = load_upstream_components(args.jpm_repo)
    to_cqm = QiskitQAOAOptimizer._docplex_model_to_cqm
    q = 0.5

    # ---- Freeze the iteration-0 snapshot, matching the S0 setup ----
    cov_df, corr_df, ret_s = load_local_jpm_data(args.data_dir)
    cov_df, corr_df, ret_s = subset_problem(cov_df, corr_df, ret_s, args.n_assets, args.subset_seed)
    cardinality = len(ret_s) // 2
    np.random.seed(args.base_seed); random.seed(args.base_seed)
    pipe = Pipeline(corr_df.values, cov_df.values, ret_s.values,
                    denoiser=Denoiser(active=True, q=0.5, q_fit=True),
                    cluster=Clustering(active=True, clustering_method="louvain", take_absolute_value=True),
                    optimize_func=SimulatedAnnealingDwave(num_reads=200, seed=args.base_seed))
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

    ci = args.community
    C = communities[ci]; K_C = int(subcard[ci]); n = len(C)
    sub = cov[np.ix_(C, C)]; x_C = current_state[C]
    adjusted = ret[C] - 2.0 * q_eff * (v[C] - sub @ x_C)
    print(f"Community #{ci}: n={n}, K={K_C}")

    # ---- Simulator QAOA reference with shared theta* ----
    cfg = QAOAConfig(p=args.qaoa_p, restarts=args.qaoa_restarts, maxiter=args.qaoa_maxiter,
                     seed=args.base_seed, optimizer="COBYLA", shots=args.shots, decode="shots",
                     warmstart=(not args.no_warmstart))
    sim = run_sim(get_model, to_cqm, adjusted, sub, K_C, q_eff, cfg, n)

    def global_dO(cand_Corder):
        st = current_state.copy(); st[C] = np.asarray(cand_Corder, dtype=int)
        return float(evaluate_state_vector(st, cov_df, ret_s, risk_factor=q)["objective"] - base_obj)

    dO_sim = global_dO(sim["sim_cand"]); acc_sim = dO_sim <= -1e-5
    print(f"[sim] E_sim={sim['E_sim']:.6f}  ΔO_sim={dO_sim:+.6f}  accepted={acc_sim}  θ*={np.round(sim['theta'],4).tolist()}")

    record = {"community": ci, "n": n, "K": K_C, "p": args.qaoa_p,
              "n_assets": args.n_assets, "subset_seed": args.subset_seed,
              "warmstart": bool(cfg.warmstart),
              "physical_qubits": physical_qubits,
              "layout_method": args.layout_method,
              "theta_star": sim["theta"].tolist(),
              "E_sim": sim["E_sim"], "dO_sim": dO_sim, "accepted_sim": bool(acc_sim)}

    # ---- Hardware-side evaluation ----
    hw_prog = build_qaoa_prog(sim["h"], sim["J"], sim["theta"], sim["p"], sim["ws_theta_hw"], n, sim["var_labels"])
    run_hw = args.submit or args.local
    if not run_hw:
        if CHIP_BACKEND is not None:
            mapped, _ = _transpile_prog_for_chip(hw_prog)
            ops = mapped.count_ops() if hasattr(mapped, "count_ops") else "?"
        else:
            ops = "(no chip_backend available; skipping transpilation and gate statistics)"
        print(f"[dry-run] Circuit built without execution. Native gate statistics: {ops}")
        print("Use --local for CPUQVM validation or --submit to run on WK_C180.")
        record["mode"] = "dry_run"; record["native_ops"] = str(ops)
    else:
        if args.local:
            E_hw = hw_energy(hw_prog, sim["h"], sim["J"], sim["offset"], sim["var_labels"])
            hw_cand, counts, qpu_debug = hw_sample(hw_prog, n, K_C, args.shots, adjusted, sub, q_eff)
        else:
            E_hw, hw_cand, counts, qpu_debug = hw_counts_energy_and_sample(
                hw_prog,
                n,
                K_C,
                args.shots,
                adjusted,
                sub,
                q_eff,
                sim["h"],
                sim["J"],
                sim["offset"],
                sim["var_labels"],
            )
        fallback_used = hw_cand is None
        if fallback_used:
            print("[hw] No feasible sample satisfied the cardinality constraint; mark this as a hardware failure instead of falling back to all-zero.")
            hw_cand = np.zeros(n, dtype=int)
            dO_hw = None
            acc_hw = False
            hamming = None
            ratio = float("nan")
        else:
            dO_hw = global_dO(hw_cand)
            acc_hw = dO_hw <= -1e-5
            hamming = int(np.sum(hw_cand != sim["sim_cand"]))
            ratio = (dO_hw / dO_sim) if dO_sim != 0 else float("nan")
        tag = "LOCAL(CPUQVM,noiseless)" if args.local else f"QPU({mach})"
        crit = bool((not fallback_used) and acc_hw and acc_sim and dO_hw < 0 and ratio >= args.tau)
        dO_text = "NA(no feasible sample)" if dO_hw is None else f"{dO_hw:+.6f}"
        ham_text = "NA" if hamming is None else str(hamming)
        ratio_text = "NA" if fallback_used else f"{ratio:.3f}"
        print(f"[hw-{tag}] E_hw={E_hw:.6f} (ΔE={E_hw-sim['E_sim']:+.2e})  ΔO_hw={dO_text}  accepted={acc_hw}  Hamming(hw,sim)={ham_text}  ΔO_hw/ΔO_sim={ratio_text}")
        print(f"Pre-declared criterion (τ={args.tau}): {'PASS ✓' if crit else 'FAIL'}")
        record.update({"mode": "local" if args.local else "submit", "E_hw": E_hw, "dO_hw": dO_hw,
                       "accepted_hw": bool(acc_hw), "hamming_hw_sim": hamming, "ratio_dO": ratio,
                       "fallback_used": bool(fallback_used),
                       "criterion_pass": crit, "hw_candidate": hw_cand.tolist(),
                       "counts_total": int(sum(counts.values())) if qpu_debug.get("distribution_kind") == "raw_counts" else 0,
                       "distribution_total": float(sum(counts.values())),
                       "counts_distinct": int(len(counts)),
                       "counts_full": counts, "qpu_debug": qpu_debug})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"HW_qaoa_{('submit' if args.submit else 'dryrun')}_{ts}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
