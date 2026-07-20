import argparse
import inspect
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import dimod
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

current_dir = Path(__file__).resolve().parent
caf_root = current_dir.parent.parent
if str(caf_root) not in sys.path:
    sys.path.insert(0, str(caf_root))

from caf_core import (
    attach_local_problem_metadata,
    evaluate_portfolio,
    normalize_local_solution,
    run_caf_pipeline,
)
from experiments._common import (
    BUNDLED_DATA_DIR as bundled_data_dir,
    extract_communities_from_partitions,
    load_local_jpm_data,
    load_upstream_components,
    split_cardinality_constraint,
    subset_problem,
    to_jsonable,
)


def _relax_local_problem(adjusted_returns, sub_covariance, budget, risk_factor):
    """Continuous relaxation used for Egger-style warm-start initialization."""
    n = int(len(adjusted_returns))
    if budget <= 0:
        return np.zeros(n, dtype=float)
    if budget >= n:
        return np.ones(n, dtype=float)
    x0 = np.full(n, float(budget) / float(n), dtype=float)
    res = minimize(
        lambda x: float(risk_factor) * (x @ sub_covariance @ x) - adjusted_returns @ x,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - budget}],
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not res.success:
        return np.clip(x0, 0.0, 1.0)
    return np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)


def _build_warmstart_from_local_problem(
    adjusted_returns,
    sub_covariance,
    budget,
    risk_factor,
    weights,
):
    relaxed = _relax_local_problem(
        np.asarray(adjusted_returns, dtype=float),
        np.asarray(sub_covariance, dtype=float),
        int(budget),
        float(risk_factor),
    )
    return {weights[j].name: float(relaxed[j]) for j in range(len(weights))}


@dataclass
class QAOAConfig:
    p: int = 1
    restarts: int = 6
    maxiter: int = 80
    seed: int = 123
    optimizer: str = "COBYLA"
    shots: int = 4000
    decode: str = "shots"  # "shots" = sample N shots, return lowest-energy FEASIBLE sampled state (hardware-aligned); "argmax_prob" = legacy highest-probability feasible (reproduces earlier Table III)
    warmstart: bool = False  # Egger warm-start: init from continuous relaxation, mixer RY(θ)RZ(2β)RY(-θ)


class QiskitQAOAOptimizer:
    """
    Small-scale local QAOA optimizer for docplex models.
    Intended for feasibility experiments only.
    """

    def __init__(self, config: QAOAConfig, warmstart_c=None):
        self.config = config
        self.shots = int(config.shots)
        self.decode = str(config.decode)
        self.warmstart_c = warmstart_c  # dict {var_label: float in [0,1]} for Egger warm-start, or None
        self.last_theta = None       # optimized θ*, exposed for hardware-in-the-loop (shared-θ*) use
        self.last_probs = None
        self.last_energies = None
        self.params = {
            "name": "QiskitQAOAOptimizer",
            "p": int(config.p),
            "restarts": int(config.restarts),
            "maxiter": int(config.maxiter),
            "seed": int(config.seed),
            "optimizer": str(config.optimizer),
            "shots": self.shots,
            "decode": self.decode,
            "warmstart": bool(config.warmstart),
        }

    def __call__(self, model, weights=None, unique_identifier=None, log_best_feasible=False):
        del unique_identifier, log_best_feasible  # unused; kept for interface compatibility
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        cqm = self._docplex_model_to_cqm(model)
        bqm, _ = dimod.cqm_to_bqm(cqm)
        var_labels = list(bqm.variables)
        n = len(var_labels)
        if n == 0:
            return np.array([]), 0.0, 0.0

        if n > 16:
            raise ValueError(f"QAOA simulator backend supports local n<=16, got n={n}")

        warmstart_c = self.warmstart_c
        if warmstart_c is None and self.config.warmstart and weights is not None:
            adjusted_returns = getattr(model, "_caf_adjusted_returns", None)
            sub_covariance = getattr(model, "_caf_sub_covariance", None)
            budget = getattr(model, "_caf_budget", None)
            local_risk_factor = getattr(model, "_caf_risk_factor", None)
            if (
                adjusted_returns is not None
                and sub_covariance is not None
                and budget is not None
                and local_risk_factor is not None
            ):
                warmstart_c = _build_warmstart_from_local_problem(
                    adjusted_returns,
                    sub_covariance,
                    budget,
                    local_risk_factor,
                    weights,
                )

        h, J, offset = bqm.to_ising()
        _best_obj, best_probs, best_theta, energies = self._optimize_qaoa(
            h,
            J,
            offset,
            n,
            var_labels,
            warmstart_c=warmstart_c,
        )
        self.last_theta = best_theta
        self.last_probs = best_probs
        self.last_energies = energies

        sample = self._select_candidate(best_probs, energies, var_labels, cqm)
        if sample is None:
            # No feasible state was sampled (mirrors hardware behavior on infeasible outcomes).
            values_by_label = {label: 0 for label in var_labels}
        else:
            values_by_label = sample

        if weights is None:
            solution = np.array([values_by_label[label] for label in var_labels], dtype=float)
        else:
            solution = np.array([values_by_label.get(w.name, 0) for w in weights], dtype=float)

        cpu_end = time.process_time()
        wall_end = time.perf_counter()
        return solution, wall_end - wall_start, cpu_end - cpu_start

    @staticmethod
    def _docplex_model_to_cqm(model):
        # Keep identical behavior with the existing SA optimizer path in this project.
        import threading

        thread_id = threading.get_ident()
        tmp_path = os.path.abspath(f"temp_dimod_model_qaoa_{thread_id}_{uuid.uuid4().hex}.lp")
        with open(tmp_path, "w") as tmp:
            tmp.write(model.export_as_lp_string())
        try:
            return dimod.lp.load(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _optimize_qaoa(self, h, J, offset, n, var_labels, warmstart_c=None):
        dim = 1 << n
        states = np.arange(dim, dtype=np.uint64)
        bit_matrix = ((states[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
        # dimod.to_ising() uses s = 2x - 1, so x=1 maps to the +1 eigenstate.
        spins = 2 * bit_matrix - 1
        energies = np.full(dim, float(offset), dtype=float)
        label_to_pos = {label: pos for pos, label in enumerate(var_labels)}
        if h:
            for i, coeff in h.items():
                energies += float(coeff) * spins[:, label_to_pos[i]]
        if J:
            for (i, j), coeff in J.items():
                energies += float(coeff) * spins[:, label_to_pos[i]] * spins[:, label_to_pos[j]]

        rng = np.random.default_rng(self.config.seed)

        def state_from_params(theta):
            gammas = theta[: self.config.p]
            betas = theta[self.config.p :]
            ws = warmstart_c
            ws_theta = None
            if ws:
                ws_theta = np.array([
                    2.0 * np.arcsin(np.sqrt(min(1.0, max(0.0, float(ws.get(lbl, 0.5))))))
                    for lbl in var_labels
                ])
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.ry(float(ws_theta[q]), q) if ws else qc.h(q)
            for layer in range(self.config.p):
                gamma = float(gammas[layer])
                beta = float(betas[layer])
                for i, coeff in h.items():
                    qc.rz(2.0 * gamma * float(coeff), label_to_pos[i])
                for (i, j), coeff in J.items():
                    qc.rzz(2.0 * gamma * float(coeff), label_to_pos[i], label_to_pos[j])
                for q in range(n):
                    if ws:
                        # Egger warm-start mixer: rotation around the warm-start axis (keeps initial state, explores perpendicular)
                        qc.ry(-float(ws_theta[q]), q)
                        qc.rz(2.0 * beta, q)
                        qc.ry(float(ws_theta[q]), q)
                    else:
                        qc.rx(2.0 * beta, q)
            return Statevector.from_instruction(qc)

        def objective(theta):
            psi = state_from_params(theta)
            probs = psi.probabilities()
            return float(np.dot(probs, energies))

        best_obj = float("inf")
        best_theta = None
        n_params = 2 * self.config.p
        method = str(self.config.optimizer).strip().upper()
        if method == "COBYLA":
            minimize_options = {"maxiter": int(self.config.maxiter), "rhobeg": 0.5, "tol": 1e-3}
        elif method == "NELDER-MEAD":
            minimize_options = {"maxiter": int(self.config.maxiter), "xatol": 1e-3, "fatol": 1e-3}
        elif method == "POWELL":
            minimize_options = {"maxiter": int(self.config.maxiter), "xtol": 1e-3, "ftol": 1e-3}
        else:
            raise ValueError(f"Unsupported QAOA classical optimizer: {self.config.optimizer}")
        for _ in range(max(1, self.config.restarts)):
            x0 = np.zeros(n_params, dtype=float)
            x0[: self.config.p] = rng.uniform(0.0, np.pi, size=self.config.p)
            x0[self.config.p :] = rng.uniform(0.0, np.pi / 2.0, size=self.config.p)
            res = minimize(
                objective,
                x0=x0,
                method=method,
                options=minimize_options,
            )
            val = float(res.fun)
            if val < best_obj:
                best_obj = val
                best_theta = np.asarray(res.x, dtype=float)

        if best_theta is None:
            best_theta = np.zeros(n_params, dtype=float)
        psi = state_from_params(best_theta)
        probs = psi.probabilities()
        return best_obj, probs, best_theta, energies

    @staticmethod
    def _decode_state(idx, var_labels):
        return {label: int((idx >> bit_pos) & 1) for bit_pos, label in enumerate(var_labels)}

    def _select_candidate(self, probabilities, energies, var_labels, cqm):
        """Choose the returned candidate from the optimized QAOA distribution.

        decode="shots" (default, hardware-aligned): draw `self.shots` bitstrings
            from the exact distribution (mimicking hardware counts), then return
            the lowest-energy FEASIBLE sampled state. A good state must carry
            enough probability to be sampled -- this preserves the QAOA-quality
            signal and matches the real-hardware protocol (best feasible sample).
        decode="argmax_prob" (legacy): return the highest-probability feasible
            state regardless of its objective. Reproduces earlier Table III
            numbers; can systematically miss the optimum when the QAOA peak sits
            on a worse feasible state.
        """
        rng = np.random.default_rng(int(self.config.seed) + 7)
        if self.decode == "argmax_prob":
            for idx in np.argsort(-probabilities):
                sample = self._decode_state(int(idx), var_labels)
                if cqm.check_feasible(sample):
                    return sample
            return None
        # shot-based decoding (default)
        shots = max(1, int(self.shots))
        sampled = rng.choice(len(probabilities), size=shots, p=probabilities)
        best_energy, best_sample = None, None
        for idx in set(sampled.tolist()):
            sample = self._decode_state(int(idx), var_labels)
            if cqm.check_feasible(sample):
                e = float(energies[int(idx)])
                if best_energy is None or e < best_energy:
                    best_energy, best_sample = e, sample
        return best_sample  # None if no feasible state was sampled (mirrors hardware)

class SQAOptimizer:
    """
    Simulated Quantum Annealing optimizer based on Ocean's path-integral sampler.
    Intended as an annealing-style reference backend.
    """

    def __init__(self, num_reads=20, num_sweeps=1000, seed=None):
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.seed = None if seed is None else int(seed)
        self.params = {
            "name": "SQAOptimizer",
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
            "seed": self.seed,
        }

    def __call__(self, model, weights=None, unique_identifier=None, log_best_feasible=False):
        del unique_identifier, log_best_feasible
        from dwave.samplers import PathIntegralAnnealingSampler

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        import threading

        thread_id = threading.get_ident()
        tmp_path = os.path.abspath(f"temp_dimod_model_sqa_{thread_id}_{uuid.uuid4().hex}.lp")
        with open(tmp_path, "w") as tmp:
            tmp.write(model.export_as_lp_string())
        try:
            cqm = dimod.lp.load(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        bqm, _ = dimod.cqm_to_bqm(cqm)
        var_labels = list(bqm.variables)
        if len(var_labels) == 0:
            return np.array([]), 0.0, 0.0

        sampler = PathIntegralAnnealingSampler()
        sample_kwargs = {
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
        }
        if self.seed is not None:
            sample_kwargs["seed"] = self.seed
        sampleset = sampler.sample(bqm, **sample_kwargs)

        values_dict = sampleset.first.sample
        if cqm.check_feasible(sampleset.first.sample):
            if weights is None:
                solution = np.array([values_dict[label] for label in var_labels], dtype=float)
            else:
                solution = np.array([values_dict.get(w.name, 0) for w in weights], dtype=float)
        else:
            # Conservative fallback keeps interface aligned with other optimizers.
            if weights is None:
                solution = np.zeros(len(var_labels), dtype=float)
            else:
                solution = np.zeros(len(weights), dtype=float)

        cpu_end = time.process_time()
        wall_end = time.perf_counter()
        return solution, wall_end - wall_start, cpu_end - cpu_start


class SAOptimizer:
    """
    Local simulated annealing optimizer using neal/dimod.
    Intended as a stable classical reference backend when the upstream wrapper is unavailable.
    """

    def __init__(self, num_reads=20, num_sweeps=1000, seed=None):
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.seed = None if seed is None else int(seed)
        self.params = {
            "name": "SAOptimizer",
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
            "seed": self.seed,
        }

    def __call__(self, model, weights=None, unique_identifier=None, log_best_feasible=False):
        del unique_identifier, log_best_feasible
        import neal

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        import threading

        thread_id = threading.get_ident()
        tmp_path = os.path.abspath(f"temp_dimod_model_sa_{thread_id}_{uuid.uuid4().hex}.lp")
        with open(tmp_path, "w") as tmp:
            tmp.write(model.export_as_lp_string())
        try:
            cqm = dimod.lp.load(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        bqm, _ = dimod.cqm_to_bqm(cqm)
        var_labels = list(bqm.variables)
        if len(var_labels) == 0:
            return np.array([]), 0.0, 0.0

        sampler = neal.SimulatedAnnealingSampler()
        sample_kwargs = {
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
        }
        if self.seed is not None:
            sample_kwargs["seed"] = self.seed
        sampleset = sampler.sample(bqm, **sample_kwargs)

        feasible_sample = None
        for sample in sampleset.data(fields=["sample"]):
            values = dict(sample.sample)
            if cqm.check_feasible(values):
                feasible_sample = values
                break
        if feasible_sample is None:
            feasible_sample = {label: 0 for label in var_labels}

        if weights is None:
            solution = np.array([feasible_sample[label] for label in var_labels], dtype=float)
        else:
            solution = np.array([feasible_sample.get(w.name, 0) for w in weights], dtype=float)

        cpu_end = time.process_time()
        wall_end = time.perf_counter()
        return solution, wall_end - wall_start, cpu_end - cpu_start


def make_denoiser_factory(Denoiser):
    try:
        from dcmppln.utils.correlation_rmt import split_covariance_matrices
        from dcmppln.utils.utils import timeit
    except ImportError:
        return lambda: Denoiser(active=True, q=0.5, q_fit=True)

    try:
        params = inspect.signature(split_covariance_matrices).parameters
    except (TypeError, ValueError):
        params = {}

    if "q" in params:
        return lambda: Denoiser(active=True, q=0.5, q_fit=True)

    if "beta" in params:
        class CompatibilityDenoiser(Denoiser):
            @timeit
            def denoise(self, C):
                if not self.active:
                    return C
                _, filtered_covariance, _ = split_covariance_matrices(C, beta=self.q, q_fit=self.q_fit)
                return filtered_covariance

        return lambda: CompatibilityDenoiser(active=True, q=0.5, q_fit=True)

    return lambda: Denoiser(active=True, q=0.5, q_fit=True)


def aggregate_results(records):
    arr_obj_static = np.asarray([r["static"]["objective"] for r in records], dtype=float)
    arr_obj_caf = np.asarray([r["caf"]["objective"] for r in records], dtype=float)
    arr_risk_static = np.asarray([r["static"]["risk"] for r in records], dtype=float)
    arr_risk_caf = np.asarray([r["caf"]["risk"] for r in records], dtype=float)
    arr_ret_static = np.asarray([r["static"]["expected_return"] for r in records], dtype=float)
    arr_ret_caf = np.asarray([r["caf"]["expected_return"] for r in records], dtype=float)
    arr_time_static = np.asarray([r["static_time_sec"] for r in records], dtype=float)
    arr_time_caf = np.asarray([r["caf_time_sec"] for r in records], dtype=float)
    rel_impr = ((arr_obj_static - arr_obj_caf) / np.abs(arr_obj_static)) * 100.0
    abs_delta = arr_obj_caf - arr_obj_static
    wins = np.sum(arr_obj_caf < arr_obj_static - 1e-7)
    
    # Component-wise improvements (safe from near-zero distortion since risk and return are typically strictly positive and larger)
    risk_reduction_pct = ((arr_risk_static - arr_risk_caf) / np.abs(arr_risk_static)) * 100.0
    return_increase_pct = ((arr_ret_caf - arr_ret_static) / np.abs(arr_ret_static)) * 100.0

    out = {
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
        "objective_improvement_pct_mean": float(rel_impr.mean()),
        "objective_improvement_pct_std": float(rel_impr.std()),
        "absolute_delta_mean": float(abs_delta.mean()),
        "absolute_delta_std": float(abs_delta.std()),
        "risk_reduction_pct_mean": float(risk_reduction_pct.mean()),
        "risk_reduction_pct_std": float(risk_reduction_pct.std()),
        "return_increase_pct_mean": float(return_increase_pct.mean()),
        "return_increase_pct_std": float(return_increase_pct.std()),
        "wins": int(wins),
        "total_runs": int(len(records)),
    }
    diagnostic_rows = []
    for r in records:
        for row in r.get("proposal_diagnostics", []):
            diagnostic_rows.append(row)
    if diagnostic_rows:
        num_proposals = len(diagnostic_rows)
        num_feasible = int(sum(1 for d in diagnostic_rows if d.get("candidate_feasible", False)))
        num_local_improved = int(sum(1 for d in diagnostic_rows if d.get("local_improved", False)))
        num_global_accepted = int(sum(1 for d in diagnostic_rows if d.get("accepted_globally", False)))
        num_local_improved_rejected = int(
            sum(1 for d in diagnostic_rows if d.get("local_improved", False) and not d.get("accepted_globally", False))
        )
        out["diagnostics"] = {
            "num_proposals": int(num_proposals),
            "num_feasible": int(num_feasible),
            "num_local_improved": int(num_local_improved),
            "num_global_accepted": int(num_global_accepted),
            "num_local_improved_but_rejected": int(num_local_improved_rejected),
            "local_improvement_rate_over_feasible": float(num_local_improved / max(1, num_feasible)),
            "global_accept_rate_over_local_improved": float(num_global_accepted / max(1, num_local_improved)),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Small-scale QAOA backend feasibility experiment for CAF.")
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_qaoa"), help="Output folder.")
    parser.add_argument("--summary-file", type=Path, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--runs", type=int, default=10, help="Number of stochastic runs.")
    parser.add_argument("--n-assets", type=int, default=24, help="Subinstance size.")
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
    parser.add_argument("--qaoa-p", type=int, default=1, help="QAOA depth p.")
    parser.add_argument(
        "--static-qaoa-p",
        type=int,
        default=None,
        help="Optional static-baseline QAOA depth p. If omitted, uses --qaoa-p.",
    )
    parser.add_argument("--qaoa-restarts", type=int, default=6, help="Random restarts for QAOA parameter search.")
    parser.add_argument("--qaoa-maxiter", type=int, default=80, help="Max COBYLA iterations per restart.")
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
    parser.add_argument(
        "--include-sa-reference",
        action="store_true",
        help="Also run SA baseline pair (Static+SA vs CAF+SA) as reference.",
    )
    parser.add_argument(
        "--include-sqa-reference",
        action="store_true",
        help="Also run SQA baseline pair (Static+SQA vs CAF+SQA) as reference.",
    )
    return parser.parse_args()


def run_one_setting(
    cov,
    corr,
    ret,
    cardinality,
    risk_factor,
    runs,
    base_seed,
    iterations,
    patience,
    optimizer_factory_static,
    optimizer_factory_caf,
    Clustering,
    denoiser_factory,
    Pipeline,
    get_model,
    rebalancing_risk_factor,
    max_workers=None,
):
    # Fix upstream decomposition to isolate backend variance, then solve both the
    # static baseline and CAF proposals against that same decomposition.
    _ = rebalancing_risk_factor  # kept for API compatibility with existing callers
    np.random.seed(int(base_seed))
    random.seed(int(base_seed))
    t0 = time.perf_counter()
    template_pipeline = Pipeline(
        corr.values,
        cov.values,
        ret.values,
        denoiser=denoiser_factory(),
        cluster=Clustering(active=True, clustering_method="louvain", take_absolute_value=True),
        optimize_func=optimizer_factory_static(int(base_seed)),
    )
    _matrix_to_cluster, partitions = template_pipeline.run(
        run_optimizer=False,
        risk_rebalancing=True,
        cluster_on_correlation=True,
        optimize_on_correlation=False,
        input_risk_factor=risk_factor,
    )
    decomposition_time = time.perf_counter() - t0
    communities = extract_communities_from_partitions(partitions)
    subproblem_cardinalities = split_cardinality_constraint([len(c) for c in communities], cardinality)
    # The folded quantum study is meant to test whether backend-generated local
    # candidates are consistent with the same global objective used by CAF's
    # acceptance rule. For this supplementary path we therefore keep the local
    # proposal objective on the original q instead of the rebalanced surrogate q.
    proposal_risk_factor = float(risk_factor)

    records = []
    for run_idx in range(runs):
        run_seed = int(base_seed + run_idx + 1000)
        np.random.seed(run_seed)
        random.seed(run_seed)

        t_static = time.perf_counter()
        static_optimizer = optimizer_factory_static(run_seed)
        static_state = np.zeros(len(ret), dtype=int)
        for community, community_cardinality in zip(communities, subproblem_cardinalities):
            local_returns = ret.values[community]
            sub_covariance = cov.values[np.ix_(community, community)]
            model, weights = get_model(
                local_returns,
                sub_covariance,
                budget=int(community_cardinality),
                risk_factor=proposal_risk_factor,
            )
            attach_local_problem_metadata(
                model,
                local_returns,
                sub_covariance,
                int(community_cardinality),
                proposal_risk_factor,
            )
            local_solution, _, _ = static_optimizer(model, weights)
            local_solution = normalize_local_solution(
                local_solution,
                local_state_len=len(community),
                community_cardinality=int(community_cardinality),
            )
            static_state[community] = local_solution
        static_time = decomposition_time + (time.perf_counter() - t_static)
        static_selected = np.where(static_state.astype(np.int64) == 1)[0].tolist()
        static_metrics = evaluate_portfolio(static_selected, cov, ret, risk_factor)

        t1 = time.perf_counter()
        caf_res = run_caf_pipeline(
            cov,
            ret,
            communities,
            subproblem_cardinalities,
            get_model_func=get_model,
            optimizer_factory=lambda num_reads: optimizer_factory_caf(run_seed, num_reads=num_reads),
            risk_factor=risk_factor,
            effective_risk_factor=proposal_risk_factor,
            initial_selected_indices=static_selected,
            iterations=iterations,
            patience=patience,
            use_incremental=True,
            return_diagnostics=True,
            max_workers=max_workers,
        )
        caf_time = time.perf_counter() - t1

        records.append(
            {
                "run_idx": int(run_idx),
                "seed": int(run_seed),
                "community_sizes": [int(len(c)) for c in communities],
                "subproblem_cardinalities": [int(k) for k in subproblem_cardinalities],
                "static": static_metrics,
                "caf": caf_res["metrics"],
                "static_time_sec": float(static_time),
                "caf_time_sec": float(caf_time),
                "proposal_diagnostics": caf_res.get("proposal_diagnostics", []),
            }
        )
    return records


def run_reference_backend(
    optimizer_cls,
    seed_offset,
    cov,
    corr,
    ret,
    cardinality,
    risk_factor,
    runs,
    base_seed,
    iterations,
    patience,
    Clustering,
    denoiser_factory,
    Pipeline,
    get_model,
    rebalancing_risk_factor,
    max_workers=None,
    default_num_reads=20,
    num_sweeps=1000,
):
    def optimizer_factory(seed, num_reads=None):
        reads = int(default_num_reads if num_reads is None else num_reads)
        return optimizer_cls(num_reads=reads, num_sweeps=int(num_sweeps), seed=int(seed))

    records = run_one_setting(
        cov=cov,
        corr=corr,
        ret=ret,
        cardinality=cardinality,
        risk_factor=risk_factor,
        runs=runs,
        base_seed=int(base_seed) + int(seed_offset),
        iterations=iterations,
        patience=patience,
        optimizer_factory_static=lambda seed: optimizer_factory(seed),
        optimizer_factory_caf=lambda seed, num_reads: optimizer_factory(seed, num_reads=num_reads),
        Clustering=Clustering,
        denoiser_factory=denoiser_factory,
        Pipeline=Pipeline,
        get_model=get_model,
        rebalancing_risk_factor=rebalancing_risk_factor,
        max_workers=max_workers,
    )
    summary = aggregate_results(records)
    return summary, records


def main():
    args = parse_args()
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

    qaoa_cfg = QAOAConfig(
        p=int(args.qaoa_p),
        restarts=int(args.qaoa_restarts),
        maxiter=int(args.qaoa_maxiter),
        seed=int(args.base_seed),
        optimizer=str(args.qaoa_optimizer),
        warmstart=bool(args.warmstart),
    )
    static_qaoa_p = int(args.static_qaoa_p) if args.static_qaoa_p is not None else int(qaoa_cfg.p)

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

    def qaoa_factory_static(seed):
        cfg = QAOAConfig(
            p=static_qaoa_p,
            restarts=qaoa_cfg.restarts,
            maxiter=qaoa_cfg.maxiter,
            seed=int(seed),
            optimizer=qaoa_cfg.optimizer,
            warmstart=qaoa_cfg.warmstart,
        )
        return QiskitQAOAOptimizer(cfg)

    print("=== Running QAOA feasibility experiment ===")
    print(
        f"N={len(ret)}, K={cardinality}, runs={args.runs}, q={args.risk_factor}, "
        f"p={qaoa_cfg.p}, warmstart={qaoa_cfg.warmstart}"
    )
    qaoa_records = run_one_setting(
        cov=cov,
        corr=corr,
        ret=ret,
        cardinality=cardinality,
        risk_factor=args.risk_factor,
        runs=args.runs,
        base_seed=args.base_seed,
        iterations=args.iterations,
        patience=args.patience,
        optimizer_factory_static=lambda seed: qaoa_factory_static(seed),
        optimizer_factory_caf=lambda seed, num_reads: qaoa_factory(seed, num_reads=num_reads),
        Clustering=Clustering,
        denoiser_factory=denoiser_factory,
        Pipeline=Pipeline,
        get_model=get_model,
        rebalancing_risk_factor=rebalancing_risk_factor,
        max_workers=args.max_workers,
    )
    qaoa_summary = aggregate_results(qaoa_records)
    print("QAOA summary (CAF vs Static):")
    print(f"  wins/runs: {qaoa_summary['wins']}/{qaoa_summary['total_runs']}")
    print(f"  absolute delta: {qaoa_summary['absolute_delta_mean']:.6f} +/- {qaoa_summary['absolute_delta_std']:.6f}")
    print(f"  Risk reduction: {qaoa_summary['risk_reduction_pct_mean']:.3f}% +/- {qaoa_summary['risk_reduction_pct_std']:.3f}%")
    print(f"  Return increase: {qaoa_summary['return_increase_pct_mean']:.3f}% +/- {qaoa_summary['return_increase_pct_std']:.3f}%")

    sa_summary = None
    sa_records = None
    if args.include_sa_reference:
        print("\n=== Running SA reference experiment ===")
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
        print("SA summary (CAF vs Static):")
        print(f"  wins/runs: {sa_summary['wins']}/{sa_summary['total_runs']}")
        print(f"  absolute delta: {sa_summary['absolute_delta_mean']:.6f} +/- {sa_summary['absolute_delta_std']:.6f}")
        print(f"  Risk reduction: {sa_summary['risk_reduction_pct_mean']:.3f}% +/- {sa_summary['risk_reduction_pct_std']:.3f}%")
        print(f"  Return increase: {sa_summary['return_increase_pct_mean']:.3f}% +/- {sa_summary['return_increase_pct_std']:.3f}%")

    sqa_summary = None
    sqa_records = None
    if args.include_sqa_reference:
        print("\n=== Running SQA reference experiment ===")
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
        print("SQA summary (CAF vs Static):")
        print(f"  wins/runs: {sqa_summary['wins']}/{sqa_summary['total_runs']}")
        print(f"  absolute delta: {sqa_summary['absolute_delta_mean']:.6f} +/- {sqa_summary['absolute_delta_std']:.6f}")
        print(f"  Risk reduction: {sqa_summary['risk_reduction_pct_mean']:.3f}% +/- {sqa_summary['risk_reduction_pct_std']:.3f}%")
        print(f"  Return increase: {sqa_summary['return_increase_pct_mean']:.3f}% +/- {sqa_summary['return_increase_pct_std']:.3f}%")

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
            "qaoa": {
                "p": int(qaoa_cfg.p),
                "restarts": int(qaoa_cfg.restarts),
                "maxiter": int(qaoa_cfg.maxiter),
                "optimizer": str(qaoa_cfg.optimizer),
                "warmstart": bool(qaoa_cfg.warmstart),
            },
            "include_sa_reference": bool(args.include_sa_reference),
            "include_sqa_reference": bool(args.include_sqa_reference),
        },
        "qaoa": {"summary": qaoa_summary, "records": qaoa_records},
        "sa_reference": None if sa_summary is None else {"summary": sa_summary, "records": sa_records},
        "sqa_reference": None if sqa_summary is None else {"summary": sqa_summary, "records": sqa_records},
    }
    summary_path = args.summary_file if args.summary_file is not None else (out_dir / "qaoa_feasibility_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
