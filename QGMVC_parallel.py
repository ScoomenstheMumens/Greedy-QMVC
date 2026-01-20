from __future__ import annotations

# =======================
# Environment safeguards
# =======================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =======================
# Standard imports
# =======================
import sys
import random
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
from datetime import datetime

# =======================
# Optional rustworkx
# =======================
try:
    import rustworkx as rx
    RxGraph = rx.PyGraph
except ImportError:
    rx = None
    RxGraph = tuple()

# =======================
# CPLEX + project imports
# =======================
from docplex.mp.model import Model

from quantum_greedy_util import (
    mixer_from_graph,
    expectation_value_cost_shifted,
    greedy_optimize,
    greedy_optimize_seq
)

from classical_runtime_guarantee_util import (
    mvc_exact_cplex,
    mvc_lp_relaxation,
    greedy_degree_vertex_cover,
    mvc_primal_dual_weighted,
)

# ============================================================
# STEP A — one independent (n, graph) experiment
# ============================================================
def run_single_graph(args):
    (n, graph_seed, case, graph_type, degree, shots, n_stat) = args

    rng = np.random.default_rng(graph_seed)

    # ----- graph generation -----
    if graph_type == "regular":
        G = nx.random_regular_graph(
            degree if n % 2 == 0 else degree + 1,
            n,
            seed=int(graph_seed),
        )
    elif graph_type == "erdos":
        while True:
            G = nx.erdos_renyi_graph(n=n, p=degree, seed=int(graph_seed))
            if nx.is_connected(G):
                break
    else:
        raise ValueError("Unknown graph type")

    # ----- weights -----
    if case == "unweighted":
        c = {i: 1.0 for i in G.nodes()}
    else:
        c = {i: random.uniform(0.40, 0.70) for i in G.nodes()}

    out = {}

    # ----- exact MVC -----
    C_opt = mvc_exact_cplex(G, c)
    opt_cost = sum(c[i] for i in C_opt)

    out["optimal"] = opt_cost
    out["worst_case"] = sum(c.values()) / opt_cost
    out["LP"] = sum(c[i] for i in mvc_lp_relaxation(G, c)) / opt_cost
    out["Dual-Primal"] = sum(c[i] for i in mvc_primal_dual_weighted(G, c)) / opt_cost
    out["Greedy vertex degree"] = (
        sum(c[i] for i in greedy_degree_vertex_cover(G, c)) / opt_cost
    )

    # ----- quantum greedy -----
    qc, betas, Gn = mixer_from_graph(G, c)
    C_cost = {i: c[i] for i in Gn.nodes()}

    beta_bias={i: np.pi/4+(i/len(C_cost))*0.1 for i in C_cost} #depth dependent 
    beta_unbias = {i: 0.5 * np.pi / 2 for i in C_cost}

    E_bias, E_unbias = [], []

    for _ in range(n_stat):
        sol = greedy_optimize(qc, betas, C_cost, beta_bias, shots=shots)
        E_bias.append(
            expectation_value_cost_shifted(qc, betas, C_cost, sol, shots=shots)
        )

        sol = greedy_optimize(qc, betas, C_cost, beta_unbias, shots=shots)
        E_unbias.append(
            expectation_value_cost_shifted(qc, betas, C_cost, sol, shots=shots)
        )

    out["Quantum greedy bias"] = np.mean(E_bias) / opt_cost
    out["Quantum greedy unbias"] = np.mean(E_unbias) / opt_cost

    sol_bias = greedy_optimize_seq(qc, betas, C_cost, beta_bias, shots=shots)
    E = expectation_value_cost_shifted(qc, betas, C_cost, sol_bias, shots=shots)
    out["Quantum greedy seq bias"] = E / opt_cost

    sol_unbias = greedy_optimize_seq(qc, betas, C_cost, beta_unbias, shots=shots)
    E = expectation_value_cost_shifted(qc, betas, C_cost, sol_unbias, shots=shots)
    out["Quantum greedy seq unbias"] = E / opt_cost

    return n, out


# ============================================================
# STEP B — parallel driver over (n, graph_seed)
# ============================================================
def run_experiment_stats_weighted(
    n_values,
    N_graphs=10,
    n_stat=1,
    shots=None,
    case="weighted",
    graph_type="regular",
    degree=3,
    seed=None,
):
    methods = [
        "optimal",
        "worst_case",
        "LP",
        "Dual-Primal",
        "Greedy vertex degree",
        "Quantum greedy bias",
        "Quantum greedy unbias",
        "Quantum greedy seq bias",
        "Quantum greedy seq unbias",
    ]

    results = {n: {m: [] for m in methods} for n in n_values}

    rng = np.random.default_rng(seed)

    # ----- build flat task list -----
    tasks = []
    for n in n_values:
        for _ in range(N_graphs):
            graph_seed = rng.integers(1e9)
            tasks.append(
                (n, graph_seed, case, graph_type, degree, shots, n_stat)
            )

    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    print(f"Using {nproc} worker processes")

    with Pool(processes=nproc) as pool:
        for n, out in pool.imap_unordered(run_single_graph, tasks):
            for k, v in out.items():
                results[n][k].append(v)

    return results


# ============================================================
# Utilities: save + summarize
# ============================================================
def save_results(
    results,
    n_values,
    N_graphs,
    n_stat,
    case,
    graph_type,
    degree,
    shots,
    seed,
    prefix="mvc_results",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fname = (
        f"{prefix}_{case}_{graph_type}_deg{degree}_"
        f"nG{N_graphs}_nStat{n_stat}_shots{shots}_seed{seed}_{timestamp}.npz"
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, fname)

    np.savez_compressed(
        out_path,
        results=results,
        n_values=np.array(n_values),
        N_graphs=N_graphs,
        n_stat=n_stat,
        case=case,
        graph_type=graph_type,
        degree=degree,
        shots=shots,
        seed=seed,
    )

    print(f"Results saved to:\n{out_path}")


# ============================================================
# Main
# ============================================================
def main(n_values, N_graphs, n_stat, case, graph_type, degree, shots, seed):

    results = run_experiment_stats_weighted(
        n_values=n_values,
        N_graphs=N_graphs,
        n_stat=n_stat,
        case=case,
        graph_type=graph_type,
        degree=degree,
        shots=shots,
        seed=seed,
    )

    save_results(
        results,
        n_values,
        N_graphs,
        n_stat,
        case,
        graph_type,
        degree,
        shots,
        seed,
    )


if __name__ == "__main__":

    # Usage:
    # python QGMVC.py N_graphs n_stat case graph_type degree shots

    N_graphs = int(sys.argv[1])
    n_stat = int(sys.argv[2])
    case = sys.argv[3]
    graph_type = sys.argv[4]
    degree = float(sys.argv[5])
    shots = int(sys.argv[6])

    if graph_type == "regular":
        degree = int(degree)
    elif graph_type == "erdos":
        if not (0.0 <= degree <= 1.0):
            raise ValueError("Erdos p must be in [0,1]")
    else:
        raise ValueError("graph_type must be 'regular' or 'erdos'")

    n_values = [6, 8, 10, 12, 14, 16, 18, 20]
    seed = 0

    print("N_graphs =", N_graphs)
    print("n_stat =", n_stat)
    print("case =", case)
    print("graph_type =", graph_type)
    print("degree =", degree)
    print("shots =", shots)

    main(
        n_values=n_values,
        N_graphs=N_graphs,
        n_stat=n_stat,
        case=case,
        graph_type=graph_type,
        degree=degree,
        shots=shots,
        seed=seed,
    )
