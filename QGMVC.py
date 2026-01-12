from __future__ import annotations
from typing import Union
import random
import math

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Optional rustworkx support
try:
    import rustworkx as rx
    RxGraph = rx.PyGraph
except ImportError:
    rx = None
    RxGraph = tuple()

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RXGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import Aer
from qiskit import transpile

# CPLEX import
from docplex.mp.model import Model
from quantum_greedy_util import mixer_from_graph, expectation_value_cost_shifted, greedy_optimize, greedy_optimize_seq,greedy_optimize_seq_rev,node_order_by_cost_degree,mean_field_cost_degree_order_init
#from classical_heuristics_util import is_vertex_cover, local_search_vertex_cover, ga_vertex_cover
from classical_runtime_guarantee_util import mvc_exact_cplex, mvc_lp_relaxation, greedy_degree_vertex_cover, greedy_edge_vertex_cover,mvc_primal_dual_weighted



# ---------------- Experiment & plotting ----------------
def run_experiment_stats_weighted(
    n_values,
    N_graphs=10,
    n_stat=1,
    shots=None,
    case="weighted",
    graph_type="regular",
    degree=3,
    seed=0
):
    methods = [
        "LP",
        "Dual-Primal",
        "Greedy vertex degree",
        "Greedy random edge",
        #"Quantum greedy bias",
        #"Quantum greedy seq bias",
        #"Quantum greedy seq rev bias",
        "Quantum greedy unbias",
        #"Quantum greedy seq unbias",
        #"Quantum greedy seq rev unbias",

    ]

    results = {n: {m: [] for m in methods} for n in n_values}
    rng = np.random.default_rng(seed)

    for n in n_values:

        print(f"\n===== n = {n} =====")
        for g in range(N_graphs):
            print(g)
            graph_seed = rng.integers(1e9)

            if graph_type == "regular":
                G = nx.random_regular_graph(
                    degree if n % 2 == 0 else degree + 1,
                    n,
                    seed=int(graph_seed)
                )
            elif graph_type == "erdos":
                while True:
                    G = nx.erdos_renyi_graph(n=n, p=degree)
                    if nx.is_connected(G):
                        break

            else:
                raise ValueError("Unknown graph type")
            
            if case == "unweighted":
                c = {i: 1.0 for i in G.nodes()}
            else:
                c = {i: random.uniform(0.25, 0.75) for i in G.nodes()}

            C_opt = mvc_exact_cplex(G, c)
            opt_cost = sum(c[i] for i in C_opt)

            C_lp = mvc_lp_relaxation(G, c)
            results[n]["LP"].append(sum(c[i] for i in C_lp) / opt_cost)

            C_pd = mvc_primal_dual_weighted(G,  c)
            results[n]["Dual-Primal"].append(sum(c[i] for i in C_pd) / opt_cost)

            C_gd = greedy_degree_vertex_cover(G,c)
            results[n]["Greedy vertex degree"].append(sum(c[i] for i in C_gd) / opt_cost)
            qc, betas, Gn = mixer_from_graph(G,c)
            C_cost = {i: c[i] for i in Gn.nodes()}                                          
            for s in range(n_stat):
                run_seed = rng.integers(1e9)
                random.seed(int(run_seed))
                np.random.seed(int(run_seed))

                C_ge = greedy_edge_vertex_cover(G,c)
                results[n]["Greedy random edge"].append(sum(c[i] for i in C_ge) / opt_cost)
            order = node_order_by_cost_degree(G, C_cost)
            beta_bias1 = mean_field_cost_degree_order_init(G, order, C_cost, alpha=1,beta=1.0, gamma=1.0, delta=0.7, n_iter=50)
            beta_bias2={i: 0.8*np.pi/2 for i in C_cost}

            #sol_weighted = greedy_optimize(qc, betas, C_cost, beta_bias2, shots=shots)
            #E_weighted = expectation_value_cost_shifted(qc, betas, C_cost, sol_weighted, shots=shots)
            #results[n]["Quantum greedy bias"].append(E_weighted / opt_cost)
            #sol_weighted = greedy_optimize_seq(qc, betas, C_cost, beta_bias2, shots=shots)
            #E_weighted = expectation_value_cost_shifted(qc, betas, C_cost, sol_weighted, shots=shots)
            #results[n]["Quantum greedy seq bias"].append(E_weighted / opt_cost)
            #sol_weighted = greedy_optimize_seq_rev(qc, betas, C_cost, beta_bias2, shots=shots)
            #E_weighted = expectation_value_cost_shifted(qc, betas, C_cost, sol_weighted, shots=shots)
            #results[n]["Quantum greedy seq rev bias"].append(E_weighted / opt_cost)
            sol_weighted = greedy_optimize_seq(qc, betas, C_cost, beta_bias1, shots=shots)
            E_weighted = expectation_value_cost_shifted(qc, betas, C_cost, sol_weighted, shots=shots)
            results[n]["Quantum greedy unbias"].append(E_weighted / opt_cost)
            #sol_weighted = greedy_optimize_seq(qc, betas, C_cost, beta_bias1, shots=shots)
            #E_weighted = expectation_value_cost_shifted(qc, betas, C_cost, sol_weighted, shots=shots)
            #results[n]["Quantum greedy seq unbias"].append(E_weighted / opt_cost)
            #sol_weighted = greedy_optimize_seq_rev(qc, betas, C_cost, beta_bias1, shots=shots)
            #E_weighted = expectation_value_cost_shifted(qc, betas, C_cost, sol_weighted, shots=shots)
            #results[n]["Quantum greedy seq rev unbias"].append(E_weighted / opt_cost)


        print(f"n={n} done")

    return results

def summarize_results(results, n_values):
    means = {}
    stds = {}

    for method in next(iter(results.values())).keys():
        means[method] = []
        stds[method] = []
        for n in n_values:
            vals = np.array(results[n][method])
            means[method].append(vals.mean())
            stds[method].append(vals.std())

    return means, stds

from datetime import datetime

def plot_with_error_bars(n_values, means, stds, prefix="vertex_cover_performance"):
    plt.figure(figsize=(9, 6))
    for method in means:
        plt.errorbar(
            n_values,
            means[method],
            yerr=stds[method],
            marker="o",
            capsize=4,
            label=method
        )
    plt.xlabel("Graph size (n)")
    plt.ylabel(r"Performance $|C| / |C^*|$")
    plt.title("Vertex Cover Performance vs Graph Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Figure saved as: {filename}")

# ---------------- Main ----------------
def main():
    n_values = [8,10,12,14,16,18,20]

    results = run_experiment_stats_weighted(
        n_values,
        N_graphs=20,
        n_stat=1,
        case="weighted",
        graph_type="regular",
        degree=4,
        shots=10000,
        seed=None
    )

    means, stds = summarize_results(results, n_values)
    plot_with_error_bars(n_values, means, stds)

if __name__ == "__main__":
    main()
