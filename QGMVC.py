from __future__ import annotations
import random
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
import sys
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
        "worst_case",
        "optimal",
        "LP",
        "Dual-Primal",
        "Greedy vertex degree",
        "Greedy random edge",
        "Quantum greedy bias",
        "Quantum greedy seq bias",
        "Quantum greedy seq rev bias",
        "Quantum greedy unbias",
        "Quantum greedy seq unbias",
        "Quantum greedy seq rev unbias",
        "Quantum greedy mean field",
        "Quantum greedy seq mean field",
        "Quantum greedy seq rev mean field",
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
                c = {i: random.uniform(0.40, 0.70) for i in G.nodes()}

            C_opt = mvc_exact_cplex(G, c)
            opt_cost = sum(c[i] for i in C_opt)
            results[n]["optimal"].append(opt_cost)  # by definition

            results[n]["worst_case"].append(sum(c[i] for i in G.nodes())/opt_cost)

            C_lp = mvc_lp_relaxation(G, c)
            results[n]["LP"].append((sum(c[i] for i in C_lp) )/ opt_cost)

            C_pd = mvc_primal_dual_weighted(G,  c)
            results[n]["Dual-Primal"].append(sum(c[i] for i in C_pd) / opt_cost)

            C_gd = greedy_degree_vertex_cover(G,c)
            results[n]["Greedy vertex degree"].append(sum(c[i] for i in C_gd) / opt_cost)
            qc, betas, Gn = mixer_from_graph(G,c)
            C_cost = {i: c[i] for i in Gn.nodes()}   
            order = node_order_by_cost_degree(G, C_cost)
            beta_mean = mean_field_cost_degree_order_init(G, order, C_cost, alpha=1,beta=1.0, gamma=1.0, delta=0.7, n_iter=50)
            beta_bias={i: 0.8*np.pi/2 for i in C_cost}
            beta_unbias={i: 0.5*np.pi/2 for i in C_cost}

            C_ge=[]     
            E_unbias=[]   
            E_bias=[] 
            E_mean=[]                              
            for _ in range(n_stat):

                c_ge = greedy_edge_vertex_cover(G,c)
                C_ge.append(sum(c[i] for i in c_ge))
                
                sol_unbias = greedy_optimize(qc, betas, C_cost, beta_unbias, shots=shots)
                E= expectation_value_cost_shifted(qc, betas, C_cost, sol_unbias, shots=shots)
                E_unbias.append(E)

                sol_bias = greedy_optimize(qc, betas, C_cost, beta_bias, shots=shots)
                E= expectation_value_cost_shifted(qc, betas, C_cost, sol_bias, shots=shots)
                E_bias.append(E)
                
                sol_mean = greedy_optimize(qc, betas, C_cost, beta_mean, shots=shots)
                E= expectation_value_cost_shifted(qc, betas, C_cost, sol_mean, shots=shots)
                E_mean.append(E)

            results[n]["Greedy random edge"].append(np.mean(np.array(C_ge))/opt_cost)
            results[n]["Quantum greedy bias"].append(np.mean(np.array(E_bias))/ opt_cost)
            results[n]["Quantum greedy unbias"].append(np.mean(np.array(E_unbias))/ opt_cost)
            results[n]["Quantum greedy mean field"].append(np.mean(np.array(E_mean))/ opt_cost)

            
            

            sol_bias = greedy_optimize_seq(qc, betas, C_cost, beta_bias, shots=shots)
            E = expectation_value_cost_shifted(qc, betas, C_cost, sol_bias, shots=shots)
            results[n]["Quantum greedy seq bias"].append(E / opt_cost)

            sol_unbias = greedy_optimize_seq(qc, betas, C_cost, beta_unbias, shots=shots)
            E = expectation_value_cost_shifted(qc, betas, C_cost, sol_unbias, shots=shots)
            results[n]["Quantum greedy seq unbias"].append(E / opt_cost)

            sol_mean = greedy_optimize_seq(qc, betas, C_cost, beta_mean, shots=shots)
            E = expectation_value_cost_shifted(qc, betas, C_cost, sol_mean, shots=shots)
            results[n]["Quantum greedy seq mean field"].append(E / opt_cost)


            sol_bias = greedy_optimize_seq_rev(qc, betas, C_cost, beta_bias, shots=shots)
            E = expectation_value_cost_shifted(qc, betas, C_cost, sol_bias, shots=shots)
            results[n]["Quantum greedy seq rev bias"].append(E / opt_cost)

            sol_unbias = greedy_optimize_seq_rev(qc, betas, C_cost, beta_unbias, shots=shots)
            E = expectation_value_cost_shifted(qc, betas, C_cost, sol_unbias, shots=shots)
            results[n]["Quantum greedy seq rev unbias"].append(E / opt_cost)

            sol_mean = greedy_optimize_seq_rev(qc, betas, C_cost, beta_mean, shots=shots)
            E = expectation_value_cost_shifted(qc, betas, C_cost, sol_mean, shots=shots)
            results[n]["Quantum greedy seq rev mean field"].append(E / opt_cost)



        print(f"n={n} done")

    return results
import os
import numpy as np
from datetime import datetime

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
    prefix="mvc_results"
):
    """
    Save experiment results with a filename encoding all key parameters.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fname = (
        f"{prefix}_"
        f"{case}_"
        f"{graph_type}_deg{degree}_"
        f"nG{N_graphs}_"
        f"nStat{n_stat}_"
        f"shots{shots}_"
        f"seed{seed}_"
        f"{timestamp}.npz"
    )

    # Save in same directory as script
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

    # Filter methods (exclude optimal)
    methods = [m for m in means if m.lower() != "optimal"]

    # Choose a colormap with enough distinct colors
    cmap = plt.get_cmap("tab20")
    colors = cmap.colors

    for i, method in enumerate(methods):
        plt.errorbar(
            n_values,
            means[method],
            yerr=stds[method],
            marker="o",
            capsize=4,
            color=colors[i % len(colors)],
            label=method
        )

    plt.xlabel("Graph size (n)")
    plt.ylabel(r"Performance $|C| / |C^*|$")
    plt.title("Vertex Cover Performance vs Graph Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()

    print(f"Figure saved as: {filename}")

# ---------------- Main ----------------
def main(n_values, N_graphs, n_stat, case, graph_type, degree, shots, seed=None):

    results = run_experiment_stats_weighted(
        n_values,
        N_graphs=N_graphs,
        n_stat=n_stat,
        case=case,
        graph_type=graph_type,
        degree=degree,
        shots=shots,
        seed=None
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
    seed
    )
    means, stds = summarize_results(results, n_values)
    plot_with_error_bars(n_values, means, stds)

if __name__ == "__main__":
    # Expected usage:
    # python run_experiment.py N_graphs n_stat case graph_type degree shots
    N_graphs = int(sys.argv[1])
    n_stat = int(sys.argv[2])
    case = str(sys.argv[3])              # "weighted" or "unweighted"
    graph_type = str(sys.argv[4])        # "regular" or "erdos"
    degree = float(sys.argv[5])          # degree or p
    shots = int(sys.argv[6])
    # Conditional typing for degree
    if graph_type == "regular":
        degree = int(degree)
    elif graph_type == "erdos":
        degree = float(degree)
        if not (0.0 <= degree <= 1.0):
            raise ValueError("For erdos graphs, degree must be a probability in [0,1]")
    else:
        raise ValueError("graph_type must be 'regular' or 'erdos'")
    # Fixed values (edit if needed)
    n_values = [6,8,10,12,14,16,18,20]
    seed = None

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
        seed=seed
    )

