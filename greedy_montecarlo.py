from __future__ import annotations
import random
import math
import networkx as nx
import numpy as np

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
from qiskit.circuit.library import RXGate,XGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import Aer
from qiskit import transpile

from collections import defaultdict

from util_greedy import mixer_from_graph,mixer_from_graph_subset,node_order_by_cost_degree, expectation_value_cost_shifted,echo_commutator_circuit,echo_fidelity


def particle_probabilities(particles):
    """Estimate marginal probabilities from particles."""
    nodes = particles[0].keys()
    probs = {}

    for v in nodes:
        probs[v] = np.mean([p[v] for p in particles])

    return probs



def conditional_smc_mvc(G, beta_values, C, k=1, node_order=None, num_particles=100):
    """
    Sequential Monte Carlo along quantum tree, returns expectation value directly
    """
    if node_order is None:
        node_order = list(G.nodes())

    # Initialize particles: all vertices in cover
    particles = [{v: 1 for v in G.nodes()} for _ in range(num_particles)]

    for _ in range(k):
        for v in node_order:
            theta_v = beta_values[v]/k
            cos2 = math.cos(theta_v)**2
            sin2 = math.sin(theta_v)**2

            for p in range(num_particles):
                assigned = particles[p]

                # Check if removal is allowed (all neighbors still in cover)
                removable = all(assigned[u] == 1 for u in G.neighbors(v))

                if removable:
                    # Branch according to quantum amplitudes
                    choice = random.choices([1, 0], weights=[cos2, sin2])[0]
                else:
                    choice = 1

                assigned[v] = choice



    # Compute expectation value directly
    exp_val = 0.0
    for p in range(num_particles):
        particle_cost = sum(C[v] for v, val in particles[p].items() if val == 1)
        exp_val += particle_cost/num_particles
    

    weights = np.ones(num_particles)  # Uniform weights since we compute expectation directly
    return exp_val, particles

def greedy_montecarlo_vertex_elimination_n2_reduced(
    G,
    C,
    beta_values_init,
    node_order=None,
    p=1,
    shots=None,
):
    G = nx.convert_node_labels_to_integers(G)
    
    # Keep original labels mapping
    original_nodes = list(G.nodes())
    
    values = beta_values_init.copy()
    global_energy = np.inf
    global_vertices = values.copy()
    
    current_graph = G.copy()
    
    while current_graph.number_of_nodes() > 0:
        
        n = current_graph.number_of_nodes()
        current_nodes = list(current_graph.nodes())
        
        # Rebuild parameters only for active nodes
        betas = {i: Parameter(f"β_{i}") for i in current_nodes}
        C = {i: 1.0 for i in current_graph.nodes()}
        new_order=node_order_by_cost_degree(current_graph,C)
        # Build mixer on reduced graph

        best_energy = np.inf
        best_vertex = None

        for v in current_nodes:

            trial_vals = values.copy()
            trial_vals[v] = 0
            E,particles=conditional_smc_mvc(current_graph, trial_vals, C, k=1, node_order=new_order, num_particles=shots)
            E=E+len(beta_values_init)-len(current_graph.nodes())

            if E < best_energy:
                best_energy = E
                best_vertex = v

        # Update global best
        if best_energy < global_energy:
            global_energy = best_energy
            global_vertices = values.copy()

        # Fix chosen vertex
        del values[len(current_graph.nodes())-1]  # Remove last entry which corresponds to the removed vertex

        # Remove vertex from graph (and all edges)
        current_graph.remove_node(best_vertex)
        #print(best_vertex)
        #print(current_graph.nodes())
        #print(current_graph.edges())
        mapping = {old: new for new, old in enumerate(current_graph.nodes())}
        current_graph = nx.relabel_nodes(current_graph, mapping)
        #print(current_graph.nodes())
        #print(current_graph.edges())
    print("Global energy:", global_energy)
    return global_energy

def greedy_two_phase_vertex_elimination_montecarlo(
    G,
    c,
    C,
    p=1,
    shots=None,
    initial_beta=np.pi / 4,
):
    """
    Two-phase greedy rounding WITH vertex elimination.

    When β_v = 0:
        - vertex is removed from graph
        - circuit rebuilt on reduced graph
        - previous assignments preserved

    Returns:
        best_configuration, best_energy
    """

    # --------------------------------------------------
    # bookkeeping
    # --------------------------------------------------
    original_vertices = list(G.nodes())

    active_graph = G.copy()
    fixed_values = {}              # eliminated vertices (β=0)
    values = {v: initial_beta for v in original_vertices}

    global_energy = np.inf
    global_vertices = None

    # ==================================================
    # PHASE 1 — quadratic fixing
    # ==================================================
    unfixed = set(active_graph.nodes())

    while unfixed:

        active_nodes = list(active_graph.nodes())

        best_energy = np.inf
        best_vertex = None
        best_value = None

        # evaluate ALL candidates
        for v in unfixed:

            for candidate in (0, np.pi / 2):

                trial_vals = values.copy()
                trial_vals[v] = candidate

                E, particles = conditional_smc_mvc(
                    active_graph,
                    trial_vals,
                    C,
                    k=1,
                    node_order=None,
                    num_particles=shots,
                )
                E = E + len(trial_vals) - len(active_graph.nodes())

                tol = 1e-6
                if E < best_energy-tol:
                    best_energy = E
                    best_vertex = v
                    best_value = candidate

        # ----- stopping condition -----
        if best_vertex is None:
            print("No improving move found, stopping Phase 1.")
            break

        # ----- fix one vertex -----
        values[best_vertex] = best_value
        unfixed.remove(best_vertex)

        # eliminate only if β=0
        if best_value == 0:
            fixed_values[best_vertex] = 0
            active_graph.remove_node(best_vertex)

        if best_energy < global_energy:
            global_energy = best_energy
            global_vertices = values.copy()

    print("Phase 1 complete. Energy:", global_energy)
    print("Remaining vertices after Phase 1 (β=π/2):", [v for v in values if values[v] == np.pi/2])

    # ==================================================
    # PHASE 2 — greedy elimination
    # ==================================================
    remaining = [v for v in values if values[v] == np.pi / 2]
    active_graph = G.subgraph(remaining).copy()

    while len(active_graph.nodes()) > 0:

        active_nodes = list(active_graph.nodes())
        best_energy = np.inf
        best_vertex = None

        for v in active_nodes:

            trial_vals = values.copy()
            trial_vals[v] = 0

            E, particles = conditional_smc_mvc(
                active_graph,
                trial_vals,
                C,
                k=1,
                node_order=None,
                num_particles=shots,
            )

            # optionally penalize remaining active nodes
            E = E + len(trial_vals) - len(active_graph.nodes())
    
            tol = 1e-6
            if E < best_energy-tol:
                best_energy = E
                best_vertex = v

        if best_vertex is None:
            break

        values[best_vertex] = 0
        active_graph.remove_node(best_vertex)

        if best_energy < global_energy:
            global_energy = best_energy
            global_vertices = values.copy()

    print(f"global_energy = {global_energy}")
    return global_vertices, global_energy