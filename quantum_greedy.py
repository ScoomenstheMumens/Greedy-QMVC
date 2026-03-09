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

from util_greedy import mixer_from_graph,mixer_from_graph_subset,node_order_by_cost_degree, expectation_value_cost_shifted,echo_commutator_circuit,echo_fidelity,mixer_fixed_beta,qubit_one_probabilities





# ---------------------------------------------------------
# Greedy vertex elimination
# ---------------------------------------------------------

def quantum_vertex_greedy(
    G,
    p=1,
    shots=None,
    tol=1e-8
):
    """
    Parameter-free quantum greedy elimination.

    Returns:
        cost = number of removed vertices
    """

    G = nx.convert_node_labels_to_integers(G)
    current_graph = G.copy()

    removed_vertices = []

    while current_graph.number_of_nodes() > 0:

        # build circuit
        qc = mixer_fixed_beta(current_graph, p=p)

        # compute probabilities
        probs = qubit_one_probabilities(qc, shots=shots)
        #print(probs)
        # stopping condition:
        if np.max(probs) - np.min(probs) < tol:
            break

        # select vertex with highest probability
        remove_idx = int(np.argmax(probs))
        removed_vertices.append(remove_idx)

        # remove vertex
        current_graph.remove_node(remove_idx)

        # relabel graph
        mapping = {old: new for new, old
                   in enumerate(current_graph.nodes())}
        current_graph = nx.relabel_nodes(current_graph, mapping)

    cost = len(removed_vertices)

    #print("Quantum greedy cost:", cost)
    return cost




def Quadratic_quantum_greedy(
    G,
    c,
    C,
    beta_values_init,
    p=1,
    shots=None,
):
    n = G.number_of_nodes()
    betas = {i: Parameter(f"β_{i}") for i in range(n)}

    values = beta_values_init.copy()
    unfixed = list(range(n))
    fixed = []
    global_energy=np.inf
    global_vertices=None
    while unfixed:

        qc = mixer_from_graph(G, c, betas=betas, p=p)

        # Current baseline energy
        current_energy = expectation_value_cost_shifted(
            qc, betas, C, values, shots
        )

        best_vertex = None
        best_value = None
        if len(unfixed)==n:
            best_energy=np.inf
        else:
            best_energy = np.inf

        improvement_found = False

        for v in unfixed:

            trial_vals = values.copy()
            trial_vals[v] = 0   # or try both 0 and 1 if needed

            E = expectation_value_cost_shifted(
                qc, betas, C, trial_vals, shots
            )
            #print(E,v)

            if E < best_energy:  # strictly better
                best_energy = E
                best_value = 0
                best_vertex = v
                improvement_found = True
        #print(best_vertex)
        # STOP CONDITION
        '''
        if not improvement_found:
            print("No further improvement possible. Stopping.")
            break
        '''
        if best_energy<global_energy:
            global_energy=best_energy
            global_vertices=values

        # Apply best move
        values[best_vertex] = best_value
        fixed.append(best_vertex)
        unfixed.remove(best_vertex)

    return global_vertices,global_energy




def Commuting_quantum_greedy(
    G,
    c,
    betas,
    beta_values,
    p=1,
    shots=None,
):
    """
    Removes vertex whose mixer least commutes with active block,
    measured via echo fidelity.
    """

    n = G.number_of_nodes()
    values = beta_values.copy()

    unfixed = list(range(n))
    fixed = []

    while unfixed:

        best_vertex = None
        worst_fidelity = 1.0

        active_nodes = unfixed.copy()
        #print(active_nodes)
        for v in unfixed:

            trial_active = [u for u in active_nodes if u != v]

            qc = echo_commutator_circuit(
                G,
                c,
                active_nodes=trial_active,
                trial_node=v,
                betas=betas,
                p=p,
            )

            F = echo_fidelity(qc, betas, values, shots)
            #print(F, v)
            tol=10e-6
            if F < worst_fidelity-tol:
                worst_fidelity = F
                best_vertex = v

        if best_vertex is None:
            print("No further non-commuting structure detected.")
            break

        #print(f"Removing vertex {best_vertex}, fidelity={worst_fidelity}")
    
        fixed.append(best_vertex)
        unfixed.remove(best_vertex)
        #print("energy",n-len(unfixed))
    energy=n-len(unfixed)
    return fixed, energy

# ==========================================================
# RELABELING UTILITY (graph, values, unfixed, c)
# ==========================================================
def relabel_graph_state(G, values, unfixed, c):
    """
    Relabel EVERYTHING consistently:
        graph
        values
        unfixed
        weight dict c
    """
    nodes = list(G.nodes())
    forward = {old: i for i, old in enumerate(nodes)}
    inverse = {i: old for old, i in forward.items()}

    # ---- graph ----
    G_new = nx.relabel_nodes(G, forward, copy=True)
    # ---- values ----
    values_new = {forward[v]: val for v, val in values.items() if v in forward}
    # ---- unfixed ----
    unfixed_new = {forward[v] for v in unfixed if v in forward}
    # ---- weights c ----
    c_new = {forward[v]: w for v, w in c.items() if v in forward}

    return G_new, values_new, unfixed_new, c_new, forward, inverse

# ==========================================================
# PHASE 1 STEP (0 or pi/2)
# ==========================================================
def fix_vertex_two_options(OG,G, c, C, values, unfixed, p=1, shots=None):
    # ----- relabel EVERYTHING -----
    G, values, unfixed, c, forward, inverse = relabel_graph_state(G, values, unfixed, c)

    active_nodes = list(G.nodes())
    betas = {v: Parameter(f"β_{v}") for v in active_nodes}
    qc = mixer_from_graph(G, c, betas=betas, p=p)

    best_energy = np.inf
    best_vertex = None
    best_value = None
    tol = 1e-6

    for v in unfixed:
        for candidate in (0, np.pi / 2):
            # only keep current nodes in trial_vals
            trial_vals = {vv: values[vv] for vv in G.nodes()}
            trial_vals[v] = candidate

            E = expectation_value_cost_shifted(qc, betas, c, trial_vals, shots)
            E=E+len(OG.nodes())-len(G.nodes())
            if E <= best_energy:
                best_energy = E
                best_vertex = v
                best_value = candidate

    # apply fixing
    new_graph = G.copy()
    new_values = values.copy()
    new_unfixed = set(unfixed)

    new_values[best_vertex] = best_value
    new_unfixed.discard(best_vertex)

    if best_value == 0:
        new_graph.remove_node(best_vertex)
        new_values.pop(best_vertex, None)

    # ----- map back to original labels -----
    new_graph = nx.relabel_nodes(new_graph, inverse, copy=True)
    new_values = {inverse[v]: val for v, val in new_values.items()}
    new_unfixed = {inverse[v] for v in new_unfixed}
    best_vertex = inverse[best_vertex]

    return new_graph, new_values, new_unfixed, best_energy, best_vertex, best_value

# ==========================================================
# PHASE 2 STEP (only 0)
# ==========================================================
def fix_vertex_one_option(OG,G, c, C, values, unfixed, p=1, shots=None):
    if len(G.nodes()) == 0:
        return G, values, unfixed, None, None

    # ----- relabel EVERYTHING -----
    G, values, unfixed, c, forward, inverse = relabel_graph_state(G, values, unfixed, c)

    active_nodes = list(G.nodes())
    betas = {v: Parameter(f"β_{v}") for v in active_nodes}
    qc = mixer_from_graph(G, c, betas=betas, p=p)

    best_energy = np.inf
    best_vertex = None
    tol = 1e-6

    for v in active_nodes:
        # only keep current nodes in trial_vals
        trial_vals = {vv: values[vv] for vv in G.nodes()}
        trial_vals[v] = 0

        E = expectation_value_cost_shifted(qc, betas, c, trial_vals, shots)
        E=E+len(OG.nodes())-len(G.nodes())
        if E < best_energy:
            best_energy = E
            best_vertex = v

    if best_vertex is None:
        return G, values, unfixed, None, None

    new_graph = G.copy()
    new_graph.remove_node(best_vertex)

    new_values = values.copy()
    new_values.pop(best_vertex, None)

    new_unfixed = set(unfixed)
    new_unfixed.discard(best_vertex)

    # ----- map back -----
    new_graph = nx.relabel_nodes(new_graph, inverse, copy=True)
    new_values = {inverse[v]: val for v, val in new_values.items()}
    new_unfixed = {inverse[v] for v in new_unfixed}
    best_vertex = inverse[best_vertex]

    return new_graph, new_values, new_unfixed, best_energy, best_vertex

# ==========================================================
# FULL GREEDY ALGORITHM
# ==========================================================
def Quadratic_rounding_quantum_greedy(G, c, C, p=1, shots=None, initial_beta=np.pi/4):
    active_graph = G.copy()
    values = {v: initial_beta for v in G.nodes()}
    unfixed = set(G.nodes())

    global_energy = np.inf
    global_vertices = values.copy()

    # ================= PHASE 1 =================
    while unfixed:
        active_graph, values, unfixed, energy, vertex, value = fix_vertex_two_options(
            G,active_graph, c, C, values, unfixed, p, shots
        )
        if vertex is None:
            print("No improving move found. Stopping Phase 1.")
            break
        if energy < global_energy:
            global_energy = energy
            global_vertices = values.copy()


    # ================= PHASE 2 =================
    unfixed = set(active_graph.nodes())
    while unfixed:
        active_graph, values, unfixed, energy, vertex = fix_vertex_one_option(G,
            active_graph, c, C, values, unfixed, p, shots
        )
        if vertex is None:
            break
        if energy < global_energy:
            global_energy = energy
            global_vertices = values.copy()

    print("Phase 2 complete. Energy:", global_energy)
    return global_vertices, global_energy



