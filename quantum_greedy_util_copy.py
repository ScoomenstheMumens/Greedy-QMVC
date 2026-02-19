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


def node_order_by_cost_degree(G, C):
    """
    Order nodes by:
      1) descending cost
      2) descending degree
    """
    return sorted(
        G.nodes(),
        #key=lambda i: (i)
        key=lambda i: (-C[i],G. degree(i))
        #key=lambda i: (G.degree(i),-C[i])
        #key=lambda i: (G.degree(i)/C[i])
    )





def expectation_value_cost_shifted(qc, betas, C, beta_values, shots=None):
    bind_dict = {betas[i]: beta_values[i] for i in betas}
    #print(bind_dict)
    qc_bound = qc.assign_parameters(bind_dict)
    
    n = qc.num_qubits
    paulis = []
    coeffs = []

    for i, c_i in C.items():
        p = ["I"] * n
        p[i] = "Z"
        paulis.append("".join(p))
        coeffs.append(-0.5 * c_i)

    HZ = SparsePauliOp(paulis, coeffs)
    shift = 0.5 * sum(C.values())

    if shots is None:
        psi = Statevector.from_instruction(qc_bound)
        return shift + psi.expectation_value(HZ).real

    # ----- shot-based -----
    qc_meas = qc_bound.copy()
    qc_meas.measure_all()

    backend = Aer.get_backend("aer_simulator")
    qc_meas = transpile(qc_meas, backend)
    counts = backend.run(qc_meas, shots=shots).result().get_counts()

    exp_val = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        z_vals = np.array([1 if b == "0" else -1 for b in bitstring[::-1]])

        hz_value = 0.0
        for i, c_i in C.items():
            hz_value += -0.5 * c_i * z_vals[i]

        exp_val += prob * hz_value

    return shift + exp_val




def mixer_from_graph_subset(
    G,
    c,
    active_nodes,
    trial_node,
    fixed_nodes,
    betas,
    p=1,
    node_order=None
):
    """
    Gate order is EXACTLY:
      1) all active nodes (p layers)
      2) trial_node (once)
      3) previously fixed nodes (once)
    """

    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)

    # Initial X layer
    for i in range(n):
        qc.x(i)

    if node_order is None:
        node_order = node_order_by_cost_degree(G, c)

    present = set(active_nodes) | {trial_node} | set(fixed_nodes)

    active_order = [v for v in node_order if v in active_nodes]
    fixed_order  = [v for v in node_order if v in fixed_nodes]

    # ---- ACTIVE (p layers)
    for _ in range(p):
        for tgt in active_order:
            angle = 2 * betas[tgt] / p
            ctrls = [u for u in G.neighbors(tgt) if u in present]
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- TRIAL (first fixed)
    tgt = trial_node
    angle = 2 * betas[tgt]
    ctrls = [u for u in G.neighbors(tgt) if u in present]
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)

    # ---- PREVIOUSLY FIXED
    for tgt in fixed_order:
        angle = 2 * betas[tgt]
        ctrls = [u for u in G.neighbors(tgt) if u in present]
        if ctrls:
            qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
        else:
            qc.rx(angle, tgt)

    return qc


# ---------------------------------------------------------
# Greedy vertex elimination
# ---------------------------------------------------------

def greedy_optimize_vertex_elimination(
    G,
    c,
    C,
    beta_values_init,
    p=1,
    shots=None,
    pick_strategy="random"
):
    n = G.number_of_nodes()
    betas = {i: Parameter(f"β_{i}") for i in range(n)}

    values = beta_values_init.copy()
    unfixed = list(range(n))
    fixed = []

    while unfixed:

        v = random.choice(unfixed) if pick_strategy == "random" else unfixed[-1]

        best_E = np.inf
        best_val = None

        for candidate in (0.0, math.pi / 2):
            trial_vals = values.copy()
            trial_vals[v] = candidate

            active = [u for u in unfixed if u != v]

            qc = mixer_from_graph_subset(
                G,
                c,
                active_nodes=active,
                trial_node=v,
                fixed_nodes=fixed,
                betas=betas,
                p=p
            )
            
            E = expectation_value_cost_shifted(
                qc, betas, C, trial_vals, shots
            )
            #print(E,candidate)
            if E <= best_E:
                best_E = E
                best_val = candidate

        values[v] = best_val
        fixed.append(v)
        unfixed.remove(v)

        #print(f"Fixed vertex {v} -> {best_val}")

    #print("Final values:", values)
    #print("E",best_E)
    return values,best_E



