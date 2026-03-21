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
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_graph(G):
    """Plot a NetworkX graph with labels."""
    plt.figure(figsize=(4,4))
    pos = nx.spring_layout(G, seed=42)  # nice-looking layout

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=800,
        node_color="lightblue",
        font_size=12,
        font_weight="bold",
        edge_color="gray"
    )

    plt.title("Graph G")
    plt.axis("off")


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


def expectation_value_cost_shifted(
    qc,
    betas,
    C,
    beta_values,
    shots=None,
    backend_method="statevector",
    mps_max_bond=None
):
    """
    Compute the shifted expectation value of a Z-Hamiltonian.
    
    Args:
        qc: Parameterized QuantumCircuit.
        betas: List or dict of parameters to bind.
        C: dict mapping qubit index to coefficient.
        beta_values: Values for the parameters.
        shots: If None, exact simulation; else shot-based.
        backend_method: 'statevector' or 'matrix_product_state'.
        mps_max_bond: Maximum bond dimension (only for MPS).
    
    Returns:
        float: Shifted expectation value.
    """
    # --- Bind parameters ---
    bind_dict = {betas[i]: beta_values[i] for i in betas}
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

    # --- Shot-based simulation ---
    if shots is not None:
        qc_meas = qc_bound.copy()
        qc_meas.measure_all()
        
        backend = Aer.get_backend("aer_simulator")        
        qc_meas = transpile(qc_meas, backend)
        counts = backend.run(qc_meas, shots=shots).result().get_counts()
        
        exp_val = 0.0
        for bitstring, count in counts.items():
            prob = count / shots
            z_vals = np.array([1 if b == "0" else -1 for b in bitstring[::-1]])
            hz_value = sum([-0.5 * C[i] * z_vals[i] for i in C])
            exp_val += prob * hz_value

        return shift + exp_val

    # --- Exact simulation ---
    backend = Aer.get_backend("aer_simulator")
    psi = Statevector.from_instruction(qc_bound)
    return shift + psi.expectation_value(HZ).real

def mixer_from_graph(
    G,
    c,
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

    
    # ---- ACTIVE (p layers)
    for _ in range(p):
        for tgt in node_order:
            angle = 2 * betas[tgt] / p
            ctrls = [u for u in G.neighbors(tgt)]
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    
    

    return qc

def mixer_fixed_beta(G, p=1, beta=None,node_order=None):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    qc = QuantumCircuit(n)

    # initial layer
    qc.x(range(n))

    if node_order is None:
        C = {i: 1.0 for i in G.nodes()}
        node_order = node_order_by_cost_degree(G, C)

    if beta == None:
        beta = np.pi / 4

    for _ in range(p):
        for tgt in node_order:
            angle = 2 * beta / p
            ctrls = list(G.neighbors(tgt))

            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)),
                          ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    return qc

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


def circuit_from_graph_commutator(
    G,
    c,
    active_nodes,
    trial_node,
    betas,
    p=1,
    flag=False,
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
    if node_order is None:
        node_order = node_order_by_cost_degree(G, c)
    # Initial X layer
    for i in range(n):
        qc.x(i)
    if flag==True:
        alpha = np.pi / 4
        p_init=1
        for _ in range(p_init):
            for tgt in node_order:
                angle = 2 * alpha / p
                ctrls = list(G.neighbors(tgt))

                if ctrls:
                    qc.append(RXGate(angle).control(len(ctrls)),
                            ctrls + [tgt])
                else:
                    qc.rx(angle, tgt)

    

    # ---- ACTIVE (p layers)
    for _ in range(p):
        for tgt in active_nodes:
            angle = 2 * betas[tgt] / p
            ctrls = [u for u in G.neighbors(tgt)]
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- TRIAL (first fixed)
    tgt = trial_node
    angle = 2 * betas[tgt]
    ctrls = [u for u in G.neighbors(tgt)]
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)

    # ---- ACTIVE (p layers)
    for _ in range(p):
        for tgt in active_nodes:
            angle = -2 * betas[tgt] / p
            ctrls = [u for u in G.neighbors(tgt)]
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- TRIAL (first fixed)
    tgt = trial_node
    angle = -2 * betas[tgt]
    ctrls = [u for u in G.neighbors(tgt)]
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)
    

    if flag==True:

        for _ in range(p_init):
            for tgt in node_order[::-1]:
                angle = -2 * alpha / p
                ctrls = list(G.neighbors(tgt))

                if ctrls:
                    qc.append(RXGate(angle).control(len(ctrls)),
                            ctrls + [tgt])
                else:
                    qc.rx(angle, tgt)

    return qc


def echo_commutator_circuit(
    G,
    c,
    active_nodes,
    trial_node,
    betas_1,
    betas_2,
    p=1,
):
    """
    Parameterized echo circuit:
        U = A B A† B†
    """
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)

    # |psi> = |111...1>
    for i in range(n):
        qc.x(i)

    # ---- A (active p layers)
    for _ in range(p):
        for tgt in active_nodes:
            angle = 2 * betas_1/ p
            ctrls = list(G.neighbors(tgt))
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- B (trial)
    tgt = trial_node
    angle = 2 * betas_2
    ctrls = list(G.neighbors(tgt))
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)

    # ---- A†
    for _ in range(p):
        for tgt in active_nodes:
            angle = -2 * betas_1 / p
            ctrls = list(G.neighbors(tgt))
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- B†
    tgt = trial_node
    angle = -2 * betas_2
    ctrls = list(G.neighbors(tgt))
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)

    return qc

def echo_fidelity(
    qc,
    shots=None,
    backend_method="matrix_product_state",
    mps_max_bond=20
):

    #

    n = qc.num_qubits
    target = "1" * n

    # -------- shot-based simulation --------
    if shots is not None:

        qc_meas = qc.copy()
        qc_meas.measure_all()
        backend = Aer.get_backend("aer_simulator")
        qc_meas = transpile(qc_meas, backend)
        counts = backend.run(qc_meas, shots=shots).result().get_counts()

        return counts.get(target, 0) / shots

    # -------- exact simulation --------
    backend = Aer.get_backend("aer_simulator")
    psi = Statevector.from_instruction(qc)

    return psi.probabilities_dict().get(target, 0.0)



def qubit_one_probabilities(
    qc,
    shots=None,
    backend_method="matrix_product_state",
    mps_max_bond=20
):
    """
    Returns p_i = <(1 - Z_i)/2> for all qubits.
    """

    n = qc.num_qubits

    # -------- shot-based simulation --------
    if shots is not None:

        qc_meas = qc.copy()
        qc_meas.measure_all()

        backend = Aer.get_backend("aer_simulator")
        qc_meas = transpile(qc_meas, backend)
        counts = backend.run(qc_meas, shots=shots).result().get_counts()

        probs = np.zeros(n)

        for bitstring, count in counts.items():

            weight = count / shots
            bits = bitstring[::-1]

            for i, b in enumerate(bits):
                if b == "1":
                    probs[i] += weight

        return probs

    # -------- exact simulation --------
    backend = Aer.get_backend("aer_simulator")
    psi = Statevector.from_instruction(qc)

    probs = []

    for i in range(n):

        pauli = ["I"] * n
        pauli[n - i - 1] = "Z"

        Zi = SparsePauliOp("".join(pauli), [1.0])
        expZ = psi.expectation_value(Zi).real

        probs.append((1 - expZ) / 2)

    return np.array(probs)