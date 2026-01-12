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
'''
def greedy_distance1_coloring(G):
    """
    Standard greedy vertex coloring (distance-1).
    Returns:
        c : dict {node: color}
    """
    c = {}
    nodes = list(G.nodes())

    for v in nodes:
        forbidden = set()

        # distance 1 only
        for u in G.neighbors(v):
            if u in c:
                forbidden.add(c[u])

        # choose smallest available color
        color = 0
        while color in forbidden:
            color += 1

        c[v] = color

    return c

def greedy_distance2_coloring(G):
    """
    Greedy distance-2 coloring.
    Returns:
        c : dict {node: color}
    """
    c = {}
    nodes = list(G.nodes())

    for v in nodes:
        forbidden = set()

        # distance 1
        for u in G.neighbors(v):
            if u in c:
                forbidden.add(c[u])

        # distance 2
        for u in G.neighbors(v):
            for w in G.neighbors(u):
                if w in c:
                    forbidden.add(c[w])

        # assign smallest available color
        color = 0
        while color in forbidden:
            color += 1

        c[v] = color

    return c
def order_nodes_by_color_size(G, c):
    """
    Orders nodes so that vertices belonging to the
    largest color class come first.
    """
    color_classes = defaultdict(list)
    for v, col in c.items():
        color_classes[col].append(v)

    # sort colors by size (descending)
    sorted_colors = sorted(
        color_classes.keys(),
        key=lambda col: len(color_classes[col]),
        reverse=True
    )

    ordered_nodes = []
    for col in sorted_colors:
        ordered_nodes.extend(color_classes[col])

    return ordered_nodes
'''

def node_order_by_cost_degree(G, C):
    """
    Order nodes by:
      1) descending cost
      2) descending degree
    """
    return sorted(
        G.nodes(),
        key=lambda i: (G.degree(i),-C[i])
        #key=lambda i: (G.degree(i)/C[i])
    )



# ---------------- Graph & mixer ----------------

def mixer_from_graph(G,c):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    qc = QuantumCircuit(n)
    betas = {i: Parameter(f"β_{i}") for i in G.nodes()}

    for i in range(n):
        qc.x(i)
    # 🔥 ORDER NODES HERE
    ordered_nodes = node_order_by_cost_degree(G,c)
    for tgt in ordered_nodes:
        angle = 2 * betas[tgt]
        ctrls = list(G.neighbors(tgt))
        if ctrls:
            qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
        else:
            qc.rx(angle, tgt)

    return qc, betas, G

'''
def mixer_from_graph(G,c):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    qc = QuantumCircuit(n)
    betas = {i: Parameter(f"β_{i}") for i in G.nodes()}

    for i in range(n):
        qc.x(i)

    for tgt in G.nodes():
        angle = 2 * betas[tgt]
        ctrls = list(G.neighbors(tgt))
        if ctrls:
            qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
        else:
            qc.rx(angle, tgt)

    return qc, betas, G
'''

'''
def mixer_from_graph(G,c):
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    # 1️⃣ distance-2 coloring
    color = greedy_distance1_coloring(G)

    qc = QuantumCircuit(n)
    betas = {i: Parameter(f"β_{i}") for i in G.nodes()}

    # initialize |+>
    for i in range(n):
        qc.x(i)

    # 2️⃣ order nodes by largest color class first
    ordered_nodes = order_nodes_by_color_size(G, color)

    # 3️⃣ build mixer
    for tgt in ordered_nodes:
        angle = 2 * betas[tgt]
        ctrls = list(G.neighbors(tgt))

        if ctrls:
            qc.append(
                RXGate(angle).control(len(ctrls)),
                ctrls + [tgt]
            )
        else:
            qc.rx(angle, tgt)

    return qc, betas, G
'''
def expectation_value_cost_shifted(qc, betas, C, beta_values, shots=None):
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

def expectation_and_variance(qc, betas, C, beta_values, shots: int | None = None):
    """
    Compute the mean and variance of the cost H = sum_i c_i (1-Z_i)/2
    Supports both ideal (shots=None) and shot-based estimation.

    Args:
        qc: QuantumCircuit with parameterized mixer
        betas: dict of Qiskit Parameters
        C: dict mapping qubit -> cost coefficient
        beta_values: dict mapping qubit -> float
        shots: number of shots for measurement (None = ideal)

    Returns:
        (mean, variance)
    """
    bind_dict = {betas[i]: beta_values[i] for i in betas}
    qc_bound = qc.assign_parameters(bind_dict)

    n = qc.num_qubits

    if shots is None:
        # Ideal case using statevector
        psi = Statevector.from_instruction(qc_bound)
        mean = 0.0
        var = 0.0
        for i, c_i in C.items():
            # expectation <Z_i>
            p_str = ["I"] * n
            p_str[i] = "Z"
            Zi = SparsePauliOp("".join(p_str))
            exp_Z = psi.expectation_value(Zi).real

            # probability qubit i = 1
            p1 = (1 - exp_Z) / 2
            mean += c_i * p1
            var += c_i**2 * p1 * (1 - p1)

        return mean, var

    else:
        # Shot-based case
        qc_meas = qc_bound.copy()
        qc_meas.measure_all()

        backend = Aer.get_backend("aer_simulator")
        qc_meas = transpile(qc_meas, backend)
        counts = backend.run(qc_meas, shots=shots).result().get_counts()

        costs = []
        for bitstring, count in counts.items():
            z_vals = np.array([1 if b == "0" else -1 for b in bitstring[::-1]])
            cost = sum(c_i * (1 - z_vals[i]) / 2 for i, c_i in C.items())
            costs += [cost] * count

        costs = np.array(costs)
        mean = costs.mean()
        var = costs.var(ddof=0)  # population variance
        return mean, var

    return best

# ---------------- Greedy optimizers ----------------

def greedy_optimize(qc, betas, C, beta_values,shots=None):
    values = beta_values.copy()
    free = list(betas.keys())

    while free:
        i = random.choice(free)
        best_val = values[i]
        best_E = expectation_value_cost_shifted(qc, betas, C, values,shots)

        for candidate in (0.0, math.pi/2):
            trial = values.copy()
            trial[i] = candidate
            E = expectation_value_cost_shifted(qc, betas, C, trial,shots)
            if E < best_E:
                best_E = E
                best_val = candidate

        values[i] = best_val
        free.remove(i)

    return values

def greedy_optimize_seq(qc, betas, C, beta_values,shots= None):
    values = beta_values.copy()

    # assume betas is an ordered mapping or keys are index-like
    indices = list(betas.keys())

    # iterate from last to first
    for i in reversed(indices):
    #for i in indices :
        best_val = values[i]
        best_E = expectation_value_cost_shifted(qc, betas, C, values,shots)

        for candidate in (0.0, math.pi / 2):
            trial = values.copy()
            trial[i] = candidate
            E = expectation_value_cost_shifted(qc, betas, C, trial,shots)

            if E < best_E:
                best_E = E
                best_val = candidate

        values[i] = best_val

    return values

def greedy_optimize_seq_rev(qc, betas, C, beta_values,shots= None):
    values = beta_values.copy()

    # assume betas is an ordered mapping or keys are index-like
    indices = list(betas.keys())

    # iterate from last to first
    #for i in reversed(indices):
    for i in indices :
        best_val = values[i]
        best_E = expectation_value_cost_shifted(qc, betas, C, values,shots)

        for candidate in (0.0, math.pi / 2):
            trial = values.copy()
            trial[i] = candidate
            E = expectation_value_cost_shifted(qc, betas, C, trial,shots)

            if E < best_E:
                best_E = E
                best_val = candidate

        values[i] = best_val

    return values

def greedy_optimize_risk_aware(qc, betas, C, beta_values, shots=None, lam=0.5):
    values = beta_values.copy()
    free = list(betas.keys())
    #while free:
        #i = random.choice(free)
    for i in range(len(free)):
        mu, var = expectation_and_variance(qc, betas, C, values, shots)
        best_score = mu + lam * math.sqrt(var)
        best_val = values[i]

        for candidate in (0.0, math.pi/2):
            trial = values.copy()
            trial[i] = candidate
            mu_t, var_t = expectation_and_variance(qc, betas, C, trial, shots)
            score = mu_t + lam * math.sqrt(var_t)
            if score < best_score:
                best_score = score
                best_val = candidate

        values[i] = best_val
        free.remove(i)

    return values
def mean_field_cost_degree_order_init(G, C_cost, order, alpha=1, beta=1, gamma=1,delta=0.2, n_iter=30):
    n = len(G)
    rank = {j: k / n for k, j in enumerate(order)}
    p = {j: 0.5 for j in G.nodes()}
    maxcost= max(C_cost)
    for _ in range(n_iter):
        p_new = {}
        for j in G.nodes():
            dj = max(1, G.degree(j))
            neigh_pressure = sum(p[k] for k in G.neighbors(j)) / dj
            field = (
                alpha * C_cost[j]
                - beta * neigh_pressure
                - gamma * rank[j]
            )
            p_new[j] = 1.0 / (1.0 + np.exp(-delta*field))
        p = p_new

    return {j: np.arcsin(np.sqrt(p[j])) for j in p}
def node_order_by_cost_degree(G, C):
    """
    Order nodes by:
      1) descending cost
      2) descending degree
    """
    return sorted(
        G.nodes(),
        #key=lambda i: (i)
        key=lambda i: (G.degree(i), -C[i]),
        #key=lambda i: (-G.degree(i)/C[i])
    )
