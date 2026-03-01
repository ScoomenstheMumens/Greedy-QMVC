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

def mixer_from_graph_subset_rev(
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
    # ---- PREVIOUSLY FIXED
    for tgt in fixed_order:
        angle = 2 * betas[tgt]
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
    # ---- ACTIVE (p layers)
    for _ in range(p):
        for tgt in active_order:
            angle = 2 * betas[tgt] / p
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
    pick_strategy="random",
    strategy="forward "
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
            if strategy == "forward":
                qc = mixer_from_graph_subset(
                    G,
                    c,
                    active_nodes=active,
                    trial_node=v,
                    fixed_nodes=fixed,
                    betas=betas,
                    p=p
                )
            else:
                qc = mixer_from_graph_subset_rev(
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




def greedy_optimize_vertex_elimination_n2(
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


def echo_commutator_circuit(
    G,
    c,
    active_nodes,
    trial_node,
    betas,
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
            angle = 2 * betas[tgt] / p
            ctrls = list(G.neighbors(tgt))
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- B (trial)
    tgt = trial_node
    angle = 2 * betas[tgt]
    ctrls = list(G.neighbors(tgt))
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)

    # ---- A†
    for _ in range(p):
        for tgt in active_nodes:
            angle = -2 * betas[tgt] / p
            ctrls = list(G.neighbors(tgt))
            if ctrls:
                qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
            else:
                qc.rx(angle, tgt)

    # ---- B†
    tgt = trial_node
    angle = -2 * betas[tgt]
    ctrls = list(G.neighbors(tgt))
    if ctrls:
        qc.append(RXGate(angle).control(len(ctrls)), ctrls + [tgt])
    else:
        qc.rx(angle, tgt)

    return qc

def echo_fidelity(qc, betas, beta_values, shots=None):

    # Bind ONLY parameters present in the circuit
    present_params = qc.parameters
    bind_dict = {
        betas[i]: beta_values[i]
        for i in betas
        if betas[i] in present_params
    }

    qc_bound = qc.assign_parameters(bind_dict)

    n = qc.num_qubits
    target = "1" * n

    # ----- statevector -----
    if shots is None:
        psi = Statevector.from_instruction(qc_bound)
        return psi.probabilities_dict().get(target, 0.0)

    # ----- shot-based -----
    qc_meas = qc_bound.copy()
    qc_meas.measure_all()

    backend = Aer.get_backend("aer_simulator")
    qc_meas = transpile(qc_meas, backend)
    result = backend.run(qc_meas, shots=shots).result()
    counts = result.get_counts()

    return counts.get(target, 0) / shots

def greedy_remove_most_noncommuting(
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

            if F < worst_fidelity:
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

import numpy as np
from qiskit.circuit import Parameter

import numpy as np
from qiskit.circuit import Parameter

def greedy_two_phase_vertex_elimination_dynamic(
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

        # rebuild parameters + circuit
        betas = {v: Parameter(f"β_{v}") for v in active_nodes}
        qc = mixer_from_graph(active_graph, c, betas=betas, p=p)

        best_energy = np.inf
        best_vertex = None
        best_value = None

        # ----------------------------------------------
        # evaluate all candidates for unfixed vertices
        # ----------------------------------------------
        for v in unfixed:
            for candidate in (0, np.pi / 2):
                trial_vals = values.copy()
                trial_vals[v] = candidate

                E = expectation_value_cost_shifted(qc, betas, C, trial_vals, shots)
                tol = 1e-6
                if E < best_energy - tol:  # Use tolerance to avoid numerical issues   
                    best_energy = E
                    best_vertex = v
                    best_value = candidate

        # ----- stopping condition -----
        if best_vertex is None:
            print("No improving move found. Stopping Phase 1.")
            break

        # ----- fix one vertex per iteration -----
        values[best_vertex] = best_value
        unfixed.remove(best_vertex)

        # eliminate from graph only if β=0
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
        betas = {v: Parameter(f"β_{v}") for v in active_nodes}
        qc = mixer_from_graph(active_graph, c, betas=betas, p=p)

        best_energy = np.inf
        best_vertex = None

        for v in active_nodes:
            trial_vals = values.copy()
            trial_vals[v] = 0

            E = expectation_value_cost_shifted(qc, betas, C, trial_vals, shots)
            tol=10e-6
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
from collections import defaultdict
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


import numpy as np
from qiskit.circuit import Parameter
import numpy as np
from qiskit.circuit import Parameter

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