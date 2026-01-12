from __future__ import annotations
import random
# Optional rustworkx support
try:
    import rustworkx as rx
    RxGraph = rx.PyGraph
except ImportError:
    rx = None
    RxGraph = tuple()
# CPLEX import
from docplex.mp.model import Model


def greedy_degree_vertex_cover(G, c):
    Gc = G.copy()
    cover = set()

    while Gc.number_of_edges() > 0:
        v = max(
            Gc.nodes(),
            #key=lambda x: (c[x],G.degree(x))
            key=lambda x: Gc.degree(x) / c[x]
        )
        cover.add(v)
        Gc.remove_node(v)

    return cover
def greedy_edge_vertex_cover(G, c):
    Gc = G.copy()
    cover = set()

    while Gc.number_of_edges() > 0:
        u, v = random.choice(list(Gc.edges()))
        chosen = u if c[u] <= c[v] else v
        cover.add(chosen)
        Gc.remove_node(chosen)

    return cover



def mvc_exact_cplex(G, c):
    mdl = Model("wmvc_exact")
    x = {i: mdl.binary_var(name=f"x_{i}") for i in G.nodes()}

    for u, v in G.edges():
        mdl.add_constraint(x[u] + x[v] >= 1)

    mdl.minimize(mdl.sum(c[i] * x[i] for i in G.nodes()))
    mdl.solve(log_output=False)

    return {i for i in x if x[i].solution_value > 0.5}



def mvc_lp_relaxation(G, c):
    mdl = Model("wmvc_lp")
    x = {i: mdl.continuous_var(lb=0, ub=1, name=f"x_{i}") for i in G.nodes()}

    for u, v in G.edges():
        mdl.add_constraint(x[u] + x[v] >= 1)

    mdl.minimize(mdl.sum(c[i] * x[i] for i in G.nodes()))
    mdl.solve(log_output=False)

    return {i for i in x if x[i].solution_value >= 0.5}


def mvc_primal_dual_weighted(G, c):
    """
    Weighted Minimum Vertex Cover using primal-dual / local-ratio.
    Returns a set of vertices.
    """
    remaining_edges = set(G.edges())
    cover = set()
    vertex_weights = c.copy()

    while remaining_edges:
        # Pick an edge with minimal combined weight
        u, v = min(remaining_edges, key=lambda e: vertex_weights[e[0]] + vertex_weights[e[1]])
        # Add the vertex with smaller weight
        if vertex_weights[u] <= vertex_weights[v]:
            cover.add(u)
            # Remove all edges incident to u
            remaining_edges = {e for e in remaining_edges if u not in e}
        else:
            cover.add(v)
            remaining_edges = {e for e in remaining_edges if v not in e}
    
    return cover