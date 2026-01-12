from __future__ import annotations
import random
import math


# Optional rustworkx support
try:
    import rustworkx as rx
    RxGraph = rx.PyGraph
except ImportError:
    rx = None
    RxGraph = tuple()


def is_vertex_cover(G, C):
    return all(u in C or v in C for u, v in G.edges())

def local_search_vertex_cover(G, C_init, c, max_iters=1000):
    C = set(C_init)

    def cost(C):
        return sum(c[v] for v in C)

    for _ in range(max_iters):
        improved = False
        for v in list(C):
            C_new = C - {v}
            if is_vertex_cover(G, C_new) and cost(C_new) < cost(C):
                C = C_new
                improved = True
                break
        if not improved:
            break
    return C


def ga_vertex_cover(G, c, pop_size=50, generations=100, mutation_rate=0.05, alpha=100):
    n = G.number_of_nodes()

    def fitness(chromosome):
        cover = {i for i, x in enumerate(chromosome) if x == 1}
        uncovered_edges = sum(
            1 for u, v in G.edges() if u not in cover and v not in cover
        )
        return sum(c[i] for i in cover) + alpha * uncovered_edges

    def generate_individual():
        chromosome = [0] * n
        for u, v in G.edges():
            if chromosome[u] == 0 and chromosome[v] == 0:
                # pick cheaper vertex with probability 0.7, otherwise random
                if random.random() < 0.7:
                    chosen = u if c[u] <= c[v] else v
                else:
                    chosen = random.choice([u, v])
                chromosome[chosen] = 1
        # add some random nodes to increase diversity
        for i in range(n):
            if chromosome[i] == 0 and random.random() < 0.1:
                chromosome[i] = 1
        return chromosome

    def crossover(p1, p2):
        point = random.randint(1, n-1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def mutate(chrom):
        for i in range(n):
            if random.random() < mutation_rate:
                chrom[i] = 1 - chrom[i]
        return chrom

    # Initialize population
    population = [generate_individual() for _ in range(pop_size)]

    for _ in range(generations):
        population.sort(key=fitness)
        new_pop = population[:2]  # elitism: keep best 2
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:20], 2)  # tournament from top 20
            c1, c2 = crossover(p1, p2)
            new_pop += [mutate(c1), mutate(c2)]
        population = new_pop[:pop_size]

    best = min(population, key=fitness)
    return {i for i, x in enumerate(best) if x == 1}

def bp_vertex_cover_enforced(
    G,
    c,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    damping=0.5,
    seed=None
):
    """
    Belief Propagation for weighted Vertex Cover with HARD constraints.
    
    P(x) ∝ exp(-beta * sum_i c_i x_i) * ∏_{(i,j)} 1[x_i + x_j ≥ 1]
    """

    if seed is not None:
        random.seed(seed)

    # Messages: m[(i,j)] = (p0, p1)
    messages = {}
    for i, j in G.edges():
        messages[(i, j)] = (0.5, 0.5)
        messages[(j, i)] = (0.5, 0.5)

    for _ in range(max_iter):
        delta = 0.0
        new_messages = {}

        for (i, j), (p0_old, p1_old) in messages.items():

            # ----- x_i = 1 (vertex i IN cover) -----
            prod1 = 1.0
            for k in G.neighbors(i):
                if k == j:
                    continue
                p0, p1 = messages[(k, i)]
                prod1 *= (p0 + p1)

            m1 = math.exp(-beta * c[i]) * prod1

            # ----- x_i = 0 (vertex i OUT of cover) -----
            # All neighbors must be in
            prod0 = 1.0
            for k in G.neighbors(i):
                if k == j:
                    continue
                _, p1 = messages[(k, i)]
                prod0 *= p1

            m0 = prod0

            # ----- Normalize -----
            Z = m0 + m1
            if Z == 0:
                m0_new, m1_new = 0.0, 1.0
            else:
                m0_new = m0 / Z
                m1_new = m1 / Z

            # ----- Damping -----
            p0 = damping * p0_old + (1 - damping) * m0_new
            p1 = damping * p1_old + (1 - damping) * m1_new

            new_messages[(i, j)] = (p0, p1)
            delta = max(delta, abs(p0 - p0_old), abs(p1 - p1_old))

        messages = new_messages
        if delta < tol:
            break

    # ----- Compute marginals -----
    marginals = {}
    for i in G.nodes():
        prod1 = math.exp(-beta * c[i])
        prod0 = 1.0

        for j in G.neighbors(i):
            p0, p1 = messages[(j, i)]
            prod1 *= (p0 + p1)
            prod0 *= p1

        Z = prod0 + prod1
        marginals[i] = prod1 / Z if Z > 0 else 1.0

    return marginals

def bp_decode_vertex_cover(G, marginals):
    """
    Deterministic decoding guaranteeing feasibility.
    """
    C = set()
    uncovered_edges = set(G.edges())

    # Sort by decreasing belief of being in cover
    order = sorted(marginals, key=lambda i: -marginals[i])

    for i in order:
        incident = [(u, v) for (u, v) in uncovered_edges if u == i or v == i]
        if incident:
            C.add(i)
            for e in incident:
                uncovered_edges.remove(e)
        if not uncovered_edges:
            break

    return C

class SimulatedAnnealingWeighted:
    def __init__(self, G, c, T=100, alpha=0.99, max_iter=10000, perturbation_type='bitflip'):
        """
        G: networkx graph
        c: dict mapping node -> weight
        """
        self.G = G
        self.c = c
        self.n = G.number_of_nodes()
        self.T = T
        self.alpha = alpha
        self.max_iter = max_iter
        self.perturbation_type = perturbation_type
        self.vertex_cover = self.generate_vertex_cover()

    # ---------- FITNESS ----------
    def fitness(self, individual):
        if not self.is_feasible(individual):
            return float('inf')  # infeasible covers have very high cost
        return sum(self.c[i] for i, bit in enumerate(individual) if bit == 1)

    def is_feasible(self, individual):
        cover_nodes = {i for i, bit in enumerate(individual) if bit == 1}
        for u, v in self.G.edges():
            if u not in cover_nodes and v not in cover_nodes:
                return False
        return True

    # ---------- INITIAL COVER ----------
    def generate_vertex_cover(self):
        # start with empty bitstring
        individual = [0] * self.n
        # greedy heuristic: for each uncovered edge, pick the cheaper vertex
        uncovered_edges = set(self.G.edges())
        while uncovered_edges:
            u, v = random.choice(list(uncovered_edges))
            chosen = u if self.c[u] <= self.c[v] else v
            individual[chosen] = 1
            # remove edges covered by chosen node
            uncovered_edges = {e for e in uncovered_edges if chosen not in e}
        return individual

    # ---------- PERTURBATION ----------
    def perturb(self, vertex_cover, type='bitflip'):
        new_cover = vertex_cover.copy()
        if type == 'bitflip':
            i = random.randint(0, self.n - 1)
            new_cover[i] = 1 - new_cover[i]
            if self.is_feasible(new_cover):
                return new_cover
            else:
                return vertex_cover
        elif type == 'swap':
            i, j = random.sample(range(self.n), 2)
            new_cover[i], new_cover[j] = new_cover[j], new_cover[i]
            if self.is_feasible(new_cover):
                return new_cover
            else:
                return vertex_cover
        else:
            raise ValueError("Invalid perturbation type")

    # ---------- SIMULATED ANNEALING ----------
    def run(self):
        current = self.vertex_cover
        current_cost = self.fitness(current)
        best = current
        best_cost = current_cost

        while self.T > 1e-3:
            for _ in range(self.max_iter):
                new_cover = self.perturb(current, self.perturbation_type)
                new_cost = self.fitness(new_cover)
                if new_cost < current_cost:
                    current = new_cover
                    current_cost = new_cost
                else:
                    p = math.exp((current_cost - new_cost) / self.T)
                    if random.random() < p:
                        current = new_cover
                        current_cost = new_cost
                if current_cost < best_cost:
                    best = current
                    best_cost = current_cost
            self.T *= self.alpha

        return {i for i, bit in enumerate(best) if bit == 1}
