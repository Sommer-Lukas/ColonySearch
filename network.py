"""
network.py — NetworkX-backed topology for the ColonySearch swarm.

Owns the node registry, directed graph, and per-edge pheromone values.
Used by router.py (neighbor lookup, pheromone read), reputation.py
(pheromone and reputation writes), and visualise.py (graph layout).

Design notes:
- DiGraph so pheromone can be asymmetric (A→B trail ≠ B→A trail),
  matching standard ACO literature.
- Pheromone is floored at MIN_PHEROMONE so no edge fully dies.
- Topology factories guarantee connectivity — isolated nodes break routing.
"""

from __future__ import annotations

import random

import networkx as nx

INITIAL_PHEROMONE: float = 1.0  # τ_0 per ACO convention
MIN_PHEROMONE: float = 0.01     # floor to keep all edges alive


class Network:
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def register_node(self, node_id: str, node_obj=None, reputation: float = 1.0) -> None:
        """Add node_id to the topology. node_obj is the Node instance (may be None during setup)."""
        self._graph.add_node(node_id, node=node_obj, reputation=reputation)

    def attach_node_obj(self, node_id: str, node_obj) -> None:
        """Bind a Node instance to an already-registered node_id."""
        self._graph.nodes[node_id]["node"] = node_obj

    def get_node_obj(self, node_id: str):
        """Return the Node instance for node_id, or None if not yet attached."""
        return self._graph.nodes[node_id].get("node")

    def node_ids(self) -> list[str]:
        return list(self._graph.nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._graph

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(self, src: str, dst: str, pheromone: float = INITIAL_PHEROMONE) -> None:
        """Add directed edge src→dst with the given initial pheromone level."""
        self._graph.add_edge(src, dst, pheromone=max(pheromone, MIN_PHEROMONE))

    def add_undirected_edge(self, a: str, b: str, pheromone: float = INITIAL_PHEROMONE) -> None:
        """Add symmetric edges a→b and b→a (standard mesh connection)."""
        self.add_edge(a, b, pheromone)
        self.add_edge(b, a, pheromone)

    def has_edge(self, src: str, dst: str) -> bool:
        return self._graph.has_edge(src, dst)

    # ------------------------------------------------------------------
    # Pheromone access — router.py and reputation.py use these
    # ------------------------------------------------------------------

    def pheromone(self, src: str, dst: str) -> float:
        return self._graph[src][dst]["pheromone"]

    def set_pheromone(self, src: str, dst: str, value: float) -> None:
        self._graph[src][dst]["pheromone"] = max(value, MIN_PHEROMONE)

    def all_edges_pheromone(self) -> dict[tuple[str, str], float]:
        """Snapshot of all edge pheromone values — used by visualise.py."""
        return {(u, v): d["pheromone"] for u, v, d in self._graph.edges(data=True)}

    # ------------------------------------------------------------------
    # Reputation access — reputation.py reads and writes these
    # ------------------------------------------------------------------

    def reputation(self, node_id: str) -> float:
        return self._graph.nodes[node_id]["reputation"]

    def set_reputation(self, node_id: str, value: float) -> None:
        self._graph.nodes[node_id]["reputation"] = max(value, 0.0)

    # ------------------------------------------------------------------
    # Routing helpers — router.py uses these
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[str]:
        """Nodes reachable from node_id via a directed edge."""
        return list(self._graph.successors(node_id))

    # ------------------------------------------------------------------
    # Raw graph — visualise.py and NetworkX utilities need direct access
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    # ------------------------------------------------------------------
    # Topology factories
    # ------------------------------------------------------------------

    @classmethod
    def build_random_mesh(
        cls,
        node_ids: list[str],
        edge_probability: float = 0.4,
        seed: int = 42,
    ) -> "Network":
        """
        Erdős–Rényi random graph with guaranteed connectivity.

        Each pair (i, j) gets a symmetric edge with probability edge_probability.
        A random spanning tree is layered on top to ensure no node is isolated —
        critical for routing correctness.
        """
        net = cls()
        for nid in node_ids:
            net.register_node(nid)

        rng = random.Random(seed)
        n = len(node_ids)
        edges: set[tuple[str, str]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_probability:
                    edges.add((node_ids[i], node_ids[j]))

        # Spanning tree pass — ensures every node is reachable
        shuffled = node_ids[:]
        rng.shuffle(shuffled)
        for i in range(1, len(shuffled)):
            a, b = shuffled[i - 1], shuffled[i]
            # Canonical order avoids adding the same pair twice
            edges.add((min(a, b), max(a, b)))

        for a, b in edges:
            net.add_undirected_edge(a, b)

        return net

    @classmethod
    def build_ring(cls, node_ids: list[str]) -> "Network":
        """Ring topology: each node connects only to its two neighbours."""
        net = cls()
        for nid in node_ids:
            net.register_node(nid)
        n = len(node_ids)
        for i in range(n):
            net.add_undirected_edge(node_ids[i], node_ids[(i + 1) % n])
        return net

    @classmethod
    def build_fully_connected(cls, node_ids: list[str]) -> "Network":
        """Clique: every node has a direct edge to every other node."""
        net = cls()
        for nid in node_ids:
            net.register_node(nid)
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                net.add_undirected_edge(a, b)
        return net
