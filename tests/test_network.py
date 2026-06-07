"""Tests for network.py — node registration, edges, pheromone, and topology factories."""

import pytest
import networkx as nx

from network import Network, INITIAL_PHEROMONE, MIN_PHEROMONE


NODES = ["n0", "n1", "n2", "n3", "n4"]


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

def test_register_node_present():
    net = Network()
    net.register_node("n0")
    assert "n0" in net

def test_register_node_obj_attach():
    net = Network()
    net.register_node("n0")
    assert net.get_node_obj("n0") is None
    sentinel = object()
    net.attach_node_obj("n0", sentinel)
    assert net.get_node_obj("n0") is sentinel

def test_node_ids_returns_all():
    net = Network()
    for nid in NODES:
        net.register_node(nid)
    assert set(net.node_ids()) == set(NODES)


# ---------------------------------------------------------------------------
# Edge creation and pheromone
# ---------------------------------------------------------------------------

def test_add_directed_edge_exists():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_edge("a", "b")
    assert net.has_edge("a", "b")
    assert not net.has_edge("b", "a")

def test_add_undirected_edge_both_directions():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_undirected_edge("a", "b")
    assert net.has_edge("a", "b")
    assert net.has_edge("b", "a")

def test_initial_pheromone_default():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_edge("a", "b")
    assert net.pheromone("a", "b") == INITIAL_PHEROMONE

def test_initial_pheromone_custom():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_edge("a", "b", pheromone=3.7)
    assert net.pheromone("a", "b") == pytest.approx(3.7)

def test_set_pheromone():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_edge("a", "b")
    net.set_pheromone("a", "b", 5.0)
    assert net.pheromone("a", "b") == pytest.approx(5.0)

def test_pheromone_floored_at_minimum():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_edge("a", "b")
    net.set_pheromone("a", "b", 0.0)
    assert net.pheromone("a", "b") == pytest.approx(MIN_PHEROMONE)

def test_asymmetric_pheromone():
    """DiGraph allows different pheromone in each direction."""
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_undirected_edge("a", "b")
    net.set_pheromone("a", "b", 2.0)
    net.set_pheromone("b", "a", 4.0)
    assert net.pheromone("a", "b") == pytest.approx(2.0)
    assert net.pheromone("b", "a") == pytest.approx(4.0)

def test_all_edges_pheromone_snapshot():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.register_node("c")
    net.add_undirected_edge("a", "b")
    net.add_edge("b", "c", pheromone=2.5)
    snapshot = net.all_edges_pheromone()
    assert snapshot[("a", "b")] == pytest.approx(INITIAL_PHEROMONE)
    assert snapshot[("b", "a")] == pytest.approx(INITIAL_PHEROMONE)
    assert snapshot[("b", "c")] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def test_neighbors_directed():
    net = Network()
    for nid in ["a", "b", "c"]:
        net.register_node(nid)
    net.add_edge("a", "b")
    net.add_edge("a", "c")
    assert set(net.neighbors("a")) == {"b", "c"}
    assert net.neighbors("b") == []

def test_neighbors_after_undirected_edge():
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_undirected_edge("a", "b")
    assert "b" in net.neighbors("a")
    assert "a" in net.neighbors("b")


# ---------------------------------------------------------------------------
# Raw graph property
# ---------------------------------------------------------------------------

def test_graph_is_digraph():
    net = Network()
    assert isinstance(net.graph, nx.DiGraph)

def test_graph_shares_state():
    """net.graph must be the live graph, not a copy."""
    net = Network()
    net.register_node("a")
    net.register_node("b")
    net.add_edge("a", "b")
    assert net.graph.has_edge("a", "b")


# ---------------------------------------------------------------------------
# Topology factories
# ---------------------------------------------------------------------------

class TestRandomMesh:
    def _net(self, n=6, p=0.4, seed=0):
        ids = [f"n{i}" for i in range(n)]
        return Network.build_random_mesh(ids, edge_probability=p, seed=seed)

    def test_all_nodes_registered(self):
        net = self._net()
        assert len(net.node_ids()) == 6

    def test_connected(self):
        net = self._net()
        assert nx.is_weakly_connected(net.graph)

    def test_initial_pheromone_on_all_edges(self):
        net = self._net()
        for _, _, d in net.graph.edges(data=True):
            assert d["pheromone"] == pytest.approx(INITIAL_PHEROMONE)

    def test_deterministic(self):
        ids = [f"n{i}" for i in range(8)]
        net1 = Network.build_random_mesh(ids, seed=7)
        net2 = Network.build_random_mesh(ids, seed=7)
        assert set(net1.graph.edges()) == set(net2.graph.edges())

    def test_different_seeds_differ(self):
        ids = [f"n{i}" for i in range(10)]
        net1 = Network.build_random_mesh(ids, seed=1)
        net2 = Network.build_random_mesh(ids, seed=99)
        # Very unlikely to produce identical edge sets with n=10
        assert set(net1.graph.edges()) != set(net2.graph.edges())


class TestRing:
    def test_each_node_has_two_neighbors(self):
        ids = [f"n{i}" for i in range(5)]
        net = Network.build_ring(ids)
        for nid in ids:
            assert len(net.neighbors(nid)) == 2

    def test_connected(self):
        ids = [f"n{i}" for i in range(5)]
        net = Network.build_ring(ids)
        assert nx.is_weakly_connected(net.graph)

    def test_initial_pheromone(self):
        ids = ["a", "b", "c"]
        net = Network.build_ring(ids)
        for _, _, d in net.graph.edges(data=True):
            assert d["pheromone"] == pytest.approx(INITIAL_PHEROMONE)


class TestFullyConnected:
    def test_edge_count(self):
        ids = [f"n{i}" for i in range(4)]
        net = Network.build_fully_connected(ids)
        # 4 nodes → 4*3 directed edges (each pair gets both directions)
        assert net.graph.number_of_edges() == 4 * 3

    def test_every_node_reaches_every_other(self):
        ids = [f"n{i}" for i in range(4)]
        net = Network.build_fully_connected(ids)
        for a in ids:
            assert set(net.neighbors(a)) == set(ids) - {a}

    def test_initial_pheromone(self):
        ids = ["x", "y", "z"]
        net = Network.build_fully_connected(ids)
        for _, _, d in net.graph.edges(data=True):
            assert d["pheromone"] == pytest.approx(INITIAL_PHEROMONE)
