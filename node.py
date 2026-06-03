from flask import Flask, jsonify, request, render_template
import requests

import json
import math
import random
from copy import deepcopy
from time import sleep

import search
from network import Network

temp_search = search.search

# ── Reputation / pheromone tuning knobs ──────────────────────────────────────
# How much a node's historical reputation nudges its results vs. raw relevance.
# 0.0 = pure content score, 1.0 = pure reputation.
REP_WEIGHT: float = 0.2

# EMA learning rate for reputation.  Keeps rep bounded in [0, 1] and lets
# nodes recover or fall — recent performance matters more than early luck.
# 0 = never updates, 1 = full replace each query.
REP_DECAY: float = 0.15

# ACO pheromone evaporation rate.  Standard τ = (1−ρ)·τ + Δτ deposit.
# Without evaporation, early-winner edges dominate forever.
PHEROMONE_EVAP: float = 0.1

# ACO selection exponents for get_neighbors probabilistic sampling.
# P(j) ∝ τ(i,j)^α · rep(j)^β
# α > 1  → more exploitation (high-pheromone edges strongly preferred)
# β > 1  → more heuristic bias (high-rep nodes strongly preferred)
ACO_ALPHA: float = 1.0
ACO_BETA:  float = 1.0

# Number of neighbors sampled per hop in get_neighbors.
# Lower = faster but fewer paths explored; higher = better recall, slower.
NETWORK_TOP_K: int = 10

# ── Embedding model — loaded once at startup, never re-downloaded ─────────────
_EMBED_MODEL_NAME = "all-mpnet-base-v2"
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading sentence-transformers model '{_EMBED_MODEL_NAME}' …", flush=True)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        print("Model ready.", flush=True)
    return _embed_model

def _encode_query(query: str):
    return _get_embed_model().encode(query, show_progress_bar=False)


def sort_search_results(search_results: list):
    return sorted(search_results, key=lambda x: x['score'], reverse=True)

def calc_node_score_from_search_results(search_results: list) -> float:
    """Harmonic-weighted average of result scores → always in [0, 1].

    Rank 1 gets weight 1, rank 2 gets 1/2, rank 3 gets 1/3, etc.
    Dividing by the total weight normalises the output so a node that returns
    one perfect result and a node that returns ten perfect results both score
    1.0 — the *quality* of results matters, not the count.
    """
    if not search_results:
        return 0.0
    sorted_results = sort_search_results(search_results)
    weights = [1.0 / (i + 1) for i in range(len(sorted_results))]
    total_w = sum(weights)
    return sum(r['score'] * w for r, w in zip(sorted_results, weights)) / total_w


def endpoint_from_str(endpoint_str: str):
    return endpoint_str.split(':')

# NOTE: this function is in case you want to specify remote neighbor nodes.
# example: add to network via node_from_endpoint_str then network.add_edge(my_id, endpoint_str)
def node_from_endpoint_str(endpoint_str: str):
    # Bypass __init__ — remote nodes have no local db and no shared Network.
    # rep falls back to the plain _rep float because the property checks _network first.
    result = Node.__new__(Node)
    result.local = False
    result.db_dir = None
    result.node_id = endpoint_str
    result.remote_endpoint = endpoint_from_str(endpoint_str)
    result._network = None
    result._rep = 0.0
    return result


# debug print search results with truncated bodies because they're too fucking long for my goddamn terminal
def print_search_results(search_results: list):
    for item in search_results:
        copy = deepcopy(item)
        copy['body'] = copy['body'][:30]
        print(json.dumps(copy))
        print('----------------------------')
    print(flush=True) # flush the goddamn buffer because that's not fucking automatic for some reason
                        # Maybe this has to do with my fucking git bash or something


class Node:
    def __init__(self, db_dir: str, network: Network, node_id: str = None):
        self.local = True
        self.db_dir = db_dir
        # node_id defaults to db_dir — they're the same in the standard setup
        self.node_id = node_id if node_id is not None else db_dir
        self.remote_endpoint = None
        self._network = network

    # Reputation is owned by the Network so all routing code sees the same value.
    # Remote nodes (no network) fall back to a plain float stored in _rep.
    @property
    def rep(self) -> float:
        if self._network is None:
            return self._rep
        return self._network.reputation(self.node_id)

    @rep.setter
    def rep(self, value: float):
        if self._network is None:
            self._rep = value
        else:
            self._network.set_reputation(self.node_id, value)

    def search(self, query: str, _visited: set = None) -> list:
        # _visited is threaded through the whole query to prevent cycles in a mesh.
        # Each top-level call from the Flask layer starts with a fresh empty set.
        # is_root distinguishes the top-level call so rep blending and truncation
        # happen exactly once — applying them at every hop would inflate all scores
        # toward 1.0 and make deep results indistinguishable from shallow ones.
        is_root = _visited is None
        if _visited is None:
            _visited = set()
        if self.local:
            results = self._local_search(query, _visited)
            if is_root:
                results = self._blend_rep(results)
                # Deduplicate by URL — the same doc can arrive via multiple mesh
                # paths.  Keep the copy with the highest score.
                seen: dict[str, dict] = {}
                for r in results:
                    url = r.get("url", "")
                    if url not in seen or r["score"] > seen[url]["score"]:
                        seen[url] = r
                results = sort_search_results(list(seen.values()))[:10]
            return results
        else:
            return self._remote_search(query)

    def _blend_rep(self, results: list) -> list:
        """Apply rep blending once at the root — never inside recursive calls."""
        if not results:
            return results
        # Build rep map across every unique node_id present in the result set.
        # Nodes not registered in the network fall back to rep 1.0.
        reps = {
            r["node_id"]: (self._network.reputation(r["node_id"])
                           if r.get("node_id") in self._network else 1.0)
            for r in results if r.get("node_id")
        }
        max_rep = max(reps.values(), default=1.0) or 1.0
        for r in results:
            rep_fraction = reps.get(r.get("node_id", ""), 1.0) / max_rep
            r["score"] = (1.0 - REP_WEIGHT) * r["score"] + REP_WEIGHT * rep_fraction
        return results

    def _local_search(self, query: str, _visited: set) -> list:
        _visited.add(self.node_id)  # mark before querying neighbors so they skip us

        query_vector = _encode_query(query)
        results = [res.to_dict() for res in temp_search(self.db_dir, query, query_vector)]
        for r in results:
            r.setdefault("node_id", self.db_dir)
        print_search_results(results)

        # Exclude already-visited nodes so mesh edges don't cause cycles.
        councilor_nodes = [n for n in self.get_neighbors(NETWORK_TOP_K) if n.node_id not in _visited]

        councilor_nodes_scores = []

        for node in councilor_nodes:
            # No rep blending here — scores stay raw so the root gets clean signal.
            search_results = node.search(query, _visited)
            print_search_results(search_results)
            node_score = calc_node_score_from_search_results(search_results)
            councilor_nodes_scores.append(node_score)

            for item in search_results:
                item.setdefault("node_id", node.db_dir)

            results += search_results

        for node, score in zip(councilor_nodes, councilor_nodes_scores):
            # EMA update — reputation tracks recent performance, doesn't snowball.
            # A node that was good once but degrades will naturally fall back.
            node.rep = (1.0 - REP_DECAY) * node.rep + REP_DECAY * score

            # Standard ACO pheromone deposit with evaporation.
            # τ = (1−ρ)·τ + Δτ  — old trails fade so early-winner edges don't
            # dominate forever and genuinely better nodes can take over.
            if self._network.has_edge(self.node_id, node.node_id):
                old_pheromone = self._network.pheromone(self.node_id, node.node_id)
                self._network.set_pheromone(
                    self.node_id,
                    node.node_id,
                    (1.0 - PHEROMONE_EVAP) * old_pheromone + score
                )

        print('\n\n\n\nbetter together, no matter the weather, now, ladies and gentlemen, welcome all the search results:\n\n\n\n')
        print_search_results(results)

        # Return all collected results — the root's search() handles truncation.
        # Cutting to [:10] here would drop deep nodes' results before they reach
        # the root, making 2nd-hop nodes effectively invisible.
        return results

    def _remote_search(self, query: str):
        response = requests.get("http://{}:{}/api/search"
                                .format(self.remote_endpoint[0], self.remote_endpoint[1]),
        {
            "query": query
        });
        return response.json()

    def get_neighbors(self, top_k: int = -1) -> list:
        neighbor_ids = self._network.neighbors(self.node_id)
        if not neighbor_ids:
            return []

        # ACO probabilistic selection: P(j) ∝ τ(i,j)^α · rep(j)^β
        # This is the core ACO mechanic — high-pheromone, high-rep neighbors are
        # *likely* to be picked but not guaranteed, so undiscovered good nodes
        # still get a chance.  Pure greedy (sort + slice) kills exploration.
        weights = [
            self._network.pheromone(self.node_id, nid) ** ACO_ALPHA *
            max(self._network.reputation(nid), 1e-9) ** ACO_BETA
            for nid in neighbor_ids
        ]

        k = len(neighbor_ids) if top_k < 0 else min(top_k, len(neighbor_ids))

        # Weighted sample without replacement via Efraimidis-Spirakis reservoir keys.
        # Each item draws key = -log(U) / weight; smallest keys win.
        # This produces exact weighted sampling with no numpy dependency.
        keyed = sorted(
            zip(neighbor_ids, weights),
            key=lambda x: -math.log(random.random()) / x[1]
        )
        chosen_ids = [nid for nid, _ in keyed[:k]]
        return [self._network.get_node_obj(nid) for nid in chosen_ids]


# ── Network topology setup ────────────────────────────────────────────────────
# Random mesh: every node has ~3-4 bidirectional edges to other nodes.
# build_random_mesh guarantees connectivity via a spanning-tree pass, so no
# node is ever isolated.  The _visited set in _local_search prevents cycles.

_NODE_IDS = [f"data/dbs/Node_{i}" for i in range(1, 11)]

# edge_probability=0.4 gives ~3-4 neighbours per node on average for 10 nodes.
# seed=42 makes the topology deterministic across restarts.
_network = Network.build_random_mesh(_NODE_IDS, edge_probability=0.4, seed=42)

# Instantiate Node objects and bind them so get_node_obj works during routing.
for _nid in _NODE_IDS:
    _network.attach_node_obj(_nid, Node(db_dir=_nid, network=_network))

myself = _network.get_node_obj(_NODE_IDS[0])




def _make_snippet(body: str, max_len: int = 300) -> str:
    """Return a plain-text excerpt, truncated with ellipsis."""
    text = body.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"


def _results_for_template(raw: list, node_id: str) -> list:
    """Convert node.py result dicts to the shape results.html expects."""
    out = []
    for r in raw:
        out.append({
            "node_id": r.get("node_id", node_id),
            "title":   r.get("title") or "",
            "doc_id":  r.get("url", ""),
            "snippet": _make_snippet(r.get("body", "")),
            "score":   r.get("score", 0.0),
            "url":     r.get("url", ""),
        })
    return out


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["GET"])
def search_page():
    query = (request.args.get("query") or request.args.get("q") or "").strip()
    if not query:
        return render_template("results.html", query="", results=[], error=None)
    try:
        raw = myself.search(query)
        results = _results_for_template(raw, node_id=myself.db_dir)
        return render_template("results.html", query=query, results=results, error=None)
    except Exception as exc:
        return render_template("results.html", query=query, results=[], error=str(exc))

@app.route("/api/search", methods=["GET"])
def api_search():
    query = request.args.get("query")
    return jsonify(myself.search(query))

@app.route("/api/rep", methods=["GET"])
def rep():
    node_endpoint = request.args.get("node_endpoint")
    return jsonify(myself.get_rep(node_endpoint))

@app.route("/api/allreps", methods=["GET"])
def allreps():
    return jsonify(sorted([n.rep for n in myself.get_neighbors()], reverse=True))

if __name__ == "__main__":
    app.run(debug=True) # TODO: change to non-debug mode when in production
