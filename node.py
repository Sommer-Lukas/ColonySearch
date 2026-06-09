from flask import Flask, jsonify, request, render_template
import requests

import json
import math
import random
from copy import deepcopy
from time import sleep

from pathlib import Path

import search
from network import Network

temp_search = search.search

# ── Reputation / pheromone tuning knobs ──────────────────────────────────────
# How much a node's historical reputation nudges its results vs. raw relevance.
# 0.0 = pure content score, 1.0 = pure reputation.
REP_WEIGHT: float = 0.2

# EMA learning rate for direct-observation reputation updates.  Keeps rep
# bounded in [0, 1] and lets nodes recover or fall.
# 0 = never updates, 1 = full replace each query.
REP_DECAY: float = 0.15

# EMA learning rate for gossiped reputation updates.  Lower than REP_DECAY
# because a direct observation is more trustworthy than hearsay from a peer.
GOSSIP_WEIGHT: float = 0.1

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

# Max hops a query is forwarded before a node searches locally only.
# Prevents unbounded flooding in large meshes.
DEFAULT_TTL: int = 3

# ── Embedding model ───────────────────────────────────────────────────────────
# Loading by local path instead of model name skips all HuggingFace Hub
# network activity (no version checks, no re-downloads).  First run downloads
# once via cache_folder; every run after that loads from the local path only.
_EMBED_MODEL_NAME = "all-mpnet-base-v2"
_EMBED_LOCAL_PATH = Path(__file__).resolve().parent / "data" / "model_cache" / "sentence-transformers_all-mpnet-base-v2"
_embed_model = None
_encode_cache: dict[str, object] = {}

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if _EMBED_LOCAL_PATH.exists():
            _embed_model = SentenceTransformer(str(_EMBED_LOCAL_PATH), device=device)
        else:
            print(f"First run: downloading '{_EMBED_MODEL_NAME}' to {_EMBED_LOCAL_PATH} …", flush=True)
            _embed_model = SentenceTransformer(_EMBED_MODEL_NAME, cache_folder=str(_EMBED_LOCAL_PATH.parent), device=device)
        print(f"Model ready (device={device}).", flush=True)
    return _embed_model

def _encode_query(query: str):
    if query not in _encode_cache:
        _encode_cache[query] = _get_embed_model().encode(query, show_progress_bar=False)
    return _encode_cache[query]


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
    result = Node.__new__(Node)
    result.local = False
    result.db_dir = None
    result.node_id = endpoint_str
    result.remote_endpoint = endpoint_from_str(endpoint_str)
    result._network = None
    result._rep_table = {}
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
        # Each node maintains its own local view of peer reputations.
        # Unknown peers default to 1.0 (benefit of the doubt).
        self._rep_table: dict[str, float] = {}

    def get_rep(self, node_id: str) -> float:
        """Return this node's local reputation estimate for node_id."""
        return self._rep_table.get(node_id, 1.0)

    def update_rep(self, node_id: str, observed_score: float) -> None:
        """EMA-update rep from a direct observation, then gossip the new value to neighbors."""
        old = self._rep_table.get(node_id, 1.0)
        new = max((1.0 - REP_DECAY) * old + REP_DECAY * observed_score, 0.0)
        self._rep_table[node_id] = new
        self._gossip_rep(node_id, new)

    def receive_gossip(self, about_id: str, value: float) -> None:
        """Merge a reputation update gossiped from a neighbor.

        Uses a lower weight than direct observation — hearsay is less
        trustworthy than what this node measured itself.
        """
        old = self._rep_table.get(about_id, 1.0)
        self._rep_table[about_id] = max(
            (1.0 - GOSSIP_WEIGHT) * old + GOSSIP_WEIGHT * value, 0.0
        )

    def _gossip_rep(self, about_id: str, value: float) -> None:
        """Push a reputation update one hop to all direct neighbors.

        One-hop only — cascading gossip would let reputation flood the
        network and make values harder to trace back to an observation.
        """
        if self._network is None:
            return
        for nid in self._network.neighbors(self.node_id):
            neighbor_obj = self._network.get_node_obj(nid)
            if neighbor_obj is None or neighbor_obj.node_id == about_id:
                continue
            if neighbor_obj.local:
                neighbor_obj.receive_gossip(about_id, value)
            else:
                # Remote node — fire-and-forget HTTP POST, ignore failures.
                try:
                    requests.post(
                        "http://{}:{}/api/gossip_rep".format(
                            neighbor_obj.remote_endpoint[0],
                            neighbor_obj.remote_endpoint[1],
                        ),
                        json={"about": about_id, "value": value},
                        timeout=0.5,
                    )
                except Exception:
                    pass

    def search(self, query: str, _visited: set = None, ttl: int = DEFAULT_TTL) -> list:
        # _visited travels with the query to prevent cycles.  The root starts
        # with an empty set; each hop adds itself before forwarding.
        # In remote calls _visited is serialised into the request so cycle state
        # lives in the message, not in shared process memory.
        # is_root distinguishes the top-level call so rep blending and
        # deduplication happen exactly once.
        is_root = _visited is None
        if _visited is None:
            _visited = set()
        if self.local:
            results = self._local_search(query, _visited, ttl)
            if is_root:
                results = self._blend_rep(results)
                # Deduplicate by URL — same doc can arrive via multiple paths.
                seen: dict[str, dict] = {}
                for r in results:
                    url = r.get("url", "")
                    if url not in seen or r["score"] > seen[url]["score"]:
                        seen[url] = r
                results = sort_search_results(list(seen.values()))[:10]
            return results
        else:
            return self._remote_search(query, _visited, ttl)

    def _blend_rep(self, results: list) -> list:
        """Apply rep blending once at the root — never inside recursive calls."""
        if not results:
            return results
        # Build rep map using this node's local reputation table.
        # Unknown nodes default to 1.0 inside get_rep.
        reps = {
            r["node_id"]: self.get_rep(r["node_id"])
            for r in results if r.get("node_id")
        }
        max_rep = max(reps.values(), default=1.0) or 1.0
        for r in results:
            rep_fraction = reps.get(r.get("node_id", ""), 1.0) / max_rep
            r["score"] = (1.0 - REP_WEIGHT) * r["score"] + REP_WEIGHT * rep_fraction
        return results

    def _local_search(self, query: str, _visited: set, ttl: int) -> list:
        _visited.add(self.node_id)  # mark before querying neighbors so they skip us

        query_vector = _encode_query(query)
        results = [res.to_dict() for res in temp_search(self.db_dir, query, query_vector)]
        for r in results:
            r.setdefault("node_id", self.db_dir)
        print_search_results(results)

        if ttl <= 0:
            # TTL exhausted — return own results only, no forwarding.
            return results

        # Exclude already-visited nodes so mesh edges don't cause cycles.
        councilor_nodes = [n for n in self.get_neighbors(NETWORK_TOP_K) if n.node_id not in _visited]

        # Collect the full subtree results per direct neighbour so we can use
        # them for two distinct signals:
        #   - pheromone on self→councilor: path quality = everything that flowed
        #     through that edge, including the councilor's downstream nodes.
        #     This makes ACO reward the route, not just the endpoint.
        #   - rep per node: only that node's own results (node_id == nid),
        #     so content quality and routing quality don't blur together.
        # Both signals are computed in one pass to avoid re-iterating results.
        branch_results: dict[str, list] = {}
        for node in councilor_nodes:
            sub = node.search(query, _visited, ttl - 1)
            print_search_results(sub)
            for item in sub:
                item.setdefault("node_id", node.node_id)
            branch_results[node.node_id] = sub

        node_own_results: dict[str, list] = {}
        for councilor in councilor_nodes:
            sub = branch_results[councilor.node_id]

            # Accumulate per-node own results across all branches for rep.
            for r in sub:
                nid = r.get("node_id", "")
                if nid:
                    node_own_results.setdefault(nid, []).append(r)

            # Pheromone on the direct edge rewards the full path through this
            # councilor.  The recursive call already did the same for the next
            # hop (councilor→its neighbours), so every edge on the traversed
            # tree gets incrementally updated with path-level signal.
            if self._network.has_edge(self.node_id, councilor.node_id):
                path_score = calc_node_score_from_search_results(sub)
                old_ph = self._network.pheromone(self.node_id, councilor.node_id)
                self._network.set_pheromone(
                    self.node_id, councilor.node_id,
                    (1.0 - PHEROMONE_EVAP) * old_ph + path_score,
                )

        for nid, own in node_own_results.items():
            self.update_rep(nid, calc_node_score_from_search_results(own))

        all_peer_results = [r for sub in branch_results.values() for r in sub]

        print('\n\n\n\nbetter together, no matter the weather, now, ladies and gentlemen, welcome all the search results:\n\n\n\n')
        results += all_peer_results
        print_search_results(results)

        # Return all collected results — the root's search() handles truncation.
        return results

    def _remote_search(self, query: str, visited: set, ttl: int) -> list:
        try:
            response = requests.get(
                "http://{}:{}/api/search".format(
                    self.remote_endpoint[0], self.remote_endpoint[1]
                ),
                params={
                    "query": query,
                    "visited": json.dumps(list(visited)),
                    "ttl": ttl,
                },
                timeout=5.0,
            )
            return response.json()
        except Exception:
            return []

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
            max(self.get_rep(nid), 1e-9) ** ACO_BETA
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
    ttl = int(request.args.get("ttl", str(DEFAULT_TTL)))
    try:
        visited = set(json.loads(request.args.get("visited", "[]")))
    except (json.JSONDecodeError, TypeError):
        visited = set()
    return jsonify(myself.search(query, _visited=visited, ttl=ttl))

@app.route("/api/rep", methods=["GET"])
def rep():
    node_endpoint = request.args.get("node_endpoint")
    return jsonify(myself.get_rep(node_endpoint))

@app.route("/api/allreps", methods=["GET"])
def allreps():
    return jsonify(sorted(
        [myself.get_rep(n.node_id) for n in myself.get_neighbors()],
        reverse=True,
    ))

@app.route("/api/gossip_rep", methods=["POST"])
def gossip_rep():
    """Receive a gossiped reputation update from a peer node."""
    data = request.get_json(silent=True) or {}
    about = data.get("about")
    value = data.get("value")
    if about is None or value is None:
        return jsonify({"error": "missing 'about' or 'value'"}), 400
    myself.receive_gossip(about, float(value))
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True) # TODO: change to non-debug mode when in production
