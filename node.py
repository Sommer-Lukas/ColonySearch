from flask import Flask, jsonify, request, render_template
import requests

import json
from copy import deepcopy
from time import sleep
import uuid

import search
temp_search = search.search

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

def calc_node_score_from_search_results(search_results: list):
    sorted_search_results = sort_search_results(search_results)

    total_score = 0

    for i, item in zip(range(len(sorted_search_results)), sorted_search_results):
        # TODO: this shit is just the first thing I thought of, probably we should make this better
        total_score += item['score'] * 1/(i+1)

    return total_score


def endpoint_from_str(endpoint_str: str):
    return endpoint_str.split(':')

# NOTE: this function is in case you want to specify remote neighbor nodes.
# example: myself = Node(db_dir, [ node_from_endpoint_str('http://mynode.com:12345') ])
def node_from_endpoint_str(endpoint_str: str):
    result = Node()
    result.local = False
    result.db_dir = None
    result.remote_endpoint = endpoint_from_str(endpoint_str)
    result.neighbors = None
    result.rep = 0
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


# NOTES ABOUT ROUTING:
# Two parameters: depth and req_id
# Depth is set by the user when making the request, determines how deep the search
# is propagated into the network.
# req_id is generated when the search is first requested and then forwarded to each node that
# is indirectly queried to fulfill the request. If Node A knows Node B and Node B also knows Node A,
# or any number of more complicated loops in the routing graph, then the req_id helps to avoid
# useless querying. Each node has a req_id_cache and if it sees multiple requests come in with the same req_id,
# it will just ignore the second one (or return nothing, or something, idk yet, TODO)
# in order to save time.


def gen_req_id():
    result = uuid.uuid4()
    print(f"DEBUG: gen_req_id called. result: {result}")
    return result


class Node:
    def __init__(self, db_dir: str, neighbors: list, req_id_cache_max_length: int):
        self.local = True
        self.db_dir = db_dir
        self.remote_endpoint = None
        self.neighbors = neighbors
        self.rep = 0

        # cache of recent request IDs. Used for avoiding looping in the request routing.
        # The depth parameter is the other mechanism that avoids this, but the cache makes it
        # better.
        self.req_id_cache = []
        self.req_id_cache_max_length = req_id_cache_max_length

    def _req_id_cache_new_entry(self, new_req_id):
        # remove existing cache entry if it exists
        for i, req_id in zip(range(len(self.req_id_cache)), self.req_id_cache):
            if req_id == new_req_id:
                self.req_id_cache.remove(i)

        # place new cache entry at input end of FIFO
        self.req_id_cache.append(new_req_id)

        # remove oldest cache entry if cache is too large now
        if len(self.req_id_cache) > self.req_id_cache_max_length:
            self.req_id_cache.remove(0)

    def _req_id_cache_query(self, query_req_id):
        return query_req_id in self.req_id_cache

    def search(self, query: str, remaining_depth: int, req_id):
        if req_id == None:
            req_id = gen_req_id()
        else:
            if self._req_id_cache_query(req_id) == True:
                self._req_id_cache_new_entry(req_id) # in this case we STILL put the entry in the cache, because it could come again
                                                     # and we want to refresh its entry
                return [] # do as little as possible to conserve effort and simply return empty result list

        self._req_id_cache_new_entry(req_id)

        if (self.local): return self._local_search(query, remaining_depth, req_id)
        else:            return self._remote_search(query, remaining_depth, req_id)

    def _local_search(self, query: str, remaining_depth: int, req_id):
        query_vector = _encode_query(query)
        results = [res.to_dict() for res in temp_search(self.db_dir, query, query_vector)]
        for r in results:
            r.setdefault("node_id", self.db_dir)
        print_search_results(results)

        if remaining_depth > 0:
            remaining_depth -= 1;

            councilor_nodes = self.get_neighbors(10)

            councilor_nodes_scores = []

            for node in councilor_nodes:
                search_results = node.search(query, remaining_depth, req_id)
                print_search_results(search_results)
                node_score = calc_node_score_from_search_results(search_results)
                councilor_nodes_scores.append(node_score)

                for item in search_results:
                    item.setdefault("node_id", node.db_dir)
                    # TODO: make sure changing item changes search_results.
                    # TODO: think harder about whether we should use the old or new rep, maybe using the new rep has some advantage. It would effectively give a little more
                    # weight to the score of the results themselves. But I suppose a better way to do this would be to have a multiplier/slider for how much to consider
                    # the nodes rep vs how much to consider the quality of the results. The quality of the results can never 100% capture their actual quality,
                    # so historical data in the form of the running node reps is super useful because it completes the picture, at least to a certain degree.
                    # But how much weight to give one vs the other?
                    # Idk.
                    item["score"] += node.rep # add old node rep to its results. The calculated node score shouldn't be considered for the search results it was calculated from

                results += search_results

            for node, score in zip(councilor_nodes, councilor_nodes_scores):
                node.rep += score

        print('\n\n\n\nbetter together, no matter the weather, now, ladies and gentlemen, welcome all the search results:\n\n\n\n')
        print_search_results(results)

        return sort_search_results(results)[:10]

    def _remote_search(self, query: str, remaining_depth: int, req_id):
        response = requests.get("http://{}:{}/api/search"
                                .format(self.remote_endpoint[0], self.remote_endpoint[1]),
        {
            "query": query,
            "remaining_depth": remaining_depth,
            "req_id": req_id
        });
        return response.json()

    def get_neighbors(self, top_k:int=-1):
        return sorted(self.neighbors, key=lambda x: x.rep, reverse=True)[:top_k]


myself = Node("data/dbs/Node_1", [
    Node("data/dbs/Node_2",  [], 8),
    Node("data/dbs/Node_3",  [], 8),
    Node("data/dbs/Node_4",  [], 8),
    Node("data/dbs/Node_5",  [], 8),
    Node("data/dbs/Node_6",  [], 8),
    Node("data/dbs/Node_7",  [], 8),
    Node("data/dbs/Node_8",  [], 8),
    Node("data/dbs/Node_9",  [], 8),
    Node("data/dbs/Node_10", [], 8),
], 8)





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
        # TODO: get depth from configurable value on the search panel in the frontend
        depth = 3
        raw = myself.search(query, depth, None) # req_id arg is always None here because this is brand new request
        results = _results_for_template(raw, node_id=myself.db_dir)
        return render_template("results.html", query=query, results=results, error=None)
    except Exception as exc:
        return render_template("results.html", query=query, results=[], error=str(exc))

@app.route("/api/search", methods=["GET"])
def api_search():
    query  = request.args.get("query")
    depth  = request.args.get("depth")
    req_id = request.args.get("req_id")
    return jsonify(myself.search(query, depth, req_id))

@app.route("/api/rep", methods=["GET"])
def rep():
    node_endpoint = request.args.get("node_endpoint")
    return jsonify(myself.get_rep(node_endpoint))

@app.route("/api/allreps", methods=["GET"])
def allreps():
    return jsonify(sorted([neighbor.rep for neighbor in myself.neighbors], reverse=True))

if __name__ == "__main__":
    app.run(debug=True) # TODO: change to non-debug mode when in production
