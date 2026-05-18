from flask import Flask, jsonify, request
import requests

import search
temp_search = search.search


def sort_search_results(search_results: list):
    return sorted(search_results, key=lambda x: x.score, reverse=True)

def calc_node_score_from_search_results(search_results: list):
    sorted_search_results = sort_search_results(search_results)

    total_score = 0

    for i, item in zip(range(len(sorted_search_results)), sorted_search_results):
        # TODO: this shit is just the first thing I thought of, probably we should make this better
        total_score += item['score'] * 1/(i+1)

    return total_score

class Node:
    @staticmethod
    def create_from_endpoint_str(endpoint_str: str):
        result = Node()
        result.local = False
        result.db_dir = None
        result.remote_endpoint = endpoint_from_str(endpoint_str)
        result.neighbors = None
        result.rep = 0
        return result

    def __init__(self, db_dir: str, neighbor_endpoint_strings: list = []):
        self.local = True
        self.db_dir = db_dir
        self.remote_endpoint = None
        self.neighbors = set(Node.create_from_endpoint_str(endpt_str)
                             for endpt_str in neighbor_endpoint_strings)
        self.rep = 0

    def search(self, query: str):
        if (self.local): return self._local_search(query)
        else:            return self._remote_search(query)

    def _local_search(self, query: str):
        # TODO: figure out what to do for the vector argument
        results = temp_search(self.db_dir, query, None)

        councilor_nodes = self.get_neighbors(10)

        councilor_nodes_scores = []

        for node in councilor_nodes:
            search_results = node.search(query)
            node_score = calc_node_score_from_search_results(search_results)
            councilor_nodes_scores.append(node_score)

            for item in search_results:
                # TODO: make sure changing item changes search_results.
                # TODO: think harder about whether we should use the old or new rep, maybe using the new rep has some advantage. It would effectively give a little more
                # weight to the score of the results themselves. But I suppose a better way to do this would be to have a multiplier/slider for how much to consider
                # the nodes rep vs how much to consider the quality of the results. The quality of the results can never 100% capture their actual quality,
                # so historical data in the form of the running node reps is super useful because it completes the picture, at least to a certain degree.
                # But how much weight to give one vs the other?
                # Idk.
                item["score"] += node.rep # add old node rep to its results. The calculated node score shouldn't be considered for the search results it was calculated from

            results.append(search_results)

        for node, score in zip(councilor_nodes, councilor_nodes_scores):
            node.rep += score

        return sort_search_results(results)[:10]

    def _remote_search(self, query: str):
        response = requests.get("http://{}:{}/api/search"
                                .format(self.remote_endpoint[0], self.remote_endpoint[1]),
        {
            "query": query
        });
        return response.json()

    def get_neighbors(self, top_k:int=-1):
        return sorted(self.neighbors, key=lambda x: x.rep)[:top_k]


myself = Node("data/dbs/Node_1", [
    # endpoint strings go here:
    # format: host:port
    # example: 127.0.0.1:80
    # example: localhost:80
    # example: example.com:12345
])





app = Flask(__name__)

@app.route("/")
def index():
    # TODO: return search page here
    return jsonify({
        "message": "Hello from Flask!"
    })

@app.route("/search", methods=["GET"])
def search():
    # TODO: return as an html document with a listing of search results
    query = request.args.get("query")
    return jsonify(myself.search(query))

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
    return jsonify(sorted([neighbor.rep for neighbor in myself.neighbors], reverse=True))

if __name__ == "__main__":
    app.run(debug=True) # TODO: change to non-debug mode when in production
