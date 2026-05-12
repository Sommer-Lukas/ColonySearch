from flask import Flask, jsonify, request
import requests

import index.py as index


class Node:
    @staticmethod
    def create_from_endpoint_str(endpoint_str: str):
        result = Node()
        result.local = False
        result.remote_endpoint = endpoint_from_str(endpoint_str)
        result.neighbors = None
        return result

    def __init__(self, neighbor_endpoint_strings: list = []):
        self.local = True
        self.remote_endpoint = None
        self.neighbors = set(Node.create_from_endpoint_str(endpt_str)
                             for endpt_str in neighbor_endpoint_strings)

    def search(self, query: str):
        if (self.local): return self._local_search(query)
        else:            return self._remote_search(query)

    def _local_search(self, query: str):
        results = index.search(query)

        councilor_nodes = self.get_neighbors(10)
        for node in councilor_nodes:
            results.append(node.search(query))

        return index.sort_search_results(results)

    def _remote_search(self, query: str):
        response = requests.get("http://{}:{}/api/search"
                                .format(self.remote_endpoint[0], self.remote_endpoint[1]),
        {
            "query": query
        });
        return response.json()

    def get_neighbors(self, top_n:int=-1):
        return sort(self.neighbors)[:top_n]


myself = Node([
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

@app.route("/api/search", methods=["GET"])
def search():
    query = request.get_json().get("query")
    return jsonify(myself.search(query))

@app.route("/api/rep", methods=["GET"])
def rep():
    data = request.get_json()
    node_endpoint = data.get("node_endpoint")
    return jsonify(myself.get_rep(node_endpoint))

if __name__ == "__main__":
    app.run(debug=True) # TODO: change to non-debug mode when in production
