# ColonySearch — Agent Context

## Project
Decentralized search engine. Queries propagate through a node graph via
pheromone-based reputation routing (ACO-inspired). No central index.
This is a proof-of-concept MVP for a university module.

## Stack
- Python 3.11+
- Flask — inter-node HTTP
- SQLite FTS5 — local full-text index per node (stdlib, no install)
- NetworkX — graph topology + routing
- Matplotlib / Plotly — visualisation
- pytest — tests

## Run
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python api/server.py
pytest tests/
```

## Architecture
nodes/ Node class, SQLite FTS5 index, local cache
swarm/ ACO routing, pheromone update, NetworkX graph
api/ Flask endpoints: /search /forward /status
data/corpus/ Static .txt files — no web crawling
visualisation/ Query path + network plots
evaluation/ Swarm vs. broadcast baseline benchmark

## Key Rules
- NO central coordinator — every decision is local to a node
- NO web crawling — fixed corpus only
- NO external DB — SQLite per node, in-memory NetworkX graph
- Final result ranking = BM25 score × node reputation (tunable alpha)
- Pheromone update happens AFTER a query resolves, not during indexing
- Touch only files relevant to the current task — no drive-by refactors
- Explain *why* in comments, not *what*
- Run `pytest` before marking any task done