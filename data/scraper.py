"""
ColonySearch corpus scraper — multi-source, multi-topic, cross-topic-aware.

Sources
-------
wikipedia       Wikipedia Action API (clean plain-text extracts + article links)
openalex        OpenAlex Works API (scientific paper abstracts + concepts)
arxiv           arXiv Atom API (paper abstracts)
wikidata        Wikidata entity search → Wikipedia title discovery (no body,
                just expands the wikipedia queue)
html            Generic HTML scraper used for NASA, Python docs, Stack Overflow

Crawl strategy
--------------
Each topic has its own round-robin deque.  The main loop takes one item from
each topic in turn so the corpus stays balanced regardless of --max.

After saving a document, the body is scored against every other topic's
keyword list.  If hits exceed CROSS_TOPIC_THRESHOLD, one unused seed from
the related topic is injected into that topic's queue — this is what drives
cross-topic relation discovery.

Output
------
One JSON file per document: { url, title, body, links, source, topic }

Usage
-----
    python data/scraper.py                        # defaults: depth 1, 24 pages
    python data/scraper.py --max 40 --depth 2
    python data/scraper.py --seeds data/seeds.txt # flat URL list → topic "custom"
    python data/scraper.py --seed 42              # reproducible shuffle
"""

import argparse
import json
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────────────
# Topic cluster definitions
# ──────────────────────────────────────────────────────────────────────────────

# Seed format: (source_type, value)
#   wikipedia / wikidata  →  article title or concept string
#   openalex / arxiv      →  free-text search query
#   html / stackoverflow  →  absolute URL

TOPIC_CLUSTERS: dict[str, dict] = {

    # ── Climate change ────────────────────────────────────────────────────────
    "climate_change": {
        "keywords": [
            "greenhouse gas", "carbon dioxide", "IPCC", "sea level rise",
            "net zero", "fossil fuel", "tipping point", "decarbonisation",
            "methane", "Keeling Curve", "permafrost", "albedo",
        ],
        "seeds": [
            ("wikipedia",        "Keeling_Curve"),
            ("wikipedia",        "Arctic_sea_ice_decline"),
            ("wikipedia",        "Climate_tipping_points"),
            ("wikipedia",        "IPCC_Sixth_Assessment_Report"),
            ("wikipedia",        "Carbon_budget"),
            ("openalex",         "Arctic sea ice decline feedback loop"),
            ("openalex",         "permafrost thaw methane emission"),
            ("arxiv",            "climate tipping point abrupt transition"),
            ("arxiv",            "carbon budget net zero emission pathway"),
            ("semantic_scholar", "climate change tipping point cascade"),
            ("wikidata",         "greenhouse gas"),
            ("html",             "https://climate.nasa.gov/vital-signs/carbon-dioxide/"),
            ("html",             "https://climate.nasa.gov/vital-signs/sea-level/"),
            ("html",             "https://climate.nasa.gov/vital-signs/arctic-sea-ice/"),
        ],
    },

    # ── Sustainable materials ─────────────────────────────────────────────────
    "sustainable_materials": {
        "keywords": [
            "bioplastic", "polyhydroxyalkanoate", "mycelium", "lignin",
            "biodegradable polymer", "life cycle assessment", "bio-based",
            "compostable", "cellulose composite", "upcycling", "cradle to cradle",
        ],
        "seeds": [
            ("wikipedia",        "Polyhydroxyalkanoates"),
            ("wikipedia",        "Mycelium_materials"),
            ("wikipedia",        "Lignin"),
            ("wikipedia",        "Life-cycle_assessment"),
            ("wikipedia",        "Bioremediation"),
            ("openalex",         "polyhydroxyalkanoate PHA bioplastic production"),
            ("openalex",         "mycelium biocomposite mechanical properties"),
            ("arxiv",            "cellulose nanofiber biodegradable composite"),
            ("arxiv",            "lignin valorisation bio-based material"),
            ("semantic_scholar", "biodegradable polymer cradle to cradle lifecycle"),
            ("wikidata",         "biopolymer"),
        ],
    },

    # ── Space ─────────────────────────────────────────────────────────────────
    "space": {
        "keywords": [
            "Artemis", "Perseverance rover", "JWST", "exoplanet", "Starship",
            "launch vehicle", "gravitational wave", "neutron star", "lunar gateway",
            "orbital mechanics", "reusable rocket", "SpaceX",
        ],
        "seeds": [
            ("wikipedia",        "Artemis_program"),
            ("wikipedia",        "Perseverance_(rover)"),
            ("wikipedia",        "James_Webb_Space_Telescope"),
            ("wikipedia",        "Gravitational_wave"),
            ("wikipedia",        "Lunar_Gateway"),
            ("wikipedia",        "SpaceX_Starship"),
            ("openalex",         "JWST first light infrared galaxy observation"),
            ("arxiv",            "exoplanet biosignature atmosphere transmission spectroscopy"),
            ("arxiv",            "gravitational wave neutron star merger LIGO"),
            ("semantic_scholar", "reusable launch vehicle propulsion landing"),
            ("wikidata",         "Artemis program"),
            ("html",             "https://www.nasa.gov/missions/artemis/"),
            ("html",             "https://science.nasa.gov/missions/webb/"),
        ],
    },

    # ── Programming ───────────────────────────────────────────────────────────
    "programming": {
        "keywords": [
            "Python", "async await", "garbage collector", "type hint", "GIL",
            "compiler", "bytecode", "hash table", "decorator", "generator",
            "scikit-learn", "cross-validation", "random forest", "CPython",
        ],
        "seeds": [
            ("wikipedia",        "CPython"),
            ("wikipedia",        "Async/await"),
            ("wikipedia",        "Hash_table"),
            ("wikipedia",        "Scikit-learn"),
            ("stackoverflow",    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python"),
            ("stackoverflow",    "https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python"),
            ("stackoverflow",    "https://stackoverflow.com/questions/1024559/when-to-use-os-name-posix-vs-os-name-nt"),
            ("html",             "https://docs.python.org/3/reference/datamodel.html"),
            ("html",             "https://docs.python.org/3/library/asyncio-task.html"),
            ("devto",            "python"),
            ("devto",            "machinelearning"),
            ("github",           "scikit-learn/scikit-learn"),
            ("github",           "psf/requests"),
            ("github",           "tiangolo/fastapi"),
            # scikit-learn — supervised
            ("scikit",           "https://scikit-learn.org/stable/modules/ensemble.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/cross_validation.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/linear_model.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/svm.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/tree.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/neighbors.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/naive_bayes.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/neural_networks_supervised.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/gaussian_process.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/isotonic.html"),
            # scikit-learn — unsupervised & preprocessing
            ("scikit",           "https://scikit-learn.org/stable/modules/clustering.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/decomposition.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/preprocessing.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/feature_selection.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/manifold.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/mixture.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/outlier_detection.html"),
            # scikit-learn — model selection & inspection
            ("scikit",           "https://scikit-learn.org/stable/modules/model_evaluation.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/pipeline.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/grid_search.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/learning_curve.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/permutation_importance.html"),
            ("openalex",         "Python programming language performance optimization"),
            ("arxiv",            "scikit-learn machine learning Python benchmark"),
        ],
    },

    # ── Machine learning algorithms ───────────────────────────────────────────
    # World models, transformers, diffusion, RL, GNNs, mechanistic interpretability.
    # arXiv + Semantic Scholar are the primary sources — most content lives there.
    "ml_algorithms": {
        "keywords": [
            "world model", "transformer", "diffusion model", "reinforcement learning",
            "attention mechanism", "graph neural network", "variational autoencoder",
            "contrastive learning", "model-based RL", "sparse autoencoder",
            "state space model", "Mamba", "RLHF", "latent space",
        ],
        "seeds": [
            ("wikipedia",        "Transformer_(deep_learning_architecture)"),
            ("wikipedia",        "Diffusion_model"),
            ("wikipedia",        "Variational_autoencoder"),
            ("wikipedia",        "Reinforcement_learning"),
            ("wikipedia",        "Graph_neural_network"),
            ("wikipedia",        "Attention_(machine_learning)"),
            ("wikipedia",        "Generative_adversarial_network"),
            # arXiv — algorithm-level papers
            ("arxiv",            "world model latent space model-based reinforcement learning"),
            ("arxiv",            "dreamer world model imagination planning"),
            ("arxiv",            "diffusion model score matching denoising probabilistic"),
            ("arxiv",            "sparse autoencoder interpretability mechanistic superposition"),
            ("arxiv",            "state space model Mamba selective scan sequence"),
            ("arxiv",            "contrastive learning self-supervised representation SimCLR"),
            ("arxiv",            "graph neural network message passing node classification"),
            ("arxiv",            "RLHF reward model human feedback alignment PPO"),
            ("arxiv",            "neural scaling law emergent capability large model"),
            ("arxiv",            "mixture of experts sparse gating language model"),
            ("arxiv",            "flow matching rectified flow generative model"),
            # Semantic Scholar — survey papers with citation graphs
            ("semantic_scholar", "world model model-based reinforcement learning survey"),
            ("semantic_scholar", "transformer self-attention survey vision language"),
            ("semantic_scholar", "diffusion generative model image synthesis"),
            ("semantic_scholar", "mechanistic interpretability circuits features neural network"),
            # GitHub — key framework READMEs
            ("github",           "google-deepmind/dreamerv3"),
            ("github",           "huggingface/diffusers"),
            ("github",           "state-spaces/mamba"),
            ("github",           "openai/gym"),
            # Stack Overflow
            ("stackoverflow",    "https://stackoverflow.com/questions/55243483/what-is-the-difference-between-model-free-and-model-based-reinforcement-learning"),
            ("stackoverflow",    "https://stackoverflow.com/questions/65703260/what-is-the-intuition-behind-the-attention-mechanism-in-neural-networks"),
            ("openalex",         "world model model-based reinforcement learning survey"),
            ("openalex",         "diffusion probabilistic model generative image"),
        ],
    },

    # ── Football / Soccer ─────────────────────────────────────────────────────
    "football": {
        "keywords": [
            "association football", "pressing", "expected goals", "tiki-taka",
            "gegenpressing", "UEFA Champions League", "FIFA World Cup",
            "offside trap", "high press", "false nine", "ball possession",
            "xG", "transfer market",
        ],
        "seeds": [
            ("wikipedia",        "Pressing_(association_football)"),
            ("wikipedia",        "Expected_goals"),
            ("wikipedia",        "Tiki-taka"),
            ("wikipedia",        "UEFA_Champions_League"),
            ("wikipedia",        "History_of_association_football"),
            ("wikipedia",        "Association_football_tactics_and_skills"),
            ("openalex",         "expected goals football match prediction model"),
            ("openalex",         "pressing intensity football tactical analysis"),
            ("arxiv",            "football soccer match outcome prediction machine learning"),
            ("arxiv",            "player tracking event data football performance"),
            ("semantic_scholar", "expected goals xG football analytics sports science"),
            ("wikidata",         "association football tactic"),
            ("html",             "https://www.bbc.com/sport/football/european"),
            ("html",             "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
        ],
    },

    # ── Human biology & medicine ──────────────────────────────────────────────
    "medicine": {
        "keywords": [
            "CRISPR", "mRNA vaccine", "microbiome", "immunotherapy", "oncology",
            "stem cell", "antibiotic resistance", "gene editing", "proteomics",
            "clinical trial", "epigenetics", "pathogen",
        ],
        "seeds": [
            ("wikipedia",        "CRISPR"),
            ("wikipedia",        "MRNA_vaccine"),
            ("wikipedia",        "Gut_microbiota"),
            ("wikipedia",        "Cancer_immunotherapy"),
            ("wikipedia",        "Antibiotic_resistance"),
            ("openalex",         "CRISPR Cas9 gene editing therapeutic application"),
            ("openalex",         "mRNA vaccine immunogenicity clinical trial"),
            ("arxiv",            "microbiome gut bacteria disease machine learning"),
            ("arxiv",            "protein structure prediction AlphaFold drug discovery"),
            ("semantic_scholar", "CRISPR base editing prime editing therapeutic"),
            ("wikidata",         "gene therapy"),
        ],
    },

    # ── Quantum computing ─────────────────────────────────────────────────────
    # Qubits, quantum gates, error correction, variational algorithms, NISQ.
    # Sources: Wikipedia, arXiv, Semantic Scholar, GitHub (Qiskit).
    "quantum_computing": {
        "keywords": [
            "qubit", "quantum gate", "superposition", "entanglement",
            "quantum error correction", "variational quantum eigensolver",
            "NISQ", "quantum supremacy", "Shor's algorithm", "quantum annealing",
            "topological qubit", "quantum circuit",
        ],
        "seeds": [
            ("wikipedia",        "Quantum_computing"),
            ("wikipedia",        "Qubit"),
            ("wikipedia",        "Shor%27s_algorithm"),
            ("wikipedia",        "Quantum_error_correction"),
            ("wikipedia",        "Variational_quantum_eigensolver"),
            ("wikipedia",        "Quantum_supremacy"),
            ("arxiv",            "variational quantum eigensolver NISQ hybrid algorithm"),
            ("arxiv",            "quantum error correction surface code fault tolerant"),
            ("arxiv",            "quantum machine learning kernel classification"),
            ("arxiv",            "topological qubit anyons quantum computation"),
            ("semantic_scholar", "NISQ quantum advantage near-term algorithm"),
            ("semantic_scholar", "quantum error correction logical qubit threshold"),
            ("openalex",         "quantum computing algorithm optimization survey"),
            ("github",           "Qiskit/qiskit"),
            ("github",           "quantumlib/Cirq"),
        ],
    },

    # ── Neuroscience ──────────────────────────────────────────────────────────
    # Neural circuits, connectomics, synaptic plasticity, brain-computer interfaces.
    # Sources: Wikipedia, arXiv, Semantic Scholar, OpenAlex.
    "neuroscience": {
        "keywords": [
            "neuron", "synapse", "connectome", "synaptic plasticity",
            "default mode network", "hippocampus", "dopamine", "cortex",
            "action potential", "brain-computer interface", "neuroplasticity",
            "predictive coding", "free energy principle",
        ],
        "seeds": [
            ("wikipedia",        "Neuron"),
            ("wikipedia",        "Connectome"),
            ("wikipedia",        "Synaptic_plasticity"),
            ("wikipedia",        "Default_mode_network"),
            ("wikipedia",        "Predictive_coding"),
            ("wikipedia",        "Brain%E2%80%93computer_interface"),
            ("arxiv",            "predictive coding free energy principle brain"),
            ("arxiv",            "connectome neural circuit C elegans mapping"),
            ("arxiv",            "brain computer interface EEG decoding motor"),
            ("arxiv",            "dopamine reward prediction error reinforcement"),
            ("semantic_scholar", "synaptic plasticity long-term potentiation memory"),
            ("semantic_scholar", "free energy principle active inference Karl Friston"),
            ("openalex",         "brain computer interface neural decoding prosthetic"),
            ("openalex",         "hippocampus memory consolidation spatial navigation"),
        ],
    },

    # ── Economics & game theory ───────────────────────────────────────────────
    # Mechanism design, behavioral economics, market microstructure, auctions.
    # Sources: Wikipedia, arXiv, Semantic Scholar, OpenAlex.
    "economics": {
        "keywords": [
            "game theory", "Nash equilibrium", "mechanism design", "auction",
            "behavioral economics", "nudge", "market microstructure",
            "monetary policy", "inflation", "Pareto efficiency",
            "public goods", "externality", "information asymmetry",
        ],
        "seeds": [
            ("wikipedia",        "Game_theory"),
            ("wikipedia",        "Nash_equilibrium"),
            ("wikipedia",        "Mechanism_design"),
            ("wikipedia",        "Behavioral_economics"),
            ("wikipedia",        "Auction_theory"),
            ("wikipedia",        "Information_asymmetry"),
            ("arxiv",            "mechanism design algorithmic game theory auction"),
            ("arxiv",            "behavioral economics nudge choice architecture"),
            ("arxiv",            "market microstructure order book price impact"),
            ("semantic_scholar", "Nash equilibrium evolutionary game theory"),
            ("semantic_scholar", "auction design revenue equivalence Vickrey"),
            ("openalex",         "behavioral economics prospect theory Kahneman"),
            ("openalex",         "monetary policy inflation central bank"),
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Company / NGO / policy clusters  (used by --companies)
# Topics match TOPIC_CLUSTERS so cross-topic detection still works.
# Sticks to sources known to serve plain HTML (no JS rendering required).
# ──────────────────────────────────────────────────────────────────────────────

COMPANY_CLUSTERS: dict[str, dict] = {

    # Climate & sustainability content from WEF, UN, Greenpeace, WWF, EU
    "climate_change": {
        "keywords": TOPIC_CLUSTERS["climate_change"]["keywords"],
        "seeds": [
            ("html", "https://www.weforum.org/agenda/archive/climate-change/"),
            ("html", "https://www.unep.org/topics/climate-action"),
            ("html", "https://www.greenpeace.org/international/explore-topics/climate/"),
            ("html", "https://www.worldwildlife.org/initiatives/fighting-climate-change-with-natural-solutions"),
            ("html", "https://www.oecd.org/en/topics/climate-action.html"),
            ("html", "https://ec.europa.eu/clima/en"),
            ("html", "https://www.imf.org/en/Topics/climate-change"),
            ("html", "https://www.iea.org/topics/clean-energy-transitions"),
        ],
    },

    # Sustainable materials content from Ellen MacArthur Foundation, UNEP, EU
    "sustainable_materials": {
        "keywords": TOPIC_CLUSTERS["sustainable_materials"]["keywords"],
        "seeds": [
            ("html", "https://ellenmacarthurfoundation.org/topics/circular-economy/overview"),
            ("html", "https://www.unep.org/topics/chemicals-waste/plastics"),
            ("html", "https://ec.europa.eu/environment/topics/plastics_en"),
            ("html", "https://www.weforum.org/agenda/archive/circular-economy/"),
        ],
    },

    # Space content from NASA, ESA, SpaceX (news pages)
    "space": {
        "keywords": TOPIC_CLUSTERS["space"]["keywords"],
        "seeds": [
            ("html", "https://www.nasa.gov/news/"),
            ("html", "https://www.esa.int/Newsroom"),
            ("html", "https://blogs.nasa.gov/artemis/"),
            ("html", "https://www.nasa.gov/humans-in-space/commercial-space/"),
        ],
    },

    # AI & ML content from Microsoft Research, Google DeepMind, Accenture, WEF
    "ml_algorithms": {
        "keywords": TOPIC_CLUSTERS["ml_algorithms"]["keywords"],
        "seeds": [
            ("html", "https://www.microsoft.com/en-us/research/blog/"),
            ("html", "https://deepmind.google/discover/blog/"),
            ("html", "https://www.weforum.org/agenda/archive/artificial-intelligence/"),
            ("html", "https://www.accenture.com/us-en/insights/artificial-intelligence"),
            ("html", "https://research.google/blog/"),
            ("html", "https://openai.com/news/"),
            ("html", "https://www.anthropic.com/news"),
        ],
    },

    # Economics content from IMF, OECD, WEF, World Bank
    "economics": {
        "keywords": TOPIC_CLUSTERS["economics"]["keywords"],
        "seeds": [
            ("html", "https://www.imf.org/en/Publications/WEO"),
            ("html", "https://www.oecd.org/en/topics/economy.html"),
            ("html", "https://www.weforum.org/agenda/archive/economics/"),
            ("html", "https://www.worldbank.org/en/topic/macroeconomics"),
            ("html", "https://www.mckinsey.com/mgi/our-research"),
        ],
    },

    # Medicine & health content from WHO, NIH, WEF health
    "medicine": {
        "keywords": TOPIC_CLUSTERS["medicine"]["keywords"],
        "seeds": [
            ("html", "https://www.who.int/news-room/fact-sheets"),
            ("html", "https://www.nih.gov/news-events/news-releases"),
            ("html", "https://www.weforum.org/agenda/archive/health-and-healthcare/"),
            ("html", "https://www.accenture.com/us-en/insights/health"),
        ],
    },

    # Football analytics from BBC Sport, UEFA, FBref
    "football": {
        "keywords": TOPIC_CLUSTERS["football"]["keywords"],
        "seeds": [
            ("html", "https://www.bbc.com/sport/football"),
            ("html", "https://www.uefa.com/uefachampionsleague/news/"),
            ("html", "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
            ("html", "https://www.fifaindex.com/"),
        ],
    },
}

HEADERS = {
    "User-Agent": (
        "ColonySearch/1.0 (university distributed search engine research; "
        "non-commercial; github.com/example/colonysearch)"
    )
}

MAX_BODY_CHARS        = 8_000
MAX_LINKS_PER_PAGE    = 30
CROSS_TOPIC_THRESHOLD = 3   # keyword hits in body before we inject a cross-topic seed
CROSS_TOPIC_INJECT    = 1   # seeds injected per detected relation per document

# Hosts whose child links we will follow during HTML depth crawl
CRAWLABLE_HTML_HOSTS = frozenset({
    # encyclopaedic
    "en.wikipedia.org", "simple.wikipedia.org",
    # technical docs
    "docs.python.org", "developer.mozilla.org", "scikit-learn.org",
    # space agencies
    "climate.nasa.gov", "www.nasa.gov", "blogs.nasa.gov", "science.nasa.gov",
    "www.esa.int",
    # policy / NGO
    "www.weforum.org", "www.unep.org", "www.oecd.org",
    "www.imf.org", "www.worldbank.org", "ec.europa.eu",
    "ellenmacarthurfoundation.org", "www.iea.org",
    # health
    "www.who.int", "www.nih.gov",
    # sports
    "www.bbc.com", "www.uefa.com", "fbref.com",
    # AI / tech company blogs
    "www.microsoft.com", "research.google", "deepmind.google",
})

_WIKI_SKIP   = re.compile(
    r"/(Special|Talk|User|File|Help|Category|Wikipedia|Portal|Template):"
)
_SO_SKIP     = re.compile(r"/users/|/tags/|/questions/tagged/|/jobs/|/tour")
_PY_SKIP     = re.compile(r"genindex|py-modindex|search\.html")
# Skip auto-generated API reference pages — too many, too granular for FTS
_SCIKIT_SKIP = re.compile(r"/stable/api/|generated/sklearn\.|#examples|_downloads")

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get(url: str, _retries: int = 3, **kwargs) -> requests.Response | None:
    for attempt in range(_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=14, **kwargs)
            if r.status_code == 429:
                wait = 15 * (2 ** attempt)   # 15 s → 30 s → 60 s
                print(f"  [rate-limit 429] sleeping {wait}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  [skip] {url!r:.80}: {e}")
            return None
    print(f"  [skip] {url!r:.80}: gave up after {_retries} attempts")
    return None


def _doc(*, url: str, title: str, body: str, links: list[str],
         source: str, topic: str) -> dict:
    return {"url": url, "title": title, "body": body,
            "links": links, "source": source, "topic": topic}


def _wiki_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"


def _title_from_wiki_url(url: str) -> str:
    return urlparse(url).path.split("/wiki/")[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Wikipedia Action API
# ──────────────────────────────────────────────────────────────────────────────

def fetch_wikipedia(title: str, topic: str) -> list[dict]:
    r = _get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "format": "json",
            "titles": title.replace("_", " "),
            "prop": "extracts|links",
            "exlimit": 1, "explaintext": True, "exsectionformat": "plain",
            "pllimit": MAX_LINKS_PER_PAGE, "plnamespace": 0,
        },
    )
    if not r:
        return []

    docs = []
    for page_id, page in r.json().get("query", {}).get("pages", {}).items():
        if page_id == "-1" or not page.get("extract"):
            continue
        url   = _wiki_url(page["title"])
        body  = page["extract"][:MAX_BODY_CHARS]
        links = [_wiki_url(lk["title"]) for lk in page.get("links", [])]
        docs.append(_doc(url=url, title=page["title"], body=body,
                         links=links, source="wikipedia", topic=topic))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Wikidata entity search → Wikipedia title discovery
# ──────────────────────────────────────────────────────────────────────────────

def fetch_wikidata(concept: str, topic: str) -> list[dict]:
    """
    Searches Wikidata for entities matching concept and returns lightweight
    placeholder docs pointing at their English Wikipedia articles.
    These are not saved directly — the crawler re-queues them as wikipedia seeds.
    """
    search_r = _get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbsearchentities", "search": concept,
            "language": "en", "format": "json", "limit": 5,
        },
    )
    if not search_r:
        return []

    ids = [item["id"] for item in search_r.json().get("search", [])]
    if not ids:
        return []

    ents_r = _get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities", "ids": "|".join(ids),
            "props": "sitelinks", "format": "json",
        },
    )
    if not ents_r:
        return []

    docs = []
    for entity in ents_r.json().get("entities", {}).values():
        en_wiki_title = entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
        if en_wiki_title:
            docs.append(_doc(
                url=_wiki_url(en_wiki_title), title=en_wiki_title,
                body="", links=[], source="wikidata_ref", topic=topic,
            ))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# OpenAlex Works API
# ──────────────────────────────────────────────────────────────────────────────

def _rebuild_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as inverted index; reconstruct word order."""
    if not inv:
        return ""
    pairs = [(pos, word) for word, positions in inv.items() for pos in positions]
    return " ".join(w for _, w in sorted(pairs))


def fetch_openalex(query: str, topic: str) -> list[dict]:
    r = _get(
        "https://api.openalex.org/works",
        params={
            "search": query, "per-page": 5,
            "select": "id,title,abstract_inverted_index,concepts,doi",
            "mailto": "research@example.com",
        },
    )
    if not r:
        return []

    docs = []
    for work in r.json().get("results", []):
        title = work.get("title") or ""
        abstract = _rebuild_abstract(work.get("abstract_inverted_index"))
        # Append concept labels so FTS picks up topic terms even in short abstracts
        concept_text = " ".join(c["display_name"] for c in work.get("concepts", []))
        body = f"{abstract} {concept_text}".strip()[:MAX_BODY_CHARS]

        url = work.get("doi") or work.get("id", "")
        if not url or not title or not body:
            continue

        docs.append(_doc(url=url, title=title, body=body,
                         links=[], source="openalex", topic=topic))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# arXiv Atom API
# ──────────────────────────────────────────────────────────────────────────────

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv(query: str, topic: str) -> list[dict]:
    r = _get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": 5},
    )
    if not r:
        return []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []

    docs = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        def _text(tag: str) -> str:
            el = entry.find(tag, _ATOM_NS)
            return el.text.strip() if el is not None and el.text else ""

        title   = _text("atom:title")
        body    = _text("atom:summary")[:MAX_BODY_CHARS]
        url     = _text("atom:id").replace("http://", "https://")
        links   = [
            a.get("href", "")
            for a in entry.findall("atom:link", _ATOM_NS)
            if a.get("href")
        ]

        if url and title and body:
            docs.append(_doc(url=url, title=title, body=body,
                             links=links, source="arxiv", topic=topic))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Scholar API  (free, no auth; returns abstracts + AI-generated TLDRs)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_semantic_scholar(query: str, topic: str) -> list[dict]:
    r = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query, "limit": 6,
            "fields": "title,abstract,tldr,externalIds,openAccessPdf",
        },
    )
    if not r:
        return []

    docs = []
    for paper in r.json().get("data", []):
        title    = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        tldr     = (paper.get("tldr") or {}).get("text") or ""
        # Combine abstract + TLDR so short papers still have searchable text
        body     = f"{abstract}\n\nTL;DR: {tldr}".strip()[:MAX_BODY_CHARS] if tldr else abstract[:MAX_BODY_CHARS]

        ext = paper.get("externalIds") or {}
        doi = ext.get("DOI")
        url = f"https://doi.org/{doi}" if doi else \
              (paper.get("openAccessPdf") or {}).get("url") or ""

        if title and body and url:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="semantic_scholar", topic=topic))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# DEV.to public API  (programming articles, no auth needed)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_devto(tag: str, topic: str) -> list[dict]:
    r = _get(
        "https://dev.to/api/articles",
        params={"tag": tag, "per_page": 6, "top": 1},
    )
    if not r:
        return []

    docs = []
    for article in r.json():
        url   = article.get("url") or ""
        title = article.get("title") or ""
        # body_html is present; strip tags with BS4
        raw   = BeautifulSoup(article.get("body_html") or "", "html.parser").get_text(" ", strip=True)
        body  = " ".join(raw.split())[:MAX_BODY_CHARS]
        if url and title and body:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="devto", topic=topic))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# GitHub README scraper  (raw.githubusercontent.com — no JS, no auth needed)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_github(repo: str, topic: str) -> list[dict]:
    # Try main then master branch
    for branch in ("main", "master"):
        raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
        r = _get(raw_url)
        if r and r.status_code == 200:
            break
    else:
        return []

    # Strip markdown syntax for clean FTS body
    text = re.sub(r"```.*?```", " ", r.text, flags=re.DOTALL)  # code blocks
    text = re.sub(r"`[^`]+`", " ", text)                        # inline code
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)                # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)        # links → text
    text = re.sub(r"#+\s*", "", text)                            # headings
    text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", text)  # bold/italic
    body = " ".join(text.split())[:MAX_BODY_CHARS]

    gh_url = f"https://github.com/{repo}"
    title  = repo.split("/")[-1].replace("-", " ").replace("_", " ").title()
    return [_doc(url=gh_url, title=title, body=body, links=[], source="github", topic=topic)]


# ──────────────────────────────────────────────────────────────────────────────
# Generic HTML scraper (NASA, Python docs, Stack Overflow)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_html(url: str, topic: str, source: str) -> list[dict]:
    r = _get(url)
    if not r or "text/html" not in r.headers.get("Content-Type", ""):
        return []

    soup  = BeautifulSoup(r.text, "html.parser")
    host  = urlparse(url).netloc
    title = (soup.find("title") or soup.new_tag("x")).get_text(strip=True) or url

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    content = _pick_content(soup, host)
    body    = " ".join(content.get_text(" ", strip=True).split())[:MAX_BODY_CHARS] if content else ""
    links   = _extract_html_links(soup, url, host)

    return [_doc(url=url, title=title, body=body, links=links, source=source, topic=topic)]


def _pick_content(soup: BeautifulSoup, host: str):
    if "wikipedia.org"     in host: return soup.find(id="mw-content-text")
    if "stackoverflow.com" in host: return soup.find(id="question") or soup.find(id="answers")
    if "docs.python.org"   in host: return soup.find("div", role="main") or soup.find(class_="body")
    if "scikit-learn.org"  in host: return soup.find("div", role="main") or soup.find(class_="bd-article")
    if "nasa.gov"          in host: return soup.find("main") or soup.find(class_="page-content")
    return soup.find("main") or soup.find("article") or soup.body


def _extract_html_links(soup: BeautifulSoup, base_url: str, host: str) -> list[str]:
    base = urlparse(base_url)
    skip = (
        _SO_SKIP     if "stackoverflow.com" in host else
        _PY_SKIP     if "python.org"        in host else
        _SCIKIT_SKIP if "scikit-learn.org"  in host else
        _WIKI_SKIP   if "wikipedia.org"     in host else
        None
    )
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        href  = urljoin(base_url, a["href"])
        p     = urlparse(href)
        clean = p._replace(fragment="").geturl()
        if p.netloc != base.netloc or clean in seen or clean == base_url:
            continue
        if skip and skip.search(clean):
            continue
        seen.add(clean)
        links.append(clean)
        if len(links) >= MAX_LINKS_PER_PAGE:
            break
    return links


# ──────────────────────────────────────────────────────────────────────────────
# Source dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def dispatch(source: str, value: str, topic: str) -> list[dict]:
    match source:
        case "wikipedia":        return fetch_wikipedia(value, topic)
        case "wikidata":         return fetch_wikidata(value, topic)
        case "openalex":         return fetch_openalex(value, topic)
        case "arxiv":            return fetch_arxiv(value, topic)
        case "semantic_scholar": return fetch_semantic_scholar(value, topic)
        case "devto":            return fetch_devto(value, topic)
        case "github":           return fetch_github(value, topic)
        case "scikit":           return fetch_html(value, topic, "scikit")
        case "html":             return fetch_html(value, topic, "html")
        case "stackoverflow":    return fetch_html(value, topic, "stackoverflow")
        case _:                  return []


# ──────────────────────────────────────────────────────────────────────────────
# Cross-topic relation scoring
# ──────────────────────────────────────────────────────────────────────────────

def detect_related_topics(body: str, current_topic: str) -> list[str]:
    """
    Returns topics whose keywords appear >= CROSS_TOPIC_THRESHOLD times in body,
    excluding the document's own topic.
    """
    body_lower = body.lower()
    related = []
    for topic, conf in TOPIC_CLUSTERS.items():
        if topic == current_topic:
            continue
        hits = sum(1 for kw in conf["keywords"] if kw.lower() in body_lower)
        if hits >= CROSS_TOPIC_THRESHOLD:
            related.append(topic)
    return related


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _url_to_filename(url: str, topic: str) -> str:
    p    = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "_", p.path.lower()).strip("_") or "index"
    host = re.sub(r"[^a-z0-9]+", "_", p.netloc)
    # topic prefix keeps each cluster in its own namespace → no cross-topic overwrites
    return f"{topic}__{host}__{slug[:50]}.json"


def save(data: dict, out_dir: Path, overwrite: bool = False) -> Path | None:
    """Returns None (and skips) if the file already exists and overwrite is False."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _url_to_filename(data["url"], data["topic"])
    if path.exists() and not overwrite:
        return None
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Main crawl loop
# ──────────────────────────────────────────────────────────────────────────────

def crawl(
    clusters:   dict[str, dict],
    depth:      int,
    max_pages:  int,
    out_dir:    Path,
    delay:      float,
    overwrite:  bool = False,
) -> None:
    # One deque per topic; shuffled so repeated runs explore different seeds first
    queues: dict[str, deque] = {}
    for topic, conf in clusters.items():
        seeds = conf["seeds"][:]
        random.shuffle(seeds)
        queues[topic] = deque((src, val, 0) for src, val in seeds)

    # Spare seed pool for cross-topic injection (each topic keeps its full list)
    spare_pools: dict[str, list] = {t: c["seeds"][:] for t, c in clusters.items()}

    # Visited keys: f"{source}::{value}" for API seeds, URL for HTML seeds
    visited: set[str] = set()
    saved   = 0
    skipped = 0   # already-exists counter — printed once at the end
    topics  = list(queues.keys())
    # Track which (doc_topic, related_topic) cross-links have already been injected
    # to avoid flooding a topic queue from one very cross-topic-rich page
    injected_pairs: set[tuple[str, str]] = set()

    while saved < max_pages:
        made_progress = False

        for topic in topics:
            q = queues[topic]
            if not q:
                continue

            source, value, depth_level = q.popleft()
            visit_key = f"{source}::{value}"
            if visit_key in visited:
                continue
            visited.add(visit_key)
            made_progress = True

            print(f"[{topic}][{source}][d={depth_level}] {value[:80]}")
            docs = dispatch(source, value, topic)

            for doc in docs:
                # wikidata_ref docs are purely for discovery — re-queue as wikipedia
                if doc["source"] == "wikidata_ref":
                    wiki_title = _title_from_wiki_url(doc["url"])
                    ref_key    = f"wikipedia::{wiki_title}"
                    if ref_key not in visited:
                        q.append(("wikipedia", wiki_title, depth_level))
                    continue

                if not doc["body"]:
                    continue

                path = save(doc, out_dir, overwrite=overwrite)
                if path is None:
                    skipped += 1
                    continue
                print(
                    f"  -> {path.name}  "
                    f"({len(doc['body'])} chars, {len(doc['links'])} links)"
                )
                saved += 1

                # ── Cross-topic relation discovery ─────────────────────────
                for related in detect_related_topics(doc["body"], topic):
                    pair = (topic, related)
                    if pair in injected_pairs:
                        continue
                    pool = spare_pools[related]
                    random.shuffle(pool)
                    injected = 0
                    for src2, val2 in pool:
                        if f"{src2}::{val2}" not in visited:
                            queues[related].appendleft((src2, val2, 0))
                            injected += 1
                            if injected >= CROSS_TOPIC_INJECT:
                                break
                    if injected:
                        print(f"  [cross-topic {topic} → {related}] injected {injected} seed(s)")
                        injected_pairs.add(pair)

                # ── Child-link expansion (depth > 0) ──────────────────────
                host = urlparse(doc["url"]).netloc
                if depth_level < depth and source in ("html", "stackoverflow", "scikit") \
                        and host in CRAWLABLE_HTML_HOSTS:
                    next_source = source  # keep "scikit" label for scikit pages
                    for link in doc["links"]:
                        link_key = f"{next_source}::{link}"
                        if link_key not in visited:
                            q.append((next_source, link, depth_level + 1))

            if delay > 0:
                time.sleep(delay)
            if saved >= max_pages:
                break

        if not made_progress:
            break

    skip_note = f", {skipped} skipped (already exist)" if skipped else ""
    print(f"\nDone. {saved} pages saved to {out_dir}/{skip_note}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_seeds_file(path: str) -> dict[str, dict]:
    urls = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return {"custom": {"keywords": [], "seeds": [("html", u) for u in urls]}}


def main() -> None:
    p = argparse.ArgumentParser(
        description="ColonySearch corpus scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python data/scraper.py                              # all topics, depth 1, 24 pages
  python data/scraper.py --scikit                     # programming + ML via wiki/arxiv/scikit
  python data/scraper.py --companies                  # corporate/NGO sites, all topics
  python data/scraper.py --companies --topics climate_change,economics
  python data/scraper.py --topics football,space      # only those two clusters
  python data/scraper.py --max 40 --depth 2           # deeper crawl, more pages
""",
    )
    p.add_argument("--seeds",     help="Flat text file of seed URLs (one per line)")
    p.add_argument("--topics",    help="Comma-separated topic names to crawl (default: all)")
    p.add_argument("--scikit",    action="store_true",
                   help="Shortcut: programming + ml_algorithms via wiki/arxiv/scikit-learn")
    p.add_argument("--companies", action="store_true",
                   help="Crawl corporate/NGO sites (WEF, UN, Microsoft Research, etc.) "
                        "instead of the default academic/wiki sources; "
                        "combine with --topics to restrict to specific clusters")
    p.add_argument("--depth",     type=int,   default=1,   help="HTML crawl depth (default 1)")
    p.add_argument("--max",       type=int,   default=24,  help="Max pages to save (default 24)")
    p.add_argument("--out",       default="data/corpus",   help="Output directory")
    p.add_argument("--delay",     type=float, default=1.0, help="Seconds between requests")
    p.add_argument("--seed",      type=int,                help="RNG seed for reproducible shuffles")
    p.add_argument("--overwrite", action="store_true",     help="Re-fetch and overwrite existing files")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.seeds:
        clusters = _load_seeds_file(args.seeds)
    elif args.companies:
        base = COMPANY_CLUSTERS
        if args.topics:
            wanted = {t.strip() for t in args.topics.split(",")}
            unknown = wanted - set(base)
            if unknown:
                p.error(f"Unknown topic(s) for --companies: {', '.join(sorted(unknown))}. "
                        f"Available: {', '.join(base)}")
            base = {t: base[t] for t in base if t in wanted}
        clusters = base
    elif args.scikit:
        # Preset: programming + ml_algorithms, focused sources only
        _SCIKIT_SOURCES = {
            "wikipedia", "arxiv", "scikit", "stackoverflow",
            "html", "semantic_scholar", "github", "devto",
        }
        clusters = {}
        for t in ("programming", "ml_algorithms"):
            conf = TOPIC_CLUSTERS[t]
            clusters[t] = {
                "keywords": conf["keywords"],
                "seeds": [(s, v) for s, v in conf["seeds"] if s in _SCIKIT_SOURCES],
            }
    elif args.topics:
        wanted = {t.strip() for t in args.topics.split(",")}
        unknown = wanted - set(TOPIC_CLUSTERS)
        if unknown:
            p.error(f"Unknown topic(s): {', '.join(sorted(unknown))}. "
                    f"Available: {', '.join(TOPIC_CLUSTERS)}")
        clusters = {t: TOPIC_CLUSTERS[t] for t in TOPIC_CLUSTERS if t in wanted}

    else:
        clusters = TOPIC_CLUSTERS

    crawl(
        clusters  = clusters,
        depth     = args.depth,
        max_pages = args.max,
        out_dir   = Path(args.out),
        delay     = args.delay,
        overwrite = args.overwrite,
    )


if __name__ == "__main__":
    main()
