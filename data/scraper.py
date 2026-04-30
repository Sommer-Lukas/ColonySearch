"""
ColonySearch corpus scraper.

Modes
-----
  (default)      Academic/API sources — Wikipedia, arXiv, OpenAlex, Semantic
                 Scholar, DEV.to, GitHub READMEs, scikit-learn docs.
  --companies    Corporate/NGO HTML sources — WEF, UNEP, NASA, MIT News,
                 Quanta Magazine, WHO, IEA, OECD, IMF, BBC Sport, …
  --scikit       programming + ml_algorithms clusters only, all non-API sources.
  --expand       Follow links already recorded in your corpus JSON files,
                 randomly sample them, and scrape to the given depth.
  --topics a,b   Restrict any mode to the named clusters.
  --seeds file   Plain text file of seed URLs → single "custom" cluster.

Output: one JSON file per page  { url, title, body, links, source, topic }
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


# ── TOPIC CLUSTERS ────────────────────────────────────────────────────────────
# Seed format: (source_type, value)
#   wikipedia / wikidata  →  article title or concept string
#   openalex / arxiv / semantic_scholar / devto  →  search query or tag
#   github  →  "owner/repo"
#   html / scikit / stackoverflow  →  full URL

TOPIC_CLUSTERS: dict[str, dict] = {
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
            # scikit-learn modules
            ("scikit",           "https://scikit-learn.org/stable/modules/ensemble.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/cross_validation.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/linear_model.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/svm.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/tree.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/neighbors.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/naive_bayes.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/neural_networks_supervised.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/gaussian_process.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/clustering.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/decomposition.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/preprocessing.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/feature_selection.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/manifold.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/mixture.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/outlier_detection.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/model_evaluation.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/pipeline.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/grid_search.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/learning_curve.html"),
            ("scikit",           "https://scikit-learn.org/stable/modules/permutation_importance.html"),
            ("openalex",         "Python programming language performance optimization"),
            ("arxiv",            "scikit-learn machine learning Python benchmark"),
        ],
    },

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
            ("semantic_scholar", "world model model-based reinforcement learning survey"),
            ("semantic_scholar", "transformer self-attention survey vision language"),
            ("semantic_scholar", "diffusion generative model image synthesis"),
            ("semantic_scholar", "mechanistic interpretability circuits features neural network"),
            ("github",           "google-deepmind/dreamerv3"),
            ("github",           "huggingface/diffusers"),
            ("github",           "state-spaces/mamba"),
            ("github",           "openai/gym"),
            ("stackoverflow",    "https://stackoverflow.com/questions/55243483/what-is-the-difference-between-model-free-and-model-based-reinforcement-learning"),
            ("openalex",         "world model model-based reinforcement learning survey"),
            ("openalex",         "diffusion probabilistic model generative image"),
        ],
    },

    "football": {
        "keywords": [
            "association football", "pressing", "expected goals", "tiki-taka",
            "gegenpressing", "UEFA Champions League", "FIFA World Cup",
            "high press", "false nine", "ball possession", "xG", "transfer market",
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
            ("html",             "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
        ],
    },

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
            ("arxiv",            "variational quantum eigensolver NISQ hybrid algorithm"),
            ("arxiv",            "quantum error correction surface code fault tolerant"),
            ("arxiv",            "quantum machine learning kernel classification"),
            ("arxiv",            "topological qubit anyons quantum computation"),
            ("semantic_scholar", "NISQ quantum advantage near-term algorithm"),
            ("openalex",         "quantum computing algorithm optimization survey"),
            ("github",           "Qiskit/qiskit"),
            ("github",           "quantumlib/Cirq"),
        ],
    },

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
            ("arxiv",            "connectome neural circuit mapping"),
            ("arxiv",            "brain computer interface EEG decoding motor"),
            ("arxiv",            "dopamine reward prediction error reinforcement"),
            ("semantic_scholar", "free energy principle active inference Karl Friston"),
            ("openalex",         "hippocampus memory consolidation spatial navigation"),
        ],
    },

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
            ("semantic_scholar", "Nash equilibrium evolutionary game theory"),
            ("semantic_scholar", "auction design revenue equivalence Vickrey"),
            ("openalex",         "behavioral economics prospect theory Kahneman"),
            ("openalex",         "monetary policy inflation central bank"),
        ],
    },
}


# ── COMPANY / NGO CLUSTERS ────────────────────────────────────────────────────
# Only sites that reliably serve plain HTML without JS rendering.
# Quanta Magazine, MIT News, WEF, UNEP, OECD, IMF, WHO, NASA blogs, BBC Sport.
# Topics mirror TOPIC_CLUSTERS so cross-topic detection still fires.

COMPANY_CLUSTERS: dict[str, dict] = {
    "climate_change": {
        "keywords": TOPIC_CLUSTERS["climate_change"]["keywords"],
        "seeds": [
            ("html", "https://www.weforum.org/agenda/archive/climate-change/"),
            ("html", "https://www.unep.org/topics/climate-action"),
            ("html", "https://www.iea.org/topics/clean-energy-transitions"),
            ("html", "https://www.oecd.org/en/topics/climate-action.html"),
            ("html", "https://www.imf.org/en/Topics/climate-change"),
            ("html", "https://www.quantamagazine.org/tag/climate/"),
            ("html", "https://climate.nasa.gov/news/"),
        ],
    },
    "sustainable_materials": {
        "keywords": TOPIC_CLUSTERS["sustainable_materials"]["keywords"],
        "seeds": [
            ("html", "https://ellenmacarthurfoundation.org/topics/circular-economy/overview"),
            ("html", "https://www.unep.org/topics/chemicals-waste/plastics"),
            ("html", "https://www.weforum.org/agenda/archive/circular-economy/"),
        ],
    },
    "space": {
        "keywords": TOPIC_CLUSTERS["space"]["keywords"],
        "seeds": [
            ("html", "https://www.nasa.gov/news/"),
            ("html", "https://www.esa.int/Newsroom"),
            ("html", "https://blogs.nasa.gov/artemis/"),
            ("html", "https://www.quantamagazine.org/tag/cosmology/"),
        ],
    },
    "ml_algorithms": {
        "keywords": TOPIC_CLUSTERS["ml_algorithms"]["keywords"],
        "seeds": [
            ("html", "https://www.microsoft.com/en-us/research/research-area/artificial-intelligence/"),
            ("html", "https://news.mit.edu/topic/artificial-intelligence2"),
            ("html", "https://www.quantamagazine.org/tag/artificial-intelligence/"),
            ("html", "https://www.weforum.org/agenda/archive/artificial-intelligence/"),
        ],
    },
    "programming": {
        "keywords": TOPIC_CLUSTERS["programming"]["keywords"],
        "seeds": [
            ("html", "https://news.mit.edu/topic/computers"),
            ("html", "https://www.weforum.org/agenda/archive/data-science/"),
        ],
    },
    "economics": {
        "keywords": TOPIC_CLUSTERS["economics"]["keywords"],
        "seeds": [
            ("html", "https://www.imf.org/en/Publications/WEO"),
            ("html", "https://www.oecd.org/en/topics/economy.html"),
            ("html", "https://www.weforum.org/agenda/archive/economics/"),
            ("html", "https://www.worldbank.org/en/topic/macroeconomics"),
        ],
    },
    "medicine": {
        "keywords": TOPIC_CLUSTERS["medicine"]["keywords"],
        "seeds": [
            ("html", "https://www.who.int/news-room/fact-sheets"),
            ("html", "https://www.nih.gov/news-events/news-releases"),
            ("html", "https://www.quantamagazine.org/tag/biology/"),
        ],
    },
    "football": {
        "keywords": TOPIC_CLUSTERS["football"]["keywords"],
        "seeds": [
            ("html", "https://www.bbc.com/sport/football"),
            ("html", "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
        ],
    },
    "quantum_computing": {
        "keywords": TOPIC_CLUSTERS["quantum_computing"]["keywords"],
        "seeds": [
            ("html", "https://www.quantamagazine.org/tag/quantum-computing/"),
            ("html", "https://news.mit.edu/topic/quantum-computing"),
        ],
    },
    "neuroscience": {
        "keywords": TOPIC_CLUSTERS["neuroscience"]["keywords"],
        "seeds": [
            ("html", "https://www.quantamagazine.org/tag/neuroscience/"),
            ("html", "https://www.nih.gov/research-training/research-topics/neuroscience"),
        ],
    },
}


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "ColonySearch/1.0 (university distributed search engine; "
        "non-commercial; github.com/example/colonysearch)"
    )
}

MAX_BODY          = 8_000
MAX_LINKS         = 25
CROSS_THRESHOLD   = 3   # keyword hits before cross-topic seed injection
CROSS_INJECT      = 1   # seeds injected per detected relation

# Hosts whose discovered child-links we follow (depth > 0)
CRAWLABLE = frozenset({
    "en.wikipedia.org", "simple.wikipedia.org",
    "docs.python.org", "scikit-learn.org",
    "climate.nasa.gov", "www.nasa.gov", "blogs.nasa.gov", "science.nasa.gov",
    "www.esa.int",
    "www.weforum.org", "www.unep.org", "www.oecd.org",
    "www.imf.org", "www.worldbank.org",
    "ellenmacarthurfoundation.org", "www.iea.org",
    "www.who.int", "www.nih.gov",
    "www.bbc.com", "fbref.com",
    "news.mit.edu", "www.quantamagazine.org",
    "www.microsoft.com",
})

# Per-host patterns for links to skip
_SKIP: dict[str, re.Pattern] = {
    "en.wikipedia.org":    re.compile(r"/(Special|Talk|User|File|Help|Category|Wikipedia|Portal|Template):"),
    "simple.wikipedia.org":re.compile(r"/(Special|Talk|User|File|Help|Category|Wikipedia):"),
    "docs.python.org":     re.compile(r"genindex|py-modindex|search\.html"),
    "scikit-learn.org":    re.compile(r"/stable/api/|generated/sklearn\.|_downloads"),
    "stackoverflow.com":   re.compile(r"/users/|/tags/|/questions/tagged/|/jobs/"),
}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(url: str, _retries: int = 3, **kwargs) -> requests.Response | None:
    for attempt in range(_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=14, **kwargs)
            if r.status_code == 429:
                wait = 15 * (2 ** attempt)
                print(f"  [rate-limit] sleeping {wait}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  [skip] {str(url)[:80]}: {e}")
            return None
    return None


# ── SHARED HELPERS ────────────────────────────────────────────────────────────

def _doc(*, url: str, title: str, body: str, links: list[str],
         source: str, topic: str) -> dict:
    return {"url": url, "title": title, "body": body,
            "links": links, "source": source, "topic": topic}

def _wiki_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

def _title_from_wiki_url(url: str) -> str:
    return urlparse(url).path.split("/wiki/")[-1]


# ── ADAPTERS ──────────────────────────────────────────────────────────────────

def fetch_wikipedia(title: str, topic: str) -> list[dict]:
    r = _get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "format": "json",
        "titles": title.replace("_", " "),
        "prop": "extracts|links",
        "exlimit": 1, "explaintext": True, "exsectionformat": "plain",
        "pllimit": MAX_LINKS, "plnamespace": 0,
    })
    if not r:
        return []
    docs = []
    for pid, page in r.json().get("query", {}).get("pages", {}).items():
        if pid == "-1" or not page.get("extract"):
            continue
        url   = _wiki_url(page["title"])
        body  = page["extract"][:MAX_BODY]
        links = [_wiki_url(lk["title"]) for lk in page.get("links", [])]
        docs.append(_doc(url=url, title=page["title"], body=body,
                         links=links, source="wikipedia", topic=topic))
    return docs


def fetch_wikidata(concept: str, topic: str) -> list[dict]:
    """Searches Wikidata and returns Wikipedia article URLs to re-queue."""
    r = _get("https://www.wikidata.org/w/api.php", params={
        "action": "wbsearchentities", "search": concept,
        "language": "en", "format": "json", "limit": 5,
    })
    if not r:
        return []
    ids = [x["id"] for x in r.json().get("search", [])]
    if not ids:
        return []
    r2 = _get("https://www.wikidata.org/w/api.php", params={
        "action": "wbgetentities", "ids": "|".join(ids),
        "props": "sitelinks", "format": "json",
    })
    if not r2:
        return []
    docs = []
    for entity in r2.json().get("entities", {}).values():
        title = entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
        if title:
            docs.append(_doc(url=_wiki_url(title), title=title,
                             body="", links=[], source="wikidata_ref", topic=topic))
    return docs


def _rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    return " ".join(w for _, w in sorted(
        (pos, word) for word, positions in inv.items() for pos in positions
    ))

def fetch_openalex(query: str, topic: str) -> list[dict]:
    r = _get("https://api.openalex.org/works", params={
        "search": query, "per-page": 5,
        "select": "id,title,abstract_inverted_index,concepts,doi",
        "mailto": "research@example.com",
    })
    if not r:
        return []
    docs = []
    for work in r.json().get("results", []):
        title   = work.get("title") or ""
        body    = _rebuild_abstract(work.get("abstract_inverted_index"))
        body   += " " + " ".join(c["display_name"] for c in work.get("concepts", []))
        body    = body.strip()[:MAX_BODY]
        url     = work.get("doi") or work.get("id", "")
        if title and body and url:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="openalex", topic=topic))
    return docs


def fetch_arxiv(query: str, topic: str) -> list[dict]:
    r = _get("https://export.arxiv.org/api/query", params={
        "search_query": f"all:{query}", "start": 0, "max_results": 5,
    })
    if not r:
        return []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    docs = []
    for entry in root.findall("a:entry", ns):
        def _t(tag: str) -> str:
            el = entry.find(tag, ns)
            return el.text.strip() if el is not None and el.text else ""
        title = _t("a:title")
        body  = _t("a:summary")[:MAX_BODY]
        url   = _t("a:id").replace("http://", "https://")
        if title and body and url:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="arxiv", topic=topic))
    return docs


def fetch_semantic_scholar(query: str, topic: str) -> list[dict]:
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search", params={
        "query": query, "limit": 6,
        "fields": "title,abstract,tldr,externalIds,openAccessPdf",
    })
    if not r:
        return []
    docs = []
    for paper in r.json().get("data", []):
        title    = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        tldr     = (paper.get("tldr") or {}).get("text") or ""
        body     = (f"{abstract}\n\nTL;DR: {tldr}" if tldr else abstract)[:MAX_BODY]
        ext      = paper.get("externalIds") or {}
        doi      = ext.get("DOI")
        url      = (f"https://doi.org/{doi}" if doi else
                    (paper.get("openAccessPdf") or {}).get("url") or "")
        if title and body and url:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="semantic_scholar", topic=topic))
    return docs


def fetch_devto(tag: str, topic: str) -> list[dict]:
    r = _get("https://dev.to/api/articles", params={"tag": tag, "per_page": 6, "top": 1})
    if not r:
        return []
    docs = []
    for article in r.json():
        url   = article.get("url") or ""
        title = article.get("title") or ""
        raw   = BeautifulSoup(article.get("body_html") or "", "html.parser").get_text(" ", strip=True)
        body  = " ".join(raw.split())[:MAX_BODY]
        if url and title and body:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="devto", topic=topic))
    return docs


def fetch_github(repo: str, topic: str) -> list[dict]:
    for branch in ("main", "master"):
        r = _get(f"https://raw.githubusercontent.com/{repo}/{branch}/README.md")
        if r and r.status_code == 200:
            break
    else:
        return []
    text = re.sub(r"```.*?```", " ", r.text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", text)
    body  = " ".join(text.split())[:MAX_BODY]
    title = repo.split("/")[-1].replace("-", " ").replace("_", " ").title()
    return [_doc(url=f"https://github.com/{repo}", title=title, body=body,
                 links=[], source="github", topic=topic)]


def fetch_html(url: str, topic: str, source: str) -> list[dict]:
    r = _get(url)
    if not r or "text/html" not in r.headers.get("Content-Type", ""):
        return []
    soup  = BeautifulSoup(r.text, "html.parser")
    host  = urlparse(url).netloc
    title = (soup.find("title") or soup.new_tag("x")).get_text(strip=True) or url
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    content = _body(soup, host)
    body    = " ".join(content.get_text(" ", strip=True).split())[:MAX_BODY] if content else ""
    links   = _links(soup, url, host)
    return [_doc(url=url, title=title, body=body, links=links, source=source, topic=topic)]


def _body(soup: BeautifulSoup, host: str):
    """Pick the most relevant content container for a given host."""
    if "wikipedia.org"      in host: return soup.find(id="mw-content-text")
    if "stackoverflow.com"  in host: return soup.find(id="question") or soup.find(id="answers")
    if "docs.python.org"    in host: return soup.find("div", role="main")
    if "scikit-learn.org"   in host: return soup.find("div", role="main") or soup.find(class_="bd-article")
    if "quantamagazine.org" in host: return soup.find(class_="article__content") or soup.find("article")
    if "weforum.org"        in host: return soup.find(class_="article-body") or soup.find("article")
    return soup.find("main") or soup.find("article") or soup.body


def _links(soup: BeautifulSoup, base_url: str, host: str) -> list[str]:
    base  = urlparse(base_url)
    skip  = next((p for h, p in _SKIP.items() if h in host), None)
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
        if len(links) >= MAX_LINKS:
            break
    return links


# ── DISPATCH ──────────────────────────────────────────────────────────────────

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
        case "stackoverflow":    return fetch_html(value, topic, "stackoverflow")
        case "html":             return fetch_html(value, topic, "html")
        case _:                  return []


# ── CROSS-TOPIC DETECTION ─────────────────────────────────────────────────────

def detect_related(body: str, exclude: str, clusters: dict) -> list[str]:
    low = body.lower()
    return [
        t for t, conf in clusters.items()
        if t != exclude
        and sum(1 for kw in conf["keywords"] if kw.lower() in low) >= CROSS_THRESHOLD
    ]


# ── PERSISTENCE ───────────────────────────────────────────────────────────────

def _filename(url: str, topic: str) -> str:
    p    = urlparse(url)
    slug = re.sub(r"[^a-z0-9]+", "_", p.path.lower()).strip("_") or "index"
    host = re.sub(r"[^a-z0-9]+", "_", p.netloc)
    return f"{topic}__{host}__{slug[:50]}.json"


def save(data: dict, out_dir: Path, overwrite: bool = False) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _filename(data["url"], data["topic"])
    if path.exists() and not overwrite:
        return None
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def load_corpus_links(corpus_dir: Path) -> list[tuple[str, str]]:
    """
    Reads every JSON file in corpus_dir, collects (url, topic) pairs from
    their 'links' field, and returns only those not already scraped.
    """
    files = list(corpus_dir.glob("*.json"))
    if not files:
        return []

    existing: set[str] = set()
    raw: list[tuple[str, str]] = []

    for f in files:
        try:
            doc = json.loads(f.read_text())
            existing.add(doc.get("url", ""))
        except (json.JSONDecodeError, KeyError):
            continue

    seen: set[str] = set(existing)
    for f in files:
        try:
            doc   = json.loads(f.read_text())
            topic = doc.get("topic", "custom")
            for link in doc.get("links", []):
                if link and link not in seen:
                    raw.append((link, topic))
                    seen.add(link)
        except (json.JSONDecodeError, KeyError):
            continue

    return raw


# ── CRAWL ENGINE ─────────────────────────────────────────────────────────────

def crawl(
    clusters:  dict[str, dict],
    depth:     int,
    max_pages: int,
    out_dir:   Path,
    delay:     float,
    overwrite: bool = False,
) -> None:
    queues: dict[str, deque] = {}
    for topic, conf in clusters.items():
        seeds = conf["seeds"][:]
        random.shuffle(seeds)
        queues[topic] = deque((src, val, 0) for src, val in seeds)

    spare: dict[str, list] = {t: c["seeds"][:] for t, c in clusters.items()}
    visited:        set[str]             = set()
    injected_pairs: set[tuple[str,str]]  = set()
    topics  = list(queues.keys())
    saved   = 0
    skipped = 0

    while saved < max_pages:
        progress = False

        for topic in topics:
            q = queues[topic]
            if not q:
                continue

            source, value, dlevel = q.popleft()
            key = f"{source}::{value}"
            if key in visited:
                continue
            visited.add(key)
            progress = True

            print(f"[{topic}][{source}][d={dlevel}] {value[:80]}")
            docs = dispatch(source, value, topic)

            for doc in docs:
                # Wikidata returns placeholder docs → re-queue as wikipedia
                if doc["source"] == "wikidata_ref":
                    wt  = _title_from_wiki_url(doc["url"])
                    wk  = f"wikipedia::{wt}"
                    if wk not in visited:
                        q.append(("wikipedia", wt, dlevel))
                    continue

                if not doc["body"]:
                    continue

                path = save(doc, out_dir, overwrite)
                if path is None:
                    skipped += 1
                    continue

                print(f"  -> {path.name}  ({len(doc['body'])} chars, {len(doc['links'])} links)")
                saved += 1

                # Cross-topic injection
                for related in detect_related(doc["body"], topic, clusters):
                    pair = (topic, related)
                    if pair in injected_pairs:
                        continue
                    pool = spare[related][:]
                    random.shuffle(pool)
                    for src2, val2 in pool:
                        if f"{src2}::{val2}" not in visited:
                            queues[related].appendleft((src2, val2, 0))
                            print(f"  [cross → {related}] {src2}:{val2[:50]}")
                            injected_pairs.add(pair)
                            break

                # Child-link expansion
                host = urlparse(doc["url"]).netloc
                if dlevel < depth and source in ("html", "scikit", "stackoverflow") \
                        and host in CRAWLABLE:
                    for link in doc["links"]:
                        lk = f"{source}::{link}"
                        if lk not in visited:
                            q.append((source, link, dlevel + 1))

            if delay > 0:
                time.sleep(delay)
            if saved >= max_pages:
                break

        if not progress:
            break

    note = f", {skipped} already existed" if skipped else ""
    print(f"\nDone. {saved} pages saved to {out_dir}/{note}")


def expand(
    corpus_dir: Path,
    sample:     int,
    depth:      int,
    max_pages:  int,
    out_dir:    Path,
    delay:      float,
    overwrite:  bool,
) -> None:
    """
    Reads links from existing corpus files, randomly samples `sample` of them,
    then crawls those URLs (and follows their links to `depth` levels).
    """
    candidates = load_corpus_links(corpus_dir)
    if not candidates:
        print("No unscraped links found in corpus.")
        return

    random.shuffle(candidates)
    sampled = candidates[:sample]
    print(f"Sampled {len(sampled)} / {len(candidates)} candidate links.")

    # Group by topic so the round-robin engine still balances coverage
    by_topic: dict[str, list] = {}
    for url, topic in sampled:
        by_topic.setdefault(topic, []).append(("html", url))

    clusters = {
        t: {
            "keywords": TOPIC_CLUSTERS.get(t, {}).get("keywords", []),
            "seeds":    seeds,
        }
        for t, seeds in by_topic.items()
    }

    crawl(clusters, depth=depth, max_pages=max_pages,
          out_dir=out_dir, delay=delay, overwrite=overwrite)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _seeds_from_file(path: str) -> dict[str, dict]:
    urls = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    return {"custom": {"keywords": [], "seeds": [("html", u) for u in urls]}}


def main() -> None:
    p = argparse.ArgumentParser(
        description="ColonySearch corpus scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python data/scraper.py                                # all topics, academic sources
  python data/scraper.py --companies                    # WEF/UN/NASA/Quanta/MIT News/…
  python data/scraper.py --companies --topics climate_change,economics
  python data/scraper.py --scikit --max 40              # programming + ML, all sources
  python data/scraper.py --expand --expand-sample 30    # follow links from existing corpus
  python data/scraper.py --topics football,space --max 20
  python data/scraper.py --seed 42 --max 50 --depth 2
""",
    )
    p.add_argument("--seeds",          help="Text file with one seed URL per line")
    p.add_argument("--topics",         help="Comma-separated cluster names (default: all)")
    p.add_argument("--companies",      action="store_true",
                   help="Use corporate/NGO HTML sources instead of academic APIs")
    p.add_argument("--scikit",         action="store_true",
                   help="programming + ml_algorithms clusters, all source types")
    p.add_argument("--expand",         action="store_true",
                   help="Follow links from existing corpus files")
    p.add_argument("--expand-sample",  type=int, default=20,
                   help="How many corpus links to randomly sample (default 20)")
    p.add_argument("--depth",          type=int,   default=1,
                   help="HTML link-follow depth (default 1)")
    p.add_argument("--max",            type=int,   default=24,
                   help="Max pages to save (default 24)")
    p.add_argument("--out",            default="data/corpus",
                   help="Output directory (default data/corpus)")
    p.add_argument("--delay",          type=float, default=1.0,
                   help="Seconds between requests (default 1.0)")
    p.add_argument("--seed",           type=int,
                   help="RNG seed for reproducible shuffles")
    p.add_argument("--overwrite",      action="store_true",
                   help="Re-fetch and overwrite existing files")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.out)

    # ── Mode selection ────────────────────────────────────────────────────────
    if args.expand:
        corpus_dir = out_dir
        expand(
            corpus_dir     = corpus_dir,
            sample         = args.expand_sample,
            depth          = args.depth,
            max_pages      = args.max,
            out_dir        = out_dir,
            delay          = args.delay,
            overwrite      = args.overwrite,
        )
        return

    if args.seeds:
        clusters = _seeds_from_file(args.seeds)

    elif args.companies:
        base = COMPANY_CLUSTERS
        if args.topics:
            wanted  = {t.strip() for t in args.topics.split(",")}
            unknown = wanted - set(base)
            if unknown:
                p.error(f"Unknown --companies topic(s): {', '.join(sorted(unknown))}. "
                        f"Available: {', '.join(base)}")
            base = {t: base[t] for t in base if t in wanted}
        clusters = base

    elif args.scikit:
        allowed = {"wikipedia", "arxiv", "scikit", "stackoverflow",
                   "html", "semantic_scholar", "github", "devto"}
        clusters = {}
        for t in ("programming", "ml_algorithms"):
            conf = TOPIC_CLUSTERS[t]
            clusters[t] = {
                "keywords": conf["keywords"],
                "seeds":    [(s, v) for s, v in conf["seeds"] if s in allowed],
            }

    elif args.topics:
        wanted  = {t.strip() for t in args.topics.split(",")}
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
        out_dir   = out_dir,
        delay     = args.delay,
        overwrite = args.overwrite,
    )


if __name__ == "__main__":
    main()
