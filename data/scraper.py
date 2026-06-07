"""
ColonySearch corpus scraper.

Modes
-----
  (default)      All topic clusters, all source types.
  --companies    Corporate/NGO HTML sources — WEF, UNEP, NASA, MIT News,
                 Quanta Magazine, WHO, IEA, OECD, IMF, BBC Sport, …
  --expand       Follow links already recorded in your corpus JSON files,
                 randomly sample them, and scrape to the given depth.
  --topics a,b   Restrict any mode to the named clusters.
  --sources a,b  Restrict to specific source types (wikipedia, arxiv,
                 openalex, semantic_scholar, devto, github, html,
                 stackoverflow, wikidata).
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
from urllib.parse import unquote, urljoin, urlparse

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
            "global warming", "Paris Agreement", "carbon capture",
            "renewable energy", "climate model", "extreme weather",
        ],
        "seeds": [
            ("wikipedia",        "Keeling_Curve"),
            ("wikipedia",        "Arctic_sea_ice_decline"),
            ("wikipedia",        "Climate_tipping_points"),
            ("wikipedia",        "IPCC_Sixth_Assessment_Report"),
            ("wikipedia",        "Carbon_budget"),
            ("wikipedia",        "Global_warming"),
            ("wikipedia",        "Paris_Agreement"),
            ("wikipedia",        "Carbon_capture_and_storage"),
            ("wikipedia",        "Renewable_energy"),
            ("wikipedia",        "Climate_change_mitigation"),
            ("wikipedia",        "Ocean_acidification"),
            ("wikipedia",        "Deforestation"),
            ("wikipedia",        "Carbon_tax"),
            ("wikipedia",        "Electric_vehicle"),
            ("wikipedia",        "Solar_power"),
            ("wikipedia",        "Wind_power"),
            ("wikipedia",        "Climate_change"),
            ("wikipedia",        "Greenhouse_gas"),
            ("wikipedia",        "Ice_core"),
            ("openalex",         "Arctic sea ice decline feedback loop"),
            ("openalex",         "permafrost thaw methane emission"),
            ("openalex",         "Paris Agreement climate policy implementation"),
            ("openalex",         "ocean acidification coral reef bleaching"),
            ("openalex",         "solar wind energy transition electricity grid"),
            ("openalex",         "carbon capture storage geological sequestration"),
            ("openalex",         "extreme weather events climate change attribution"),
            ("openalex",         "net zero emission pathway decarbonisation cost"),
            ("arxiv",            "climate tipping point abrupt transition"),
            ("arxiv",            "carbon budget net zero emission pathway"),
            ("arxiv",            "sea level rise ice sheet collapse projection"),
            ("arxiv",            "climate model feedback sensitivity equilibrium"),
            ("arxiv",            "methane emission wetland permafrost global warming"),
            ("arxiv",            "carbon dioxide removal direct air capture"),
            ("arxiv",            "renewable energy storage grid integration"),
            ("arxiv",            "climate change extreme precipitation flood drought"),
            ("arxiv",            "solar geoengineering stratospheric aerosol injection"),
            ("semantic_scholar", "climate change tipping point cascade"),
            ("semantic_scholar", "IPCC mitigation adaptation vulnerability assessment"),
            ("semantic_scholar", "carbon pricing emissions trading scheme effectiveness"),
            ("semantic_scholar", "renewable energy transition cost learning curve"),
            ("wikidata",         "greenhouse gas"),
            ("wikidata",         "climate change"),
            ("wikidata",         "Paris Agreement"),
            ("github",           "openclimatefix/nowcasting"),
            ("github",           "pangeo-data/pangeo"),
            ("html",             "https://climate.nasa.gov/vital-signs/carbon-dioxide/"),
            ("html",             "https://climate.nasa.gov/vital-signs/sea-level/"),
            ("html",             "https://climate.nasa.gov/vital-signs/arctic-sea-ice/"),
            ("html",             "https://www.ipcc.ch/report/ar6/syr/"),
            ("html",             "https://www.carbonbrief.org/explainers/"),
            ("html",             "https://www.carbonbrief.org/"),
            ("html",             "https://phys.org/tags/climate+change/"),
            ("html",             "https://www.sciencedaily.com/news/earth_climate/global_warming/"),
            ("html",             "https://www.eco2050.de/"),
            ("html",             "https://ec.europa.eu/climate/"),
            ("html",             "https://www.un.org/en/climatechange/"),
            ("html",             "https://www.bundestag.de/dokumente/textarchiv/themenbereich-klima"),
            ("rss",              "https://climate.nasa.gov/news/rss.xml"),
            ("rss",              "https://www.carbonbrief.org/feed"),
            ("rss",              "https://www.sciencedaily.com/rss/earth_climate/global_warming.xml"),
            ("rss",              "https://phys.org/rss-feed/earth-news/environment/"),
        ],
    },

    "sustainable_materials": {
        "keywords": [
            "bioplastic", "polyhydroxyalkanoate", "mycelium", "lignin",
            "biodegradable polymer", "life cycle assessment", "bio-based",
            "compostable", "cellulose composite", "upcycling", "cradle to cradle",
            "circular economy", "recycling", "biomass", "natural fibre",
            "sustainable packaging", "green chemistry",
        ],
        "seeds": [
            ("wikipedia",        "Polyhydroxyalkanoates"),
            ("wikipedia",        "Mycelium_materials"),
            ("wikipedia",        "Lignin"),
            ("wikipedia",        "Life-cycle_assessment"),
            ("wikipedia",        "Bioremediation"),
            ("wikipedia",        "Circular_economy"),
            ("wikipedia",        "Bioplastic"),
            ("wikipedia",        "Recycling"),
            ("wikipedia",        "Natural_fiber"),
            ("wikipedia",        "Green_chemistry"),
            ("wikipedia",        "Biomass"),
            ("wikipedia",        "Polylactic_acid"),
            ("wikipedia",        "Hemp"),
            ("wikipedia",        "Cellulose"),
            ("wikipedia",        "Sustainable_packaging"),
            ("wikipedia",        "Biocomposite"),
            ("wikipedia",        "Algae_fuel"),
            ("openalex",         "polyhydroxyalkanoate PHA bioplastic production"),
            ("openalex",         "mycelium biocomposite mechanical properties"),
            ("openalex",         "circular economy waste reduction material recovery"),
            ("openalex",         "life cycle assessment biobased material carbon footprint"),
            ("openalex",         "cellulose nanocrystal sustainable composite material"),
            ("openalex",         "green chemistry solvent-free synthesis bio-based"),
            ("openalex",         "sustainable packaging biodegradable barrier properties"),
            ("arxiv",            "cellulose nanofiber biodegradable composite"),
            ("arxiv",            "lignin valorisation bio-based material"),
            ("arxiv",            "bioplastic mechanical thermal properties review"),
            ("arxiv",            "circular economy material flow analysis sustainability"),
            ("arxiv",            "mycelium fungal composite building material"),
            ("arxiv",            "algae biofuel lipid extraction microalgae"),
            ("semantic_scholar", "biodegradable polymer cradle to cradle lifecycle"),
            ("semantic_scholar", "circular economy design waste valorisation"),
            ("semantic_scholar", "natural fibre composite mechanical performance review"),
            ("wikidata",         "biopolymer"),
            ("wikidata",         "circular economy"),
            ("github",           "IndEcol/brightway2"),
            ("github",           "openLCA/olca-app"),
            ("github",           "symbiflow/symbiflow-examples"),
            ("devto",            "sustainability"),
            ("devto",            "greentech"),
            ("stackoverflow",    "https://stackoverflow.com/questions/tagged/life-cycle-assessment"),
            ("html",             "https://ellenmacarthurfoundation.org/topics/circular-economy/overview"),
            ("html",             "https://www.unep.org/topics/chemicals-waste/plastics"),
            ("html",             "https://www.weforum.org/agenda/archive/circular-economy/"),
            ("html",             "https://phys.org/tags/biodegradable/"),
            ("html",             "https://www.sciencedaily.com/news/matter_energy/biomaterials/"),
            ("html",             "https://www.oecd.org/en/topics/circular-economy.html"),
            ("html",             "https://www.eco2050.de/"),
            ("html",             "https://ec.europa.eu/environment/circular-economy/"),
            ("html",             "https://www.unep.org/resources/report"),
            ("html",             "https://www.materialstoday.com/"),
            ("html",             "https://www.azom.com/green-chemistry.aspx"),
            ("rss",              "https://www.sciencedaily.com/rss/matter_energy/biomaterials.xml"),
            ("rss",              "https://phys.org/rss-feed/technology-news/materials-science/"),
        ],
    },

    "space": {
        "keywords": [
            "Artemis", "Perseverance rover", "JWST", "exoplanet", "Starship",
            "launch vehicle", "gravitational wave", "neutron star", "lunar gateway",
            "orbital mechanics", "reusable rocket", "SpaceX",
            "black hole", "dark matter", "International Space Station",
            "Mars", "telescope", "satellite", "cosmos",
        ],
        "seeds": [
            ("wikipedia",        "Artemis_program"),
            ("wikipedia",        "Perseverance_(rover)"),
            ("wikipedia",        "James_Webb_Space_Telescope"),
            ("wikipedia",        "Gravitational_wave"),
            ("wikipedia",        "Lunar_Gateway"),
            ("wikipedia",        "SpaceX_Starship"),
            ("wikipedia",        "International_Space_Station"),
            ("wikipedia",        "Black_hole"),
            ("wikipedia",        "Dark_matter"),
            ("wikipedia",        "Exoplanet"),
            ("wikipedia",        "Mars_colonization"),
            ("wikipedia",        "Hubble_Space_Telescope"),
            ("wikipedia",        "Orbital_mechanics"),
            ("wikipedia",        "Neutron_star"),
            ("wikipedia",        "Voyager_program"),
            ("wikipedia",        "Asteroid_mining"),
            ("wikipedia",        "Solar_wind"),
            ("wikipedia",        "Cosmic_microwave_background"),
            ("wikipedia",        "Event_Horizon_Telescope"),
            ("openalex",         "JWST first light infrared galaxy observation"),
            ("openalex",         "dark matter detection direct experiment"),
            ("openalex",         "gravitational wave black hole merger LIGO Virgo"),
            ("openalex",         "exoplanet atmosphere characterisation spectroscopy"),
            ("openalex",         "Mars sample return mission geology astrobiology"),
            ("openalex",         "reusable launch vehicle commercial space economy"),
            ("openalex",         "cosmic microwave background inflation power spectrum"),
            ("arxiv",            "exoplanet biosignature atmosphere transmission spectroscopy"),
            ("arxiv",            "gravitational wave neutron star merger LIGO"),
            ("arxiv",            "JWST galaxy formation high redshift observation"),
            ("arxiv",            "dark matter candidate WIMP axion detection"),
            ("arxiv",            "Mars atmospheric composition habitability astrobiology"),
            ("arxiv",            "black hole shadow Event Horizon Telescope imaging"),
            ("arxiv",            "lunar resource utilisation in-situ water ice"),
            ("arxiv",            "orbital debris mitigation space sustainability"),
            ("arxiv",            "pulsar timing array gravitational wave background"),
            ("semantic_scholar", "reusable launch vehicle propulsion landing"),
            ("semantic_scholar", "exoplanet transit photometry radial velocity survey"),
            ("semantic_scholar", "cosmic inflation CMB primordial gravitational wave"),
            ("wikidata",         "Artemis program"),
            ("wikidata",         "black hole"),
            ("wikidata",         "exoplanet"),
            ("github",           "astropy/astropy"),
            ("github",           "poliastro/poliastro"),
            ("github",           "spacetelescope/jwst"),
            ("html",             "https://www.nasa.gov/missions/artemis/"),
            ("html",             "https://science.nasa.gov/missions/webb/"),
            ("html",             "https://www.esa.int/Science_Exploration/Space_Science"),
            ("html",             "https://phys.org/tags/space/"),
            ("html",             "https://www.sciencedaily.com/news/space_time/"),
            ("html",             "https://spacenews.com/"),
            ("rss",              "https://www.nasa.gov/news-release/feed/"),
            ("rss",              "https://www.sciencedaily.com/rss/space_time.xml"),
            ("rss",              "https://phys.org/rss-feed/space-news/"),
        ],
    },

    "programming": {
        "keywords": [
            "Python", "async await", "garbage collector", "type hint", "GIL",
            "compiler", "bytecode", "hash table", "decorator", "generator",
            "REST API", "Docker", "Kubernetes", "microservices", "CI/CD",
            "data structure", "algorithm", "concurrency", "memory management",
            "functional programming", "object-oriented", "test-driven development",
        ],
        "seeds": [
            ("wikipedia",        "CPython"),
            ("wikipedia",        "Async/await"),
            ("wikipedia",        "Hash_table"),
            ("wikipedia",        "Python_(programming_language)"),
            ("wikipedia",        "Object-oriented_programming"),
            ("wikipedia",        "Functional_programming"),
            ("wikipedia",        "Garbage_collection_(computer_science)"),
            ("wikipedia",        "Docker_(software)"),
            ("wikipedia",        "Kubernetes"),
            ("wikipedia",        "Microservices"),
            ("wikipedia",        "Representational_state_transfer"),
            ("wikipedia",        "Git"),
            ("wikipedia",        "Linux_kernel"),
            ("wikipedia",        "Compiler"),
            ("wikipedia",        "Dynamic_programming"),
            ("wikipedia",        "Big_O_notation"),
            ("wikipedia",        "Concurrent_computing"),
            ("wikipedia",        "Test-driven_development"),
            ("wikipedia",        "Continuous_integration"),
            ("wikipedia",        "SQL"),
            ("wikipedia",        "Type_system"),
            ("wikipedia",        "Memory_management"),
            ("wikipedia",        "Event-driven_programming"),
            ("stackoverflow",    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python"),
            ("stackoverflow",    "https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python"),
            ("stackoverflow",    "https://stackoverflow.com/questions/419163/what-does-if-name-main-do"),
            ("stackoverflow",    "https://stackoverflow.com/questions/1024559/when-to-use-os-name-posix-vs-os-name-nt"),
            ("stackoverflow",    "https://stackoverflow.com/questions/2709821/what-is-the-difference-between-str-and-repr"),
            ("stackoverflow",    "https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference-in-python"),
            ("stackoverflow",    "https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary"),
            ("stackoverflow",    "https://stackoverflow.com/questions/739654/how-to-make-function-decorators-and-chain-them-together"),
            ("stackoverflow",    "https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters"),
            ("stackoverflow",    "https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument"),
            ("stackoverflow",    "https://stackoverflow.com/questions/4166/how-do-you-calculate-the-big-o-notation-of-a-function"),
            ("stackoverflow",    "https://stackoverflow.com/questions/3047006/is-it-possible-to-run-python-on-android"),
            ("stackoverflow",    "https://stackoverflow.com/questions/6470428/catch-multiple-exceptions-in-one-line"),
            ("html",             "https://docs.python.org/3/reference/datamodel.html"),
            ("html",             "https://docs.python.org/3/library/asyncio-task.html"),
            ("html",             "https://docs.python.org/3/tutorial/classes.html"),
            ("html",             "https://docs.python.org/3/library/collections.html"),
            ("html",             "https://docs.python.org/3/library/functools.html"),
            ("html",             "https://docs.python.org/3/howto/sorting.html"),
            ("html",             "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),
            ("html",             "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise"),
            ("html",             "https://realpython.com/python-gil/"),
            ("html",             "https://realpython.com/python-concurrency/"),
            ("html",             "https://realpython.com/python-decorators/"),
            ("html",             "https://realpython.com/python-type-checking/"),
            ("devto",            "python"),
            ("devto",            "machinelearning"),
            ("devto",            "webdev"),
            ("devto",            "docker"),
            ("devto",            "typescript"),
            ("devto",            "rust"),
            ("github",           "psf/requests"),
            ("github",           "tiangolo/fastapi"),
            ("github",           "pallets/flask"),
            ("github",           "django/django"),
            ("github",           "python/cpython"),
            ("github",           "rust-lang/rust"),
            ("github",           "golang/go"),
            ("github",           "microsoft/TypeScript"),
            ("github",           "nodejs/node"),
            ("openalex",         "Python programming language performance optimization"),
            ("openalex",         "software engineering testing continuous integration survey"),
            ("openalex",         "microservices architecture patterns cloud native"),
            ("arxiv",            "compiler optimisation static analysis program verification"),
            ("arxiv",            "concurrent programming lock-free data structure"),
            ("html",             "https://www.th-nuernberg.de/"),
            ("html",             "https://www.accenture.com/en-us/industries"),
            ("rss",              "https://phys.org/rss-feed/technology-news/computer-sciences/"),
            ("rss",              "https://news.mit.edu/rss/topic/computer-science-and-technology"),
        ],
    },

    "ml_algorithms": {
        "keywords": [
            "world model", "transformer", "diffusion model", "reinforcement learning",
            "attention mechanism", "graph neural network", "variational autoencoder",
            "contrastive learning", "model-based RL", "sparse autoencoder",
            "state space model", "Mamba", "RLHF", "latent space",
            "large language model", "neural network", "deep learning",
            "gradient descent", "backpropagation", "transfer learning",
        ],
        "seeds": [
            ("wikipedia",        "Transformer_(deep_learning_architecture)"),
            ("wikipedia",        "Diffusion_model"),
            ("wikipedia",        "Variational_autoencoder"),
            ("wikipedia",        "Reinforcement_learning"),
            ("wikipedia",        "Graph_neural_network"),
            ("wikipedia",        "Attention_(machine_learning)"),
            ("wikipedia",        "Generative_adversarial_network"),
            ("wikipedia",        "Large_language_model"),
            ("wikipedia",        "Convolutional_neural_network"),
            ("wikipedia",        "Recurrent_neural_network"),
            ("wikipedia",        "Backpropagation"),
            ("wikipedia",        "Gradient_descent"),
            ("wikipedia",        "Random_forest"),
            ("wikipedia",        "Support-vector_machine"),
            ("wikipedia",        "Transfer_learning"),
            ("wikipedia",        "Federated_learning"),
            ("wikipedia",        "Prompt_engineering"),
            ("wikipedia",        "Retrieval-augmented_generation"),
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
            ("arxiv",            "large language model instruction tuning fine-tuning"),
            ("arxiv",            "vision transformer image recognition patch embedding"),
            ("arxiv",            "retrieval augmented generation RAG knowledge grounding"),
            ("semantic_scholar", "world model model-based reinforcement learning survey"),
            ("semantic_scholar", "transformer self-attention survey vision language"),
            ("semantic_scholar", "diffusion generative model image synthesis"),
            ("semantic_scholar", "mechanistic interpretability circuits features neural network"),
            ("openalex",         "world model model-based reinforcement learning survey"),
            ("openalex",         "diffusion probabilistic model generative image"),
            ("openalex",         "large language model benchmark evaluation reasoning"),
            ("openalex",         "federated learning privacy preserving distributed training"),
            ("github",           "google-deepmind/dreamerv3"),
            ("github",           "huggingface/diffusers"),
            ("github",           "state-spaces/mamba"),
            ("github",           "openai/gym"),
            ("github",           "huggingface/transformers"),
            ("github",           "pytorch/pytorch"),
            ("github",           "tensorflow/tensorflow"),
            ("stackoverflow",    "https://stackoverflow.com/questions/55243483/what-is-the-difference-between-model-free-and-model-based-reinforcement-learning"),
            ("devto",            "deeplearning"),
            ("devto",            "ai"),
            ("html",             "https://phys.org/tags/artificial+intelligence/"),
            ("html",             "https://www.sciencedaily.com/news/computers_math/artificial_intelligence/"),
            ("html",             "https://paperswithcode.com/sota"),
            ("html",             "https://distill.pub/"),
            ("rss",              "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml"),
            ("rss",              "https://phys.org/rss-feed/technology-news/artificial-intelligence-news/"),
            ("rss",              "https://www.quantamagazine.org/feed/"),
        ],
    },

    "football": {
        "keywords": [
            "association football", "pressing", "expected goals", "tiki-taka",
            "gegenpressing", "UEFA Champions League", "FIFA World Cup",
            "high press", "false nine", "ball possession", "xG", "transfer market",
            "Premier League", "VAR", "football analytics", "football tactics",
        ],
        "seeds": [
            ("wikipedia",        "Pressing_(association_football)"),
            ("wikipedia",        "Expected_goals"),
            ("wikipedia",        "Tiki-taka"),
            ("wikipedia",        "UEFA_Champions_League"),
            ("wikipedia",        "History_of_association_football"),
            ("wikipedia",        "Association_football_tactics_and_skills"),
            ("wikipedia",        "Premier_League"),
            ("wikipedia",        "FIFA_World_Cup"),
            ("wikipedia",        "Association_football"),
            ("wikipedia",        "Pep_Guardiola"),
            ("wikipedia",        "Video_assistant_referee"),
            ("wikipedia",        "Football_analytics"),
            ("wikipedia",        "Offside_(association_football)"),
            ("wikipedia",        "Transfer_window"),
            ("wikipedia",        "UEFA_Europa_League"),
            ("wikipedia",        "Serie_A"),
            ("wikipedia",        "Bundesliga"),
            ("openalex",         "expected goals football match prediction model"),
            ("openalex",         "pressing intensity football tactical analysis"),
            ("openalex",         "football transfer market player valuation machine learning"),
            ("openalex",         "football tracking data player performance GPS"),
            ("openalex",         "football injury prevention workload monitoring"),
            ("arxiv",            "football soccer match outcome prediction machine learning"),
            ("arxiv",            "player tracking event data football performance"),
            ("arxiv",            "expected goals xG model football shot quality"),
            ("arxiv",            "football formation pressing tactics graph network"),
            ("arxiv",            "sports analytics football passing network possession"),
            ("semantic_scholar", "expected goals xG football analytics sports science"),
            ("semantic_scholar", "football tactical pressing high block analysis"),
            ("semantic_scholar", "sports analytics player performance prediction model"),
            ("wikidata",         "association football tactic"),
            ("wikidata",         "FIFA World Cup"),
            ("github",           "friends-of-tracking-data/LaurieOnTracking"),
            ("github",           "devinpleuler/analytics-handbook"),
            ("github",           "metrica-sports/sample-data"),
            ("devto",            "football"),
            ("devto",            "sportsanalytics"),
            ("stackoverflow",    "https://stackoverflow.com/questions/tagged/sports-analytics"),
            ("html",             "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
            ("html",             "https://www.bbc.com/sport/football"),
            ("html",             "https://statsbomb.com/articles/"),
            ("html",             "https://understat.com/"),
            ("html",             "https://theathletic.com/football/"),
            ("html",             "https://www.optasports.com/news/"),
            ("html",             "https://totalfootballanalysis.com/"),
            ("html",             "https://www.transfermarkt.com/"),
            ("html",             "https://www.uefa.com/uefachampionsleague/news/"),
            ("html",             "https://www.fifa.com/fifaplus/en/articles"),
            ("html",             "https://www.skysports.com/football"),
            ("html",             "https://www.goal.com/en"),
            ("html",             "https://www.theguardian.com/football"),
            ("html",             "https://tacticaljournal.com/"),
            ("rss",              "https://www.bbc.com/sport/football/rss.xml"),
            ("rss",              "https://www.theguardian.com/football/rss"),
        ],
    },

    "medicine": {
        "keywords": [
            "CRISPR", "mRNA vaccine", "microbiome", "immunotherapy", "oncology",
            "stem cell", "antibiotic resistance", "gene editing", "proteomics",
            "clinical trial", "epigenetics", "pathogen",
            "Alzheimer", "cancer", "diabetes", "COVID-19", "drug discovery",
        ],
        "seeds": [
            ("wikipedia",        "CRISPR"),
            ("wikipedia",        "MRNA_vaccine"),
            ("wikipedia",        "Gut_microbiota"),
            ("wikipedia",        "Cancer_immunotherapy"),
            ("wikipedia",        "Antibiotic_resistance"),
            ("wikipedia",        "Alzheimer%27s_disease"),
            ("wikipedia",        "Cancer"),
            ("wikipedia",        "Diabetes_mellitus"),
            ("wikipedia",        "COVID-19"),
            ("wikipedia",        "Vaccine"),
            ("wikipedia",        "Stem_cell"),
            ("wikipedia",        "Gene_therapy"),
            ("wikipedia",        "Proteomics"),
            ("wikipedia",        "Epigenetics"),
            ("wikipedia",        "Drug_discovery"),
            ("wikipedia",        "Personalized_medicine"),
            ("wikipedia",        "Oncology"),
            ("openalex",         "CRISPR Cas9 gene editing therapeutic application"),
            ("openalex",         "mRNA vaccine immunogenicity clinical trial"),
            ("openalex",         "gut microbiome dysbiosis disease biomarker"),
            ("openalex",         "cancer immunotherapy checkpoint inhibitor PD-1"),
            ("openalex",         "antibiotic resistance mechanism horizontal gene transfer"),
            ("openalex",         "Alzheimer neurodegeneration amyloid tau pathology"),
            ("openalex",         "drug discovery machine learning molecular docking"),
            ("arxiv",            "microbiome gut bacteria disease machine learning"),
            ("arxiv",            "protein structure prediction AlphaFold drug discovery"),
            ("arxiv",            "cancer genomics mutation driver gene expression"),
            ("arxiv",            "CRISPR off-target edit safety in-vivo delivery"),
            ("arxiv",            "clinical trial design randomised controlled evidence"),
            ("arxiv",            "mRNA lipid nanoparticle vaccine delivery immune"),
            ("semantic_scholar", "CRISPR base editing prime editing therapeutic"),
            ("semantic_scholar", "cancer immunotherapy response biomarker tumour microenvironment"),
            ("semantic_scholar", "gut microbiome brain axis neurological mental health"),
            ("semantic_scholar", "antibiotic resistance ESKAPE pathogen clinical threat"),
            ("wikidata",         "gene therapy"),
            ("wikidata",         "mRNA vaccine"),
            ("github",           "deepmind/alphafold"),
            ("github",           "facebookresearch/esm"),
            ("github",           "chemprop/chemprop"),
            ("html",             "https://www.nih.gov/news-events/news-releases"),
            ("html",             "https://www.cdc.gov/nceh/features/"),
            ("html",             "https://www.mayoclinic.org/diseases-conditions"),
            ("html",             "https://phys.org/tags/medicine/"),
            ("html",             "https://www.sciencedaily.com/news/health_medicine/"),
            ("rss",              "https://www.sciencedaily.com/rss/health_medicine.xml"),
            ("rss",              "https://phys.org/rss-feed/health-news/"),
            ("rss",              "https://www.nih.gov/rss/alldiscoveries.xml"),
        ],
    },

    "quantum_computing": {
        "keywords": [
            "qubit", "quantum gate", "superposition", "entanglement",
            "quantum error correction", "variational quantum eigensolver",
            "NISQ", "quantum supremacy", "Shor's algorithm", "quantum annealing",
            "topological qubit", "quantum circuit",
            "quantum cryptography", "quantum key distribution",
        ],
        "seeds": [
            ("wikipedia",        "Quantum_computing"),
            ("wikipedia",        "Qubit"),
            ("wikipedia",        "Shor%27s_algorithm"),
            ("wikipedia",        "Quantum_error_correction"),
            ("wikipedia",        "Variational_quantum_eigensolver"),
            ("wikipedia",        "Quantum_entanglement"),
            ("wikipedia",        "Quantum_superposition"),
            ("wikipedia",        "Quantum_gate"),
            ("wikipedia",        "Topological_quantum_computer"),
            ("wikipedia",        "Quantum_cryptography"),
            ("wikipedia",        "Quantum_supremacy"),
            ("wikipedia",        "Quantum_annealing"),
            ("wikipedia",        "Grover%27s_algorithm"),
            ("wikipedia",        "Quantum_teleportation"),
            ("arxiv",            "variational quantum eigensolver NISQ hybrid algorithm"),
            ("arxiv",            "quantum error correction surface code fault tolerant"),
            ("arxiv",            "quantum machine learning kernel classification"),
            ("arxiv",            "topological qubit anyons quantum computation"),
            ("arxiv",            "quantum key distribution BB84 quantum cryptography"),
            ("arxiv",            "quantum advantage computational complexity oracle"),
            ("arxiv",            "quantum circuit noise mitigation error mitigation"),
            ("arxiv",            "quantum simulation chemistry Hamiltonian variational"),
            ("arxiv",            "post-quantum cryptography lattice-based algorithm"),
            ("semantic_scholar", "NISQ quantum advantage near-term algorithm"),
            ("semantic_scholar", "quantum computing hardware superconducting transmon qubit"),
            ("semantic_scholar", "quantum error correction code logical qubit threshold"),
            ("openalex",         "quantum computing algorithm optimization survey"),
            ("openalex",         "quantum cryptography key distribution protocol security"),
            ("openalex",         "variational quantum algorithm hybrid classical quantum"),
            ("github",           "Qiskit/qiskit"),
            ("github",           "quantumlib/Cirq"),
            ("github",           "microsoft/QuantumKatas"),
            ("github",           "PennyLaneAI/pennylane"),
            ("html",             "https://www.quantamagazine.org/tag/quantum-computing/"),
            ("html",             "https://phys.org/tags/quantum+computing/"),
            ("html",             "https://www.sciencedaily.com/news/computers_math/quantum_computers/"),
            ("html",             "https://research.ibm.com/quantum-computing"),
            ("rss",              "https://www.sciencedaily.com/rss/computers_math/quantum_computers.xml"),
            ("rss",              "https://phys.org/rss-feed/technology-news/quantum-physics-news/"),
            ("rss",              "https://www.quantamagazine.org/feed/"),
        ],
    },

    "neuroscience": {
        "keywords": [
            "neuron", "synapse", "connectome", "synaptic plasticity",
            "default mode network", "hippocampus", "dopamine", "cortex",
            "action potential", "brain-computer interface", "neuroplasticity",
            "predictive coding", "free energy principle",
            "neurotransmitter", "neurodegeneration", "cerebral cortex",
        ],
        "seeds": [
            ("wikipedia",        "Neuron"),
            ("wikipedia",        "Connectome"),
            ("wikipedia",        "Synaptic_plasticity"),
            ("wikipedia",        "Default_mode_network"),
            ("wikipedia",        "Predictive_coding"),
            ("wikipedia",        "Brain%E2%80%93computer_interface"),
            ("wikipedia",        "Hippocampus"),
            ("wikipedia",        "Dopamine"),
            ("wikipedia",        "Neuroplasticity"),
            ("wikipedia",        "Action_potential"),
            ("wikipedia",        "Cerebral_cortex"),
            ("wikipedia",        "Amygdala"),
            ("wikipedia",        "Neurotransmitter"),
            ("wikipedia",        "Neurodegeneration"),
            ("wikipedia",        "Serotonin"),
            ("wikipedia",        "Long-term_potentiation"),
            ("arxiv",            "predictive coding free energy principle brain"),
            ("arxiv",            "connectome neural circuit mapping"),
            ("arxiv",            "brain computer interface EEG decoding motor"),
            ("arxiv",            "dopamine reward prediction error reinforcement"),
            ("arxiv",            "hippocampus place cell grid cell spatial memory"),
            ("arxiv",            "synaptic plasticity long-term potentiation memory consolidation"),
            ("arxiv",            "cortical oscillation neural synchrony gamma theta"),
            ("arxiv",            "neurodegenerative disease protein aggregation propagation"),
            ("semantic_scholar", "free energy principle active inference Karl Friston"),
            ("semantic_scholar", "connectomics electron microscopy synapse reconstruction"),
            ("semantic_scholar", "brain computer interface neural decoding prosthetic"),
            ("openalex",         "hippocampus memory consolidation spatial navigation"),
            ("openalex",         "dopamine reward learning striatum basal ganglia"),
            ("openalex",         "neuroplasticity cortical reorganisation learning recovery"),
            ("openalex",         "default mode network resting state fMRI connectivity"),
            ("openalex",         "neurodegenerative disease Alzheimer Parkinson protein aggregation"),
            ("wikidata",         "synapse"),
            ("wikidata",         "neurotransmitter"),
            ("github",           "NeuralEnsemble/elephant"),
            ("github",           "brian-team/brian2"),
            ("html",             "https://phys.org/tags/neuroscience/"),
            ("html",             "https://www.sciencedaily.com/news/mind_brain/"),
            ("html",             "https://www.nih.gov/news-events/nih-research-matters"),
            ("html",             "https://www.quantamagazine.org/tag/neuroscience/"),
            ("rss",              "https://www.sciencedaily.com/rss/mind_brain.xml"),
            ("rss",              "https://phys.org/rss-feed/health-news/neuroscience/"),
        ],
    },

    "economics": {
        "keywords": [
            "game theory", "Nash equilibrium", "mechanism design", "auction",
            "behavioral economics", "nudge", "market microstructure",
            "monetary policy", "inflation", "Pareto efficiency",
            "public goods", "externality", "information asymmetry",
            "Keynesian", "fiscal policy", "GDP", "interest rate",
        ],
        "seeds": [
            ("wikipedia",        "Game_theory"),
            ("wikipedia",        "Nash_equilibrium"),
            ("wikipedia",        "Mechanism_design"),
            ("wikipedia",        "Behavioral_economics"),
            ("wikipedia",        "Auction_theory"),
            ("wikipedia",        "Information_asymmetry"),
            ("wikipedia",        "Keynesian_economics"),
            ("wikipedia",        "Fiscal_policy"),
            ("wikipedia",        "Monetary_policy"),
            ("wikipedia",        "International_trade"),
            ("wikipedia",        "Stock_market"),
            ("wikipedia",        "Supply_and_demand"),
            ("wikipedia",        "Gross_domestic_product"),
            ("wikipedia",        "Inflation"),
            ("wikipedia",        "Comparative_advantage"),
            ("wikipedia",        "Externality"),
            ("arxiv",            "mechanism design algorithmic game theory auction"),
            ("arxiv",            "behavioral economics nudge choice architecture"),
            ("arxiv",            "monetary policy inflation central bank interest rate"),
            ("arxiv",            "asset pricing risk premium stock market return"),
            ("arxiv",            "trade policy tariff welfare comparative advantage"),
            ("semantic_scholar", "Nash equilibrium evolutionary game theory"),
            ("semantic_scholar", "auction design revenue equivalence Vickrey"),
            ("semantic_scholar", "fiscal policy multiplier government spending output"),
            ("openalex",         "behavioral economics prospect theory Kahneman"),
            ("openalex",         "monetary policy inflation central bank"),
            ("openalex",         "market microstructure liquidity bid ask spread"),
            ("openalex",         "income inequality Gini coefficient redistribution"),
            ("openalex",         "international trade comparative advantage supply chain"),
            ("wikidata",         "game theory"),
            ("wikidata",         "behavioral economics"),
            ("html",             "https://www.imf.org/en/Publications/WEO"),
            ("html",             "https://www.oecd.org/en/topics/economy.html"),
            ("html",             "https://www.weforum.org/agenda/archive/economics/"),
            ("html",             "https://www.worldbank.org/en/topic/macroeconomics"),
            ("html",             "https://phys.org/tags/economics/"),
            ("html",             "https://www.sciencedaily.com/news/mind_brain/economics/"),
            ("rss",              "https://www.sciencedaily.com/rss/mind_brain/economics.xml"),
            ("rss",              "https://phys.org/rss-feed/social-sciences-news/economics/"),
        ],
    },

    "smart_manufacturing": {
        "keywords": [
            "Industry 4.0", "digital twin", "IIoT", "industrial IoT",
            "additive manufacturing", "cyber-physical system",
            "predictive maintenance", "smart factory", "CNC machining",
            "SCADA", "MES", "PLM", "lean manufacturing", "cobots",
            "robot automation", "3D printing", "quality control",
        ],
        "seeds": [
            ("wikipedia",        "Industry_4.0"),
            ("wikipedia",        "Digital_twin"),
            ("wikipedia",        "Additive_manufacturing"),
            ("wikipedia",        "Cyber-physical_system"),
            ("wikipedia",        "Industrial_Internet_of_Things"),
            ("wikipedia",        "Predictive_maintenance"),
            ("wikipedia",        "Computer-integrated_manufacturing"),
            ("wikipedia",        "Collaborative_robot"),
            ("wikipedia",        "Smart_manufacturing"),
            ("wikipedia",        "SCADA"),
            ("wikipedia",        "Manufacturing_execution_system"),
            ("wikipedia",        "Lean_manufacturing"),
            ("wikipedia",        "Six_Sigma"),
            ("wikipedia",        "Computer_numerical_control"),
            ("wikipedia",        "Flexible_manufacturing_system"),
            ("wikipedia",        "Digital_manufacturing"),
            ("openalex",         "Industry 4.0 smart factory digital twin survey"),
            ("openalex",         "predictive maintenance machine learning industrial IoT"),
            ("openalex",         "cyber-physical system manufacturing automation"),
            ("openalex",         "additive manufacturing 3D printing process parameters"),
            ("openalex",         "lean manufacturing waste reduction Industry 4.0"),
            ("openalex",         "industrial robot collaborative human-robot interaction"),
            ("arxiv",            "digital twin manufacturing simulation real-time"),
            ("arxiv",            "IIoT industrial internet of things anomaly detection"),
            ("arxiv",            "additive manufacturing 3D printing process optimisation"),
            ("arxiv",            "federated learning industrial IoT edge computing"),
            ("arxiv",            "reinforcement learning robot arm control manipulation"),
            ("arxiv",            "quality control defect detection computer vision manufacturing"),
            ("arxiv",            "smart factory energy efficiency scheduling optimisation"),
            ("semantic_scholar", "Industry 4.0 smart manufacturing review challenges"),
            ("semantic_scholar", "digital twin cyber-physical production system"),
            ("semantic_scholar", "predictive maintenance deep learning vibration sensor"),
            ("github",           "apache/plc4x"),
            ("github",           "eclipse/ditto"),
            ("github",           "ros-planning/navigation2"),
            ("github",           "Industrial-IoT/Industrial-IoT"),
            ("github",           "Azure/Industrial-IoT"),
            ("github",           "eclipse-cyclonedds/cyclonedds"),
            ("devto",            "iot"),
            ("devto",            "robotics"),
            ("devto",            "industry40"),
            ("stackoverflow",    "https://stackoverflow.com/questions/tagged/industrial-automation"),
            ("stackoverflow",    "https://stackoverflow.com/questions/tagged/plc"),
            ("html",             "https://www.nist.gov/manufacturing"),
            ("html",             "https://www.weforum.org/agenda/archive/advanced-manufacturing/"),
            ("html",             "https://phys.org/tags/manufacturing/"),
            ("html",             "https://www.sciencedaily.com/news/matter_energy/engineering/"),
            ("html",             "https://www.iea.org/topics/industry"),
            ("html",             "https://news.mit.edu/topic/manufacturing"),
            ("html",             "https://www.siemens.com/trends/industrialization-services"),
            ("html",             "https://www.siemens.com/industrial-automation"),
            ("html",             "https://www.th-nuernberg.de/"),
            ("html",             "https://www.accenture.com/en-us/industries"),
            ("html",             "https://ec.europa.eu/growth/industry/policy/"),
            ("html",             "https://www.automationworld.com/"),
            ("html",             "https://www.manufacturingtomorrow.com/"),
            ("html",             "https://www.industryweek.com/technology-and-iiot"),
            ("rss",              "https://www.sciencedaily.com/rss/matter_energy/engineering.xml"),
            ("rss",              "https://phys.org/rss-feed/technology-news/engineering/"),
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
            ("html", "https://www.carbonbrief.org/"),
            ("html", "https://www.carbonbrief.org/explainers/"),
            ("html", "https://phys.org/tags/climate+change/"),
            ("html", "https://www.sciencedaily.com/news/earth_climate/global_warming/"),
            ("html", "https://www.ipcc.ch/report/ar6/syr/"),
            ("html", "https://www.eco2050.de/"),
            ("html", "https://ec.europa.eu/climate/"),
            ("html", "https://www.un.org/en/climatechange/"),
            ("html", "https://www.bundestag.de/dokumente/textarchiv/themenbereich-klima"),
        ],
    },
    "sustainable_materials": {
        "keywords": TOPIC_CLUSTERS["sustainable_materials"]["keywords"],
        "seeds": [
            ("html", "https://ellenmacarthurfoundation.org/topics/circular-economy/overview"),
            ("html", "https://www.unep.org/topics/chemicals-waste/plastics"),
            ("html", "https://www.weforum.org/agenda/archive/circular-economy/"),
            ("html", "https://phys.org/tags/biodegradable/"),
            ("html", "https://www.sciencedaily.com/news/matter_energy/biomaterials/"),
            ("html", "https://www.oecd.org/en/topics/circular-economy.html"),
            ("html", "https://www.eco2050.de/"),
            ("html", "https://ec.europa.eu/environment/circular-economy/"),
            ("html", "https://www.unep.org/resources/report"),
        ],
    },
    "space": {
        "keywords": TOPIC_CLUSTERS["space"]["keywords"],
        "seeds": [
            ("html", "https://www.nasa.gov/news/"),
            ("html", "https://www.esa.int/Newsroom"),
            ("html", "https://blogs.nasa.gov/artemis/"),
            ("html", "https://www.quantamagazine.org/tag/cosmology/"),
            ("html", "https://spacenews.com/"),
            ("html", "https://phys.org/tags/space/"),
            ("html", "https://www.sciencedaily.com/news/space_time/"),
        ],
    },
    "ml_algorithms": {
        "keywords": TOPIC_CLUSTERS["ml_algorithms"]["keywords"],
        "seeds": [
            ("html", "https://www.microsoft.com/en-us/research/research-area/artificial-intelligence/"),
            ("html", "https://news.mit.edu/topic/artificial-intelligence2"),
            ("html", "https://www.quantamagazine.org/tag/artificial-intelligence/"),
            ("html", "https://www.weforum.org/agenda/archive/artificial-intelligence/"),
            ("html", "https://phys.org/tags/artificial+intelligence/"),
            ("html", "https://www.sciencedaily.com/news/computers_math/artificial_intelligence/"),
            ("html", "https://paperswithcode.com/sota"),
            ("html", "https://distill.pub/"),
        ],
    },
    "programming": {
        "keywords": TOPIC_CLUSTERS["programming"]["keywords"],
        "seeds": [
            ("html", "https://news.mit.edu/topic/computers"),
            ("html", "https://www.weforum.org/agenda/archive/data-science/"),
            ("html", "https://phys.org/tags/computer+science/"),
            ("html", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"),
            ("html", "https://realpython.com/python-gil/"),
            ("html", "https://realpython.com/python-concurrency/"),
        ],
    },
    "economics": {
        "keywords": TOPIC_CLUSTERS["economics"]["keywords"],
        "seeds": [
            ("html", "https://www.imf.org/en/Publications/WEO"),
            ("html", "https://www.oecd.org/en/topics/economy.html"),
            ("html", "https://www.weforum.org/agenda/archive/economics/"),
            ("html", "https://www.worldbank.org/en/topic/macroeconomics"),
            ("html", "https://phys.org/tags/economics/"),
            ("html", "https://www.sciencedaily.com/news/mind_brain/economics/"),
        ],
    },
    "medicine": {
        "keywords": TOPIC_CLUSTERS["medicine"]["keywords"],
        "seeds": [
            ("html", "https://www.who.int/news-room/fact-sheets"),
            ("html", "https://www.nih.gov/news-events/news-releases"),
            ("html", "https://www.quantamagazine.org/tag/biology/"),
            ("html", "https://www.cdc.gov/nceh/features/"),
            ("html", "https://www.mayoclinic.org/diseases-conditions"),
            ("html", "https://phys.org/tags/medicine/"),
            ("html", "https://www.sciencedaily.com/news/health_medicine/"),
        ],
    },
    "football": {
        "keywords": TOPIC_CLUSTERS["football"]["keywords"],
        "seeds": [
            ("html", "https://www.bbc.com/sport/football"),
            ("html", "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
            ("html", "https://statsbomb.com/articles/"),
            ("html", "https://understat.com/"),
        ],
    },
    "quantum_computing": {
        "keywords": TOPIC_CLUSTERS["quantum_computing"]["keywords"],
        "seeds": [
            ("html", "https://www.quantamagazine.org/tag/quantum-computing/"),
            ("html", "https://news.mit.edu/topic/quantum-computing"),
            ("html", "https://phys.org/tags/quantum+computing/"),
            ("html", "https://www.sciencedaily.com/news/computers_math/quantum_computers/"),
            ("html", "https://research.ibm.com/quantum-computing"),
        ],
    },
    "neuroscience": {
        "keywords": TOPIC_CLUSTERS["neuroscience"]["keywords"],
        "seeds": [
            ("html", "https://www.quantamagazine.org/tag/neuroscience/"),
            ("html", "https://www.nih.gov/research-training/research-topics/neuroscience"),
            ("html", "https://phys.org/tags/neuroscience/"),
            ("html", "https://www.sciencedaily.com/news/mind_brain/"),
        ],
    },
    "smart_manufacturing": {
        "keywords": TOPIC_CLUSTERS["smart_manufacturing"]["keywords"],
        "seeds": [
            ("html", "https://www.weforum.org/agenda/archive/advanced-manufacturing/"),
            ("html", "https://www.iea.org/topics/industry"),
            ("html", "https://news.mit.edu/topic/manufacturing"),
            ("html", "https://www.oecd.org/en/topics/industry-and-entrepreneurship.html"),
            ("html", "https://phys.org/tags/manufacturing/"),
            ("html", "https://www.sciencedaily.com/news/matter_energy/engineering/"),
            ("html", "https://www.nist.gov/manufacturing"),
            ("html", "https://www.siemens.com/trends/industrialization-services"),
            ("html", "https://www.siemens.com/industrial-automation"),
            ("html", "https://www.th-nuernberg.de/"),
            ("html", "https://www.accenture.com/en-us/industries"),
            ("html", "https://ec.europa.eu/growth/industry/policy/"),
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

CROSS_THRESHOLD   = 3   # keyword hits before cross-topic seed injection
WIKI_FOLLOW_LIMIT = 30  # max Wikipedia links re-queued per article at depth > 0
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
    # science news
    "phys.org", "www.sciencedaily.com",
    "www.carbonbrief.org",
    "spacenews.com",
    # programming
    "realpython.com", "developer.mozilla.org",
    # medicine
    "www.mayoclinic.org", "www.cdc.gov",
    # quantum / ML
    "research.ibm.com", "paperswithcode.com", "distill.pub",
    # football analytics
    "statsbomb.com", "understat.com",
    # manufacturing
    "www.nist.gov",
    # jury contacts & industry
    "www.siemens.com", "th-nuernberg.de", "www.accenture.com",
    "www.eco2050.de",
    # EU & international
    "ec.europa.eu", "www.un.org", "www.bundestag.de", "www.bundesrat.de",
})

# Per-host patterns for links to skip
_SKIP: dict[str, re.Pattern] = {
    "en.wikipedia.org":    re.compile(r"/(Special|Talk|User|File|Help|Category|Wikipedia|Portal|Template):"),
    "simple.wikipedia.org":re.compile(r"/(Special|Talk|User|File|Help|Category|Wikipedia):"),
    "docs.python.org":     re.compile(r"genindex|py-modindex|search\.html"),
    "scikit-learn.org":    re.compile(r"/stable/api/|generated/sklearn\.|_downloads"),
    "stackoverflow.com":   re.compile(r"/users/|/tags/|/questions/tagged/|/jobs/"),
    "phys.org":            re.compile(r"/search/|/account/|/sitemap"),
    "www.sciencedaily.com":re.compile(r"/releases/\d{4}/|/search/"),
    "realpython.com":      re.compile(r"/search/|/account/|/community/"),
    "developer.mozilla.org": re.compile(r"/en-US/search|/en-US/plus"),
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

def _base_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}" if p.netloc else url

def _doc(*, url: str, title: str, body: str, links: list[str],
         source: str, source_base: str, topic: str) -> dict:
    return {"url": url, "title": title, "body": body,
            "links": links, "source": source, "source_base": source_base, "topic": topic}

def _wiki_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

def _title_from_wiki_url(url: str) -> str:
    return urlparse(url).path.split("/wiki/")[-1]


# ── ADAPTERS ──────────────────────────────────────────────────────────────────

def fetch_wikipedia(title: str, topic: str,
                    max_body: int | None = None, max_links: int | None = None) -> list[dict]:
    r = _get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "format": "json",
        "titles": title.replace("_", " "),
        "prop": "extracts|links",
        "exlimit": 1, "explaintext": True, "exsectionformat": "plain",
        "pllimit": max_links or 500, "plnamespace": 0,
    })
    if not r:
        return []
    docs = []
    for pid, page in r.json().get("query", {}).get("pages", {}).items():
        if pid == "-1" or not page.get("extract"):
            continue
        url   = _wiki_url(page["title"])
        body  = page["extract"][:max_body] if max_body else page["extract"]
        links = [_wiki_url(lk["title"]) for lk in page.get("links", [])]
        docs.append(_doc(url=url, title=page["title"], body=body,
                         links=links, source="wikipedia", source_base=_base_url(url),
                         topic=topic))
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
            url = _wiki_url(title)
            docs.append(_doc(url=url, title=title, body="", links=[],
                             source="wikidata_ref", source_base=_base_url(url),
                             topic=topic))
    return docs


def fetch_wikipedia_category(category: str, topic: str,
                             max_members: int = 200, **_) -> list[dict]:
    """Return stub docs for every article in a Wikipedia category.

    Returned docs have source='wikidata_ref' so the crawl engine re-queues
    them as proper wikipedia fetches — reusing the existing stub-requeue path.
    Strips a leading 'Category:' prefix if the caller included it.
    """
    title = category.removeprefix("Category:").removeprefix("category:")
    r = _get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{title}",
        "cmlimit": max_members, "cmnamespace": 0, "format": "json",
    })
    if not r:
        return []
    docs = []
    for member in r.json().get("query", {}).get("categorymembers", []):
        art_title = member.get("title", "")
        if art_title:
            url = _wiki_url(art_title)
            docs.append(_doc(url=url, title=art_title, body="", links=[],
                             source="wikidata_ref", source_base=_base_url(url),
                             topic=topic))
    return docs


def _rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    return " ".join(w for _, w in sorted(
        (pos, word) for word, positions in inv.items() for pos in positions
    ))

def fetch_openalex(query: str, topic: str,
                   max_body: int | None = None, max_links: int | None = None,
                   max_results: int = 5) -> list[dict]:
    per_page = min(max_results, 200)  # API hard cap is 200
    docs: list[dict] = []
    page = 1
    while len(docs) < max_results:
        r = _get("https://api.openalex.org/works", params={
            "search": query, "per-page": per_page, "page": page,
            "select": "id,title,abstract_inverted_index,concepts,doi",
            "mailto": "research@example.com",
        })
        if not r:
            break
        results = r.json().get("results", [])
        if not results:
            break
        for work in results:
            title = work.get("title") or ""
            body  = _rebuild_abstract(work.get("abstract_inverted_index"))
            body += " " + " ".join(c["display_name"] for c in work.get("concepts", []))
            body  = body.strip()
            body  = body[:max_body] if max_body else body
            url   = work.get("doi") or work.get("id", "")
            if title and body and url:
                docs.append(_doc(url=url, title=title, body=body,
                                 links=[], source="openalex", source_base=_base_url(url),
                                 topic=topic))
            if len(docs) >= max_results:
                break
        if len(results) < per_page:
            break  # last page
        page += 1
        if page > 1:
            time.sleep(0.5)
    return docs


def fetch_arxiv(query: str, topic: str,
                max_body: int | None = None, max_links: int | None = None,
                max_results: int = 5) -> list[dict]:
    per_page = min(max_results, 100)  # stay well within arxiv's rate-limit guidance
    ns = {"a": "http://www.w3.org/2005/Atom"}
    docs: list[dict] = []
    start = 0
    while len(docs) < max_results:
        r = _get("https://export.arxiv.org/api/query", params={
            "search_query": f"all:{query}", "start": start, "max_results": per_page,
        })
        if not r:
            break
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            break
        entries = root.findall("a:entry", ns)
        if not entries:
            break
        for entry in entries:
            def _t(tag: str) -> str:
                el = entry.find(tag, ns)
                return el.text.strip() if el is not None and el.text else ""
            title = _t("a:title")
            raw   = _t("a:summary")
            body  = raw[:max_body] if max_body else raw
            url   = _t("a:id").replace("http://", "https://")
            if title and body and url:
                docs.append(_doc(url=url, title=title, body=body,
                                 links=[], source="arxiv", source_base=_base_url(url),
                                 topic=topic))
            if len(docs) >= max_results:
                break
        if len(entries) < per_page:
            break  # last page
        start += per_page
        if start > 0:
            time.sleep(3.0)  # arxiv asks for a 3-second delay between paged requests
    return docs


def fetch_semantic_scholar(query: str, topic: str,
                           max_body: int | None = None, max_links: int | None = None,
                           max_results: int = 6) -> list[dict]:
    per_page = min(max_results, 100)  # API hard cap is 100
    docs: list[dict] = []
    offset = 0
    while len(docs) < max_results:
        r = _get("https://api.semanticscholar.org/graph/v1/paper/search", params={
            "query": query, "limit": per_page, "offset": offset,
            "fields": "title,abstract,tldr,externalIds,openAccessPdf",
        })
        if not r:
            break
        data = r.json().get("data", [])
        if not data:
            break
        for paper in data:
            title    = paper.get("title") or ""
            abstract = paper.get("abstract") or ""
            tldr     = (paper.get("tldr") or {}).get("text") or ""
            raw      = f"{abstract}\n\nTL;DR: {tldr}" if tldr else abstract
            body     = raw[:max_body] if max_body else raw
            ext      = paper.get("externalIds") or {}
            doi      = ext.get("DOI")
            url      = (f"https://doi.org/{doi}" if doi else
                        (paper.get("openAccessPdf") or {}).get("url") or "")
            if title and body and url:
                docs.append(_doc(url=url, title=title, body=body,
                                 links=[], source="semantic_scholar", source_base=_base_url(url),
                                 topic=topic))
            if len(docs) >= max_results:
                break
        if len(data) < per_page:
            break  # last page
        offset += per_page
        if offset > 0:
            time.sleep(1.0)
    return docs


def fetch_devto(tag: str, topic: str,
                max_body: int | None = None, max_links: int | None = None) -> list[dict]:
    r = _get("https://dev.to/api/articles", params={"tag": tag, "per_page": 6, "top": 1})
    if not r:
        return []
    docs = []
    for article in r.json():
        url   = article.get("url") or ""
        title = article.get("title") or ""
        raw   = BeautifulSoup(article.get("body_html") or "", "html.parser").get_text(" ", strip=True)
        raw   = " ".join(raw.split())
        body  = raw[:max_body] if max_body else raw
        if url and title and body:
            docs.append(_doc(url=url, title=title, body=body,
                             links=[], source="devto", source_base=_base_url(url),
                             topic=topic))
    return docs


_ATOM_NS    = "http://www.w3.org/2005/Atom"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"


def fetch_rss(url: str, topic: str,
              max_body: int | None = None, max_links: int | None = None,
              max_results: int = 20) -> list[dict]:
    """Fetch an RSS 2.0 or Atom feed and return one doc per item/entry.

    Handles both formats transparently. Body is taken from <content:encoded>
    → <description> (RSS) or <content> → <summary> (Atom), with HTML stripped.
    """
    r = _get(url)
    if not r:
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []

    # Detect format: Atom feeds use a namespaced <feed> root tag
    is_atom = _ATOM_NS in (root.tag or "")
    items   = (root.findall(f".//{{{_ATOM_NS}}}entry") if is_atom
               else root.findall(".//item"))

    docs: list[dict] = []
    for item in items:
        if len(docs) >= max_results:
            break

        if is_atom:
            def _a(tag: str) -> str:
                el = item.find(f"{{{_ATOM_NS}}}{tag}")
                return (el.text or "").strip() if el is not None else ""
            title   = _a("title")
            link_el = item.find(f"{{{_ATOM_NS}}}link")
            link    = (link_el.get("href") or "").strip() if link_el is not None else ""
            raw     = _a("content") or _a("summary")
        else:
            def _r(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""
            title   = _r("title")
            link    = _r("link") or _r("guid")
            enc     = item.find(f"{{{_CONTENT_NS}}}encoded")
            raw     = (enc.text or "") if enc is not None else _r("description")

        if not title or not link:
            continue

        body = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True) if raw else ""
        body = " ".join(body.split())
        body = body[:max_body] if max_body else body
        if not body:
            continue

        docs.append(_doc(url=link, title=title, body=body,
                         links=[], source="rss", source_base=_base_url(link),
                         topic=topic))
    return docs


def fetch_github(repo: str, topic: str,
                 max_body: int | None = None, max_links: int | None = None) -> list[dict]:
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
    raw   = " ".join(text.split())
    body  = raw[:max_body] if max_body else raw
    url   = f"https://github.com/{repo}"
    title = repo.split("/")[-1].replace("-", " ").replace("_", " ").title()
    return [_doc(url=url, title=title, body=body,
                 links=[], source="github", source_base=_base_url(url),
                 topic=topic)]


def fetch_html(url: str, topic: str, source: str,
               max_body: int | None = None, max_links: int | None = None) -> list[dict]:
    r = _get(url)
    if not r or "text/html" not in r.headers.get("Content-Type", ""):
        return []
    soup  = BeautifulSoup(r.text, "html.parser")
    host  = urlparse(url).netloc
    title = (soup.find("title") or soup.new_tag("x")).get_text(strip=True) or url
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    content = _body(soup, host)
    raw     = " ".join(content.get_text(" ", strip=True).split()) if content else ""
    body    = raw[:max_body] if max_body else raw
    links   = _links(soup, url, host, max_links)
    return [_doc(url=url, title=title, body=body, links=links,
                 source=source, source_base=_base_url(url), topic=topic)]


def _body(soup: BeautifulSoup, host: str):
    """Pick the most relevant content container for a given host."""
    if "wikipedia.org"      in host: return soup.find(id="mw-content-text")
    if "stackoverflow.com"  in host: return soup.find(id="question") or soup.find(id="answers")
    if "docs.python.org"    in host: return soup.find("div", role="main")
    if "scikit-learn.org"   in host: return soup.find("div", role="main") or soup.find(class_="bd-article")
    if "quantamagazine.org" in host: return soup.find(class_="article__content") or soup.find("article")
    if "weforum.org"        in host: return soup.find(class_="article-body") or soup.find("article")
    return soup.find("main") or soup.find("article") or soup.body


def _links(soup: BeautifulSoup, base_url: str, host: str,
           max_links: int | None = None) -> list[str]:
    skip  = next((p for h, p in _SKIP.items() if h in host), None)
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        href  = urljoin(base_url, a["href"])
        p     = urlparse(href)
        if not p.netloc or p.scheme not in ("http", "https"):
            continue
        clean = p._replace(fragment="").geturl()
        if clean in seen or clean == base_url:
            continue
        if skip and skip.search(clean):
            continue
        seen.add(clean)
        links.append(clean)
        if max_links and len(links) >= max_links:
            break
    return links


# ── DISPATCH ──────────────────────────────────────────────────────────────────

def dispatch(source: str, value: str, topic: str,
             max_body: int | None = None, max_links: int | None = None,
             max_results: int | None = None) -> list[dict]:
    kw = {"max_body": max_body, "max_links": max_links}
    if max_results is not None:
        kw["max_results"] = max_results
    match source:
        case "wikipedia":          return fetch_wikipedia(value, topic, **kw)
        case "wikipedia_category": return fetch_wikipedia_category(value, topic, **kw)
        case "wikidata":           return fetch_wikidata(value, topic)
        case "openalex":         return fetch_openalex(value, topic, **kw)
        case "arxiv":            return fetch_arxiv(value, topic, **kw)
        case "semantic_scholar": return fetch_semantic_scholar(value, topic, **kw)
        case "devto":            return fetch_devto(value, topic, **kw)
        case "rss":              return fetch_rss(value, topic, **kw)
        case "github":           return fetch_github(value, topic, **kw)
        case "scikit":           return fetch_html(value, topic, "scikit", **kw)
        case "stackoverflow":    return fetch_html(value, topic, "stackoverflow", **kw)
        case "html":             return fetch_html(value, topic, "html", **kw)
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

def _filename(url: str, topic: str, title: str = "") -> str:
    if title:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:80] or "index"
    else:
        p    = urlparse(url)
        slug = re.sub(r"[^a-z0-9]+", "_", p.path.lower()).strip("_") or "index"
    return f"{topic}__{slug}.json"


def save(data: dict, out_dir: Path, overwrite: bool = False) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _filename(data["url"], data["topic"], data.get("title", ""))
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
    clusters:     dict[str, dict],
    depth:        int,
    max_pages:    int,
    out_dir:      Path,
    delay:        float,
    overwrite:    bool = False,
    max_body:     int | None = None,
    max_links:    int | None = None,
    max_results:  int | None = None,
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
            docs = dispatch(source, value, topic, max_body=max_body, max_links=max_links,
                            max_results=max_results)

            for doc in docs:
                # Wikidata returns placeholder docs → re-queue as wikipedia
                if doc["source"] == "wikidata_ref":
                    wt  = unquote(_title_from_wiki_url(doc["url"]))
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
                if dlevel < depth:
                    if source == "wikipedia":
                        # Links are Wikipedia URLs — extract title and re-queue
                        # via the API rather than raw HTML. Shuffle so each run
                        # explores a different neighbourhood.
                        wiki_links = doc["links"][:]
                        random.shuffle(wiki_links)
                        for link in wiki_links[:WIKI_FOLLOW_LIMIT]:
                            art = unquote(_title_from_wiki_url(link))
                            lk  = f"wikipedia::{art}"
                            if lk not in visited:
                                q.append(("wikipedia", art, dlevel + 1))
                    elif source in ("html", "scikit", "stackoverflow"):
                        for link in doc["links"]:
                            if urlparse(link).netloc not in CRAWLABLE:
                                continue
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
    max_body:     int | None = None,
    max_links:    int | None = None,
    max_results:  int | None = None,
    topics:       set[str] | None = None,
) -> None:
    """
    Reads links from existing corpus files, randomly samples `sample` of them,
    then crawls those URLs (and follows their links to `depth` levels).
    """
    candidates = load_corpus_links(corpus_dir)
    if topics:
        candidates = [(url, t) for url, t in candidates if t in topics]
    if not candidates:
        print("No unscraped links found in corpus.")
        return

    random.shuffle(candidates)
    sampled = candidates[:sample]
    print(f"Sampled {len(sampled)} / {len(candidates)} candidate links.")

    # Group by topic so the round-robin engine still balances coverage.
    # Route Wikipedia URLs through the API adapter, not raw HTML.
    by_topic: dict[str, list] = {}
    for url, topic in sampled:
        if "wikipedia.org/wiki/" in url:
            by_topic.setdefault(topic, []).append(
                ("wikipedia", _title_from_wiki_url(url))
            )
        else:
            by_topic.setdefault(topic, []).append(("html", url))

    clusters = {
        t: {
            "keywords": TOPIC_CLUSTERS.get(t, {}).get("keywords", []),
            "seeds":    seeds,
        }
        for t, seeds in by_topic.items()
    }

    crawl(clusters, depth=depth, max_pages=max_pages,
          out_dir=out_dir, delay=delay, overwrite=overwrite,
          max_body=max_body, max_links=max_links, max_results=max_results)


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
  python data/scraper.py                                      # all topics, all sources
  python data/scraper.py --companies                          # WEF/UN/NASA/Quanta/MIT News/…
  python data/scraper.py --companies --topics climate_change,economics
  python data/scraper.py --topics programming,ml_algorithms --sources wikipedia,arxiv,github
  python data/scraper.py --topics smart_manufacturing --max 40
  python data/scraper.py --expand --expand-sample 30          # follow links from existing corpus
  python data/scraper.py --topics football,space --max 20
  python data/scraper.py --seed 42 --max 50 --depth 2
""",
    )
    p.add_argument("--seeds",          help="Text file with one seed URL per line")
    p.add_argument("--topics",         help="Comma-separated cluster names (default: all)")
    p.add_argument("--sources",        help="Comma-separated source types to include "
                   "(wikipedia, arxiv, openalex, semantic_scholar, devto, github, "
                   "html, stackoverflow, wikidata, rss)")
    p.add_argument("--wiki-cats",      help="Comma-separated Wikipedia category names to "
                   "scrape (e.g. 'Climate_change,Renewable_energy'). Seeds are injected "
                   "into every selected topic cluster.")
    p.add_argument("--companies",      action="store_true",
                   help="Use corporate/NGO HTML sources instead of academic APIs")
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
    p.add_argument("--max-body",        type=int, default=None,
                   help="Truncate body text to N characters (default: no limit)")
    p.add_argument("--max-links",       type=int, default=None,
                   help="Keep only the first N links per page (default: no limit)")
    p.add_argument("--max-results",     type=int, default=None,
                   help="Max results per query for academic APIs — openalex, arxiv, "
                   "semantic_scholar (default: 5/5/6). Use e.g. 50 to paginate deeper.")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.out)

    # ── Mode selection ────────────────────────────────────────────────────────
    if args.expand:
        corpus_dir = out_dir
        expand(
            corpus_dir = corpus_dir,
            sample     = args.expand_sample,
            depth      = args.depth,
            max_pages  = args.max,
            out_dir    = out_dir,
            delay      = args.delay,
            overwrite  = args.overwrite,
            max_body    = args.max_body,
            max_links   = args.max_links,
            max_results = args.max_results,
            topics      = {t.strip() for t in args.topics.split(",")} if args.topics else None
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

    else:
        base = TOPIC_CLUSTERS
        if args.topics:
            wanted  = {t.strip() for t in args.topics.split(",")}
            unknown = wanted - set(base)
            if unknown:
                p.error(f"Unknown topic(s): {', '.join(sorted(unknown))}. "
                        f"Available: {', '.join(base)}")
            base = {t: base[t] for t in base if t in wanted}
        clusters = base

    if args.sources:
        allowed = {s.strip() for s in args.sources.split(",")}
        clusters = {
            t: {
                "keywords": conf["keywords"],
                "seeds":    [(s, v) for s, v in conf["seeds"] if s in allowed],
            }
            for t, conf in clusters.items()
        }

    if args.wiki_cats:
        cats = [c.strip() for c in args.wiki_cats.split(",") if c.strip()]
        # Copy seeds lists before appending so TOPIC_CLUSTERS is never mutated.
        clusters = {t: {**conf, "seeds": list(conf["seeds"])} for t, conf in clusters.items()}
        for topic in clusters:
            for cat in cats:
                clusters[topic]["seeds"].append(("wikipedia_category", cat))
        print(f"[wiki-cats] injected {len(cats)} category seed(s) into "
              f"{len(clusters)} topic(s): {', '.join(cats)}")

    crawl(
        clusters    = clusters,
        depth       = args.depth,
        max_pages   = args.max,
        out_dir     = out_dir,
        delay       = args.delay,
        overwrite   = args.overwrite,
        max_body    = args.max_body,
        max_links   = args.max_links,
        max_results = args.max_results,
    )


if __name__ == "__main__":
    main()
