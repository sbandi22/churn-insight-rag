# Churn Insight RAG

> A production-ready **Retrieval-Augmented Generation (RAG)** system for customer churn intelligence. Ask natural language questions and get grounded, cited answers backed by Telco customer data, research papers, and industry reports.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Architecture

```
                         CHURN INSIGHT RAG — SYSTEM ARCHITECTURE

 ┌─────────────────────────────── DATA SOURCES ────────────────────────────────┐
 │                                                                              │
 │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   │
 │  │  Telco Customer  │   │ Research Papers  │   │  Industry Reports    │   │
 │  │  Churn Dataset   │   │  (5 synthetic)   │   │  (3 synthetic)       │   │
 │  │  (Kaggle CSV)    │   │                  │   │                      │   │
 │  │  ~7,000 rows     │   │  • RFM Analysis  │   │  • SaaS Benchmarks   │   │
 │  │  converted to    │   │  • NLP Tickets   │   │  • Telecom Benchmks  │   │
 │  │  NL summaries    │   │  • Survival Anal │   │  • Retention Econ.   │   │
 │  │                  │   │  • Ensembles     │   │                      │   │
 │  │                  │   │  • Deep Seq Mdl  │   │                      │   │
 │  └────────┬─────────┘   └────────┬─────────┘   └──────────┬───────────┘   │
 └───────────┼──────────────────────┼──────────────────────────┼──────────────┘
             │                       │                          │
             └───────────────────────┼──────────────────────────┘
                                     │
                            ┌────────▼─────────┐
                            │   INGESTION      │
                            │  load_documents  │
                            │  chunking (512t) │
                            │  embeddings      │
                            └────────┬─────────┘
                                     │
                            ┌────────▼─────────┐
                            │   VECTOR STORE   │
                            │   ChromaDB       │
                            │   (local, disk)  │
                            └────────┬─────────┘
                                     │
           ┌─────────────────────────┼────────────────────────┐
           │                         │                        │
  ┌────────▼─────────┐    ┌──────────▼──────────┐   ┌───────▼──────────┐
  │   EMBEDDINGS     │    │     RETRIEVAL        │   │    GENERATION    │
  │                  │    │                      │   │                  │
  │  Offline:        │    │  Top-5 chunks w/     │   │  GPT-4o-mini     │
  │  sentence-       │    │  MMR diversity       │   │  Claude Haiku    │
  │  transformers    │    │  (LangChain)         │   │  or Offline LLM  │
  │  all-MiniLM-L6   │    │                      │   │                  │
  │                  │    │  Returns chunks +    │   │  Prompt template │
  │  Online (opt):   │    │  source metadata     │   │  + citations     │
  │  OpenAI          │    │                      │   │                  │
  │  text-emb-3-small│    └──────────────────────┘   └──────────────────┘
  └──────────────────┘                                        │
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │   STREAMLIT UI       │
                                                    │                      │
                                                    │  • Chat interface    │
                                                    │  • Sidebar stats     │
                                                    │  • Source citations  │
                                                    │  • Example questions │
                                                    └──────────────────────┘
```

---

## Project Structure

```
churn-insight-rag/
├── data/                        # Knowledge base documents
│   ├── papers/                  # 5 research paper summaries (Markdown)
│   │   ├── 01_rfm_segmentation.md
│   │   ├── 02_nlp_support_tickets.md
│   │   ├── 03_survival_analysis.md
│   │   ├── 04_ensemble_models.md
│   │   └── 05_deep_sequence_models.md
│   ├── reports/                 # 3 industry report excerpts (Markdown)
│   │   ├── 01_saas_churn_benchmarks.md
│   │   ├── 02_telecom_churn_benchmarks.md
│   │   └── 03_retention_economics.md
│   └── download_telco.py        # Downloads Telco CSV from Kaggle
│
├── ingestion/                   # Data ingestion pipeline
│   ├── __init__.py
│   ├── build_index.py           # Main indexing entry point
│   ├── load_documents.py        # Document loaders for all 3 source types
│   ├── chunking.py              # Text splitting with citation metadata
│   ├── embeddings.py            # Embedding model factory (offline/OpenAI)
│   └── telco_summarizer.py      # Converts CSV rows to NL summaries
│
├── retrieval/                   # Retrieval layer
│   ├── __init__.py
│   └── retriever.py             # ChurnRetriever with MMR top-5
│
├── generation/                  # Answer generation layer
│   ├── __init__.py
│   └── answer.py                # ChurnAnswerEngine with citations
│
├── ui/                          # Streamlit application
│   ├── __init__.py
│   └── app.py                   # Full chat UI with sidebar + sources
│
├── eval/                        # Evaluation scripts
│   ├── __init__.py
│   └── eval_retrieval.py        # 10 scored questions, keyword overlap metric
│
├── config.py                    # Central configuration (paths, model names)
├── requirements.txt             # Pinned Python dependencies
├── .env.example                 # Example environment variables
├── .gitignore
├── Dockerfile                   # Multi-stage production Docker image
└── docker-compose.yml           # Compose with volume persistence
```

---

## Quick Start

### Option A: Local (no Docker)

```bash
# 1. Clone and install
git clone https://github.com/sbandi22/churn-insight-rag.git
cd churn-insight-rag
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment (optional — retrieval works fully offline)
cp .env.example .env
# Edit .env to add API keys if you want LLM-generated answers

# 3. (Optional) Download the Telco dataset from Kaggle
python data/download_telco.py

# 4. Build the vector index
python -m ingestion.build_index

# 5. Launch the Streamlit UI
streamlit run ui/app.py
# Open http://localhost:8501
```

### Option B: Docker

```bash
# Build and start
docker compose up --build

# Build the index inside the container
docker compose run --rm app python -m ingestion.build_index

# Then restart the UI service
docker compose up -d
# Open http://localhost:8501
```

---

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```env
# LLM provider: "openai", "anthropic", or "offline"
LLM_PROVIDER=offline

# OpenAI (optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-20240307

# Embedding model: "sentence-transformers" (offline) or "openai"
EMBEDDING_MODEL=sentence-transformers
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
```

**Offline mode** (no API keys): The retrieval layer works 100% offline using `sentence-transformers`. Answers are composed from retrieved chunks without a generative LLM.

---

## Evaluation

Run the evaluation script against 10 predefined churn questions:

```bash
# Basic run
python -m eval.eval_retrieval

# Verbose with per-question details
python -m eval.eval_retrieval --verbose

# Save results to JSON
python -m eval.eval_retrieval --output eval/results.json
```

Sample output:
```
Q01 PASS [83%] (12ms) Why are enterprise customers churning?
Q02 PASS [67%] (11ms) What churn rate is typical for SaaS companies?
Q03 PASS [83%] (10ms) Which customer segments are highest risk?
Q04 PASS [100%] ( 9ms) How does contract type affect churn probability?
Q05 PASS [100%] (12ms) What does RFM analysis reveal about churn prediction?
...

============================================================
EVALUATION SUMMARY
============================================================
Questions evaluated : 10
Passed (score >= 40%): 10/10  (100%)
Avg relevance score : 83.50%
Avg retrieval time  : 11 ms
```

---

## Data Sources

| Source | Type | Count | Description |
|--------|------|-------|-------------|
| Telco Customer Churn | CSV | ~7,000 rows | IBM Watson Telco dataset from Kaggle, converted to natural language summaries |
| Research Papers | Markdown | 5 | Synthetic abstracts covering RFM, NLP, survival analysis, ensembles, deep sequence models |
| Industry Reports | Markdown | 3 | SaaS benchmarks, telecom benchmarks, retention economics |

---

## RAG Pipeline Details

**Chunking**: `RecursiveCharacterTextSplitter` with 512-token chunks and 50-token overlap. Each chunk carries metadata: `source`, `source_type`, `chunk_id`, `page`.

**Embeddings**: Configurable — defaults to `all-MiniLM-L6-v2` (384-dimensional, offline) or `text-embedding-3-small` (OpenAI, 1536-dim).

**Vector Store**: ChromaDB persisted to `./chroma_db/`. Collection name: `churn_insight`.

**Retrieval**: `ChurnRetriever` uses Maximal Marginal Relevance (MMR) with `k=5` and `fetch_k=20` to balance relevance and diversity.

**Generation**: `ChurnAnswerEngine` builds a structured prompt with the retrieved context, instructs the LLM to cite sources, and parses source references from the response.

---

## Example Questions

These are pre-loaded in the sidebar of the Streamlit UI:

- "Why are enterprise customers churning?"
- "What churn rate is typical for SaaS companies?"
- "Which customer segments are highest risk?"
- "How does contract type affect churn probability?"
- "What does RFM analysis reveal about churn prediction?"
- "How can NLP on support tickets predict churn?"
- "What are the most effective retention strategies for telecom companies?"
- "How does tenure affect churn likelihood in the Telco dataset?"

---

## Screenshots

> _Screenshots placeholder — run `streamlit run ui/app.py` to see the live UI._

**Chat Interface with Source Citations:**
```
┌─────────────────────────────────────────────────────────┐
│ 💬 Churn Insight Assistant                              │
│                                                         │
│  You: What churn rate is typical for SaaS companies?   │
│                                                         │
│  Assistant: Based on the indexed industry reports,     │
│  SaaS companies typically experience annual churn rates │
│  of 5-7% for enterprise accounts and 10-15% for SMB... │
│                                                         │
│  📌 3 cited sources                                    │
│  ├─ 📊 01_saas_churn_benchmarks.md · chunk 3           │
│  ├─ 📄 02_rfm_segmentation.md · chunk 1                │
│  └─ 👤 Customer ID 7590-VHVEG · chunk 421              │
└─────────────────────────────────────────────────────────┘
```

---

## Development

```bash
# Run tests
pytest

# Run evaluation
python -m eval.eval_retrieval --verbose

# Rebuild index from scratch
python -m ingestion.build_index --reset
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| RAG Framework | LangChain 0.3 |
| Vector Store | ChromaDB 0.6 |
| Embeddings (offline) | sentence-transformers all-MiniLM-L6-v2 |
| Embeddings (online) | OpenAI text-embedding-3-small |
| LLM (optional) | OpenAI GPT-4o-mini / Anthropic Claude Haiku |
| UI | Streamlit 1.45 |
| Data | Pandas 2.2 |
| Containerization | Docker + Compose |

---

## License

MIT — see [LICENSE](LICENSE) for details.
