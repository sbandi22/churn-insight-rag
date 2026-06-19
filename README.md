TEST_PASTE_123
LINE2# Churn Insight RAG

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions about customer churn with **grounded, cited answers**.

It ingests three kinds of knowledge into a local vector database and serves them through a Streamlit chat UI:

1. **Telco Customer Churn dataset** (Kaggle) — each row is converted into a natural-language summary before indexing.
2. **5 research-paper abstracts** on churn prediction (RFM, NLP on support tickets, survival analysis, ensemble models, deep learning).
3. **3 industry-report excerpts** on SaaS and telecom churn benchmarks.

The system runs **fully offline** using `sentence-transformers` (`all-MiniLM-L6-v2`) so you can demo the retrieval layer with **no API key**. Add an OpenAI or Anthropic key to enable LLM-synthesized answers.

---

## Architecture

```
                        +-------------------------------+
                        |          Data Sources         |
                        |  Telco CSV | Papers | Reports  |
                        +---------------+---------------+
                                        |
                          ingestion/ (load -> summarize)
                                        |
                                        v
                        +-------------------------------+
                        |   Chunking (RecursiveSplitter) |
                        +---------------+---------------+
                                        |
                       Embeddings (all-MiniLM-L6-v2 / OpenAI)
                                        |
                                        v
                        +-------------------------------+
                        |   ChromaDB (local persistent)  |
                        +---------------+---------------+
                                        |
                          retrieval/ (MMR, top-5)
                                        |
                                        v
                        +-------------------------------+
                        | generation/ (LLM + citations)  |
                        |  OpenAI / Claude / offline      |
                        +---------------+---------------+
                                        |
                                        v
                        +-------------------------------+
                        |   ui/ Streamlit chat + sources |
                        +-------------------------------+
```

## Project Structure

```
churn-insight-rag/
├── config.py                 # Central settings (paths, models, retrieval params)
├── ingestion/
│   ├── __init__.py
│   ├── load_documents.py     # Load papers, reports, and Telco summaries
│   ├── telco_summarizer.py   # Row -> natural language summary
│   ├── chunking.py           # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings.py         # sentence-transformers / OpenAI factory
│   └── build_index.py        # Build/refresh the Chroma vector store
├── retrieval/
│   ├── __init__.py
│   └── retriever.py          # MMR top-5 retriever over Chroma
├── generation/
│   ├── __init__.py
│   └── answer.py             # Prompt + LLM call + citation formatting
├── ui/
│   └── app.py                # Streamlit chat interface
├── eval/
│   └── run_eval.py           # 10-question retrieval relevance eval
├── data/
│   ├── download_telco.py     # Download Telco dataset from Kaggle
│   ├── papers/               # 5 research abstracts (.md)
│   └── reports/              # 3 industry report excerpts (.md)
├── docs/
│   └── screenshots/          # README screenshot placeholders
├── requirements.txt
├── .env.example
├── .gitignore
└── Dockerfile
```

## Quickstart (offline, no API key)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Download the Telco dataset. Synthetic fallback is used if unavailable.
python data/download_telco.py

# 3. Build the vector index
python -m ingestion.build_index

# 4. Launch the chat UI
streamlit run ui/app.py
```

The first run downloads the `all-MiniLM-L6-v2` model (~80 MB). After that, everything is local.

## Enabling LLM answers

Copy `.env.example` to `.env` and set one of:

```
LLM_PROVIDER=openai          # or "anthropic" or "offline"
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

With `LLM_PROVIDER=offline` (default), answers are composed extractively from the retrieved chunks — still fully cited, just not paraphrased by an LLM.

## Evaluation

```bash
python -m eval.run_eval
```

Runs 10 sample questions, retrieves chunks for each, and scores retrieval relevance using keyword-overlap and expected-source-hit metrics. Prints a per-question table and an aggregate score.

## Docker

```bash
docker build -t churn-insight-rag .
docker run -p 8501:8501 churn-insight-rag
```

Then open http://localhost:8501.

## Screenshots

| Chat answer with citations | Sidebar index status |
|----------------------------|----------------------|
| ![Chat](docs/screenshots/chat.png) | ![Sidebar](docs/screenshots/sidebar.png) |

> Screenshot placeholders — replace the files in `docs/screenshots/` after your first run.

## Example Questions

- Why are enterprise customers churning?
- What churn rate is typical for SaaS companies?
- Which customer segments are highest risk?
- How does month-to-month contract affect churn?
- What modeling approaches work best for churn prediction?

## Tech Stack

Python 3.11 · LangChain · ChromaDB · Sentence-Transformers · OpenAI / Anthropic API · Streamlit · Pandas · Docker

## License

MIT — synthetic research/report content is provided for demonstration only.
