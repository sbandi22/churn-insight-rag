.PHONY: install setup data index ui eval test clean help

PYTHON := python
VENV   := .venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

setup: install  ## Full project setup (install + copy .env)
	@[ -f .env ] || cp .env.example .env
	@echo "Edit .env with your API keys (optional)."

data:  ## Download Telco dataset from Kaggle
	$(PYTHON) data/download_telco.py

index:  ## Build the ChromaDB vector index
	$(PYTHON) -m ingestion.build_index

index-reset:  ## Rebuild the index from scratch
	$(PYTHON) -m ingestion.build_index --reset

ui:  ## Launch the Streamlit chat UI
	streamlit run ui/app.py

eval:  ## Run retrieval evaluation (10 questions)
	$(PYTHON) -m eval.eval_retrieval --verbose

eval-save:  ## Run evaluation and save JSON results
	$(PYTHON) -m eval.eval_retrieval --verbose --output eval/results.json

test:  ## Run pytest unit tests
	pytest --tb=short

docker-build:  ## Build Docker image
	docker compose build

docker-up:  ## Start all services
	docker compose up -d

docker-index:  ## Build index inside Docker container
	docker compose run --rm app python -m ingestion.build_index

docker-eval:  ## Run eval inside Docker container
	docker compose run --rm app python -m eval.eval_retrieval --verbose

clean:  ## Remove ChromaDB index and __pycache__
	rm -rf chroma_db __pycache__ ingestion/__pycache__ retrieval/__pycache__ generation/__pycache__ ui/__pycache__ eval/__pycache__
	find . -name "*.pyc" -delete

clean-all: clean  ## Remove venv and data
	rm -rf $(VENV) data/WA_Fn-UseC_-Telco-Customer-Churn.csv
