.PHONY: up down fetch-peps ingest ingest-sample eval serve
up:            ## start postgres+pgvector
	docker compose up -d
down:
	docker compose down
fetch-peps:    ## download the Python PEPs corpus into data/corpus/
	bash scripts/fetch_peps.sh
ingest:        ## index the PEPs corpus
	python -m rag_lab.sync.reconcile data/corpus
ingest-sample: ## index the tiny built-in sample corpus
	python -m rag_lab.sync.reconcile sample_corpus
eval:          ## run dense/sparse/hybrid comparison (doc-level)
	python -m rag_lab.eval.harness
serve:         ## start the query API
	uvicorn rag_lab.api.main:app --reload
