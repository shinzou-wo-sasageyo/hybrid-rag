.PHONY: up down ingest eval serve
up:        ## start postgres+pgvector
	docker compose up -d
down:
	docker compose down
ingest:    ## index the sample corpus
	python -m rag_lab.sync.reconcile sample_corpus
eval:      ## run dense/sparse/hybrid comparison
	python -m rag_lab.eval.harness
serve:     ## start the query API
	uvicorn rag_lab.api.main:app --reload
