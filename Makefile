.PHONY: install test demo report

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

demo:
	.venv/bin/python -m agent_model_router.demo

report:
	.venv/bin/python -m agent_model_router.ledger router_ledger.jsonl
