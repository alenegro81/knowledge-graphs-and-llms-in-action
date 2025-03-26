PIP=../../venv/bin/pip
PYTHON=../../venv/bin/python

OPENAI_KEY="your OPENAI API key"
OPENAI_BASE_URL="https://api.openai.com/v1"
OPENAI_MODEL="gpt-4"


init:
	$(PIP) install -r requirements.lock

import:
	OPENAI_KEY=$(OPENAI_KEY) OPENAI_BASE_URL=$(OPENAI_BASE_URL) OPENAI_MODEL=$(OPENAI_MODEL) PYTHONPATH=../../ \
	$(PYTHON) importer/ingest_and_process.py



