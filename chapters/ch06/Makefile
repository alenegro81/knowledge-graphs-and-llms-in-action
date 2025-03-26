PIP=../../venv/bin/pip
PYTHON=../../venv/bin/python

init:
	$(PIP) install -r requirements.lock
	$(PYTHON) -m spacy download en_core_web_sm

import:
	PYTHONPATH=../../ $(PYTHON) importer/step1__import_bbc.py
	PYTHONPATH=../../ $(PYTHON) importer/step2__enrich_organizations.py
	PYTHONPATH=../../ $(PYTHON) importer/step3__enrich_by_ownerships.py



