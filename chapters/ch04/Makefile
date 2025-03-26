.PHONY: init import analysis
PIP=../../venv/bin/pip
PYTHON=../../venv/bin/python

init:
	$(PIP) install -r requirements.lock

import:
	PYTHONPATH=../../ $(PYTHON) importer/import_seed.py

analysis:
	PYTHONPATH=../../ $(PYTHON) analysis/louvain_cluster_analysis.py
	PYTHONPATH=../../ $(PYTHON) analysis/pharma_analysis.py

