PIP=../../venv/bin/pip
PYTHON=../../venv/bin/python
STREAMLIT=../../venv/bin/streamlit

# Neo4j Environment Variables:
# NEO4J_URI="neo4j://localhost:7687"
# NEO4J_USER="neo4j"
# NEO4J_PASSWORD="password"
# NEO4J_DATABASE="chicago.ila"

# AI Environment Variables:
# OPENAI_API_KEY=""
# AZURE_OPENAI_API_KEY=""
# AZURE_OPENAI_ENDPOINT=""
# AZURE_OPENAI_DEPLOYMENT=""
# AZURE_OPENAI_API_VERSION=""


init:
	$(PIP) install -r requirements.lock

import:
	PYTHONPATH=../../ $(PYTHON) importer/import_seed.py

app:
	PYTHONPATH=../../ $(STREAMLIT)  run app.py

