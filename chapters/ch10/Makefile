PIP=../../venv/bin/pip
PYTHON=../../venv/bin/python

#APIKEY=

init:
	$(PIP) install -r requirements.lock

download:
	mkdir -p ../../dataset/ontology/snomed/
	curl "https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/mlb/utsauth/USExt/SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z.zip&apiKey=$(APIKEY)" -o ../../dataset/ontology/snomed/SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z.zip
	unzip ../../dataset/ontology/snomed/SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z.zip -d ../../dataset/ontology/snomed/
	mv ../../dataset/ontology/snomed/SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z/Full/Terminology/sct2_Relationship_Full_US1000124_20220901.txt ../../dataset/ontology/snomed/sct2_Relationship_Full_US1000124_20220901.txt
	mv ../../dataset/ontology/snomed/SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z/Full/Terminology/sct2_Description_Full-en_US1000124_20220901.txt ../../dataset/ontology/snomed/sct2_Description_Full-en_US1000124_20220901.txt
	mv ../../dataset/ontology/snomed/SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z/Full/Terminology/sct2_TextDefinition_Full-en_US1000124_20220901.txt ../../dataset/ontology/snomed/sct2_TextDefinition_Full-en_US1000124_20220901.txt

import:
	PYTHONPATH=../../ $(PYTHON) importer/import_snomed_rels.py
	PYTHONPATH=../../ $(PYTHON) importer/import_snomed_names.py
	PYTHONPATH=../../ $(PYTHON) importer/propagate_snomed_categories.py

disambiguate:
	PYTHONPATH=../../ $(PYTHON) disambiguation/main.py
