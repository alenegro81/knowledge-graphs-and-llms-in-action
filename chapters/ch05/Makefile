PIP=../../venv/bin/pip
PYTHON=../../venv/bin/python

init:
	$(PIP) install -r requirements.lock

download:
	mkdir -p ../../dataset/hmdd/miRDB
	mkdir -p ../../dataset/hmdd/dbDEMC
	mkdir -p ../../dataset/hmdd/miRBase
	mkdir -p ../../dataset/hmdd/misim
	mkdir -p ../../dataset/hmdd/miR2Disease
	curl https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz -o ../../dataset/hmdd/miRDB/miRDB_v6.0_prediction_result.txt.gz
	curl http://www.cuilab.cn/static/hmdd3/data/alldata.txt -o ../../dataset/hmdd/HMDD_v3.2.txt
	curl https://www.biosino.org/dbDEMC/download/MiRExpAll -o ../../dataset/hmdd/dbDEMC/miRExpAll.txt
	curl http://watson.compbio.iupui.edu:8080/miR2Disease/download/AllEntries.txt -o ../../dataset/hmdd/miR2Disease/AllEntries.txt
	curl https://mirbase.org/download/miRNA.dat -o  ../../dataset/hmdd/miRBase/miRNA.dat
	curl http://www.cuilab.cn/files/images/cuilab/misim.zip -o ../../dataset/hmdd/misim/misim.zip
	gunzip -f ../../dataset/hmdd/miRDB/miRDB_v6.0_prediction_result.txt.gz
	unzip -d  ../../dataset/hmdd/misim/ ../../dataset/hmdd/misim/misim.zip
	mv "../../dataset/hmdd/misim/miRNA similarity matrix.txt" ../../dataset/hmdd/misim/similarityMatrix.txt
	mv "../../dataset/hmdd/misim/microRNA name.xls" ../../dataset/hmdd/misim/microRNA.xls

import:
	PYTHONPATH=../../ $(PYTHON) importer/import_miRNA_hmdd.py
	PYTHONPATH=../../ $(PYTHON) importer/import_miRNA_dbDEMC.py
	PYTHONPATH=../../ $(PYTHON) importer/import_miRNA_miR2Disease.py
	PYTHONPATH=../../ $(PYTHON) importer/import_miRNA_EMBL.py
	PYTHONPATH=../../ $(PYTHON) importer/import_miRNA_RDB.py
	PYTHONPATH=../../ $(PYTHON) importer/import_miRNA_sim.py

reconciliate:
	PYTHONPATH=../../ $(PYTHON) reconciliation/reconciliate_disease.py


