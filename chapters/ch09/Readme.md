## Named Entity Disambiguation

### Getting started
The construction of the Knowledge Graph requires to download multiple ontologies from the Unified Medical Language System (UMLS) platform. For the access to these files, you need to sign in in the following page: https://uts.nlm.nih.gov/uts/login using one of the proposed identity providers, including Google or Microsoft. After the authentication step, you can get your API key by clicking the link "*Get Your API Key*" and copy it in the `Makefile`. Further information on the programmatic access is available here: https://documentation.uts.nlm.nih.gov/automating-downloads.html.


#### Install requirements
This chapter's `Makefile` assumes you have a virtual environemnt folder called `venv` 
at the reopository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

#### Download the datasets
Run this command to create a `dataset` folder at the repository root folder containing 
the raw data required.
```shell
make download
```

#### Import the datasets
Run this command to execute the importer code for each dataset.
```shell
make import
```

