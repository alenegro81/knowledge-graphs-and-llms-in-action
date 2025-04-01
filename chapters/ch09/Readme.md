## Named Entity Disambiguation

### Getting started
The construction of the Knowledge Graph requires to download multiple ontologies from the Unified Medical Language System (UMLS) platform.

To access these files, follow these steps:

1. Sign in at UMLS Login using one of the available identity providers, such as Google or Microsoft: https://uts.nlm.nih.gov/uts/login.
2. After authentication, retrieve your API key by clicking on "*Get Your API Key*".
3. Copy the API key into the `Makefile`.

For more details on programmatic access, refer to the official documentation: https://documentation.uts.nlm.nih.gov/automating-downloads.html.


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

