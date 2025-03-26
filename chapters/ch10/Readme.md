# NED with Open LLMs and Domain Ontologies

## Getting started
To construct the Knowledge Graph, you need the SNOMED ontology, which is available through the Unified Medical Language System (UMLS) platform.

To access these files, follow these steps:

1. Sign in at UMLS Login using one of the available identity providers, such as Google or Microsoft: https://uts.nlm.nih.gov/uts/login.
2. After authentication, retrieve your API key by clicking on "*Get Your API Key*".
3. Copy the API key into the `Makefile`.

For more details on programmatic access, refer to the official documentation: https://documentation.uts.nlm.nih.gov/automating-downloads.html.

## Neo4j settings
This chapter has been tested with Neo4j version `5.20` with the APOC and GDS plugins.

You have to update the Neo4j settings as follows:

```bash
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
```

## Ollama settings
To locally run the Llama 3.1 8B model, you have to download the Ollama tool from: https://ollama.com/download/mac.

After the installation process, to dowload and serve the model you can run the following commands: 

```bash
ollama pull llama3.1:latest
ollama serve
```

## Install requirements
This chapter's `Makefile` assumes you have a virtual environemnt folder called `venv` 
at the reopository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

## Download the SNOMED ontology
Downloading the SNOMED ontology requires to create an API key on the UMLS platform.
```shell
make init
```

## Perform disambiguation
To run the disambiguation pipeline, you can use the following command:
```shell
make disambiguate
```
