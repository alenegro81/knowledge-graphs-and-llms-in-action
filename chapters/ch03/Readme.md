## Create your first knowledge graph from ontologies

### Getting started
The code provided in this chapter requires that you have installed the Neosemantics plugin in your Neo4j instance.

#### Install requirements
This chapter's `Makefile` assumes you have a virtual environemnt folder called `venv` 
at the reopository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

#### Download and import the datasets
Run this command to execute the importer code for each dataset.
```shell
make import
```