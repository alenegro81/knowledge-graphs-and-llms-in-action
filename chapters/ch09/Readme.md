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

### Notes for Mac Users
If you're having trouble installing libraries related to scispacy, you can run the code from this chapter via a Docker container by following these steps:

#### Create a Suitable Container 
Create an ephemeral container called venv_ch5:

```bash
docker run -e NEO4J_URI=bolt://host.docker.internal:7687 -e PYTHONPATH=../../ \
       -w /work/chapters/ch09 -d -it --name venv_ch9 -v $PWD/../../:/work python:3.9-slim
```
This command assumes you'll use a Neo4j instance running on your local machine. 
The container will mount this repository under `/work` and set the working directory accordingly.

#### Update the Makefile
Change the first lines of the Makefile to:

```makefile
PIP=docker exec -it venv_ch9 pip  
PYTHON=docker exec -it venv_ch9 python
```

#### Run the Make Commands
Now you can run the make commands as usual:
```bash
make init     # Install required libraries inside the container
make download # Download datasets (data will be available within the container)
make import   # Execute the import in the container (data will be written to the Neo4j instance on your local machine)
make disambiguate # Execute disambiguation within the container (data will be written to the Neo4j instance on your local machine)
```