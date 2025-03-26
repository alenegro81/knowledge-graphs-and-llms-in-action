## Question Answering System using LangGraph and Streamlit

### Getting started

#### Install requirements
This chapter's `Makefile` assumes you have a virtual environment folder called `venv` 
at the repository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

#### Download the investigative graph 
Run this command to initialize the graph content in your Neo4j database."
```shell
make import
```
To provide extra information such as the name of the database, set the environment variables accordingly to the makefile
```shell
NEO4J_DATABASE=mydb make import
```
will create a graph database named `mydb` for example

#### Launch the application
To execute the Streamlit-based application, run the following command:
```shell
make app
```
Make sure to set the relevant environment variables for both the Neo4j database and the AI provider:
```shell
NEO4J_DATABASE=mydb \
AZURE_OPENAI_API_KEY=xyz \
AZURE_OPENAI_API_VERSION=abc \
AZURE_OPENAI_DEPLOYMENT=model-name \
AZURE_OPENAI_ENDPOINT=https://mydomain.openai.azure.com/ \
make app
```
This will set up an Azure-based GPT model as the AI provider and will use the `mydb` graph as the database.

To switch AI providers, refer to `chains/investigator.py`, which contains examples of configurations for other AI providers.
