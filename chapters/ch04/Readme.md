## More complex knowledge graphs: biomedical examples

### Getting started

#### Install requirements
This chapter's `Makefile` assumes you have a virtual environment folder called `venv` 
at the repository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

#### Download the PPi graph 
Run this command to initialize the graph content in your Neo4j database."
```shell
make import
```

#### Launch the analysis scripts
To execute the analysis scripts, run the following command:
```shell
make analysis
```
