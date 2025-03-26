## Build knowlede Graph from structured Sources

### Getting started

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

#### Reconciliate diseases
Run this command to use scispacy model to resolve diseases coming from different datasets
```shell
make reconciliate
```