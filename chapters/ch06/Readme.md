## Knowledge graphs and natural language processing

### Getting started

#### Install requirements
This chapter's `Makefile` assumes you have a virtual environemnt folder called `venv` 
at the reopository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

#### Import the datasets
Run this command to execute the importer code.
```shell
make import
```

##### Cached requests

During the enrichment steps, many requests are made to the Wikidata REST API. 
As do any public APIs, the Wikidata REST API has certain rate limits in place to prevent misuse and abuse. 
To be considerate of these limits, the enrichment steps include important throttling mechanisms that slow down the process.

To mitigate this performance issue while still maintaining some flexibility, responses from Wikidata API requests are cached 
so that each request is issued at most once. 
Afterward, the cached file representing the response is used instead. 
This behavior is enabled by default and the cached responses are available in this Git repository. 
As a result, even the first time the process runs, it will use the cached Wikidata responses, resulting in quick execution.

This behavior can be overridden by simply removing one or more JSON files in the `cache_org` and/or the `cache_owners` 
subfolder of the `data` folder.

Please note that using the cached responses provided in this repository does not guarantee that not a single request will 
be issued to the Wikidata API. 
The output of the Spacy pipeline may differ slightly depending on the model version used, 
and it is possible that entities that are not cached may be added.