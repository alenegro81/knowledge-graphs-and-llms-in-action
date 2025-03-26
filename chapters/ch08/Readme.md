## Building knowledge graphs with Large Language Models

### Getting started

#### Install requirements
This chapter's `Makefile` assumes you have a virtual environemnt folder called `venv` 
at the reopository root folder. You can edit the`Makefile` and redefine the `PIP` variable
at the first line to match your configuration.
```shell
make init
```

#### Import the datasets
Run this command to execute the importer code for each dataset.
```shell
make import
```

##### Cached LLM responses

Once loaded, diary entries are processed using OpenAI ChatGPT by default. 
The responses from ChatGPT are cached in the `data/cached_llm`  folder to avoid executing the same prompt multiple
times and incurring additional costs. 
This caching mechanism is enabled by default, and a populated version of the cache is included in this Git repository, 
allowing the entire process to run even without a valid OpenAI key.

The caching mechanism is simple and tied to the name of the diary page identifier. 
If the model or prompt is changed since the last time the cache was populated, the cache will not be invalidated, 
and unmodified responses may be used. 
In such scenarios, it is sufficient to remove the contents of the cache folder or copy them to another location.