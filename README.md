# Knowledge Graph Applied

### Getting started with this repository

Make sure that the Neo4j instance you want to use is up and running. Follow the instructions Appendix B for installation directions.

Update the [config.ini](config.ini) file with the relevant neo4j credentials.

It is reccomanded to set up a python virtual environment for this project. For example:
```shell
$ python -m venv venv
```

Unless otherwise stated, the code in this repo is tested with python version 3.8/3.9/3.10

Chapters make use of a `MakeFiles` based approach to simplfy operations, make sure you can run 
the make command:

```shell
$ make -version
```

Generally the `GNU make` is available on many package managers for a wide range of OSes.

```shell
$ choco install make # Windows Os
$ apt install make # Debian & derivated OSes (including ubuntu)
$ yum install make # Centos 
```

macOS's users should have `make` available through XCode - Command line tools:

```shell
$ xcode-select --install
```
alteratively  `GNU Make` can be installed via brew
```shell
brew install make
```

For further information refere to the Readme.md available in each chapter's directory
