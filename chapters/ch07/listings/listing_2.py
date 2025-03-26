prompt_segments = dict()

prompt_segments['task'] = """You are an expert on constructing Knowledge Graphs from texts using named entity recognition and relation extraction.
Given a prompt, identify as many entities and relations among them as possible and output a list of relations in the format 
[ENTITY 1, ENTITY 1 TYPE, RELATION, ENTITY 2, ENTITY 2 TYPE]. 
The relations are directed, so the order matters."""

prompt_segments['example'] = "J.R.Smith (Prof. Phys.) is employed by MIT and mentioned another scientist, Mary Hodge, who works on cyclotron research."

prompt_segments['example_output'] = """["J. R. Smith", "person", "has title", "Professor of Physics", "title"]
["J. R. Smith", "person", "works for", "MIT", "organization"]
["J. R. Smith", "person", "talked about", "Mary", "person"] 
["Mary", "person", "works on", "cyclotron", "occupation"]"""