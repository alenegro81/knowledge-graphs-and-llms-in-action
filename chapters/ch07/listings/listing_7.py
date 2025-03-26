prompt_segments = dict()

prompt_segments['task'] = """You are an expert on constructing Knowledge Graphs from texts using named entity recognition and relation extraction.
Given a prompt which represents diary entry of Warren Weaver (often referred to as WW), identify as many entities and relations among them as possible and output a list of relations in the format [ENTITY 1, ENTITY 1 TYPE, RELATION, ENTITY 2, ENTITY 2 TYPE]. 
The relations are directed, so the order matters.

Entities of interest: person, location, organization, date, occupation (a.k.a. person's work, specialization, research discipline, interests, occupation, technology).
Top relations of interest: "works for", "works with", "student of" (relates students with their teachers/advisors), "talked about" (a person talking about another person), "talked with" (a person talking with another person), "works on" (relates persons to their occupation, work, specialisation, research discipline, interests, technology etc.).

Note that persons are often first referenced by their full name, and then mentioned only by their surname or initials, for example: "A. N. Richards" becomes "Richards", "ANR", or just "R.".
Note that organizations (universities, their departments) are often shortened, for example: "University of California" is written as "U. of Cal." or just "U. Cal.", "Department of Physics" is written as "Dept. Phys." etc."""

prompt_segments['example'] = "J.R.Smith, Prof. Phys. is employed by MIT and mentioned another colleague Mary Hodge, who studies along with her master's student John Smith radioisotopes produced by cyclotron."

prompt_segments['example_output'] = """["J. R. Smith", "person", "has title", "Professor of Physics", "title"]
["J. R. Smith", "person", "works for", "MIT", "organization"]
["J. R. Smith", "person", "talked about", "Mary Hodge", "person"] 
["Mary Hodge", "person", "works for", "MIT", "organization"]
["John Smith", "person", "student of", "Mary Hodge", "person"]
["Mary Hodge", "person", "works on", "radioisotopes", "occupation"]
["Mary Hodge", "person", "works on", "cyclotron", "occupation"]"""