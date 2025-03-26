
KG_SCHEMA = """Nodes:
(:Person {{name: str (full person name), titles: [], pagerank: float, betweenness: float}}) // concrete individuals
(:Organization {{name: str}})
(:Occupation {{name: str, type: str}})


Relationships:
(:Person)-[:WORKS_FOR]->(:Organization)
(:Person)-[:WORKS_WITH]->(:Person) // this relationship indicates colleagues, therefore its direction is irrelevant; use it without specifying direction
(:Person)-[:TALKED_WITH {{conversation_type: str}}]->(:Person) // this relationship's direction is also irrelevant; use it without specifying direction
(:Person)-[:TALKED_ABOUT]->(:Person)
(:Person)-[:WORKS_ON]->(:Occupation)
(:Person)-[:HAS_TITLE]->(:Title)
(:Occupation)-[:SIMILAR_OCCUPATION]->(:Occupation) // semantically similar occupations
(:Organization)-[:SIMILAR_ORGANIZATION]->(:Organization) // similar organizations (typically alternate names)
"""