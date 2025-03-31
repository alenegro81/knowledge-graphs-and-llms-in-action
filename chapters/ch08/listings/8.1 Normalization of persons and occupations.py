from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")
NEO4J_DB = "neo4j"

REMOVE_TITLES = ["dr.", "prof.", "dean", "president", "pres.", "sir", "mr.", "mrs."]
QUERY_NORM_PERSONS = """
  MATCH (e:Entity {label: "Person"})
  WITH e, CASE WHEN ANY(title IN $remove_titles WHERE toLower(e.name)
  STARTS WITH title) THEN apoc.text.join(split(e.name, " ")[1..], " ")
  ELSE e.name END AS name
  SET e.name_normalized = name
 """

QUERY_NORM_OCCUPATIONS = """
  MATCH (e:Entity {label: "Occupation"})
  SET e.name_normalized = toLower(e.name)
"""

if __name__ == "__main__":
  with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session(database=NEO4J_DB) as session:
        print("Normalising Person names")
        session.run(QUERY_NORM_PERSONS, remove_titles=REMOVE_TITLES)
        print("Normalising Occupations")
        session.run(QUERY_NORM_OCCUPATIONS)