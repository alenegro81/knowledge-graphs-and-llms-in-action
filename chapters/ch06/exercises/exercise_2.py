import sys
from pprint import pprint

sys.path.append('../importer')

from neo4j import GraphDatabase, basic_auth
from step2__enrich_organizations import query_wikidata_entity

NEO4J_DB = "news"

SPARQL_2 = """SELECT ?personX ?personXLabel ?dob (GROUP_CONCAT(DISTINCT ?occupationLabel; SEPARATOR=";") AS ?occupations)
    (GROUP_CONCAT(DISTINCT ?fieldOfWorkLabel; SEPARATOR=";") AS ?fieldsOfWork)
    (GROUP_CONCAT(DISTINCT ?ownedCompanyLabel; SEPARATOR=";") AS ?ownedCompanies)
WHERE {
  ?personX rdfs:label "%s"@en.

  ?personX wdt:P569 ?dob.
  OPTIONAL { ?personX wdt:P106 ?occupation. }
  OPTIONAL { ?personX wdt:P101 ?fieldOfWork. }
  OPTIONAL { ?personX wdt:P1830 ?ownedCompany. }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
    ?personX rdfs:label ?personXLabel.
    ?occupation rdfs:label ?occupationLabel.
    ?fieldOfWork rdfs:label ?fieldOfWorkLabel.
    ?ownedCompany rdfs:label ?ownedCompanyLabel.
  }
}
GROUP BY ?personX ?personXLabel ?dob"""


if __name__ == "__main__":
    ### Solution of the exercise in chapter 5.2:
    #  "Enrich the Person nodes by their date of birth, occupation, fields of work, and company ownership. "

    # Initialise Neo4j driver
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=basic_auth('neo4j', 'neo'))

    # Get entities to enrich
    QUERY_GET_INPUTS = "MATCH (p:Person) RETURN p.name AS name"
    with driver.session(database=NEO4J_DB) as session:
        entities = session.run(QUERY_GET_INPUTS).data()
        print(f"Retrieved {len(entities)} entities")
        print(entities[0])

        pprint(query_wikidata_entity(SPARQL_2, entities[0]['name']))

        # now do as you please with the results:)