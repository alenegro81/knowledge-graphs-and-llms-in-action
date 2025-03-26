import os
from typing import List, AnyStr
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph

load_dotenv(dotenv_path="../.env")

graph = Neo4jGraph(
        url=os.environ['NEO4J_URL'],
        username=os.environ['NEO4J_USER'],
        password=os.environ['NEO4J_PWD'],
        database=os.environ['NEO4J_DB']
    )  # requires APOC to be installed, specifically `apoc.meta.data`

RE_SELECTOR_QUERY = """MATCH (p:Page)-[:MENTIONS_ENTITY]->(m1:Entity)-->(e1:{e1_class})-[:{rel_class}]-(e2:{e2_class})<--(m2:Entity)<-[:MENTIONS_ENTITY]-(p)
WHERE e1.name = "{e1}" AND e2.name = "{e2}"
MATCH (m1)-[r:RELATED_TO_ENTITY]-(m2)
WHERE r.type = "{rel_class}"
RETURN DISTINCT p.id AS id, p.text AS text
"""

class REDiarySelectorInput(BaseModel):
    entity_source: str = Field(description="Source entity of the relationship as mentioned in the question.")
    entity_source_class: str = Field(description="Class of the source entity of the relationship. "
                                                 "Available option is only one, 'Person'."
                                    )
    entity_target: str = Field(description="Target entity of the relationship as mentioned in the question.")
    entity_target_class: str = Field(description="Class of the target entity of the relationship. "
                                                 "Available options are Person, Organization, Occupation and Title."
                                     )
    relationship: str = Field(description="Relationship class between source and target entity. "
                                       "Available options: TALKED_ABOUT, TALKED_WITH, WORKS_WITH, WORKS_ON, HAS_TITLE")


def kg_doc_selector(entity_source: str, entity_source_class: str, entity_target: str, entity_target_class: str,
                    relationship: str) -> List[AnyStr]:
    query = RE_SELECTOR_QUERY.format(e1=entity_source, e1_class=entity_source_class,
                             e2=entity_target, e2_class=entity_target_class,
                             rel_class=relationship)
    print(f"kg_doc_selector's query:\n{query}\n")
    try:
        res = graph.query(query)
        print(f"kg_doc_selector found {len(res)} matching documents")
    except Exception as e:
        print(f"Cypher execution exception: {e}")
        return []
    return [x['text'] for x in res[:3]]

if __name__ == "__main__":
    print(kg_doc_selector("August Krogh", "Person", "Lawrence Irving", "Person", "TALKED_ABOUT"))