import os
from pprint import pprint
from dotenv import load_dotenv
from typing import List, AnyStr

from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from pydantic import BaseModel, Field

from definitions import KG_SCHEMA

load_dotenv(dotenv_path="../.env")

graph = Neo4jGraph(
    url=os.environ['NEO4J_URL'],
    username=os.environ['NEO4J_USER'],
    password=os.environ['NEO4J_PWD'],
    database=os.environ['NEO4J_DB']
)  # requires APOC to be installed, specifically `apoc.meta.data`

class VectorSearchInput(BaseModel):
    question: str = Field(description="User's question / search query.")

def vector_search(question: str) -> List[AnyStr]:
    vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            url=os.environ['NEO4J_URL'],
            username=os.environ['NEO4J_USER'],
            password=os.environ['NEO4J_PWD'],
            database=os.environ['NEO4J_DB'],
            index_name='embeddings',
            node_label="Page",
            text_node_properties=['text'],
            embedding_node_property='embedding'
        )

    response = vector_index.similarity_search_with_score(question, k=3)

    return [r[0].page_content for r in response] # r[1] is a score


class KGReaderInput(BaseModel):
    question: str = Field(description="User's question / search query.")


def kg_reader(question: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(input_variables=['question'], template=(
        "From the natural language question provided below, generate a Cypher query to run against a Neo4j DB\n"
        f"with the following schema:\n{KG_SCHEMA}\n\n"
        f"The schema is authoritative - no other entities, relationships and properties are available. The relationships are only available between the two specified entity classes!\n"
        "When searching through Person node names, but only surname is provided, always use CONTAINS operator instead of "
        "exact matching, i.e. `MATCH ...(p:Person)... WHERE p.name CONTAINS \"<surname>\"``.\n"
        "When searching through Occupation nodes names, use CONTAINS operator instead of exact matching, and remove any"
        "unnecessary words that could make it harder to match all relevant Occupations.\n"
        "When asked about how are entities related/connected, search up to three-hop paths in undirected fashion, i.e. `(...)-[*1..3]-(...)`.\n"
        "Important note: Output only the Cypher query, that is all that's required. If you're unable to\n"
        "generate it, return an empty string.\n\n"
        "The question is:\n{question}\n\n"
        ))
    cypher_chain = prompt | llm

    # generate Cypher query
    cypher_query = cypher_chain.invoke({'question': question}).content
    print(f"kg_reader generated the following Cypher query:\n{cypher_query}")
    if len(cypher_query.strip()) == 0:
        return ""
    if cypher_query.lower().startswith("```cypher"):
        cypher_query = cypher_query[9:].strip()
    elif cypher_query.startswith("```"):
        cypher_query = cypher_query[3:].strip()
    if cypher_query.endswith("```"):
        cypher_query = cypher_query[:-3].strip()

    # execute Cypher
    try:
        res = graph.query(cypher_query)
        print(f"kg_reader found {len(res)} results")
    except Exception as e:
        print(f"Cypher execution exception: {e}")
        return "No results found."

    return f"Cypher query:\n{cypher_query}\n\nResponse from Neo4j:\n" + repr(res)


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
    RE_SELECTOR_QUERY = """MATCH (p:Page)-[:MENTIONS_ENTITY]->(m1:Entity)-->(e1:{e1_class})-[:{rel_class}]-(e2:{e2_class})<--(m2:Entity)<-[:MENTIONS_ENTITY]-(p)
    WHERE e1.name = "{e1}" AND e2.name = "{e2}"
    MATCH (m1)-[r:RELATED_TO_ENTITY]-(m2)
    WHERE r.type = "{rel_class}"
    RETURN DISTINCT p.id AS id, p.text AS text
    """
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
    pprint(vector_search("What did August Krogh say about Lawrence Irving?"))
    #print(kg_doc_selector("August Krogh", "Person", "Lawrence Irving", "Person", "TALKED_ABOUT"))