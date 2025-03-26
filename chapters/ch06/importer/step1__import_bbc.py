import sys
from pathlib import Path

import spacy
import pytextrank
from neo4j.exceptions import ClientError as Neo4jClientError

from util.base_importer import BaseImporter


class BBCImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "news"

        # load standard English NLP and NER models
        #  NORP entity = Nationalities or religious or political groups
        #  FAC entity = Buildings, airports, highways, bridges etc.
        #  GPE entity = Countries, cities, states
        self.nlp = spacy.load("en_core_web_sm")

        # CREATE DATABASE news
        self.create_indices()

    def create_indices(self):
        with self._driver.session(database=self._database) as session:
            query = """
            CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE;
            CREATE CONSTRAINT person_name_unique IF NOT EXISTS FOR (n:Person) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT location_name_unique IF NOT EXISTS FOR (n:Location) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT organization_name_unique IF NOT EXISTS FOR (n:Organization) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT date_name_unique IF NOT EXISTS FOR (n:Date) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT money_name_unique IF NOT EXISTS FOR (n:Money) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT number_name_unique IF NOT EXISTS FOR (n:Number) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT group_name_unique IF NOT EXISTS FOR (n:NatReligPolitGroup) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT workofart_name_unique IF NOT EXISTS FOR (n:WorkOfArt) REQUIRE n.name IS UNIQUE;
            CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS FOR (n:Keyword) REQUIRE n.name IS UNIQUE"""
            for q in query.split(";"):
                try:
                    session.run(q)
                except Neo4jClientError as e:
                    # ignore if we already have the rule in place
                    if e.code != "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists":
                        raise e

    @staticmethod
    def cleanse_entity(en: str):
        TO_IGNORE = ["the", "a"]
        tokenized = en.split()
        if tokenized[0].lower() in TO_IGNORE:
            en = en[len(tokenized[0]):]
        en = en.replace("\'s", "")
        return en.strip()

    @staticmethod
    def count_documents(data_path: Path):
        return sum(1 for _ in data_path.glob("*/*.txt"))

    def get_documents(self, data_path: Path):
        for file in data_path.glob("*/*.txt"):
            # file: xyz/topic1/123.txt => id: topic1_123
            id_ = f"{file.parent.name}_{file.name[:-4]}"
            with file.open('r') as f:
                try:
                    lines = [x.strip() for x in f if len(x.strip()) > 1]
                    document = {"topic": file.parent.name, "id": id_, "title": lines[0],
                                "text": "\n".join(lines[1:])}
                    self.enrich_document(document)
                    yield document
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError for {id}")

    def enrich_document(self, document: dict):
        processed = self.nlp(document['title'] + ".\n\n" + document['text'])
        for en in processed.ents:
            if en.label_ not in document:
                document[en.label_] = list()
            document[en.label_].append(self.cleanse_entity(en.text))

    def import_documents(self, data_path: Path):
        import_document_query = """
        UNWIND $batch as item
        MERGE (t:Topic {name: item.topic})
        
        WITH item,t
        MERGE (n:Document {id: item.id}) 
        SET n.title = item.title, n.text = item.text
        MERGE (t)-[:HAS_DOCUMENT]->(n)
        
        WITH item,t,n
        FOREACH(entity IN item.PERSON |
              MERGE (e:Person {name: entity})
              MERGE (n)-[:MENTIONS_PERSON]->(e)
            )

            FOREACH(entity IN item.ORG |
              MERGE (e:Organization {name: entity})
              MERGE (n)-[:MENTIONS_ORGANIZATION]->(e)
            )

            FOREACH(entity IN item.GPE |
              MERGE (e:Location {name: entity})
              MERGE (n)-[:MENTIONS_LOCATION]->(e)
            )

            FOREACH(entity IN item.DATE |
              MERGE (e:Date {name: toLower(entity)})
              MERGE (n)-[:MENTIONS_DATE]->(e)
            )

            FOREACH(entity IN item.MONEY |
              MERGE (e:Money {name: entity})
              MERGE (n)-[:MENTIONS_MONEY]->(e)
            )

            FOREACH(entity IN item.CARDINAL |
              MERGE (e:Number {name: entity})
              MERGE (n)-[:MENTIONS_NUMER]->(e)
            )

            FOREACH(entity IN item.NORP |
              MERGE (e:NatReligPolitGroup {name: entity})
              MERGE (n)-[:MENTIONS_GROUP]->(e)
            )

            FOREACH(entity IN item.WORK_OF_ART |
              MERGE (e:WorkOfArt {name: entity})
              MERGE (n)-[:MENTIONS_WORK_OF_ART]->(e)
            )
        """
        size = self.count_documents(data_path)
        self.batch_size = 50
        self.batch_store(import_document_query, self.get_documents(data_path), size=size, desc="importing documents")


class BBCKeywordImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "news"
        self.nlp = spacy.load("en_core_web_sm")
        # add keyword extraction to the NLP pipeline
        self.nlp.add_pipe("textrank")

    @staticmethod
    def cleanse_keyword(kw: str):
        TO_IGNORE = ["the", "their", "a"]
        tokenized = kw.split()
        if tokenized[0].lower() in TO_IGNORE:
            kw = kw[len(tokenized[0]):]
        return kw.lower().strip()

    def count_documents(self):
        count_query = "MATCH (n:Document) RETURN count(n) as total"
        with self._driver.session(database=self._database) as session:
            return session.run(count_query).single()["total"]

    def get_documents(self):
        query_documents = """
        MATCH (n:Document)
        RETURN id(n) AS id, n.title AS title, n.text AS text
        """
        with self._driver.session(database=self._database) as session:
            for document in session.run(query_documents):
                document = dict(document)
                processed = self.nlp(document['title'] + ".\n\n" + document['text'])
                document['keywords'] = [{'name': self.cleanse_keyword(x.text), 'rank': x.rank}
                                        for x in processed._.phrases
                                        if len(x.text) > 1][:30]
                yield document

    def import_keyword(self):
        query_store_keywords = """
        UNWIND $batch AS item
        MATCH (n:Document)
        WHERE id(n) = item.id
        FOREACH(entity IN item.keywords |
            MERGE (e:Keyword {name: entity.name})
            MERGE (n)-[r:MENTIONS_KEYWORD]->(e)
            SET r.rank = entity.rank
        )"""
        size = self.count_documents()
        self.batch_size = 50
        self.batch_store(query_store_keywords, self.get_documents(), size=size, desc="importing keywords")


if __name__ == "__main__":
    importer = BBCImporter(argv=sys.argv[1:])
    base_path = importer.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../data/bbc/"

    base_path = Path(base_path)

    # Ingest articles & run NLP processing
    importer.import_documents(base_path)

    importerKW = BBCKeywordImporter(argv=sys.argv[1:])
    # for each article, extract keywords and key-phrases (enrichment of the KG created in previous step)
    importerKW.import_keyword()
