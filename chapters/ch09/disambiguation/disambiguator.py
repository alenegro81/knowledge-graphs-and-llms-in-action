import sys
from pathlib import Path

import neo4j.exceptions

from chapters.ch08.disambiguation.entity_extractor import EntityExtractor
from util.base_importer import BaseImporter


class Disambiguator(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"
        self.batch_size = 50
        self.entity_extractor: EntityExtractor = None

    def setupEntityExtractor(self, metathesaurus_file):
        self.entity_extractor = EntityExtractor(metathesaurus_file)

    def get_page_count(self):
        with self._driver.session(database=self._database) as session:
            return session.run("MATCH (p:Page) WHERE not (p:NEDProcessed) RETURN count(p) as pages").single()["pages"]

    def get_entities_by_page(self):
        pages_query = """
        MATCH (p:Page) 
        WHERE not (p:NEDProcessed)
        RETURN p.id as id, p.text as text"""
        with self._driver.session(database=self._database) as session:
            result = session.run(query=pages_query)
            for page in iter(result):
                yield {
                    "id": page["id"],
                    "ents": self.entity_extractor.extract_ents(page["text"])
                }

    def ingest_entities(self):
        upload_ents_query = """
        UNWIND $batch as item

        MATCH (page:Page)
        WHERE page.id = item.id AND NOT page:NEDProcessed
        SET page:NEDProcessed
        WITH page, item
        
        UNWIND item.ents as entity
    
        MERGE(mention:EntityMention {name_normalized: toLower(apoc.text.join(apoc.text.split(trim(entity.value), "\\s+"), " "))})
        ON CREATE SET mention.name = apoc.text.join(apoc.text.split(trim(entity.value), "\\s+"), " ")
        MERGE (page)-[s:MENTIONS_MENTION {from_model: "ned"}]->(mention)
        ON CREATE SET s.start_chars= [entity.beginCharacter], 
                      s.end_chars= [entity.endCharacter], 
                      s.sentence_index = [entity.sentenceIndex],
                      s.type = toLower(entity.label)
        ON MATCH SET s.start_chars = s.start_chars + entity.beginCharacter, 
                     s.end_chars = s.end_chars + entity.endCharacter, 
                     s.sentence_index = s.sentence_index + entity.sentenceIndex
        WITH page, mention, entity     
        
        FOREACH(medical in entity |
        MERGE (dis:MedicalEntity {id: medical.selected_ned_id})
        ON CREATE SET dis.name= apoc.text.join(apoc.text.split(trim(medical.selected_ned_name), "\\s+"), " "),
                      dis.type_id = medical.selected_ned_types_id, 
                      dis.types = medical.selected_ned_types,
                      dis.type = medical.selected_ned_types[0],
                      dis.original_mention = medical.value, 
                      dis.definition = medical.selected_ned_definition, 
                      dis.aliases = medical.selected_ned_aliases,
                      dis.start_chars= [entity.beginCharacter], 
                      dis.end_chars= [entity.endCharacter], 
                      dis.sentence_index = [entity.sentenceIndex]
        ON MATCH SET dis.start_chars = dis.start_chars + entity.beginCharacter, 
                     dis.end_chars = dis.end_chars + entity.endCharacter
    
        MERGE (mention)-[r:DISAMBIGUATED_TO]->(dis)
        SET r.confidence = medical.selected_ned.confidence
        
        MERGE (page)-[t:MENTIONS_ENTITY]->(dis)
        ON CREATE SET t.sentence_index = [medical.sentenceIndex]
        ON MATCH SET t.sentence_index = t.sentence_index + medical.sentenceIndex)  
        """

        size = self.get_page_count()
        self.batch_store(upload_ents_query, self.get_entities_by_page(), size=size)

    def set_constraints(self):
        queries = ["CREATE CONSTRAINT IF NOT EXISTS FOR (n:MedicalEntity) REQUIRE n.id IS UNIQUE",
                   "CREATE INDEX medicalEntityName IF NOT EXISTS FOR (n:MedicalEntity) ON (n.name)",
                   "CREATE INDEX EntityMentionNormalizedName IF NOT EXISTS FOR (n:EntityMention) ON (n.name_normalized)",
                   "CREATE FULLTEXT INDEX medicalEntityText FOR (n:MedicalEntity) ON EACH [n.name, n.type]",
                   "CREATE FULLTEXT INDEX entityName FOR (n:EntityMention) ON EACH [n.name]"]

        for q in queries:
            with self._driver.session(database=self._database) as session:
                try:
                    session.run(q)
                except neo4j.exceptions.ClientError as e:
                    if e.code == 'Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists':
                        print("skipping", q)
                    else:
                        raise e


if __name__ == '__main__':
    importing = Disambiguator(argv=sys.argv[1:], )
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/ontology/umls/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    metathesaurus_file = base_path / "SemGroups.txt"

    if not metathesaurus_file.is_file():
        print(metathesaurus_file, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.set_constraints()
    importing.setupEntityExtractor(metathesaurus_file)
    importing.ingest_entities()
    importing.close()
