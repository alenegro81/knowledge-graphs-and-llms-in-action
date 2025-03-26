import sys

from util.base_importer import BaseImporter


class CoOccurrenceGenerator(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"
        self.batch_size = 50

    def get_page_count(self):
        with self._driver.session(database=self._database) as session:
            return session.run("MATCH (p:Page) WHERE not (p:NEDProcessed) RETURN count(p) as pages").single()["pages"]
    
    def get_medical_entities_by_sentence(self):
        medical_entities_query = """
        MATCH (p:Page:NEDProcessed)-[r:MENTIONS_ENTITY]->(me:MedicalEntity)
        WITH p, r.sentence_index as sentences, me 
        UNWIND sentences as sentence
        WITH p, sentence, collect(distinct me.id) as entities
        UNWIND range(0, size(entities)-2) as i
        UNWIND range(i+1, size(entities)-1) as j
        RETURN p.id as p, sentence, entities, i, j
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query=medical_entities_query)
            for record in iter(result):
                yield dict(record)

    def link_cooccurring_entities(self):
        link_entities = """
        UNWIND $batch as item
        MATCH (m1:MedicalEntity) WHERE m1.id = item.entities[item.i]
        MATCH (m2:MedicalEntity) WHERE m2.id = item.entities[item.j]
        MERGE (m1)-[s:COOCCURR]-(m2)
        ON CREATE SET s.count = 1, 
                      s.sentences = [item.sentence]
        ON MATCH SET s.count = s.count + 1, 
                     s.sentences = s.sentences + item.sentence
        """

        size = self.get_page_count()
        self.batch_store(link_entities, self.get_medical_entities_by_sentence(), size=size)

if __name__ == '__main__':
    linking = CoOccurrenceGenerator(argv=sys.argv[1:], )
    linking.link_cooccurring_entities()
