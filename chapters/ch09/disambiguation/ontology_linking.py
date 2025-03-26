import sys
from util.base_importer import BaseImporter


class OntologyLinking(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"
        self.batch_size = 50

    def get_medical_entities(self):
        with self._driver.session(database=self._database) as session:
            return session.run("""
            MATCH (me:MedicalEntity)
            MATCH (umls:UMLS {id: me.id})
            OPTIONAL MATCH (umls)-[:UMLS_TO_SNOMED]->(snomed:SnomedEntity)
            OPTIONAL MATCH (umls)-[:UMLS_TO_HPO]->(hpo:Hpo)
            OPTIONAL MATCH (umls)-[:UMLS_TO_DIS]->(dis:Disease)
            return count(*) as count""").single()["count"]

    def get_medical_links(self):
        medical_links_query = """
        MATCH (me:MedicalEntity)
        MATCH (umls:UMLS {id: me.id})
        OPTIONAL MATCH (umls)-[:UMLS_TO_SNOMED]->(snomed:SnomedEntity)
        OPTIONAL MATCH (umls)-[:UMLS_TO_HPO]->(hpo:Hpo)
        OPTIONAL MATCH (umls)-[:UMLS_TO_DIS]->(dis:Disease)
        RETURN me.id AS me, umls.id AS umls, snomed.id As snomed, hpo.id AS hpo, dis.id AS dis
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(query=medical_links_query)
            for record in iter(result):
                yield dict(record)

    def link_entities_to_snomed(self):
        link_entities = """
            UNWIND $batch as item
            WITH item
            WHERE item.snomed is not null
            MATCH (me:MedicalEntity {id: item.me})
            MATCH (snomed:SnomedEntity {id: item.snomed})
            MERGE (me)-[:IS_SNOMED_ENTITY]->(snomed)
        """
        size = self.get_medical_entities()
        self.batch_store(link_entities, self.get_medical_links(), size=size)

    def link_entities_to_hpo(self):
        link_entities = """
            UNWIND $batch as item
            WITH item
            WHERE item.hpo is not null
            MATCH (me:MedicalEntity {id: item.me})
            MATCH (hpo:Hpo {id: item.hpo})
            MERGE (me)-[:IS_HPO_ENTITY]->(hpo)
        """
        size = self.get_medical_entities()
        self.batch_store(link_entities, self.get_medical_links(), size=size)

    def link_entities_to_disease(self):
        link_entities = """
            UNWIND $batch as item
            WITH item
            WHERE item.dis is not null
            MATCH (me:MedicalEntity {id: item.me})
            MATCH (dis:Disease {id: item.dis})
            MERGE (me)-[:IS_DISEASE_ENTITY]->(dis)
        """
        size = self.get_medical_entities()
        self.batch_store(link_entities, self.get_medical_links(), size=size)


if __name__ == '__main__':
    linking = OntologyLinking(argv=sys.argv[1:], )

    print("linking entities to Snomed")
    linking.link_entities_to_snomed()
    print("linking entities to HPO")
    linking.link_entities_to_hpo()
    print("linking entities to Diseases")
    linking.link_entities_to_disease()
    linking.close()
