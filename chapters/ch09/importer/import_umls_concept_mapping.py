import csv
import sys
from pathlib import Path

from util.base_importer import BaseImporter


class UmlsConceptMappingsImporter(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"
        self._size = None

    def set_constraints(self):
        queries = ["CREATE CONSTRAINT IF NOT EXISTS FOR (n:UMLS) REQUIRE n.id IS UNIQUE"]

        for q in queries:
            with self._driver.session(database=self._database) as session:
                session.run(q)

    def get_csv_size(self, umls_file, encoding="utf-8"):
        if self._size is None:
            self._size = sum(1 for row in UmlsConceptMappingsImporter.get_rows(umls_file))
        return self._size

    @staticmethod
    def get_rows(umls_file):
        with open(umls_file, "r+") as in_file:
            reader = csv.reader(in_file, delimiter="|")
            for row in reader:
                yield {
                    "umls_id": row[0],
                    "type_source": row[11],
                    "other_id": row[13]
                }

    def import_umls_snomed(self, umls_file):
        umls_snomed_query = """
        UNWIND $batch as item
        MATCH (se:SnomedEntity)
        WHERE se.id = item.other_id
        WITH item, se
        MERGE (umls:UMLS {id: item.umls_id})
        MERGE (umls)-[:UMLS_TO_SNOMED]->(se)
        SET se.umls_ids = CASE 
                WHEN item.umls_id in se.umls_ids THEN se.umls_ids
                ELSE coalesce(se.umls_ids,[]) + item.umls_id END
        """

        size = self.get_csv_size(umls_file)
        self.batch_store(umls_snomed_query, self.get_rows(umls_file), size=size)

    def import_umls_hpo(self, umls_file):
        umls_hpo_query = """
        UNWIND $batch as item
        MATCH (hpo:Hpo)
        WHERE hpo.id = item.other_id
        WITH item, hpo
        MERGE (umls:UMLS {id: item.umls_id})
        MERGE (umls)-[:UMLS_TO_HPO]->(hpo)
        SET hpo.umls_ids = CASE 
                WHEN item.umls_id in hpo.umls_ids THEN hpo.umls_ids
                ELSE coalesce(hpo.umls_ids,[]) + item.umls_id END
        """
        size = self.get_csv_size(umls_file)
        self.batch_store(umls_hpo_query, self.get_rows(umls_file), size=size)

    def import_umls_disease(self, umls_file):
        umls_hpo_query = """
        UNWIND $batch as item
        MATCH (dis:Disease)
        WHERE dis.id = item.type_source + ':' + item.other_id
        WITH item, dis
        MERGE (umls:UMLS {id: item.umls_id})
        MERGE (umls)-[:UMLS_TO_DIS]->(dis)
        SET dis.umls_ids = CASE 
                WHEN item.umls_id in dis.umls_ids THEN dis.umls_ids
                ELSE coalesce(dis.umls_ids,[]) + item.umls_id END
        """
        size = self.get_csv_size(umls_file)
        self.batch_store(umls_hpo_query, self.get_rows(umls_file), size=size)


if __name__ == '__main__':
    importing = UmlsConceptMappingsImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/ontology/umls/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    umls_path = base_path / "MRCONSO.RRF"

    if not umls_path.is_file():
        print(umls_path, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.set_constraints()
    print("importing UMLS <-> HPO")
    importing.import_umls_hpo(umls_path)
    print("importing UMLS <-> Snomed")
    importing.import_umls_snomed(umls_path)
    print("importing UMLS <-> Disease")
    importing.import_umls_disease(umls_path)
    importing.close()
