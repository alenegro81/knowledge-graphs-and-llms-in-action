import csv
import sys
from pathlib import Path

from util.base_importer import BaseImporter


class SnomedRelationshipsImporter(BaseImporter):
    
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"
    
    @staticmethod
    def get_csv_size(snomedRels_file, encoding="utf-8"):
        return sum(1 for row in SnomedRelationshipsImporter.get_rows(snomedRels_file))
    
    @staticmethod
    def get_rows(snomedRels_file):
        with open(snomedRels_file, "r+") as in_file:
            reader = csv.reader(in_file, delimiter="\t")
            header = next(reader)
            for row in reader:
                record = dict(zip(header, row))

                yield {
                    "sourceId": record["sourceId"],
                    "destinationId": record["destinationId"],
                    "typeId": record["typeId"],
                }
    
    def import_snomed_rels(self, snomedRels_file):
        snomed_rels_query = """
        UNWIND $batch as item
        MERGE (e1:SnomedEntity {id: item.sourceId})
        MERGE (e2:SnomedEntity {id: item.destinationId})
        MERGE (e1)-[:SNOMED_RELATION {id: item.typeId}]->(e2)
        FOREACH(ignoreMe IN CASE WHEN item.typeId = '116680003' THEN [true] ELSE [] END|
            MERGE (e1)-[:SNOMED_IS_A]->(e2)
        )
        """
        
        size = self.get_csv_size(snomedRels_file)
        self.batch_store(snomed_rels_query, self.get_rows(snomedRels_file), size=size)
    
    
    def set_constraints(self):
        queries = ["CREATE CONSTRAINT IF NOT EXISTS FOR (n:SnomedEntity) REQUIRE n.id IS UNIQUE",
                   "CREATE INDEX snomedNodeName IF NOT EXISTS FOR (n:SnomedEntity) ON (n.name)",
                   "CREATE INDEX snomedRelationId IF NOT EXISTS FOR ()-[r:SNOMED_RELATION]-() ON (r.id)",
                   "CREATE INDEX snomedRelationType IF NOT EXISTS FOR ()-[r:SNOMED_RELATION]-() ON (r.type)",
                   "CREATE INDEX snomedRelationUmls IF NOT EXISTS FOR ()-[r:SNOMED_RELATION]-() ON (r.umls)"]

        for q in queries:
            with self._driver.session(database=self._database) as session:
                session.run(q)

if __name__ == '__main__':
    importing = SnomedRelationshipsImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/ontology/snomed/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    snomedRels_dat = base_path / "sct2_Relationship_Full_US1000124_20220901.txt"

    if not snomedRels_dat.is_file():
        print(snomedRels_dat, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.set_constraints()
    importing.import_snomed_rels(snomedRels_dat)
    importing.close()
