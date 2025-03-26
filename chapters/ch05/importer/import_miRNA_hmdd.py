import csv
import sys
from pathlib import Path

from neo4j.exceptions import ClientError as Neo4jClientError

from util.base_importer import BaseImporter


class HMDDImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"

    def get_rows(self, HMDD_file):
        with open(HMDD_file, 'r+', encoding="latin-1") as in_file:
            reader = csv.reader(in_file, delimiter='\t')
            header = next(reader)
            for row in reader:
                yield dict(zip(header, row))

    def import_HMDD(self, HMDD_file):
        query = """
            UNWIND $batch as item
            WITH trim(toLower(item.disease)) as disease, toLower(item.mir) as mir,item
            MERGE (d:Disease {name: disease})
            MERGE (m:MiRNA {name:mir})
            SET m:MiRNA_HMDD
            MERGE (m)-[r:RELATED_TO]->(d)
            SET r.description = item.description, r.pmid=item.pmid, r.category = item.category
            MERGE (ref:Reference {pubmed_id:item.pmid})
            MERGE (m)-[:HAS_REFERENCE]->(ref)
        """
        size = self.get_csv_size(HMDD_file, encoding="latin-1")
        self.batch_store(query, self.get_rows(HMDD_file), size=size, strategy="aggregate")

    def set_constraints(self):
        ver = self._driver.verify_connectivity()
        with self._driver.session(database=self._database) as session:
            if ver.startswith("Neo4j/5"):
                query = """
                    CREATE CONSTRAINT FOR (a:Disease) REQUIRE a.name IS UNIQUE;
                    CREATE CONSTRAINT FOR (a:MiRNA) REQUIRE a.name IS UNIQUE;
                    
                    CREATE CONSTRAINT FOR (a:Reference) REQUIRE a.pubmed_id IS UNIQUE;
                    CREATE CONSTRAINT FOR (a:Target) REQUIRE a.name IS UNIQUE
                """
            else:
                query = """
                    CREATE CONSTRAINT ON (a:Disease) ASSERT a.name IS UNIQUE;
                    CREATE CONSTRAINT ON (a:MiRNA) ASSERT a.name IS UNIQUE;
                    
                    CREATE CONSTRAINT ON (a:Reference) ASSERT a.pubmed_id IS UNIQUE;
                    CREATE CONSTRAINT ON (a:Target) ASSERT a.name IS UNIQUE"""
            for q in query.split(";"):
                try:
                    session.run(q)
                except Neo4jClientError as e:
                    # ignore if we already have the rule in place
                    if e.code != "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists":
                        raise e


def main():
    importing = HMDDImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/hmdd/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    HMDD_file = base_path / "HMDD_v3.2.txt"

    if not HMDD_file.is_file():
        print(HMDD_file, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.set_constraints()
    importing.import_HMDD(HMDD_file)

    importing.close()


if __name__ == '__main__':
    main()
