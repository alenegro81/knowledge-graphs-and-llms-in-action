import csv
import sys
from pathlib import Path

from util.base_importer import BaseImporter



class Mir2DiseaseImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"

    def get_rows(self, miRDB_file):
        with open(miRDB_file, 'r+') as in_file:
            reader = csv.reader(in_file, delimiter='\t')
            for row in reader:
                if len(row) < 2:
                    continue
                yield {
                    "name": row[0].lower().strip(),
                    "disease": row[1].lower().strip(),
                    "regulated": row[2]
                }

    def import_miR2Disease(self, miR2Disease_file):
        query = """
            UNWIND $batch as item
            MERGE (d:Disease {name: item.disease})
            MERGE (m:MiRNA {name:item.name})
            SET m:MiRNA_miR2Disease
            MERGE (m)-[r:REGULATES]->(d)
            SET r.regulation = item.regulated
        """
        size = self.get_csv_size(miR2Disease_file)
        self.batch_store(query, self.get_rows(miR2Disease_file), size=size, strategy="aggregate")

    def import_miR2Disease_old(self, miR2Disease_file):
        query = """
                    UNWIND $batch as item
                    MATCH (m:MiRNA)
                    WHERE m.lower_cased_name= item.name
                    SET m:MiRNAR2
                    WITH m, item
                    WITH m, item, [item.disease, replace(item.disease, "cancer", "neoplasms")] as diseases
                    UNWIND diseases as disease
                    MATCH (n:Disease)
                    WITH m, item, n, disease, [x IN split(n.lower_cased_name, ", ") | trim(x)] as terms, split(disease, " ") as splitDisease
                    WITH m, item, n, disease, apoc.coll.intersection(terms, splitDisease) as intersect, splitDisease
                    WHERE n.lower_cased_name = disease OR (size(splitDisease) = size(terms) AND size(intersect) = size(terms))
                    WITH m, item, n, disease
                    SET n:DiseaseR2, n.name_in_r2 = disease
                    MERGE (m)-[r:REGULATES {regulated: item.regulated}]->(n)
                    return id(m) as id, id(n)
                """
        size = self.get_csv_size(miR2Disease_file)
        self.batch_store(query, self.get_rows(miR2Disease_file), size=size, strategy="aggregate")


def main():
    importing = Mir2DiseaseImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/hmdd/miR2Disease/"
    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    miRNA_dat = base_path / "AllEntries.txt"

    if not miRNA_dat.is_file():
        print(miRNA_dat, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.import_miR2Disease(miRNA_dat)
    importing.close()


if __name__ == '__main__':
    main()
