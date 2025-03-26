import csv
import sys
from pathlib import Path

from util.base_importer import BaseImporter


class MiRDBImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"

    @staticmethod
    def get_csv_size(miRDB_file, encoding="utf-8"):
        reader = csv.reader(miRDB_file.open(), delimiter='\t')
        return sum(1 for record in reader if record[0].startswith("hsa"))

    def get_rows(self, miRDB_file):
        with miRDB_file.open("r") as in_file:
            reader = csv.reader(in_file, delimiter='\t')
            header = ("name", "target", "value")
            for row in reader:
                record = dict(zip(header, row))
                if not record["name"].startswith("hsa"):
                    continue
                yield record

    def import_miRDB(self, miRDB_file):
        query = """
                    UNWIND $batch as item 
                    MATCH (m:MiRNA {name: toLower(item.name)})
                    SET m:MiRNA_RDB
                    MERGE (t:Target {name: item.target})
                    MERGE (m)-[r:HAS_TARGET]->(t)
                    SET r.value= toFloat(item.value)
                """
        size = self.get_csv_size(miRDB_file)
        self.batch_store(query, self.get_rows(miRDB_file), size=size, strategy="aggregate")


def main():
    importing = MiRDBImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/hmdd/miRDB/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    miRDB_file = base_path / "miRDB_v6.0_prediction_result.txt"

    if not miRDB_file.is_file():
        print(miRDB_file, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.import_miRDB(miRDB_file)
    importing.close()


if __name__ == '__main__':
    main()
