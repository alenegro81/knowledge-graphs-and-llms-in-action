import csv
import sys
from pathlib import Path

from util.base_importer import BaseImporter


class SnomedNamesImporter(BaseImporter):

    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "ned"

    @staticmethod
    def get_csv_size(snomedNames_file, encoding="utf-8"):
        return sum(1 for row in SnomedNamesImporter.get_rows(snomedNames_file))

    @staticmethod
    def get_rows(snomedNames_file):
        with open(snomedNames_file, "r+") as in_file:
            reader = csv.reader(in_file, delimiter="\t")
            header = next(reader)
            for row in reader:
                record = dict(zip(header, row))
                record["termAsType"] = record["term"].replace(" ", "_").upper()
                yield record

    def import_snomed_names(self, snomedNames_file):
        snomed_names_concepts_query = """
        UNWIND $batch as item
        MATCH (e1:SnomedEntity)-[r:SNOMED_RELATION {id: item.conceptId}]->(e2:SnomedEntity)
        SET r.type = CASE 
                WHEN r.type IS NULL THEN item.termAsType
                ELSE r.type END,
            r.aliases = CASE 
                WHEN item.termAsType IN r.aliases THEN r.aliases
                ELSE coalesce(r.aliases,[]) + item.termAsType END
        """
        snomed_names_entities_query = """
        UNWIND $batch as item
        MATCH (e:SnomedEntity {id: item.conceptId})
        SET e.name = CASE 
                WHEN e.name IS NULL THEN item.term
                ELSE e.name END,
            e.aliases = CASE 
                WHEN item.term in e.aliases THEN  e.aliases
                ELSE coalesce(e.aliases, []) + item.term END
        """
        size = self.get_csv_size(snomedNames_file)
        self.batch_store(snomed_names_concepts_query, self.get_rows(snomedNames_file), size=size)
        self.batch_store(snomed_names_entities_query, self.get_rows(snomedNames_file), size=size)



def csv_as_dict_list(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader]


if __name__ == '__main__':
    importing = SnomedNamesImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/ontology/snomed/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    snomedNames_dat = base_path / "sct2_Description_Full-en_US1000124_20220901.txt"

    if not snomedNames_dat.is_file():
        print(snomedNames_dat, "doesn't exist in ", base_path)
        sys.exit(1)

    print("importing Description")
    importing.import_snomed_names(snomedNames_dat)

    snomedDefs_dat = base_path / "sct2_TextDefinition_Full-en_US1000124_20220901.txt"

    if not snomedDefs_dat.is_file():
        print(snomedDefs_dat, "doesn't exist in ", base_path)
        sys.exit(1)

    print("importing TextDefinition")
    importing.import_snomed_names(snomedDefs_dat)
    importing.close()

    importing.close()
