import sys
from pathlib import Path

from Bio import SeqIO

from util.base_importer import BaseImporter


class BioImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"

    @staticmethod
    def get_embl_size(miRNA_dat):
        return sum(1 for record in SeqIO.parse(miRNA_dat, "embl")
                   if record.name.startswith("hsa"))

    @staticmethod
    def get_rows(miRNA_dat):
        for record in SeqIO.parse(miRNA_dat, "embl"):
            if not record.name.startswith("hsa"):
                continue
            if len(record.name) < 2:
                continue
            yield {
                "name": record.name.lower(),
                "description": record.description,
                "seq": record.seq.__str__(),
                "comment": record.annotations.get('comment', ''),
                "references": [
                    {"authors": r.authors, "title": r.title,
                     "pubmed_id": r.pubmed_id, "journal": r.journal}
                    for r in (record.annotations
                              .get('references', []))],
                "features": [
                    {"type": r.type,
                     "accession": (r.qualifiers
                                   .get('accession', [""])[0]),
                     "name": (r.qualifiers
                              .get('product', [""])[0]
                              .lower())}
                    for r in record.features if r.type == "miRNA"]
            }

    def import_miRNA_dat(self, miRDB_file):
        query = """
                UNWIND $batch as item
                MATCH (m:MiRNA {name: item.name})
                SET
                  m:MiRNA_miRBase, 
                  m.description = item.description, 
                  m.seq = item.seq, m.comment = item.comment
                WITH m,item
                FOREACH (feature in item.features  | 
                    MERGE (f:MiRNA {name: feature.name})
                    SET f.type = feature.type, 
                        f.accession = feature.accession
                    MERGE (m)-[:HAS_FEATURE]->(f)
                )
                WITH m,item
                UNWIND item.references as reference
                MERGE (r:Reference {pubmed_id: reference.pubmed_id})
                ON CREATE SET r.authors = reference.authors, 
                              r.title = reference.title, 
                              r.journal = reference.journal
                MERGE (m)-[:HAS_REFERENCE]->(r)
            """
        size = self.get_embl_size(miRDB_file)
        self.batch_store(query, self.get_rows(miRDB_file), size=size)


#   https://www.mirbase.org/ftp/CURRENT/miRNA.dat.gz
def main():
    importing = BioImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/hmdd/miRBase/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    miRNA_dat = base_path / "miRNA.dat"

    if not miRNA_dat.is_file():
        print(miRNA_dat, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.import_miRNA_dat(miRNA_dat)
    importing.close()


if __name__ == '__main__':
    main()
