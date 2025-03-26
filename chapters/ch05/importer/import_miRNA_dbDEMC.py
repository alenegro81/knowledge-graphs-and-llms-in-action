import csv
import sys
from pathlib import Path

from util.base_importer import BaseImporter
from util.graphdb_base import GraphDBBase


class BioImporter(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"

    @staticmethod
    def get_csv_size(HMDD_file, encoding="utf-8"):
        return sum(1 for row in BioImporter.get_rows(HMDD_file))

    @staticmethod
    def get_rows(miRNA_file):
        with open(miRNA_file, 'r+') as in_file:
            reader = csv.reader(in_file, delimiter='\t')
            header = next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                record = dict(zip(header, row))
                disease = record["CancerSubtype"].lower() if len(record["CancerSubtype"]) > 1 else record[
                    "CancerType"].lower()

                if record["Species"] != "Homo sapiens":
                    continue

                # if row[0].lower() == row[1].lower():
                #    continue

                disease = (disease.replace(",", "")
                           .replace("/", " ")
                           # .replace("neoplasms", "carcinoma")
                           .replace("-", " "))

                name = record["miRBaseID"] if record["miRBaseID"] != "NA" else record["miRNA_ID"]
                name = record["miRNA_ID"].lower().strip()

                yield {
                    "name": name,
                    "disease": disease,
                    "experiment": record["ExperimentID"],
                    "regulated": record["Status"]
                }

    def import_exact_match(self, miRDB_file):
        exact_match_query = """
                UNWIND $batch as item
                MERGE (m:MiRNA {name: item.name})
                SET m:MiRNA_dbDEMC
                WITH m,item
                MERGE (n:Disease {name: item.disease})
                SET n:DiseaseDbDEMC, n.name_in_db_demc = item.disease
                MERGE (m)-[r:REGULATES {regulated: item.regulated}]->(n)
                SET r.source = 'dbDEMC', r.experiment = item.experiment
                return id(m) as id, id(n)
        """
        size = self.get_csv_size(miRDB_file)
        self.batch_store(exact_match_query, self.get_rows(miRDB_file), size=size)

    def import_miR2Disease(self, miRDB_file):
        indirect_approximate_match_query = """
                                    UNWIND $batch as item
                                    MATCH (m:MiRNA)
                                    WHERE m.name = item.name
                                    SET m:MiRNAdbDEMC
                                    MATCH (n:Disease)-[:REFERS_TO]->(resource)<-[:NAME_REFERS_TO]-(diseaseName)
                                    WHERE NOT EXISTS ((m)-[:REGULATES]->(n))
                                    WITH m, item, n, disease, diseaseName, [x IN split(diseaseName.label, " ") | trim(x)] as terms, [x in split(disease, " ") | trim(x)] as splitDisease
                                    WITH m, item, n, disease, diseaseName, apoc.coll.intersection(terms, splitDisease) as intersect, splitDisease, apoc.text.jaroWinklerDistance(disease, toLower(diseaseName.label)) as similarity
                                    WHERE toLower(diseaseName.label) = disease OR (size(splitDisease) = size(terms) AND size(intersect) = size(terms) and similarity > 0.9)
                                    WITH m, item, n, disease
                                    SET n:DiseaseDbDEMC, n.name_in_db_demc = item.disease
                                    MERGE (m)-[r:REGULATES {regulated: item.regulated}]->(n)
                                    SET r.source = 'dbDEMC', r.type = "indirect_approximate"
                                    return id(m) as id, id(n)
                                """
        size = self.get_csv_size(miRDB_file)
        self.batch_store(indirect_approximate_match_query, self.get_rows(miRDB_file), size=size)


class OLDBioImporter(GraphDBBase):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self.__database = "hmdd2.0"

    def import_miR2Disease(self, file):
        with self._driver.session(database=self.__database) as session:
            exact_match_query = """
                MATCH (m:MiRNA)
                WHERE m.name = $name
                SET m:MiRNAdbDEMC
                WITH m
                MATCH (n:Disease)
                WHERE toLower(n.name) = $disease
                SET n:DiseaseDbDEMC, n.name_in_db_demc = $disease
                MERGE (m)-[r:REGULATES {regulated: $regulated}]->(n)
                SET r.source = 'dbDEMC', r.type = "exact"
                return id(m) as id, id(n)
            """

            indirect_exact_match_query = """
                MATCH (m:MiRNA)
                WHERE m.name = $name
                SET m:MiRNAdbDEMC
                WITH m
                MATCH (n:Disease)-[:REFERS_TO]->(resource)<-[:NAME_REFERS_TO]-(diseaseName)
                WHERE toLower(diseaseName.label) = $disease AND NOT EXISTS ((m)-[:REGULATES]->(n))
                WITH m, n
                SET n:DiseaseDbDEMC, n.name_in_db_demc = $disease
                MERGE (m)-[r:REGULATES {regulated: $regulated}]->(n)
                SET r.source = 'dbDEMC', r.type = "indirec_exact"
                return id(m) as id, id(n)
            """

            direct_approximate_match_query = """
                MATCH (m:MiRNA)
                WHERE m.name = $name
                SET m:MiRNAdbDEMC
                WITH m
                WITH m, replace(replace(replace(replace($disease, ",", ""), "/", " "), "neoplasms", "carcinoma"), "-", " ") as disease
                MATCH (n:Disease)
                WHERE NOT EXISTS ((m)-[:REGULATES]->(n))
                WITH m, n, disease, [x IN split(replace(replace(replace(replace(toLower(n.name), ",", ""), "/", " "), "neoplasms", "carcinoma"), "-", " "), " ") | trim(x)] as terms, [x in split(disease, " ") | trim(x)] as splitDisease
                WITH m, n, disease, apoc.coll.intersection(terms, splitDisease) as intersect, splitDisease, apoc.text.jaroWinklerDistance(disease, toLower(n.name)) as similarity
                WHERE toLower(n.name) = disease OR (size(splitDisease) = size(terms) AND size(intersect) = size(terms) and similarity > 0.9)
                WITH m, n, disease
                SET n:DiseaseDbDEMC, n.name_in_db_demc = $disease
                MERGE (m)-[r:REGULATES {regulated: $regulated}]->(n)
                SET r.source = 'dbDEMC', r.type = "direct_approximate"
                return id(m) as id, id(n)
            """

            indirect_approximate_match_query = """
                            MATCH (m:MiRNA)
                            WHERE m.name = $name
                            SET m:MiRNAdbDEMC
                            WITH m
                            WITH m, replace(replace(replace(replace($disease, ",", ""), "/", " "), "neoplasms", "carcinoma"), "-", " ") as disease
                            MATCH (n:Disease)-[:REFERS_TO]->(resource)<-[:NAME_REFERS_TO]-(diseaseName)
                            WHERE NOT EXISTS ((m)-[:REGULATES]->(n))
                            WITH m, n, disease, diseaseName, [x IN split(replace(replace(replace(replace(toLower(diseaseName.label), ",", ""), "/", " "), "neoplasms", "carcinoma"), "-", " "), " ") | trim(x)] as terms, [x in split(disease, " ") | trim(x)] as splitDisease
                            WITH m, n, disease, diseaseName, apoc.coll.intersection(terms, splitDisease) as intersect, splitDisease, apoc.text.jaroWinklerDistance(disease, toLower(diseaseName.label)) as similarity
                            WHERE toLower(diseaseName.label) = disease OR (size(splitDisease) = size(terms) AND size(intersect) = size(terms) and similarity > 0.9)
                            WITH m, n, disease
                            SET n:DiseaseDbDEMC, n.name_in_db_demc = $disease
                            MERGE (m)-[r:REGULATES {regulated: $regulated}]->(n)
                            SET r.source = 'dbDEMC', r.type = "indirect_approximate"
                            return id(m) as id, id(n)
                        """

            tx = session.begin_transaction()
            items_processed = 0
            with open(file, 'r+') as in_file:
                reader = csv.reader(in_file, delimiter='\t')
                for row in reader:
                    if len(row) < 2:
                        print(row)
                        continue
                    disease = row[5].lower() if row[5].__len__() > 1 else row[4].lower()

                    if row[0].lower() == row[1].lower():
                        continue

                    parameters = {
                        "name": row[1].lower(),
                        "disease": disease,
                        "regulated": row[15]
                    }
                    tx.run(indirect_approximate_match_query, parameters)
                    items_processed += 1
                    if items_processed % 1000 == 0:
                        tx.commit()
                        print(items_processed, "items processed")
                        tx = session.begin_transaction()

            tx.commit()
            print(items_processed, "items processed")

        # print(names_values)


#   https://www.biosino.org/dbDEMC/download/MiRExpAll

if __name__ == '__main__':
    importing = BioImporter(argv=sys.argv[1:])
    base_path = importing.source_dataset_path

    if not base_path:
        print("source path directory is mandatory. Setting it to default.")
        base_path = "../../dataset/hmdd/dbDEMC/"

    base_path = Path(base_path)

    if not base_path.is_dir():
        print(base_path, "isn't a directory")
        sys.exit(1)

    miRNA_dat = base_path / "miRExpAll.txt"

    if not miRNA_dat.is_file():
        print(miRNA_dat, "doesn't exist in ", base_path)
        sys.exit(1)

    importing.import_exact_match(miRNA_dat)
    importing.close()
