import sys

from disease_resolver import Resolver as DiseaseResolver
from util.base_importer import BaseImporter


class Reconciliator(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "hmdd2.0"
        self.resolver = DiseaseResolver()

    def compute_statistics(self):
        with self._driver.session(database=self._database) as session:

            session.run("""
            call gds.graph.project.cypher(
                "oneHop",
                "MATCH (n:MiRNA) return id(n) as id",
                "MATCH (a:MiRNA)-[:REGULATES|RELATED_TO]->
                       ()<-[:REGULATES|RELATED_TO]-(b:MiRNA) 
                WHERE id(a)<id(b) 
                RETURN distinct 
                       id(a) as source, 
                       id(b) as target")
            """)

            d = (session.run("""
            call gds.alpha.allShortestPaths.stream('oneHop',{}) 
            yield distance 
            with distance 
            return distinct distance,count(distance)as count""").data())
            print(d)

            session.run("""
            call gds.graph.project.cypher(
                "oneHopNormalized",
                "MATCH (n:MiRNA) return id(n) as id",
                "MATCH p1=(a:MiRNA)-[:REGULATES|RELATED_TO]->()-[:REPRESENTS]->(d)
                 MATCH p2=(d)<-[:REPRESENTS]-()<-[:REGULATES|RELATED_TO]-(b:MiRNA) 
                 WHERE id(a)<id(b) 
                 RETURN distinct 
                       id(a) as source, 
                       id(b) as target")
            """)

            d = (session.run("""
            call gds.alpha.allShortestPaths.stream('oneHopNormalized',{}) 
            yield distance 
            with distance 
            return distinct distance,count(distance)as count""").data())
            print(d)



    def get_normalized_diseases(self):
        with self._driver.session(database=self._database) as session:
            # fetch diseases
            diseases_data = session.run("match (d:Disease) return id(d) as id, d.name as name").data()

        # split diseases text and id
        diseases_text = [d["name"] for d in diseases_data]
        disease_ids = [d["id"] for d in diseases_data]

        # convert items like `leukemia, lymphocytic, chronic, b-cell` into  "b-cell chronic lymphocytic leukemia"
        diseases_text = [" ".join(i for i in reversed(d.split(","))).strip() for d in diseases_text]

        # parse disease texts using scispacy model
        diseases_items = [self.resolver.nlp(disease) for disease in diseases_text]

        # normalize diseases
        disease_normalized = [self.resolver.normalize(item) for item in diseases_items]

        # convert normalized diseases into dictionary
        diseases = [{
            "source_id": disease_id,
            "name": disease_name,
            "umnls_id": disease_UMNLS_ID}
            for disease_id, (disease_name, disease_UMNLS_ID)
            in zip(disease_ids, disease_normalized)]

        return diseases

    def import_normalized_diseases(self):
        query = """
            UNWIND $batch as item
            MATCH (d:Disease)
            WHERE id(d) = item.source_id
            MERGE (nd:NormalizedDisease {name:item.name})
            SET nd.umnls_id = item.umnls_id
            MERGE (d)-[:REPRESENTS]->(nd)
        """
        diseases = self.get_normalized_diseases()
        self.batch_store(query, iter(diseases), size=len(diseases), strategy="aggregate")


def main():
    reconciliatior = Reconciliator(argv=sys.argv[1:])

    reconciliatior.import_normalized_diseases()

    reconciliatior.close()


if __name__ == '__main__':
    main()
