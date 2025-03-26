import sys
import time
from pathlib import Path

import math
import spacy
from tqdm.auto import tqdm

from util.base_importer import BaseImporter
from step2__enrich_organizations import query_wikidata_entity


class OwnershipEnricher(BaseImporter):
    def __init__(self, argv):
        super().__init__(command=__file__, argv=argv)
        self._database = "news"
        self.cache_folder = None

        self.QUERY_GET_INPUTS = """MATCH (e:Organization)
        WHERE e.wikidata_id IS NOT NULL
        RETURN id(e) AS id, e.name AS name, e.wikidata_id AS wikidata_id
        """

        self.QUERY_STORE_RESULTS = """UNWIND $inputs AS input
        MATCH (n)
        WHERE id(n) = input.id
    
        FOREACH(entity IN input.owned_by_org |
            MERGE (e:Organization {name: entity})
            SET e:Owner
            MERGE (n)-[:OWNED_BY_ORG]->(e)
        )
    
        FOREACH(entity IN input.owned_by_per |
            MERGE (e:Person {name: entity})
            SET e:Owner
            MERGE (n)-[:OWNED_BY_PER]->(e)
        )
        """

    def get_onwers_with_type(self, entity):
        SPARQL = """
        SELECT ?owned_by ?owned_byLabel ?owner_instance_of 
        WHERE {
          wd:%s wdt:P127 ?owned_by .
          ?owned_by wdt:P31 ?owner_instance_of .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }"""
        results = query_wikidata_entity(SPARQL, entity["wikidata_id"], self.cache_folder)
        owned_by_org = []
        owned_by_per = []
        for r in results:
            if r["owner_instance_of"] == "https://www.wikidata.org/entity/Q5":  # owner is person
                owned_by_org.append(r["owned_byLabel"]["value"])
            else:
                owned_by_per.append(r["owned_byLabel"]["value"])
        if len(results) > 0:
            return {'id': entity["id"],
                    'owned_by_org': owned_by_org,
                    'owned_by_per': owned_by_per}
        pass

    @staticmethod
    def get_owners(entities: list):
        """
        Retrieve wikidata IDs of owners for each entity
        :param entities: list of entities represented as dictionaries
        :return: dictionary {<entity_node_id>: list(owners)}
        """
        ents = " ".join([f"(wd:{x['wikidata_id']})" for x in entities])
        SPARQL = """
        SELECT ?org ?owned_by
        WHERE { VALUES(?org) {%s} .
                ?org wdt:P127 ?owned_by .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """

        results = query_wikidata_entity(SPARQL, ents)
        final_res = dict()
        for r in results:
            org_id = r['org']['value'].split("/")[-1].strip()
            node_id = [x['id'] for x in entities if x['wikidata_id'] == org_id][0]
            if node_id not in final_res:
                final_res[node_id] = list()
            final_res[node_id].append(r['owned_by']['value'].split("/")[-1].strip())

        return final_res

    @staticmethod
    def disambiguate_owners(entities: dict):
        # verify owner entity type (Organization or Person)
        ents = " ".join([f"(wd:{owner})" for x in entities.values() for owner in x])
        SPARQL = """
        SELECT ?ent ?entLabel ?instance_of
        WHERE {
            VALUES(?ent) {%s} .
            OPTIONAL {?ent wdt:P31 ?instance_of . }
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """

        results = query_wikidata_entity(SPARQL, ents)

        dict_owners = dict()
        for r in results:
            ent = r['ent']['value'].split("/")[-1]
            inst = r['instance_of']['value'].split("/")[-1] if 'instance_of' in r else None
            dict_owners[ent] = {'name': r['entLabel']['value'].strip(),
                                'type': "person" if inst == "Q5" else "organization"
                                }

        final_res = list()
        for en, owners in entities.items():
            final_res.append({'id': en, 'owned_by_org': list(), 'owned_by_per': list()})
            for ow in owners:
                label = 'owned_by_per' if dict_owners[ow]['type'] == "person" else 'owned_by_org'
                final_res[-1][label].append(dict_owners[ow]['name'])

        return final_res

    def run(self):
        # Get entities to enrich
        with self._driver.session(database=self._database) as session:
            entities = session.run(self.QUERY_GET_INPUTS).data()
            print(f"Retrieved {len(entities)} entities")
            results = []
            for e in tqdm(entities[:], desc="acquiring owners"):
                res = self.get_onwers_with_type(entity=e)
                if res is not None:
                    results.append(res)
            session.run(self.QUERY_STORE_RESULTS, inputs=results)
            return

            # run in batches
            NUM_BATCHES = 20
            batches = spacy.util.minibatch(entities, NUM_BATCHES)
            print(f"Entities split into {math.ceil(1.0 * len(entities) / NUM_BATCHES)} batches")
            for idx, batch in enumerate(batches):
                print(f"Processing batch {idx}")
                owners = self.get_owners(entities)
                time.sleep(1)
                results = self.disambiguate_owners(owners)
                time.sleep(1)
                session.run(self.QUERY_STORE_RESULTS, inputs=results)


if __name__ == "__main__":
    enricher = OwnershipEnricher(argv=sys.argv[1:])

    # set up a cache folder for wikidata responses
    enricher.cache_folder = Path("../../data/cache_owners")
    enricher.cache_folder.mkdir(exist_ok=True)

    enricher.run()
