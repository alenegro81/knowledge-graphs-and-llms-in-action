import sys

from util.graphdb_base import GraphDBBase

if __name__ == '__main__':
    importing = GraphDBBase(argv=sys.argv[1:])
    print("Make sure the following line existsin the neo4j.conf file")
    print("dbms.databases.seed_from_uri_providers=URLConnectionSeedProvider")

    with importing._driver.session() as session:
        # import PPI database
        print("importing PPI")
        ppi_uri = "https://downloads.graphaware.com/neo4j-db-seeds/ppi-5.26.4-2025-03-22T08-30-33.backup"
        result = session.run(f"""
            CREATE DATABASE ppi IF NOT EXISTS OPTIONS {{ existingData: "use", seedUri: "{ppi_uri}" }}
        """)
        print("system updates:", result.consume().counters.system_updates)
    importing.close()
