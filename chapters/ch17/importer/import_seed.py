import sys

from util.graphdb_base import GraphDBBase

if __name__ == '__main__':
    importing = GraphDBBase(argv=sys.argv[1:])

    with importing._driver.session() as session:
        session.run(f"""
            CREATE DATABASE {importing.database} OPTIONS {{ existingData: "use", seedUri: "https://downloads.graphaware.com/neo4j-db-seeds/chicago.ila-2025-03-14T17-38-31.backup"}}
        """)
    importing.close()
